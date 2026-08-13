from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from chronophoto.processing.alignment import FrameAligner, align_sequence
from chronophoto.processing.backends import RenderTelemetry, probe_video_pipeline
from chronophoto.processing.compositor import (
    CLEAN_PLATE_MAX_FRAMES,
    ComposeCache,
    ComposeSettings,
    build_background,
    compose_sequence,
    create_motion_mask,
)
from chronophoto.processing.effects import apply_background_effect_tracks, apply_effect_tracks
from chronophoto.processing.exports import _IncrementalAlphaCompositor, _iter_transparent_poses
from chronophoto.processing.nvidia_backend import BUNDLED_NVIDIA_PIPELINE
from chronophoto.processing.parallel import export_render_workers, ordered_parallel_map
from chronophoto.processing.pipeline import RenderPlan, RenderStage, StageArtifact
from chronophoto.processing.sources import iter_video_frames, load_video_sequence

ImageArray = np.ndarray
ProgressCallback = Callable[[int, str], None]


@dataclass(slots=True, frozen=True)
class _VideoEncoder:
    codec: str
    label: str
    options: dict[str, str]
    hardware: bool = False


@dataclass(slots=True, frozen=True)
class _TrailPoseLayer:
    source_index: int
    frame: ImageArray
    mask: NDArray[np.float32]
    bounds: tuple[int, int, int, int] | None
    regions: tuple[tuple[int, int, int, int], ...]


@dataclass(slots=True, frozen=True)
class _StreamingSetup:
    background: ImageArray | None
    reference: ImageArray | None
    width: int
    height: int
    expected_frames: int


class _IncrementalMotionCompositor:
    """Maintain one exact float32 normal-alpha trail across a moving window."""

    def __init__(self, background: ImageArray, overlap: str) -> None:
        self.background = background.astype(np.float32)
        self.overlap = overlap
        self.combined: NDArray[np.float32] | None = None
        self.previous_indices: tuple[int, ...] = ()
        self.known: dict[int, _TrailPoseLayer] = {}

    def compose(self, current: Sequence[_TrailPoseLayer]) -> ImageArray:
        if not current:
            raise ValueError("a motion trail frame requires at least one pose")
        current_indices = tuple(layer.source_index for layer in current)
        current_set = set(current_indices)
        previous_set = set(self.previous_indices)
        self.known.update((layer.source_index, layer) for layer in current)

        if self.combined is None:
            self.combined = self.background.copy()
            stack = current if self.overlap == "newest" else tuple(reversed(current))
            for layer in stack:
                _blend_layer_regions(self.combined, layer, layer.regions)
        else:
            added = [index for index in current_indices if index not in previous_set]
            removed = [index for index in self.previous_indices if index not in current_set]
            if self.overlap == "newest":
                retained = [index for index in current_indices if index not in added]
                _recompose_motion_regions(
                    self.combined,
                    self.background,
                    _merge_motion_regions(
                        [box for index in removed for box in self.known[index].regions]
                    ),
                    retained,
                    self.known,
                )
                for index in added:
                    layer = self.known[index]
                    _blend_layer_regions(self.combined, layer, layer.regions)
            else:
                dirty = [box for index in (*removed, *added) for box in self.known[index].regions]
                _recompose_motion_regions(
                    self.combined,
                    self.background,
                    _merge_motion_regions(dirty),
                    list(reversed(current_indices)),
                    self.known,
                )

        output = self.combined
        # The legacy single-frame path duplicates its only input to satisfy the
        # static compositor. Match that first-frame alpha without polluting the
        # rolling state used by frame two.
        if not self.previous_indices and len(current) == 1:
            output = self.combined.copy()
            _blend_layer_regions(output, current[0], current[0].regions)
        pixels = np.clip(output, 0.0, 255.0).astype(np.uint8)
        self.previous_indices = current_indices
        for index in tuple(self.known):
            if index not in current_set:
                del self.known[index]
        return pixels


def _blend_layer_regions(
    target: NDArray[np.float32],
    layer: _TrailPoseLayer,
    regions: Sequence[tuple[int, int, int, int]],
) -> None:
    for left, top, right, bottom in regions:
        alpha = np.clip(layer.mask[top:bottom, left:right], 0.0, 1.0)[..., None]
        source = layer.frame[top:bottom, left:right].astype(np.float32)
        destination = target[top:bottom, left:right]
        destination[:] = source * alpha + destination * (1.0 - alpha)


def _recompose_motion_regions(
    target: NDArray[np.float32],
    background: NDArray[np.float32],
    regions: Sequence[tuple[int, int, int, int]],
    stack_order: Sequence[int],
    layers: dict[int, _TrailPoseLayer],
) -> None:
    for left, top, right, bottom in regions:
        target[top:bottom, left:right] = background[top:bottom, left:right]
        for index in stack_order:
            layer = layers[index]
            if layer.bounds is None:
                continue
            layer_left, layer_top, layer_right, layer_bottom = layer.bounds
            intersection = (
                max(left, layer_left),
                max(top, layer_top),
                min(right, layer_right),
                min(bottom, layer_bottom),
            )
            if intersection[0] < intersection[2] and intersection[1] < intersection[3]:
                _blend_layer_regions(target, layer, (intersection,))


def _motion_mask_regions(
    mask: NDArray[np.float32],
    *,
    tile_size: int = 256,
) -> tuple[tuple[int, int, int, int] | None, tuple[tuple[int, int, int, int], ...]]:
    points = cv2.findNonZero((mask > 0.001).astype(np.uint8))
    if points is None:
        return None, ()
    x, y, width, height = cv2.boundingRect(points)
    bounds = (x, y, x + width, y + height)
    regions: list[tuple[int, int, int, int]] = []
    for top in range(y, y + height, tile_size):
        bottom = min(y + height, top + tile_size)
        run_left: int | None = None
        for left in range(x, x + width, tile_size):
            right = min(x + width, left + tile_size)
            if np.any(mask[top:bottom, left:right] != 0.0):
                if run_left is None:
                    run_left = left
            elif run_left is not None:
                regions.append((run_left, top, left, bottom))
                run_left = None
        if run_left is not None:
            regions.append((run_left, top, x + width, bottom))
    return bounds, tuple(_merge_motion_regions(regions))


def _merge_motion_regions(
    boxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
        left, top, right, bottom = candidate
        position = 0
        while position < len(merged):
            other = merged[position]
            if right < other[0] or other[2] < left or bottom < other[1] or other[3] < top:
                position += 1
                continue
            merged_left = min(left, other[0])
            merged_top = min(top, other[1])
            merged_right = max(right, other[2])
            merged_bottom = max(bottom, other[3])
            overlap_width = max(0, min(right, other[2]) - max(left, other[0]))
            overlap_height = max(0, min(bottom, other[3]) - max(top, other[1]))
            union_area = (
                (right - left) * (bottom - top)
                + ((other[2] - other[0]) * (other[3] - other[1]))
                - overlap_width * overlap_height
            )
            merged_area = (merged_right - merged_left) * (merged_bottom - merged_top)
            if merged_area != union_area:
                position += 1
                continue
            left, top, right, bottom = (
                merged_left,
                merged_top,
                merged_right,
                merged_bottom,
            )
            merged.pop(position)
            position = 0
        merged.append((left, top, right, bottom))
    return merged


def _can_reuse_motion_layers(settings: ComposeSettings) -> bool:
    if settings.smear_style != "none":
        return False
    for track in settings.trail_effect_tracks:
        if not track.enabled:
            continue
        if track.kind == "blend_mode" and track.option != "normal":
            return False
        if track.kind != "blend_mode" and track.timing_basis == "trail":
            return False
    return True


def can_stream_motion_trail(settings: ComposeSettings) -> bool:
    """Return whether bounded-memory rendering preserves the selected effects."""

    return _can_reuse_motion_layers(settings)


def _iter_motion_pose_layers(
    frames: Sequence[ImageArray],
    cache: ComposeCache,
    settings: ComposeSettings,
    effect_progress: Sequence[float],
    *,
    effect_pixel_scale: float,
) -> Iterator[_TrailPoseLayer]:
    pixel_effects = tuple(
        track
        for track in settings.trail_effect_tracks
        if track.enabled and track.kind != "blend_mode"
    )

    def prepare(item: tuple[int, ImageArray, NDArray[np.uint8], float]) -> _TrailPoseLayer:
        index, frame, cached_mask, position = item
        mask = cached_mask.astype(np.float32) / 255.0
        effected_frame = frame
        effected_mask = mask
        if pixel_effects:
            effected_frame, effected_mask = apply_effect_tracks(
                frame,
                mask,
                position,
                pixel_effects,
                pixel_scale=effect_pixel_scale,
            )
        bounds, regions = _motion_mask_regions(effected_mask)
        return _TrailPoseLayer(index, effected_frame, effected_mask, bounds, regions)

    height, width = frames[0].shape[:2]
    items = (
        (index, frame, cache.masks[index], float(effect_progress[index]))
        for index, frame in enumerate(frames)
    )
    yield from ordered_parallel_map(
        prepare,
        items,
        workers=export_render_workers(width, height),
    )


def _iter_reused_motion_trail_frames(
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    effect_progress: Sequence[float],
    effect_pixel_scale: float,
    frame_indices: Sequence[int],
    progress: ProgressCallback | None = None,
) -> Iterator[tuple[int, ImageArray]]:
    wanted = set(frame_indices)
    maximum_index = max(frame_indices)
    background = apply_background_effect_tracks(
        cache.background,
        settings.background_effect_tracks,
        pixel_scale=effect_pixel_scale,
    )
    compositor = _IncrementalMotionCompositor(background, settings.overlap)
    active_layers: dict[int, _TrailPoseLayer] = {}
    total = len(frame_indices)
    completed = 0
    layers = _iter_motion_pose_layers(
        frames,
        cache,
        settings,
        effect_progress,
        effect_pixel_scale=effect_pixel_scale,
    )
    for frame_index, layer in enumerate(layers):
        if frame_index > maximum_index:
            break
        active_layers[frame_index] = layer
        window = motion_trail_window(timestamps, frame_index, trail_duration)
        pixels = compositor.compose([active_layers[index] for index in window])
        for index in tuple(active_layers):
            if index < window[0]:
                del active_layers[index]
        if frame_index not in wanted:
            continue
        completed += 1
        if progress is not None:
            progress(
                round((completed / total) * 100),
                f"Rendering cached trail frame {completed} of {total}",
            )
        yield frame_index, pixels


def _iter_cached_alpha_trail_frames(
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    effect_progress: Sequence[float],
    effect_pixel_scale: float,
    frame_indices: Sequence[int],
    progress: ProgressCallback | None = None,
) -> Iterator[tuple[int, ImageArray]]:
    """Render full-resolution 8-bit frames from reusable alpha layers."""

    wanted = set(frame_indices)
    maximum_index = max(frame_indices)
    background = Image.fromarray(
        apply_background_effect_tracks(
            cache.background,
            settings.background_effect_tracks,
            pixel_scale=effect_pixel_scale,
        )
    ).convert("RGBA")
    compositor = _IncrementalAlphaCompositor(settings.overlap)
    active_poses: dict[int, Image.Image] = {}
    total = len(frame_indices)
    completed = 0
    pose_pixels = _iter_transparent_poses(
        list(frames),
        cache.masks,
        settings,
        [float(value) for value in effect_progress],
        pixel_scale=effect_pixel_scale,
    )
    for frame_index, pixels in enumerate(pose_pixels):
        if frame_index > maximum_index:
            break
        active_poses[frame_index] = Image.fromarray(pixels)
        window = motion_trail_window(timestamps, frame_index, trail_duration)
        overlay = compositor.compose([(index, active_poses[index]) for index in window])
        if frame_index == 0 and len(window) == 1:
            overlay.alpha_composite(active_poses[0])
        for index in tuple(active_poses):
            if index < window[0]:
                del active_poses[index]
        if frame_index not in wanted:
            continue
        completed += 1
        result = np.asarray(Image.alpha_composite(background, overlay).convert("RGB")).copy()
        if progress is not None:
            progress(
                round((completed / total) * 100),
                f"Rendering cached trail frame {completed} of {total}",
            )
        yield frame_index, result


def motion_trail_window(
    timestamps: Sequence[float],
    frame_index: int,
    trail_duration: float,
) -> tuple[int, ...]:
    """Return the causal timestamp window ending at ``frame_index``."""

    _validate_timestamps(timestamps)
    if not 0 <= frame_index < len(timestamps):
        raise IndexError("frame index is outside the timestamp sequence")
    if not math.isfinite(trail_duration) or trail_duration < 0:
        raise ValueError("trail duration must be a finite non-negative value")

    cutoff = timestamps[frame_index] - trail_duration
    first = bisect_left(timestamps, cutoff, 0, frame_index + 1)
    return tuple(range(first, frame_index + 1))


def compose_motion_trail_frame(
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    frame_index: int,
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    effect_progress: Sequence[float] | None = None,
    effect_pixel_scale: float = 1.0,
) -> ImageArray:
    """Render one frame from the current subject and its recent history only."""

    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have equal lengths")
    if len(cache.masks) != len(frames):
        raise ValueError("cached masks must match the video frames")
    indices = list(motion_trail_window(timestamps, frame_index, trail_duration))
    if effect_progress is None:
        positions = _normalized_progress(timestamps)
    else:
        positions = [float(value) for value in effect_progress]
        if len(positions) != len(frames):
            raise ValueError("effect progress must match the video frames")

    # The static compositor requires two chronological inputs. Duplicating the
    # current frame keeps the first/zero-duration result causal and visually unchanged.
    if len(indices) == 1:
        indices.append(indices[0])
    selected_frames = [frames[index] for index in indices]
    selected_cache = cache.select(indices)
    selected_progress = [positions[index] for index in indices]
    window_start = timestamps[indices[0]]
    window_span = timestamps[indices[-1]] - window_start
    selected_trail_progress = (
        [1.0] * len(indices)
        if window_span <= 0.0
        else [(timestamps[index] - window_start) / window_span for index in indices]
    )
    result, _ = compose_sequence(
        selected_frames,
        settings,
        return_masks=False,
        cache=selected_cache,
        effect_progress=selected_progress,
        trail_progress=selected_trail_progress,
        effect_pixel_scale=effect_pixel_scale,
        frame_contiguous=True,
    )
    return result


def render_motion_trail_sequence(
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    effect_progress: Sequence[float] | None = None,
    effect_pixel_scale: float = 1.0,
    frame_indices: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
) -> list[ImageArray]:
    """Render an in-memory sequence for interactive trail playback."""

    indices = list(range(len(frames))) if frame_indices is None else list(frame_indices)
    if any(not 0 <= index < len(frames) for index in indices):
        raise IndexError("rendered frame index is outside the video sequence")
    if not indices:
        return []
    if effect_progress is None:
        positions = _normalized_progress(timestamps)
    else:
        positions = [float(value) for value in effect_progress]
        if len(positions) != len(frames):
            raise ValueError("effect progress must match the video frames")
    if _can_reuse_motion_layers(settings) and indices == sorted(set(indices)):
        return [
            pixels
            for _index, pixels in _iter_cached_alpha_trail_frames(
                frames,
                timestamps,
                trail_duration,
                settings,
                cache,
                effect_progress=positions,
                effect_pixel_scale=effect_pixel_scale,
                frame_indices=indices,
                progress=progress,
            )
        ]
    height, width = frames[0].shape[:2]
    workers = export_render_workers(width, height)

    def render(index: int) -> ImageArray:
        return compose_motion_trail_frame(
            frames,
            timestamps,
            index,
            trail_duration,
            settings,
            cache,
            effect_progress=positions,
            effect_pixel_scale=effect_pixel_scale,
        )

    rendered: list[ImageArray] = []
    for position, pixels in enumerate(ordered_parallel_map(render, indices, workers=workers)):
        rendered.append(pixels)
        if progress is not None:
            progress(
                round(((position + 1) / len(indices)) * 100),
                f"Rendering trail preview {position + 1} of {len(indices)} · "
                f"{workers} parallel worker{'s' if workers != 1 else ''}",
            )
    return rendered


def write_motion_trail_video(
    output_path: str | Path,
    source_path: str | Path,
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    start: float,
    end: float,
    frame_rate: float,
    effect_progress: Sequence[float] | None = None,
    effect_pixel_scale: float = 1.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Render an H.264 MP4 and retain compatible source audio in the selected range."""

    target = Path(output_path)
    if target.suffix.casefold() != ".mp4":
        target = target.with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    effective_frame_rate = _resolve_frame_rate(frame_rate, timestamps)
    _validate_video_inputs(frames, timestamps, start, end, effective_frame_rate)

    temporary_paths: list[Path] = []
    silent_path = _temporary_mp4(target.parent, "chronophoto-video-")
    temporary_paths.append(silent_path)
    try:
        _write_silent_video(
            silent_path,
            frames,
            timestamps,
            trail_duration,
            settings,
            cache,
            frame_rate=effective_frame_rate,
            effect_progress=effect_progress,
            effect_pixel_scale=effect_pixel_scale,
            progress=progress,
        )
        audio_path = _temporary_mp4(target.parent, "chronophoto-audio-")
        temporary_paths.append(audio_path)
        if _mux_selected_audio(
            silent_path,
            Path(source_path),
            audio_path,
            start,
            end,
            progress,
        ):
            final_path = audio_path
            if progress is not None:
                progress(99, "Finalizing video and source audio")
        else:
            if audio_path.exists():
                audio_path.unlink()
            temporary_paths.remove(audio_path)
            final_path = silent_path
            if progress is not None:
                progress(99, "Finalizing video")
        os.replace(final_path, target)
        temporary_paths.remove(final_path)
        if progress is not None:
            progress(100, "Motion-trail video ready")
        return target
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def write_streaming_motion_trail_video(
    output_path: str | Path,
    source_path: str | Path,
    trail_duration: float,
    settings: ComposeSettings,
    *,
    start: float,
    end: float,
    frame_rate: float,
    alignment: str = "off",
    effect_pixel_scale: float = 1.0,
    progress: ProgressCallback | None = None,
) -> tuple[Path, RenderTelemetry]:
    """Render a common trail path with clip-length-independent frame memory."""

    if start < 0 or end <= start:
        raise ValueError("video export end must be later than start")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("streaming video export requires a positive frame rate")
    if not _can_reuse_motion_layers(settings):
        raise ValueError("selected effects require the compatibility renderer")
    target = Path(output_path).with_suffix(".mp4")
    source = Path(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    silent_path = _temporary_mp4(target.parent, "chronophoto-stream-video-")
    audio_path = _temporary_mp4(target.parent, "chronophoto-stream-audio-")
    temporary_paths = [silent_path, audio_path]
    capabilities = probe_video_pipeline(source)
    expected_frames = max(2, round((end - start) * frame_rate))

    def sample_stage(_dependencies, stage_progress):  # type: ignore[no-untyped-def]
        with av.open(str(source)) as container:
            video_stream = container.streams.video[0]
            source_width = int(video_stream.codec_context.width)
            source_height = int(video_stream.codec_context.height)
        gpu_supported = BUNDLED_NVIDIA_PIPELINE.supports(
            settings, alignment, source_width, source_height
        )
        gpu_available, _reason = BUNDLED_NVIDIA_PIPELINE.probe()
        if gpu_supported and gpu_available:
            return StageArtifact(
                "streaming_setup",
                _StreamingSetup(None, None, source_width, source_height, expected_frames),
                ("nvidia_streaming_setup", source, start, end),
                metadata={"gpu": True},
            )
        sample_count = min(CLEAN_PLATE_MAX_FRAMES, expected_frames)
        sequence = load_video_sequence(
            source,
            start,
            end,
            sample_count,
            progress=stage_progress,
        )
        aligned_samples = align_sequence(sequence.frames, alignment)
        background = build_background(aligned_samples, settings.background)
        height, width = background.shape[:2]
        return StageArtifact(
            "streaming_setup",
            _StreamingSetup(
                background,
                sequence.frames[0],
                width,
                height,
                expected_frames,
            ),
            ("streaming_setup", source, start, end, alignment, settings.background),
            estimated_bytes=sum(frame.nbytes for frame in aligned_samples) + background.nbytes,
            metadata={"sample_count": sample_count},
        )

    def encode_stage(dependencies, stage_progress):  # type: ignore[no-untyped-def]
        setup = dependencies["clean_plate"].value
        if not isinstance(setup, _StreamingSetup):
            raise TypeError("clean plate stage returned an invalid setup")
        if BUNDLED_NVIDIA_PIPELINE.supports(settings, alignment, setup.width, setup.height):
            available, reason = BUNDLED_NVIDIA_PIPELINE.probe()
            if available:
                try:
                    telemetry = BUNDLED_NVIDIA_PIPELINE.render_silent(
                        silent_path,
                        source,
                        settings,
                        start=start,
                        end=end,
                        frame_rate=frame_rate,
                        trail_duration=trail_duration,
                        progress=stage_progress,
                    )
                    return StageArtifact(
                        "encoded_video",
                        (silent_path, telemetry),
                        (),
                        telemetry.estimated_peak_bytes,
                        {"encoder": telemetry.encoder},
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    if exc.__class__.__name__ == "TaskCancelled":
                        raise
                    silent_path.unlink(missing_ok=True)
                    if stage_progress is not None:
                        stage_progress(0, f"NVIDIA GPU unavailable; using CPU renderer: {exc}")
            elif stage_progress is not None:
                stage_progress(0, f"Using CPU renderer: {reason}")
        if setup.background is None or setup.reference is None:
            sample_count = min(CLEAN_PLATE_MAX_FRAMES, expected_frames)
            sequence = load_video_sequence(source, start, end, sample_count)
            aligned_samples = align_sequence(sequence.frames, alignment)
            background = build_background(aligned_samples, settings.background)
            setup = _StreamingSetup(
                background,
                sequence.frames[0],
                background.shape[1],
                background.shape[0],
                expected_frames,
            )
        candidates = _video_encoder_candidates(setup.width, setup.height)
        last_error: Exception | None = None
        for position, encoder in enumerate(candidates):
            try:
                telemetry = _write_streaming_silent_video(
                    silent_path,
                    source,
                    setup,
                    trail_duration,
                    settings,
                    start=start,
                    end=end,
                    frame_rate=frame_rate,
                    alignment=alignment,
                    effect_pixel_scale=effect_pixel_scale,
                    encoder=encoder,
                    decoder_label=capabilities.decoder,
                    progress=stage_progress,
                )
                return StageArtifact(
                    "encoded_video",
                    (silent_path, telemetry),
                    (),
                    telemetry.estimated_peak_bytes,
                    {"encoder": telemetry.encoder},
                )
            except (av.error.FFmpegError, OSError) as exc:
                last_error = exc
                silent_path.unlink(missing_ok=True)
                if position + 1 < len(candidates) and stage_progress is not None:
                    stage_progress(0, f"{encoder.label} unavailable; retrying with CPU H.264")
        assert last_error is not None
        raise last_error

    def audio_stage(dependencies, stage_progress):  # type: ignore[no-untyped-def]
        encoded = dependencies["encode"].value
        silent, telemetry = encoded
        if _mux_selected_audio(silent, source, audio_path, start, end, stage_progress):
            final_path = audio_path
        else:
            audio_path.unlink(missing_ok=True)
            final_path = silent
        os.replace(final_path, target)
        if final_path in temporary_paths:
            temporary_paths.remove(final_path)
        return StageArtifact(
            "final_video",
            (target, telemetry),
            (),
            telemetry.estimated_peak_bytes,
            {"audio": final_path == audio_path},
        )

    plan = RenderPlan(
        (
            RenderStage(
                "clean_plate",
                "streaming_setup",
                sample_stage,
                mode="materialized",
                cache_key=("clean_plate", source, start, end, alignment, settings.background),
            ),
            RenderStage(
                "encode",
                "encoded_video",
                encode_stage,
                dependencies=("clean_plate",),
                mode="streaming",
            ),
            RenderStage(
                "audio",
                "final_video",
                audio_stage,
                dependencies=("encode",),
                mode="streaming",
            ),
        )
    )
    try:
        execution = plan.execute(progress=progress)
        result = execution.artifacts["audio"].value
        if progress is not None:
            progress(100, "Motion-trail video ready")
        return result
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _write_streaming_silent_video(
    path: Path,
    source: Path,
    setup: _StreamingSetup,
    trail_duration: float,
    settings: ComposeSettings,
    *,
    start: float,
    end: float,
    frame_rate: float,
    alignment: str,
    effect_pixel_scale: float,
    encoder: _VideoEncoder,
    decoder_label: str,
    progress: ProgressCallback | None,
) -> RenderTelemetry:
    if setup.background is None or setup.reference is None:
        raise ValueError("CPU streaming renderer requires materialized clean-plate frames")
    rate = Fraction(frame_rate).limit_denominator(1001)
    time_base = Fraction(1, 90_000)
    aligner = FrameAligner(setup.reference, alignment)
    background_blur = cv2.GaussianBlur(setup.background, (5, 5), 0)
    processed_background = Image.fromarray(
        apply_background_effect_tracks(
            setup.background,
            settings.background_effect_tracks,
            pixel_scale=effect_pixel_scale,
        )
    ).convert("RGBA")
    pixel_effects = tuple(
        track
        for track in settings.trail_effect_tracks
        if track.enabled and track.kind != "blend_mode"
    )
    compositor = _IncrementalAlphaCompositor(settings.overlap)
    active_poses: dict[int, Image.Image] = {}
    timestamps: list[float] = []
    peak_window = 0
    completed = 0
    started = time.perf_counter()
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream(encoder.codec, rate=rate)
        stream.width = setup.width
        stream.height = setup.height
        stream.pix_fmt = "yuv420p" if setup.width % 2 == 0 and setup.height % 2 == 0 else "yuv444p"
        stream.time_base = time_base
        stream.options = encoder.options
        first_timestamp: float | None = None
        for decoded in iter_video_frames(source, start, end):
            aligned = aligner.align(decoded.pixels)
            timestamp = decoded.timestamp
            timestamps.append(timestamp)
            position = max(0.0, min(1.0, (timestamp - start) / max(end - start, 0.001)))
            mask = create_motion_mask(
                aligned,
                setup.background,
                settings,
                background_blur=background_blur,
            )
            effected_frame, effected_mask = apply_effect_tracks(
                aligned,
                mask,
                position,
                pixel_effects,
                pixel_scale=effect_pixel_scale,
            )
            alpha = np.rint(np.clip(effected_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgba = np.empty((*effected_frame.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = effected_frame
            rgba[..., 3] = alpha
            rgba[alpha == 0, :3] = 0
            active_poses[decoded.index] = Image.fromarray(rgba)
            window = motion_trail_window(timestamps, decoded.index, trail_duration)
            overlay = compositor.compose([(index, active_poses[index]) for index in window])
            if decoded.index == 0:
                overlay.alpha_composite(active_poses[0])
            for index in tuple(active_poses):
                if index < window[0]:
                    del active_poses[index]
            peak_window = max(peak_window, len(active_poses))
            pixels = np.asarray(Image.alpha_composite(processed_background, overlay).convert("RGB"))
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            if first_timestamp is None:
                first_timestamp = timestamp
            video_frame.pts = round((timestamp - first_timestamp) / float(time_base))
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                output.mux(packet)
            completed += 1
            if progress is not None:
                progress(
                    min(99, round(((timestamp - start) / max(end - start, 0.001)) * 100)),
                    f"{encoder.label} frame {completed} · {len(active_poses)} in trail window",
                )
        if completed < 2:
            raise RuntimeError("Could not decode at least two video frames in the selected range")
        for packet in stream.encode():
            output.mux(packet)
    elapsed = max(0.001, time.perf_counter() - started)
    pixels_per_frame = setup.width * setup.height
    estimated_peak = pixels_per_frame * (24 + peak_window * 4)
    render_fps = completed / elapsed
    return RenderTelemetry(
        decoder=decoder_label,
        compositor="Incremental 8-bit alpha (CPU/Pillow)",
        encoder=encoder.label,
        rendered_frames=completed,
        seconds=elapsed,
        render_fps=render_fps,
        estimated_peak_bytes=estimated_peak,
        active_window_peak=peak_window,
        zero_copy=False,
        bottleneck=(
            "CPU decode/compositing" if render_fps < frame_rate * 0.95 else "encoder or storage"
        ),
    )


def write_layered_reference_video(
    output_path: str | Path,
    background_path: str | Path,
    overlay_paths: Sequence[str | Path],
    timestamps: Sequence[float],
    *,
    frame_rate: float,
    progress: ProgressCallback | None = None,
) -> Path:
    """Encode already-rendered Resolve layers without recomposing the trail."""

    overlays = [Path(path) for path in overlay_paths]
    if not overlays or len(overlays) != len(timestamps):
        raise ValueError("overlay paths and timestamps must be non-empty and equal in length")
    _validate_timestamps(timestamps)
    effective_frame_rate = _resolve_frame_rate(frame_rate, timestamps)
    target = Path(output_path).with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(background_path) as source_background:
        background = source_background.convert("RGBA")
    width, height = background.size
    temporary = _temporary_mp4(target.parent, "chronophoto-reference-")
    try:
        candidates = _video_encoder_candidates(width, height)
        for position, encoder in enumerate(candidates):
            try:
                _write_layered_video_with_encoder(
                    temporary,
                    background,
                    overlays,
                    timestamps,
                    frame_rate=effective_frame_rate,
                    progress=progress,
                    encoder=encoder,
                )
                break
            except (av.error.FFmpegError, OSError):
                temporary.unlink(missing_ok=True)
                if position + 1 >= len(candidates):
                    raise
                if progress is not None:
                    progress(4, f"{encoder.label} unavailable · falling back to CPU encoding")
        os.replace(temporary, target)
        if progress is not None:
            progress(100, "Layered reference video ready")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _write_layered_video_with_encoder(
    path: Path,
    background: Image.Image,
    overlay_paths: Sequence[Path],
    timestamps: Sequence[float],
    *,
    frame_rate: float,
    progress: ProgressCallback | None,
    encoder: _VideoEncoder,
) -> None:
    width, height = background.size
    rate = Fraction(frame_rate).limit_denominator(1001)
    time_base = Fraction(1, 90_000)
    workers = export_render_workers(width, height) if encoder.hardware else 1
    if progress is not None:
        progress(
            4,
            f"Reusing rendered layers · {encoder.label} · {workers} parallel workers",
        )

    def render(item: tuple[int, Path]) -> tuple[int, ImageArray]:
        index, overlay_path = item
        with Image.open(overlay_path) as source_overlay:
            overlay = source_overlay.convert("RGBA")
        if overlay.size != background.size:
            raise ValueError("reference overlay dimensions must match the background")
        pixels = np.asarray(Image.alpha_composite(background, overlay).convert("RGB"))
        return index, pixels

    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream(encoder.codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
        stream.time_base = time_base
        stream.options = encoder.options
        total = len(overlay_paths)
        first_timestamp = timestamps[0]
        rendered = ordered_parallel_map(
            render,
            enumerate(overlay_paths),
            workers=workers,
        )
        for completed, (index, pixels) in enumerate(rendered, start=1):
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            video_frame.pts = round((timestamps[index] - first_timestamp) / float(time_base))
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                output.mux(packet)
            if progress is not None:
                progress(
                    5 + round((completed / total) * 94),
                    f"Compositing layers + {encoder.label} frame {completed} of {total}",
                )
        for packet in stream.encode():
            output.mux(packet)


def write_selected_audio(
    output_path: str | Path,
    source_path: str | Path,
    *,
    start: float,
    end: float,
    progress: ProgressCallback | None = None,
) -> Path | None:
    """Write the selected source-audio range as a portable PCM WAV file."""

    if start < 0 or end <= start:
        raise ValueError("audio export end must be later than start")
    target = Path(output_path).with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}-writing{target.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        with av.open(str(source_path)) as source:
            if not source.streams.audio:
                return None
            audio_input = source.streams.audio[0]
            sample_rate = audio_input.codec_context.sample_rate or 48_000
            layout = (
                audio_input.codec_context.layout.name
                if audio_input.codec_context.layout is not None
                else "stereo"
            )
            if audio_input.time_base is not None:
                seek_timestamp = max(0, int((start - 0.5) / float(audio_input.time_base)))
                source.seek(seek_timestamp, stream=audio_input, backward=True, any_frame=False)

            encoded_samples = 0
            with av.open(str(temporary), mode="w", format="wav") as output:
                audio_output = output.add_stream("pcm_s16le", rate=sample_rate)
                audio_output.layout = layout
                resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
                reached_end = False
                for decoded_index, decoded_frame in enumerate(source.decode(audio_input)):
                    if decoded_frame.pts is None:
                        continue
                    for audio_frame in resampler.resample(decoded_frame):
                        timestamp = float(audio_frame.pts * audio_frame.time_base)
                        frame_end = timestamp + audio_frame.samples / sample_rate
                        if frame_end <= start:
                            continue
                        if timestamp >= end:
                            reached_end = True
                            break
                        first_sample = max(0, round((start - timestamp) * sample_rate))
                        last_sample = min(
                            audio_frame.samples,
                            round((end - timestamp) * sample_rate),
                        )
                        if last_sample <= first_sample:
                            continue
                        samples = audio_frame.to_ndarray()[:, first_sample:last_sample]
                        trimmed = av.AudioFrame.from_ndarray(
                            samples,
                            format="fltp",
                            layout=layout,
                        )
                        trimmed.sample_rate = sample_rate
                        trimmed.pts = encoded_samples
                        trimmed.time_base = Fraction(1, sample_rate)
                        encoded_samples += trimmed.samples
                        for packet in audio_output.encode(trimmed):
                            output.mux(packet)
                        if progress is not None and decoded_index % 24 == 0:
                            progress(96, "Encoding selected source audio")
                    if reached_end:
                        break
                for packet in audio_output.encode():
                    output.mux(packet)
            if encoded_samples <= 0:
                temporary.unlink(missing_ok=True)
                return None
        os.replace(temporary, target)
        if progress is not None:
            progress(100, "Source audio ready")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _write_silent_video(
    path: Path,
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    frame_rate: float,
    effect_progress: Sequence[float] | None,
    effect_pixel_scale: float,
    progress: ProgressCallback | None,
) -> None:
    height, width = frames[0].shape[:2]
    candidates = _video_encoder_candidates(width, height)
    for position, encoder in enumerate(candidates):
        try:
            _write_silent_video_with_encoder(
                path,
                frames,
                timestamps,
                trail_duration,
                settings,
                cache,
                frame_rate=frame_rate,
                effect_progress=effect_progress,
                effect_pixel_scale=effect_pixel_scale,
                progress=progress,
                encoder=encoder,
            )
            return
        except (av.error.FFmpegError, OSError):
            path.unlink(missing_ok=True)
            if position + 1 >= len(candidates):
                raise
            if progress is not None:
                progress(4, f"{encoder.label} unavailable · falling back to CPU encoding")


def _write_silent_video_with_encoder(
    path: Path,
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    trail_duration: float,
    settings: ComposeSettings,
    cache: ComposeCache,
    *,
    frame_rate: float,
    effect_progress: Sequence[float] | None,
    effect_pixel_scale: float,
    progress: ProgressCallback | None,
    encoder: _VideoEncoder,
) -> None:
    height, width = frames[0].shape[:2]
    rate = Fraction(frame_rate).limit_denominator(1001)
    time_base = Fraction(1, 90_000)
    if effect_progress is None:
        positions = _normalized_progress(timestamps)
    else:
        positions = [float(value) for value in effect_progress]
        if len(positions) != len(frames):
            raise ValueError("effect progress must match the video frames")
    reuse_layers = _can_reuse_motion_layers(settings)
    workers = export_render_workers(width, height) if encoder.hardware or reuse_layers else 1
    if progress is not None:
        strategy = "cached motion layers" if reuse_layers else "parallel frame rendering"
        progress(
            4,
            f"{encoder.label} · {strategy} · {workers} worker{'s' if workers != 1 else ''}",
        )

    def render(index: int) -> tuple[int, ImageArray]:
        return (
            index,
            compose_motion_trail_frame(
                frames,
                timestamps,
                index,
                trail_duration,
                settings,
                cache,
                effect_progress=positions,
                effect_pixel_scale=effect_pixel_scale,
            ),
        )

    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream(encoder.codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
        stream.time_base = time_base
        stream.options = encoder.options
        first_timestamp = timestamps[0]
        total = len(frames)
        if reuse_layers:
            rendered = _iter_cached_alpha_trail_frames(
                frames,
                timestamps,
                trail_duration,
                settings,
                cache,
                effect_progress=positions,
                effect_pixel_scale=effect_pixel_scale,
                frame_indices=range(total),
            )
        else:
            rendered = ordered_parallel_map(render, range(total), workers=workers)
        for completed, (index, pixels) in enumerate(rendered, start=1):
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            video_frame.pts = round((timestamps[index] - first_timestamp) / float(time_base))
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                output.mux(packet)
            if progress is not None:
                progress(
                    5 + round((completed / total) * 89),
                    f"Rendering + {encoder.label} frame {completed} of {total}",
                )
        for packet in stream.encode():
            output.mux(packet)


def _video_encoder_candidates(width: int, height: int) -> tuple[_VideoEncoder, ...]:
    candidates: list[_VideoEncoder] = []
    if width % 2 == 0 and height % 2 == 0:
        if sys.platform == "darwin" and _codec_available("h264_videotoolbox"):
            candidates.append(
                _VideoEncoder(
                    "h264_videotoolbox",
                    "Apple VideoToolbox",
                    # FFmpeg maps allow_sw=0 to VideoToolbox's
                    # RequireHardwareAcceleratedVideoEncoder specification.
                    {"allow_sw": "0"},
                    hardware=True,
                )
            )
        if _codec_available("h264_nvenc"):
            candidates.append(
                _VideoEncoder(
                    "h264_nvenc",
                    "NVIDIA NVENC",
                    {"cq": "18", "preset": "p4", "rc": "vbr"},
                    hardware=True,
                )
            )
    candidates.append(
        _VideoEncoder(
            "libx264",
            "CPU H.264",
            {"crf": "18", "preset": "medium"},
        )
    )
    return tuple(candidates)


def _codec_available(name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
    except (av.codec.codec.UnknownCodecError, av.error.FFmpegError):
        return False
    return True


def _mux_selected_audio(
    video_path: Path,
    source_path: Path,
    output_path: Path,
    start: float,
    end: float,
    progress: ProgressCallback | None,
) -> bool:
    with av.open(str(source_path)) as source:
        if not source.streams.audio:
            return False
        audio_input = source.streams.audio[0]
        sample_rate = audio_input.codec_context.sample_rate or 48_000
        layout = (
            audio_input.codec_context.layout.name
            if audio_input.codec_context.layout is not None
            else "stereo"
        )
        if audio_input.time_base is not None:
            seek_timestamp = max(0, int((start - 0.5) / float(audio_input.time_base)))
            source.seek(seek_timestamp, stream=audio_input, backward=True, any_frame=False)
        encoded_samples = 0
        with (
            av.open(str(video_path)) as video,
            av.open(str(output_path), mode="w", options={"movflags": "+faststart"}) as output,
        ):
            video_input = video.streams.video[0]
            video_output = output.add_stream_from_template(video_input)
            audio_output = output.add_stream("aac", rate=sample_rate)
            audio_output.layout = layout
            audio_output.bit_rate = 192_000
            for packet in video.demux(video_input):
                if packet.dts is None:
                    continue
                packet.stream = video_output
                output.mux(packet)

            resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            reached_end = False
            for decoded_index, decoded_frame in enumerate(source.decode(audio_input)):
                if decoded_frame.pts is None:
                    continue
                for audio_frame in resampler.resample(decoded_frame):
                    timestamp = float(audio_frame.pts * audio_frame.time_base)
                    frame_end = timestamp + audio_frame.samples / sample_rate
                    if frame_end <= start:
                        continue
                    if timestamp >= end:
                        reached_end = True
                        break
                    first_sample = max(0, round((start - timestamp) * sample_rate))
                    last_sample = min(
                        audio_frame.samples,
                        round((end - timestamp) * sample_rate),
                    )
                    if last_sample <= first_sample:
                        continue
                    samples = audio_frame.to_ndarray()[:, first_sample:last_sample]
                    trimmed = av.AudioFrame.from_ndarray(
                        samples,
                        format="fltp",
                        layout=layout,
                    )
                    trimmed.sample_rate = sample_rate
                    trimmed.pts = encoded_samples
                    trimmed.time_base = Fraction(1, sample_rate)
                    encoded_samples += trimmed.samples
                    for packet in audio_output.encode(trimmed):
                        output.mux(packet)
                    if progress is not None and decoded_index % 24 == 0:
                        progress(96, "Encoding selected source audio")
                if reached_end:
                    break
            for packet in audio_output.encode():
                output.mux(packet)
    return encoded_samples > 0


def _validate_timestamps(timestamps: Sequence[float]) -> None:
    if not timestamps:
        raise ValueError("at least one timestamp is required")
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("timestamps must be finite")
    if any(left > right for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("timestamps must be chronological")


def _validate_video_inputs(
    frames: Sequence[ImageArray],
    timestamps: Sequence[float],
    start: float,
    end: float,
    frame_rate: float,
) -> None:
    if len(frames) < 2 or len(frames) != len(timestamps):
        raise ValueError("video export requires matching frames and timestamps")
    _validate_timestamps(timestamps)
    if start < 0 or end <= start:
        raise ValueError("video export end must be later than start")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("video export requires a positive frame rate")


def _normalized_progress(timestamps: Sequence[float]) -> list[float]:
    first = timestamps[0]
    span = timestamps[-1] - first
    if span <= 0:
        return [0.0] * len(timestamps)
    return [(timestamp - first) / span for timestamp in timestamps]


def _resolve_frame_rate(frame_rate: float, timestamps: Sequence[float]) -> float:
    if math.isfinite(frame_rate) and frame_rate > 0:
        return frame_rate
    deltas = [
        right - left
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if right > left
    ]
    if not deltas:
        raise ValueError("video export requires a positive frame rate")
    return 1.0 / float(np.median(deltas))


def _temporary_mp4(parent: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".mp4", dir=parent)
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path

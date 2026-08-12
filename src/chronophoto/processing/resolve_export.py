from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from chronophoto import __version__
from chronophoto.processing.compositor import (
    ComposeCache,
    ComposeSettings,
    _pose_stack_order,
    compose_sequence,
)
from chronophoto.processing.effects import EffectTrack, apply_background_effect_tracks
from chronophoto.processing.exports import (
    _IncrementalAlphaCompositor,
    _iter_transparent_poses,
    _write_png_jobs,
    build_export_layers,
    build_transparent_poses,
)
from chronophoto.processing.motion_video import (
    motion_trail_window,
    write_selected_audio,
)
from chronophoto.processing.parallel import (
    export_io_workers,
    export_render_workers,
    ordered_parallel_map,
)

ImageArray = np.ndarray
ProgressCallback = Callable[[int, str], None]
PHOTO_TIMELINE_DURATION = 5.0
PHOTO_TIMELINE_FRAME_RATE = 30.0
FCPXML_VERSION = "1.10"


@dataclass(slots=True, frozen=True)
class ResolvePackageResult:
    directory: Path
    timeline: Path
    manifest: Path
    file_count: int
    width: int
    height: int
    duration: float
    frame_rate: float
    has_audio: bool


@dataclass(slots=True, frozen=True)
class _AlphaVideoEncoder:
    codec: str
    label: str
    pixel_format: str
    options: dict[str, str]


def available_resolve_package_directory(parent: Path, source_stem: str) -> Path:
    base = parent / f"{source_stem}-chronophoto-resolve"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base.name}-{suffix}"
        suffix += 1
    return candidate


def write_resolve_package(
    directory: Path,
    *,
    source_path: Path,
    frames: Sequence[ImageArray],
    cache: ComposeCache,
    settings: ComposeSettings,
    pose_indices: Sequence[int],
    pose_labels: Sequence[str],
    effect_progress: Sequence[float],
    timestamps: Sequence[float] | None,
    start: float,
    end: float,
    frame_rate: float,
    trail_duration: float,
    pixel_aspect_ratio: tuple[int, int] = (1, 1),
    focus_pose_index: int | None = None,
    progress: ProgressCallback | None = None,
) -> ResolvePackageResult:
    """Write an importable FCPXML package for Resolve without overwriting output."""

    source_frames = list(frames)
    positions = [float(value) for value in effect_progress]
    selected = [int(index) for index in pose_indices]
    is_video = timestamps is not None
    resolved_rate = (
        _resolve_frame_rate(frame_rate, timestamps) if is_video else PHOTO_TIMELINE_FRAME_RATE
    )
    _validate_package_inputs(
        directory,
        source_frames,
        cache,
        selected,
        positions,
        timestamps,
        start,
        end,
        resolved_rate,
    )
    duration = len(source_frames) / resolved_rate if is_video else PHOTO_TIMELINE_DURATION
    height, width = source_frames[0].shape[:2]
    timeline_name = f"{source_path.stem} — Chronophoto"
    pixel_aspect_ratio = _validate_pixel_aspect_ratio(pixel_aspect_ratio)

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-writing-", dir=directory.parent))
    try:
        media = temporary / "Media"
        trail_directory = media / "trail"
        pose_directory = media / "poses"
        audio_directory = media / "audio"
        original_directory = media / "original"
        directories = [media, trail_directory, audio_directory]
        if is_video:
            directories.append(original_directory)
        if not is_video:
            directories.append(pose_directory)
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)

        _notify(progress, 2, "Building editable Resolve layers")
        combined_poses: ImageArray | None = None
        if is_video:
            editable_poses: list[ImageArray] = []
            editable_background = apply_background_effect_tracks(
                cache.background,
                settings.background_effect_tracks,
            )
        else:
            selected_frames = [source_frames[index] for index in selected]
            selected_masks = [cache.masks[index] for index in selected]
            selected_progress = [positions[index] for index in selected]
            layers = build_export_layers(
                selected_frames,
                selected_masks,
                cache.background,
                settings,
                selected_progress,
                top_pose_index=focus_pose_index,
            )
            editable_poses = layers.poses
            editable_background = layers.background
            combined_poses = layers.combined_poses
        if is_video:
            _notify(progress, 6, "Prepared background and single mask track")
        else:
            _notify(progress, 6, f"Prepared {len(editable_poses)} editable pose layers")
        background_path = media / "background.png"
        pose_paths: list[Path] = []
        layer_jobs: list[tuple[Path, ImageArray | Image.Image]] = [
            (background_path, editable_background)
        ]
        for index, pixels in enumerate(editable_poses):
            path = pose_directory / f"pose-{index + 1:03d}.png"
            pose_paths.append(path)
            layer_jobs.append((path, pixels))
        list(_write_png_jobs(layer_jobs))
        if is_video:
            _notify(progress, 10, "Wrote processed background")
        else:
            _notify(progress, 10, f"Wrote {len(pose_paths)} editable pose layers")

        trail_paths: list[Path] = []
        trail_representation = "transparent_png"
        if is_video:
            assert timestamps is not None
            timestamp_values = [float(value) for value in timestamps]
            trail_relative_effects = any(
                track.enabled and track.kind != "blend_mode" and track.timing_basis == "trail"
                for track in settings.trail_effect_tracks
            )
            reusable_pose_images: dict[int, Image.Image] | None = None
            missing_pose_stream: Iterator[tuple[int, ImageArray]] | None = None
            if not trail_relative_effects:
                reusable_pose_images = {}
                missing_indices = list(range(len(source_frames)))
                missing_pixels = _iter_transparent_poses(
                    [source_frames[source_index] for source_index in missing_indices],
                    [cache.masks[source_index] for source_index in missing_indices],
                    settings,
                    [positions[source_index] for source_index in missing_indices],
                )
                missing_pose_stream = iter(zip(missing_indices, missing_pixels, strict=True))

            # The video path no longer needs the full-resolution NumPy pose copies.
            # Keep only the Pillow images that remain inside the moving trail window.
            del editable_poses

            def trail_composite_jobs() -> Iterator[
                tuple[Path, tuple[Image.Image | tuple[int, Image.Image], ...]]
            ]:
                for frame_index in range(len(source_frames)):
                    window = list(
                        motion_trail_window(timestamp_values, frame_index, trail_duration)
                    )
                    path = trail_directory / f"trail-{frame_index + 1:06d}.png"
                    if reusable_pose_images is not None:
                        for source_index in window:
                            if source_index not in reusable_pose_images:
                                assert missing_pose_stream is not None
                                missing_index, pose = next(missing_pose_stream)
                                if missing_index != source_index:
                                    raise RuntimeError(
                                        "transparent pose stream lost chronological order"
                                    )
                                reusable_pose_images[source_index] = Image.fromarray(pose)
                        expired = [
                            source_index
                            for source_index in reusable_pose_images
                            if source_index < window[0]
                        ]
                        for source_index in expired:
                            del reusable_pose_images[source_index]
                        yield (
                            path,
                            tuple(
                                (source_index, reusable_pose_images[source_index])
                                for source_index in window
                            ),
                        )
                    else:
                        window_start = timestamp_values[window[0]]
                        window_span = timestamp_values[window[-1]] - window_start
                        trail_positions = (
                            [1.0] * len(window)
                            if window_span <= 0.0
                            else [
                                (timestamp_values[index] - window_start) / window_span
                                for index in window
                            ]
                        )
                        trail_poses = build_transparent_poses(
                            [source_frames[index] for index in window],
                            [cache.masks[index] for index in window],
                            settings,
                            [positions[index] for index in window],
                            trail_progress=trail_positions,
                        )
                        yield (
                            path,
                            tuple(
                                Image.fromarray(trail_poses[local_index])
                                for local_index in _pose_stack_order(len(window), settings.overlap)
                            ),
                        )

            if reusable_pose_images is not None:
                render_workers = export_render_workers(width, height)
                rendered_trails = _compose_incremental_trail_frames(
                    trail_composite_jobs(),
                    overlap=settings.overlap,
                )
                render_label = f"incremental composite + {render_workers} pose workers"
            else:
                render_workers = export_render_workers(width, height)
                rendered_trails = ordered_parallel_map(
                    _compose_trail_frame,
                    trail_composite_jobs(),  # type: ignore[arg-type]
                    workers=render_workers,
                )
                render_label = f"{render_workers} render workers"

            alpha_encoder = _alpha_video_encoder()
            if alpha_encoder is not None:
                trail_video_path = trail_directory / "trail-alpha.mov"
                _write_alpha_video(
                    trail_video_path,
                    rendered_trails,
                    frame_rate=resolved_rate,
                    frame_count=len(source_frames),
                    encoder=alpha_encoder,
                    progress=lambda value, message: _notify(
                        progress,
                        10 + round(value * 0.65),
                        f"{message} · {render_label}",
                    ),
                )
                trail_paths.append(trail_video_path)
                trail_representation = f"alpha_video:{alpha_encoder.codec}"
            else:
                trail_representation = "png_clip_fallback"
                for frame_index, path in enumerate(
                    _write_png_jobs(rendered_trails, compress_level=3),
                    start=1,
                ):
                    trail_paths.append(path)
                    _notify(
                        progress,
                        10 + round((frame_index / len(source_frames)) * 65),
                        f"Writing transparent trail frame {frame_index} of "
                        f"{len(source_frames)} · {render_label} + "
                        f"{export_io_workers()} PNG workers",
                    )
            reusable_pose_images = None
        else:
            assert combined_poses is not None
            trail_paths.append(_save_png(trail_directory / "subject.png", combined_poses))
            _notify(progress, 45, "Wrote combined transparent subject layer")

        reference_path: Path | None
        if is_video:
            reference_path = None
            original_path = original_directory / source_path.name
            shutil.copy2(source_path, original_path)
            _notify(progress, 75, "Packaged original video without re-encoding")
        else:
            original_path = None
            reference_path = media / "composite-reference.png"
            reference, _ = compose_sequence(
                selected_frames,
                settings,
                return_masks=False,
                cache=cache.select(selected),
                effect_progress=selected_progress,
                top_pose_index=focus_pose_index,
            )
            _save_png(reference_path, reference)
            _notify(progress, 75, "Wrote flattened reference composite")

        audio_path = None
        if is_video:
            audio_path = write_selected_audio(
                audio_directory / "source-audio.wav",
                source_path,
                start=start,
                end=end,
                progress=lambda value, message: _notify(
                    progress, 75 + round(value * 0.10), message
                ),
            )
        if audio_path is None:
            audio_directory.rmdir()

        _notify(progress, 87, "Assembling Resolve timeline")
        timeline_path = temporary / f"{source_path.stem}.fcpxml"
        audio_properties = _audio_properties(audio_path) if audio_path is not None else None

        def final_media_path(path: Path) -> Path:
            return directory / path.relative_to(temporary)

        timeline_path.write_text(
            _build_fcpxml(
                timeline_name=timeline_name,
                width=width,
                height=height,
                frame_rate=resolved_rate,
                duration=duration,
                pixel_aspect_ratio=pixel_aspect_ratio,
                # Resolve leaves otherwise valid relative FCPXML URLs offline.
                # Point at the final package rather than the temporary staging
                # directory that is atomically renamed below.
                background_path=final_media_path(background_path),
                trail_paths=[final_media_path(path) for path in trail_paths],
                pose_paths=[final_media_path(path) for path in pose_paths],
                pose_labels=list(pose_labels),
                reference_path=(
                    final_media_path(reference_path) if reference_path is not None else None
                ),
                source_path=(
                    final_media_path(original_path) if original_path is not None else source_path
                ),
                source_start=start,
                audio_path=(final_media_path(audio_path) if audio_path is not None else None),
                audio_properties=audio_properties,
                timestamps=(list(timestamps) if timestamps is not None else None),
                pose_order=_pose_stack_order(len(pose_paths), settings.overlap, focus_pose_index),
            ),
            encoding="utf-8",
            newline="\n",
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _build_manifest(
                    source_path=source_path,
                    is_video=is_video,
                    width=width,
                    height=height,
                    frame_rate=resolved_rate,
                    duration=duration,
                    pixel_aspect_ratio=pixel_aspect_ratio,
                    start=start,
                    end=end,
                    trail_duration=trail_duration,
                    settings=settings,
                    pose_labels=list(pose_labels),
                    trail_paths=[path.relative_to(temporary) for path in trail_paths],
                    pose_paths=[path.relative_to(temporary) for path in pose_paths],
                    reference_path=(
                        reference_path.relative_to(temporary)
                        if reference_path is not None
                        else None
                    ),
                    original_path=(
                        original_path.relative_to(temporary) if original_path is not None else None
                    ),
                    audio_path=(
                        audio_path.relative_to(temporary) if audio_path is not None else None
                    ),
                    trail_representation=trail_representation,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _notify(progress, 98, "Finalizing Resolve package")
        if directory.exists():
            raise FileExistsError(f"Resolve export directory already exists: {directory}")
        _publish_directory(temporary, directory)
        file_count = sum(path.is_file() for path in directory.rglob("*"))
        _notify(progress, 100, "DaVinci Resolve timeline ready")
        return ResolvePackageResult(
            directory=directory,
            timeline=directory / timeline_path.name,
            manifest=directory / manifest_path.name,
            file_count=file_count,
            width=width,
            height=height,
            duration=duration,
            frame_rate=resolved_rate,
            has_audio=audio_path is not None,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _alpha_video_encoder() -> _AlphaVideoEncoder | None:
    """Choose a Resolve-compatible alpha codec without claiming GPU acceleration."""

    candidates = (
        # QuickTime Animation keeps the mask channel byte-exact in the FFmpeg
        # builds used by Chronophoto. Some prores_ks builds advertise alpha but
        # quantize hard mask edges, which is not acceptable for an editable V2.
        _AlphaVideoEncoder("qtrle", "QuickTime Animation alpha", "argb", {}),
        _AlphaVideoEncoder(
            "prores_ks",
            "ProRes 4444 alpha",
            "yuva444p10le",
            {"profile": "4", "alpha_bits": "16"},
        ),
    )
    for candidate in candidates:
        try:
            av.codec.Codec(candidate.codec, "w")
        except (av.codec.codec.UnknownCodecError, av.error.FFmpegError):
            continue
        return candidate
    return None


def _write_alpha_video(
    path: Path,
    frames: Iterator[tuple[Path, Image.Image]],
    *,
    frame_rate: float,
    frame_count: int,
    encoder: _AlphaVideoEncoder,
    progress: ProgressCallback | None = None,
) -> Path:
    """Encode one straight-alpha V2 asset instead of hundreds of timeline clips."""

    if frame_count < 1:
        raise ValueError("alpha video requires at least one frame")
    rate = Fraction(frame_rate).limit_denominator(100_000)
    time_base = Fraction(1, 90_000)
    temporary = path.with_name(f".{path.stem}-writing{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        with av.open(str(temporary), mode="w", format="mov") as output:
            stream = None
            for index, (_unused_path, image) in enumerate(frames):
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                if stream is None:
                    height, width = rgba.shape[:2]
                    stream = output.add_stream(encoder.codec, rate=rate)
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = encoder.pixel_format
                    stream.time_base = time_base
                    stream.options = encoder.options
                video_frame = av.VideoFrame.from_ndarray(rgba, format="rgba")
                video_frame.pts = round((index / frame_rate) / float(time_base))
                video_frame.time_base = time_base
                for packet in stream.encode(video_frame):
                    output.mux(packet)
                _notify(
                    progress,
                    round(((index + 1) / frame_count) * 100),
                    f"Encoding {encoder.label} frame {index + 1} of {frame_count}",
                )
            if stream is None:
                raise RuntimeError("alpha video renderer produced no frames")
            for packet in stream.encode():
                output.mux(packet)
        os.replace(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _build_fcpxml(
    *,
    timeline_name: str,
    width: int,
    height: int,
    frame_rate: float,
    duration: float,
    pixel_aspect_ratio: tuple[int, int],
    background_path: Path,
    trail_paths: list[Path],
    pose_paths: list[Path],
    pose_labels: list[str],
    reference_path: Path | None,
    source_path: Path,
    source_start: float,
    audio_path: Path | None,
    audio_properties: tuple[int, int] | None,
    timestamps: list[float] | None,
    pose_order: list[int],
) -> str:
    frame_duration = Fraction(1, 1) / Fraction(frame_rate).limit_denominator(100_000)
    timeline_duration = Fraction(str(duration)).limit_denominator(90_000)
    root = ET.Element("fcpxml", {"version": FCPXML_VERSION})
    resources = ET.SubElement(root, "resources")
    format_attributes = {
        "id": "r1",
        "name": "Chronophoto Resolve Timeline",
        "frameDuration": _fraction_time(frame_duration),
        "width": str(width),
        "height": str(height),
        "colorSpace": "1-1-1 (Rec. 709)",
        "paspH": str(pixel_aspect_ratio[0]),
        "paspV": str(pixel_aspect_ratio[1]),
    }
    ET.SubElement(
        resources,
        "format",
        format_attributes,
    )
    resource_id = 2

    def add_asset(
        path: Path,
        name: str,
        asset_duration: Fraction,
        *,
        has_video: bool = True,
        has_audio: bool = False,
        audio_rate: int | None = None,
        audio_channels: int | None = None,
    ) -> str:
        nonlocal resource_id
        identifier = f"r{resource_id}"
        resource_id += 1
        attributes = {
            "id": identifier,
            "name": name,
            "start": "0s",
            "duration": _fraction_time(asset_duration),
        }
        if has_video:
            attributes.update({"hasVideo": "1", "format": "r1", "videoSources": "1"})
        if has_audio:
            attributes.update(
                {
                    "hasAudio": "1",
                    "audioSources": "1",
                    "audioChannels": str(audio_channels or 2),
                    "audioRate": str(audio_rate or 48_000),
                }
            )
        asset = ET.SubElement(resources, "asset", attributes)
        ET.SubElement(asset, "media-rep", {"kind": "original-media", "src": _media_url(path)})
        return identifier

    background_ref = add_asset(background_path, "V1 — Background", timeline_duration)
    trail_refs: list[tuple[str, Fraction, Fraction]] = []
    if timestamps is None:
        trail_refs.append(
            (
                add_asset(trail_paths[0], "V2 — Subject", timeline_duration),
                Fraction(0),
                timeline_duration,
            )
        )
    elif len(trail_paths) == 1 and trail_paths[0].suffix.casefold() in {".mov", ".mp4"}:
        trail_refs.append(
            (
                add_asset(trail_paths[0], "V2 — Masks / trail", timeline_duration),
                Fraction(0),
                timeline_duration,
            )
        )
    else:
        for index, path in enumerate(trail_paths):
            offset = index * frame_duration
            clip_duration = frame_duration
            trail_refs.append(
                (
                    add_asset(path, f"V2 — Trail {index + 1:06d}", clip_duration),
                    offset,
                    clip_duration,
                )
            )
    pose_refs = [
        add_asset(path, f"Pose {index + 1:03d}", timeline_duration)
        for index, path in enumerate(pose_paths)
    ]
    reference_ref = (
        add_asset(
            reference_path,
            "Reference — Chronophoto Composite",
            timeline_duration,
        )
        if reference_path is not None
        else None
    )
    original_ref = (
        add_asset(
            source_path,
            "V3 — Original Video",
            Fraction(str(source_start)) + timeline_duration,
        )
        if timestamps is not None
        else None
    )
    audio_ref = None
    if audio_path is not None:
        sample_rate, channels = audio_properties or (48_000, 2)
        audio_ref = add_asset(
            audio_path,
            "A1 — Source Audio",
            timeline_duration,
            has_video=False,
            has_audio=True,
            audio_rate=sample_rate,
            audio_channels=channels,
        )

    # Resolve expects projects to inherit their library context even though the
    # FCPXML schema also permits events directly below the document root.
    library = ET.SubElement(root, "library", {"colorProcessing": "standard"})
    event = ET.SubElement(library, "event", {"name": "Chronophoto Resolve Export"})
    project = ET.SubElement(event, "project", {"name": timeline_name})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "format": "r1",
            "duration": _fraction_time(timeline_duration),
            "tcStart": "0s",
            "tcFormat": "NDF",
            "audioLayout": "stereo",
            "audioRate": "48k",
        },
    )
    spine = ET.SubElement(sequence, "spine")
    if timestamps is not None:
        primary_clips: list[ET.Element] = []
        consolidated_alpha = len(trail_refs) == 1 and trail_refs[0][2] == timeline_duration
        for index, (reference, offset, clip_duration) in enumerate(trail_refs):
            primary_clips.append(
                ET.SubElement(
                    spine,
                    "asset-clip",
                    {
                        "name": (
                            "V2 — Masks / trail"
                            if consolidated_alpha
                            else f"V2 — Trail {index + 1:06d}"
                        ),
                        "ref": reference,
                        "offset": _fraction_time(offset),
                        "start": "0s",
                        "duration": _fraction_time(clip_duration),
                        "srcEnable": "video",
                    },
                )
            )
        anchor_clip = primary_clips[0]
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": "V1 — Background",
                "ref": background_ref,
                "lane": "-1",
                "offset": "0s",
                "start": "0s",
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "video",
            },
        )
        assert original_ref is not None
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": "V3 — Original Video (disabled)",
                "ref": original_ref,
                "lane": "1",
                "offset": "0s",
                "start": _fraction_time(Fraction(str(source_start))),
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "video",
                "enabled": "0",
            },
        )
        pose_lane_base = 1
    else:
        anchor_clip = ET.SubElement(
            spine,
            "asset-clip",
            {
                "name": "V1 — Background",
                "ref": background_ref,
                "offset": "0s",
                "start": "0s",
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "video",
            },
        )
        reference, offset, clip_duration = trail_refs[0]
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": "V2 — Subject",
                "ref": reference,
                "lane": "1",
                "offset": _fraction_time(offset),
                "start": "0s",
                "duration": _fraction_time(clip_duration),
                "srcEnable": "video",
                "enabled": "0",
            },
        )
        pose_lane_base = 2

    rank_by_pose = {pose_index: rank for rank, pose_index in enumerate(pose_order)}
    for index, reference in enumerate(pose_refs):
        label = pose_labels[index] if index < len(pose_labels) else f"Pose {index + 1}"
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": f"V{rank_by_pose[index] + 3} — Pose {index + 1:03d} — {label}",
                "ref": reference,
                "lane": str(rank_by_pose[index] + pose_lane_base),
                "offset": "0s",
                "start": "0s",
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "video",
                "enabled": "0" if timestamps is not None else "1",
            },
        )
    if reference_ref is not None:
        reference_lane = len(pose_refs) + pose_lane_base
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": "REFERENCE — Finished Chronophoto Composite (disabled)",
                "ref": reference_ref,
                "lane": str(reference_lane),
                "offset": "0s",
                "start": "0s",
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "video",
                "enabled": "0",
            },
        )
    if audio_ref is not None:
        ET.SubElement(
            anchor_clip,
            "asset-clip",
            {
                "name": "A1 — Source Audio",
                "ref": audio_ref,
                "lane": "-2" if timestamps is not None else "-1",
                "offset": "0s",
                "start": "0s",
                "duration": _fraction_time(timeline_duration),
                "srcEnable": "audio",
            },
        )

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n'


def _build_manifest(
    *,
    source_path: Path,
    is_video: bool,
    width: int,
    height: int,
    frame_rate: float,
    duration: float,
    pixel_aspect_ratio: tuple[int, int],
    start: float,
    end: float,
    trail_duration: float,
    settings: ComposeSettings,
    pose_labels: list[str],
    trail_paths: list[Path],
    pose_paths: list[Path],
    reference_path: Path | None,
    original_path: Path | None,
    audio_path: Path | None,
    trail_representation: str,
) -> dict[str, object]:
    unsupported: list[str] = []
    active_blends = [
        track.option
        for track in settings.trail_effect_tracks
        if track.enabled and track.kind == "blend_mode" and track.option != "normal"
    ]
    if active_blends:
        destination = (
            "the flattened reference"
            if reference_path is not None
            else "the three-track Resolve timeline"
        )
        unsupported.append(
            f"Subject blend modes are not editable in {destination}: " + ", ".join(active_blends)
        )
    if settings.smear_style != "none":
        representation = (
            "represented by the flattened reference"
            if reference_path is not None
            else "not represented as a separate editable Resolve effect"
        )
        unsupported.append(
            f"The {settings.smear_style} procedural smear is {representation}; "
            "the alpha track contains the masked pose trail."
        )
    timeline_tracks: list[dict[str, object]]
    if is_video:
        timeline_tracks = [
            {"lane": "V1", "name": "Background", "enabled": True},
            {"lane": "V2", "name": "Masks / trail", "enabled": True},
            {"lane": "V3", "name": "Original video", "enabled": False},
        ]
    else:
        timeline_tracks = [
            {"lane": "V1", "name": "Background", "enabled": True},
            {"lane": "V2", "name": "Combined subject", "enabled": False},
            {
                "lane": "V3+",
                "name": "Individual poses",
                "enabled": True,
                "stack_order": settings.overlap,
            },
            {"lane": "reference", "name": "Composite reference", "enabled": False},
        ]
    timeline_tracks.append(
        {"lane": "A1", "name": "Source audio", "enabled": audio_path is not None}
    )
    return {
        "schema_version": 1,
        "generator": {"name": "Chronophoto", "version": __version__},
        "interchange": {
            "format": "FCPXML",
            "version": FCPXML_VERSION,
            "target": "DaVinci Resolve 21",
            "timeline_file": f"{source_path.stem}.fcpxml",
            "media_url_mode": "absolute_file",
        },
        "source": {
            "kind": "video" if is_video else "photo_stack",
            "name": source_path.name,
            "selected_in_seconds": start if is_video else None,
            "selected_out_seconds": end if is_video else None,
        },
        "timeline": {
            "name": f"{source_path.stem} — Chronophoto",
            "width": width,
            "height": height,
            "pixel_aspect_ratio": f"{pixel_aspect_ratio[0]}:{pixel_aspect_ratio[1]}",
            "frame_rate": frame_rate,
            "duration_seconds": duration,
            "tracks": timeline_tracks,
        },
        "chronophoto": {
            "overlap": settings.overlap,
            "trail_duration_seconds": trail_duration if is_video else None,
            "background_mode": settings.background,
            "smear_style": settings.smear_style,
            "pose_labels": pose_labels,
            "trail_effects": [_serialize_effect(track) for track in settings.trail_effect_tracks],
            "background_effects": [
                _serialize_effect(track) for track in settings.background_effect_tracks
            ],
        },
        "media": {
            "background": "Media/background.png",
            "trail": [path.as_posix() for path in trail_paths],
            "poses": [path.as_posix() for path in pose_paths],
            "original": original_path.as_posix() if original_path is not None else None,
            "reference": reference_path.as_posix() if reference_path is not None else None,
            "audio": audio_path.as_posix() if audio_path is not None else None,
            "trail_representation": trail_representation,
        },
        "round_trip": {
            "pixel_local_effects_baked_into_alpha_media": True,
            "background_effects_baked_into_background_media": True,
            "unsupported_mappings": unsupported,
            "reference_is_visual_authority": reference_path is not None,
        },
    }


def _serialize_effect(track: EffectTrack) -> dict[str, object]:
    return {
        "kind": track.kind,
        "enabled": track.enabled,
        "amount": track.amount,
        "option": track.option,
        "timing_basis": track.timing_basis,
        "keyframes": [
            {"progress": point.progress, "value": point.value} for point in track.keyframes
        ],
    }


def _compose_trail_frame(
    job: tuple[Path, tuple[Image.Image, ...]],
) -> tuple[Path, Image.Image]:
    path, poses = job
    if not poses:
        raise ValueError("a transparent trail frame requires at least one pose")
    combined = Image.new("RGBA", poses[0].size)
    for pose in poses:
        combined = Image.alpha_composite(combined, pose)
    return path, combined


def _compose_incremental_trail_frames(
    jobs: Iterator[tuple[Path, tuple[Image.Image | tuple[int, Image.Image], ...]]],
    *,
    overlap: str,
) -> Iterator[tuple[Path, Image.Image]]:
    """Reuse unchanged pixels between adjacent moving-window trail frames."""

    compositor = _IncrementalAlphaCompositor(overlap)
    for path, raw_poses in jobs:
        indexed_poses = tuple(raw_poses)
        if not indexed_poses or not isinstance(indexed_poses[0], tuple):
            raise ValueError("incremental trail jobs require indexed poses")
        current = tuple(indexed_poses)  # type: ignore[arg-type]
        yield path, compositor.compose(current)


def _validate_package_inputs(
    directory: Path,
    frames: list[ImageArray],
    cache: ComposeCache,
    pose_indices: list[int],
    effect_progress: list[float],
    timestamps: Sequence[float] | None,
    start: float,
    end: float,
    frame_rate: float,
) -> None:
    if directory.exists():
        raise FileExistsError(f"Resolve export directory already exists: {directory}")
    if len(frames) < 2 or len(cache.masks) != len(frames):
        raise ValueError("Resolve export requires matching frames and masks")
    if cache.background.shape != frames[0].shape:
        raise ValueError("Resolve export background must match the source dimensions")
    if not pose_indices or any(not 0 <= index < len(frames) for index in pose_indices):
        raise ValueError("Resolve export pose indices are outside the source sequence")
    if len(effect_progress) != len(frames):
        raise ValueError("Resolve export effect progress must match the source sequence")
    if timestamps is not None:
        if len(timestamps) != len(frames):
            raise ValueError("Resolve export timestamps must match the source sequence")
        if start < 0 or end <= start:
            raise ValueError("Resolve export end must be later than start")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("Resolve export requires a positive frame rate")


def _publish_directory(temporary: Path, destination: Path) -> None:
    """Atomically publish an export despite brief Windows scanner/preview locks."""

    delays = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
    for position, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if position + 1 >= len(delays) or destination.exists():
                raise


def _audio_properties(path: Path) -> tuple[int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate or 48_000
        channels = stream.codec_context.channels or 2
    return sample_rate, channels


def _validate_pixel_aspect_ratio(value: tuple[int, int]) -> tuple[int, int]:
    horizontal, vertical = value
    if horizontal <= 0 or vertical <= 0:
        raise ValueError("Resolve export pixel aspect ratio must be positive")
    ratio = Fraction(horizontal, vertical)
    return ratio.numerator, ratio.denominator


def _resolve_frame_rate(
    frame_rate: float,
    timestamps: Sequence[float] | None,
) -> float:
    if math.isfinite(frame_rate) and frame_rate > 0:
        return frame_rate
    values = list(timestamps or ())
    deltas = [right - left for left, right in zip(values, values[1:], strict=False) if right > left]
    if not deltas:
        raise ValueError("Resolve export requires a positive frame rate")
    return 1.0 / float(np.median(deltas))


def _fraction_time(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def _media_url(path: Path) -> str:
    return path.resolve(strict=False).as_uri()


def _save_png(path: Path, pixels: ImageArray) -> Path:
    Image.fromarray(pixels).save(path, compress_level=6)
    return path


def _notify(progress: ProgressCallback | None, value: int, message: str) -> None:
    if progress is not None:
        progress(max(0, min(100, value)), message)

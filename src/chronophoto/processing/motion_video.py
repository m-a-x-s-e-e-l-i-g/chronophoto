from __future__ import annotations

import math
import os
import tempfile
from bisect import bisect_left
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from chronophoto.processing.compositor import ComposeCache, ComposeSettings, compose_sequence

ImageArray = np.ndarray
ProgressCallback = Callable[[int, str], None]


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
    rendered: list[ImageArray] = []
    for position, index in enumerate(indices):
        rendered.append(
            compose_motion_trail_frame(
                frames,
                timestamps,
                index,
                trail_duration,
                settings,
                cache,
                effect_progress=effect_progress,
                effect_pixel_scale=effect_pixel_scale,
            )
        )
        if progress is not None:
            progress(
                round(((position + 1) / len(indices)) * 100),
                f"Rendering trail preview {position + 1} of {len(indices)}",
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
    rate = Fraction(frame_rate).limit_denominator(1001)
    time_base = Fraction(1, 90_000)
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
        stream.time_base = time_base
        stream.options = {"crf": "18", "preset": "medium"}
        first_timestamp = timestamps[0]
        total = len(frames)
        for index in range(total):
            pixels = compose_motion_trail_frame(
                frames,
                timestamps,
                index,
                trail_duration,
                settings,
                cache,
                effect_progress=effect_progress,
                effect_pixel_scale=effect_pixel_scale,
            )
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            video_frame.pts = round((timestamps[index] - first_timestamp) / float(time_base))
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                output.mux(packet)
            if progress is not None:
                progress(
                    5 + round(((index + 1) / total) * 89),
                    f"Encoding trail frame {index + 1} of {total}",
                )
        for packet in stream.encode():
            output.mux(packet)


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

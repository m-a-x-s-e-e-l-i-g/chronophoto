from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

ImageArray = NDArray[np.uint8]
ProgressCallback = Callable[[int, str], None]


@dataclass(slots=True)
class MediaSequence:
    frames: list[ImageArray]
    labels: list[str]
    source_size: tuple[int, int]
    timestamps: list[float] | None = None
    source_indices: list[int] | None = None


@dataclass(slots=True, frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    frame_rate: float
    frame_count: int
    pixel_aspect_ratio: tuple[int, int] = (1, 1)


@dataclass(slots=True, frozen=True)
class DecodedVideoFrame:
    index: int
    timestamp: float
    pixels: ImageArray


def probe_video(path: str | Path) -> VideoInfo:
    source = Path(path)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        frame_rate = float(stream.average_rate) if stream.average_rate else 0.0
        frame_count = int(stream.frames or round(duration * frame_rate))
        sample_aspect_ratio = (
            stream.sample_aspect_ratio or stream.codec_context.sample_aspect_ratio or Fraction(1, 1)
        )
        return VideoInfo(
            source,
            max(duration, 0.01),
            stream.width,
            stream.height,
            frame_rate,
            max(1, frame_count),
            (sample_aspect_ratio.numerator, sample_aspect_ratio.denominator),
        )


def _resize(frame: ImageArray, max_dimension: int | None) -> ImageArray:
    if not max_dimension or max(frame.shape[:2]) <= max_dimension:
        return frame
    scale = max_dimension / max(frame.shape[:2])
    size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _timestamp_label(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def load_video_sequence(
    path: str | Path,
    start: float,
    end: float,
    pose_count: int | None,
    *,
    max_dimension: int | None = None,
    progress: ProgressCallback | None = None,
) -> MediaSequence:
    """Decode every frame, or frames nearest to evenly spaced target timestamps."""

    if pose_count is not None and pose_count < 2:
        raise ValueError("pose_count must be at least 2")
    if start < 0 or end <= start:
        raise ValueError("end must be later than start")

    if pose_count is None:
        return _load_all_video_frames(path, start, end, max_dimension, progress)

    targets = np.linspace(start, end, pose_count)
    selected: list[ImageArray | None] = [None] * pose_count
    best_distances = [float("inf")] * pose_count
    source = Path(path)

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is not None:
            seek_pts = max(0, int((start - 0.5) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < start - 0.5:
                continue
            if timestamp > end + 0.5:
                break

            distances = np.abs(targets - timestamp)
            improved = distances < np.asarray(best_distances)
            if np.any(improved):
                decoded = _resize(frame.to_ndarray(format="rgb24"), max_dimension)
                for target_index in np.flatnonzero(improved):
                    selected[int(target_index)] = decoded
                    best_distances[int(target_index)] = float(distances[target_index])

            if progress:
                completed = sum(value is not None for value in selected)
                progress(
                    int((completed / pose_count) * 100), f"Reading pose {completed} of {pose_count}"
                )

    missing = [index for index, frame in enumerate(selected) if frame is None]
    if missing:
        raise RuntimeError(f"Could not decode {len(missing)} requested video frames")

    frames = [frame for frame in selected if frame is not None]
    return MediaSequence(
        frames=frames,
        labels=[_timestamp_label(float(value)) for value in targets],
        source_size=(frames[0].shape[1], frames[0].shape[0]),
        timestamps=[float(value) for value in targets],
    )


def iter_video_frames(
    path: str | Path,
    start: float,
    end: float,
    *,
    max_dimension: int | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Yield one selected RGB frame at a time without retaining the clip."""

    if start < 0 or end <= start:
        raise ValueError("end must be later than start")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is not None:
            seek_pts = max(0, int((start - 0.5) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        selected_index = 0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < start:
                continue
            if timestamp > end:
                break
            pixels = _resize(frame.to_ndarray(format="rgb24"), max_dimension)
            yield DecodedVideoFrame(selected_index, timestamp, pixels)
            selected_index += 1


def _load_all_video_frames(
    path: str | Path,
    start: float,
    end: float,
    max_dimension: int | None,
    progress: ProgressCallback | None,
) -> MediaSequence:
    frames: list[ImageArray] = []
    labels: list[str] = []
    timestamps: list[float] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is not None:
            seek_pts = max(0, int((start - 0.5) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < start:
                continue
            if timestamp > end:
                break
            frames.append(_resize(frame.to_ndarray(format="rgb24"), max_dimension))
            labels.append(_timestamp_label(timestamp))
            timestamps.append(timestamp)
            if progress:
                fraction = (timestamp - start) / max(end - start, 0.001)
                progress(min(99, max(0, round(fraction * 100))), f"Reading frame {len(frames)}")

    if len(frames) < 2:
        raise RuntimeError("Could not decode at least two video frames in the selected range")
    if progress:
        progress(100, f"Read all {len(frames)} frames")
    return MediaSequence(
        frames=frames,
        labels=labels,
        source_size=(frames[0].shape[1], frames[0].shape[0]),
        timestamps=timestamps,
        source_indices=list(range(len(frames))),
    )


def select_video_sequence(
    cached: MediaSequence,
    start: float,
    end: float,
    pose_count: int | None,
) -> MediaSequence:
    """Select a range or sampled poses from an already-decoded video cache."""

    if cached.timestamps is None or len(cached.timestamps) != len(cached.frames):
        raise ValueError("The cached sequence does not contain video timestamps")
    if pose_count is not None and pose_count < 2:
        raise ValueError("pose_count must be at least 2")
    if start < 0 or end <= start:
        raise ValueError("end must be later than start")

    timestamps = np.asarray(cached.timestamps, dtype=np.float64)
    if pose_count is None:
        indices = np.flatnonzero((timestamps >= start) & (timestamps <= end)).tolist()
    else:
        targets = np.linspace(start, end, pose_count)
        indices = [int(np.argmin(np.abs(timestamps - target))) for target in targets]
    if len(indices) < 2:
        raise RuntimeError("Could not select at least two cached frames in the range")

    source_indices = indices
    if cached.source_indices is not None:
        source_indices = [cached.source_indices[index] for index in indices]
    return MediaSequence(
        frames=[cached.frames[index] for index in indices],
        labels=[cached.labels[index] for index in indices],
        source_size=cached.source_size,
        timestamps=[cached.timestamps[index] for index in indices],
        source_indices=source_indices,
    )


def load_video_thumbnails(
    path: str | Path,
    count: int = 10,
    *,
    max_dimension: int = 180,
) -> list[ImageArray]:
    """Seek to a small set of positions without decoding the entire clip."""

    info = probe_video(path)
    safe_end = max(0.001, info.duration - max(1.0 / max(info.frame_rate, 1.0), 0.001))
    targets = np.linspace(0.0, safe_end, max(2, count))
    thumbnails: list[ImageArray] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is None:
            return []
        for target in targets:
            seek_pts = max(0, int(float(target) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            candidate: ImageArray | None = None
            for frame in container.decode(stream):
                candidate = frame.to_ndarray(format="rgb24")
                if frame.time is None or float(frame.time) >= float(target):
                    break
            if candidate is not None:
                thumbnails.append(_resize(candidate, max_dimension))
    return thumbnails


def load_image_sequence(
    paths: Sequence[str | Path],
    *,
    max_dimension: int | None = None,
    sort_mode: str = "automatic",
    progress: ProgressCallback | None = None,
) -> MediaSequence:
    if len(paths) < 2:
        raise ValueError("Choose at least two photos")

    ordered = order_image_paths(paths, sort_mode)
    frames: list[ImageArray] = []
    labels: list[str] = []
    expected_size: tuple[int, int] | None = None

    for index, path in enumerate(ordered):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            frame = np.asarray(image, dtype=np.uint8)
        frame = _resize(frame, max_dimension)
        size = (frame.shape[1], frame.shape[0])
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError("All photos must have the same dimensions")
        frames.append(np.ascontiguousarray(frame))
        labels.append(path.name)
        if progress:
            progress(int(((index + 1) / len(ordered)) * 100), f"Loading {path.name}")

    assert expected_size is not None
    return MediaSequence(frames, labels, expected_size)


def _capture_time(path: Path) -> datetime | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get(36867) or exif.get(306)
        if not raw:
            return None
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except (OSError, TypeError, ValueError):
        return None


def order_image_paths(
    paths: Sequence[str | Path],
    mode: str = "automatic",
) -> list[Path]:
    items = [Path(path) for path in paths]
    if mode == "input":
        return items
    if mode == "filename":
        return sorted(items, key=lambda item: item.name.casefold())
    if mode not in {"automatic", "capture_time"}:
        raise ValueError(f"Unsupported photo ordering mode: {mode}")

    timestamps = {path: _capture_time(path) for path in items}
    has_capture_times = sum(value is not None for value in timestamps.values())
    if mode == "automatic" and has_capture_times < 2:
        return sorted(items, key=lambda item: item.name.casefold())

    maximum = datetime.max
    return sorted(
        items,
        key=lambda item: (
            timestamps[item] is None,
            timestamps[item] or maximum,
            item.name.casefold(),
        ),
    )

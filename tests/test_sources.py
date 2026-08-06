from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image

from chronophoto.processing.sources import (
    load_image_sequence,
    load_video_sequence,
    order_image_paths,
    probe_video,
    select_video_sequence,
)


def test_image_sequence_sorts_by_filename(tmp_path: Path) -> None:
    for name, value in (("frame_02.png", 80), ("frame_01.png", 30), ("frame_03.png", 140)):
        pixels = np.full((60, 90, 3), value, dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / name)

    sequence = load_image_sequence(list(tmp_path.glob("*.png")))

    assert sequence.labels == ["frame_01.png", "frame_02.png", "frame_03.png"]
    assert [int(frame[0, 0, 0]) for frame in sequence.frames] == [30, 80, 140]
    assert sequence.source_size == (90, 60)


def test_image_sequence_requires_matching_sizes(tmp_path: Path) -> None:
    Image.new("RGB", (100, 80)).save(tmp_path / "a.png")
    Image.new("RGB", (120, 80)).save(tmp_path / "b.png")

    try:
        load_image_sequence([tmp_path / "a.png", tmp_path / "b.png"])
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("Mismatched image sizes were accepted")


def test_video_probe_and_timestamp_sampling(tmp_path: Path) -> None:
    video_path = tmp_path / "motion.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width = 160
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            pixels = np.zeros((96, 160, 3), dtype=np.uint8)
            pixels[:, :, 1] = index * 9
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    info = probe_video(video_path)
    sequence = load_video_sequence(video_path, 0.1, 1.6, 5)

    assert info.width == 160
    assert info.height == 96
    assert 1.9 <= info.duration <= 2.1
    assert len(sequence.frames) == 5
    assert sequence.source_size == (160, 96)
    green_values = [int(frame[20, 20, 1]) for frame in sequence.frames]
    assert green_values == sorted(green_values)


def test_video_sampling_can_repeat_frames_for_a_short_range(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            pixels = np.full((64, 96, 3), index * 12, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    sequence = load_video_sequence(video_path, 0.20, 0.28, 40)

    assert len(sequence.frames) == 40
    assert len(sequence.labels) == 40


def test_video_sequence_can_decode_every_frame_in_the_range(tmp_path: Path) -> None:
    video_path = tmp_path / "all-frames.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            pixels = np.full((64, 96, 3), index * 8, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    sequence = load_video_sequence(video_path, 0.25, 1.25, None)

    assert 11 <= len(sequence.frames) <= 13
    assert len(sequence.labels) == len(sequence.frames)
    values = [int(frame[10, 10, 0]) for frame in sequence.frames]
    assert values == sorted(values)


def test_cached_video_sequence_can_be_reselected_without_decoding(tmp_path: Path) -> None:
    video_path = tmp_path / "cached.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            pixels = np.full((64, 96, 3), index * 8, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    cached = load_video_sequence(video_path, 0.0, 2.0, None)
    selected_all = select_video_sequence(cached, 0.5, 1.0, None)
    sampled = select_video_sequence(cached, 0.25, 1.25, 5)

    assert cached.timestamps is not None
    assert selected_all.timestamps is not None
    assert all(0.5 <= timestamp <= 1.0 for timestamp in selected_all.timestamps)
    assert len(sampled.frames) == 5
    assert all(
        any(frame is cached_frame for cached_frame in cached.frames) for frame in sampled.frames
    )


def test_photo_order_prefers_exif_capture_time(tmp_path: Path) -> None:
    later = tmp_path / "a-later.jpg"
    earlier = tmp_path / "z-earlier.jpg"
    for path, timestamp in (
        (later, "2026:08:05 14:02:00"),
        (earlier, "2026:08:05 14:01:00"),
    ):
        exif = Image.Exif()
        exif[36867] = timestamp
        Image.new("RGB", (40, 30)).save(path, exif=exif)

    ordered = order_image_paths([later, earlier], "automatic")

    assert ordered == [earlier, later]

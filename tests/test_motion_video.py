from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

import chronophoto.processing.compositor as compositor_module
from chronophoto.processing import (
    ComposeCache,
    ComposeSettings,
    EffectKeyframe,
    EffectTrack,
    build_compose_cache,
    compose_motion_trail_frame,
    load_video_sequence,
    motion_trail_window,
    write_motion_trail_video,
)


def _moving_subject_frames(count: int = 5) -> list[np.ndarray]:
    background = np.full((48, 80, 3), (18, 28, 42), dtype=np.uint8)
    frames: list[np.ndarray] = []
    for index in range(count):
        frame = background.copy()
        cv2.rectangle(frame, (5 + index * 14, 15), (15 + index * 14, 34), (225, 75, 30), -1)
        frames.append(frame)
    return frames


def test_motion_trail_window_is_causal_and_timestamp_bounded() -> None:
    timestamps = [0.0, 0.25, 0.9, 1.0, 1.8]

    assert motion_trail_window(timestamps, 0, 1.0) == (0,)
    assert motion_trail_window(timestamps, 3, 0.75) == (1, 2, 3)
    assert motion_trail_window(timestamps, 3, 0.0) == (3,)
    assert motion_trail_window(timestamps, 4, 10.0) == (0, 1, 2, 3, 4)


def test_motion_trail_frame_excludes_old_and_future_positions() -> None:
    frames = _moving_subject_frames()
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    settings = ComposeSettings(threshold=10, feather=0, background="median")
    cache = build_compose_cache(frames, settings)

    result = compose_motion_trail_frame(
        frames,
        timestamps,
        3,
        0.6,
        settings,
        cache,
    )

    assert result[24, 36, 0] > 150  # Previous subject at t=1.0 remains.
    assert result[24, 50, 0] > 150  # Current subject at t=1.5 remains.
    assert result[24, 8, 0] < 80  # Old subject at t=0.0 has disappeared.
    assert result[24, 64, 0] < 80  # Future subject at t=2.0 is never included.


@pytest.mark.parametrize(
    ("overlap", "expected_index"),
    (("newest", 3), ("oldest", 0)),
)
def test_motion_trail_overlap_selects_the_only_top_pose(
    overlap: str,
    expected_index: int,
) -> None:
    colors = ((210, 30, 20), (30, 210, 60), (40, 70, 220), (220, 180, 30))
    frames = [np.full((4, 4, 3), color, dtype=np.uint8) for color in colors]
    timestamps = [0.0, 0.2, 0.9, 1.0]
    background = np.zeros_like(frames[0])
    masks = [np.full((4, 4), 255, dtype=np.uint8) for _ in frames]

    result = compose_motion_trail_frame(
        frames,
        timestamps,
        3,
        1.0,
        ComposeSettings(overlap=overlap),
        ComposeCache(background, masks),
    )

    assert np.array_equal(result[0, 0], colors[expected_index])


@pytest.mark.parametrize(
    ("overlap", "expected_index"),
    (("newest", 3), ("oldest", 0)),
)
def test_motion_trail_smear_repaints_only_the_selected_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    overlap: str,
    expected_index: int,
) -> None:
    frames = [np.full((4, 4, 3), index * 40, dtype=np.uint8) for index in range(4)]
    masks = [np.full((4, 4), 255, dtype=np.uint8) for _ in frames]
    painted_indices: list[int] = []
    original_blend_pose = compositor_module._blend_pose

    def record_blend_pose(result, frame, *args, **kwargs):  # type: ignore[no-untyped-def]
        painted_indices.append(int(frame[0, 0, 0] // 40))
        return original_blend_pose(result, frame, *args, **kwargs)

    monkeypatch.setattr(compositor_module, "_blend_pose", record_blend_pose)
    compose_motion_trail_frame(
        frames,
        [0.0, 0.2, 0.9, 1.0],
        3,
        1.0,
        ComposeSettings(overlap=overlap, smear_style="photographic"),
        ComposeCache(np.zeros_like(frames[0]), masks),
    )

    assert painted_indices == [expected_index]


def test_zero_duration_keeps_only_the_current_subject() -> None:
    frames = _moving_subject_frames(3)
    timestamps = [0.0, 0.5, 1.0]
    settings = ComposeSettings(threshold=10, feather=0, background="median")
    cache = build_compose_cache(frames, settings)

    result = compose_motion_trail_frame(
        frames,
        timestamps,
        1,
        0.0,
        settings,
        cache,
    )

    assert result[24, 22, 0] > 150
    assert result[24, 8, 0] < 80
    assert result[24, 36, 0] < 80


def test_trail_timed_effect_restarts_across_each_visible_window() -> None:
    frames = _moving_subject_frames()
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    opacity = EffectTrack(
        "opacity",
        (EffectKeyframe(0.0, 0.0), EffectKeyframe(1.0, 100.0)),
        timing_basis="trail",
    )
    settings = ComposeSettings(
        threshold=10,
        feather=0,
        background="median",
        trail_effect_tracks=(opacity,),
    )
    cache = build_compose_cache(frames, settings)

    result = compose_motion_trail_frame(
        frames,
        timestamps,
        3,
        0.6,
        settings,
        cache,
    )

    assert result[24, 36, 0] < 80  # Oldest visible pose starts at 0% opacity.
    assert result[24, 50, 0] > 150  # Current subject ends at 100% opacity.


@pytest.mark.parametrize("smear_style", ["photographic", "dense_clones"])
def test_motion_video_rejects_implausible_connectors_between_adjacent_frames(
    smear_style: str,
) -> None:
    background = np.full((120, 240, 3), (18, 28, 42), dtype=np.uint8)
    frames = [background.copy(), background.copy()]
    cv2.rectangle(frames[0], (12, 12), (31, 31), (225, 75, 30), -1)
    cv2.rectangle(frames[1], (208, 88), (227, 107), (225, 75, 30), -1)
    masks = []
    for frame in frames:
        mask = np.max(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2) > 0
        masks.append(mask.astype(np.uint8) * 255)
    cache = ComposeCache(background, masks)

    result = compose_motion_trail_frame(
        frames,
        [0.0, 0.04],
        1,
        1.0,
        ComposeSettings(smear_style=smear_style),
        cache,
    )

    assert np.array_equal(result[60, 120], background[60, 120])
    assert np.array_equal(result[20, 20], background[20, 20])
    assert result[98, 218, 0] > 150


def test_motion_trail_mp4_retains_audio_and_source_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_source_video_with_audio(source)
    sequence = load_video_sequence(source, 0.0, 1.75, None)
    settings = ComposeSettings(threshold=10, feather=0, background="median")
    cache = build_compose_cache(sequence.frames, settings)
    output = tmp_path / "trail.mp4"

    written = write_motion_trail_video(
        output,
        source,
        sequence.frames,
        sequence.timestamps or [],
        0.5,
        settings,
        cache,
        start=0.0,
        end=1.75,
        frame_rate=0.0,
    )

    assert written == output
    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        video = container.streams.video[0]
        assert (video.width, video.height) == (80, 48)
        assert video.codec_context.name == "h264"
        assert 1.6 <= float(container.duration / av.time_base) <= 2.1
        assert len(list(container.decode(video))) == len(sequence.frames)


def test_invalid_trail_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="trail duration"):
        motion_trail_window([0.0, 1.0], 1, -0.1)


def test_cancelled_video_render_leaves_no_partial_files(tmp_path: Path) -> None:
    frames = _moving_subject_frames()
    timestamps = [index / 4 for index in range(len(frames))]
    settings = ComposeSettings(threshold=10, feather=0, background="median")
    cache = build_compose_cache(frames, settings)
    output = tmp_path / "cancelled.mp4"

    def cancel_after_first_frame(value: int, message: str) -> None:
        del message
        if value >= 20:
            raise RuntimeError("cancelled for test")

    with pytest.raises(RuntimeError, match="cancelled for test"):
        write_motion_trail_video(
            output,
            tmp_path / "unused-source.mp4",
            frames,
            timestamps,
            0.5,
            settings,
            cache,
            start=0.0,
            end=1.0,
            frame_rate=4.0,
            progress=cancel_after_first_frame,
        )

    assert not output.exists()
    assert not list(tmp_path.glob("chronophoto-*.mp4"))


def _write_source_video_with_audio(path: Path) -> None:
    frames = _moving_subject_frames(8)
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=4)
        video.width = 80
        video.height = 48
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=48_000)
        audio.layout = "stereo"
        for frame_pixels in frames:
            frame = av.VideoFrame.from_ndarray(frame_pixels, format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        samples_per_frame = 1_024
        for start in range(0, 96_000, samples_per_frame):
            sample_count = min(samples_per_frame, 96_000 - start)
            samples = np.zeros((2, sample_count), dtype=np.float32)
            samples[:, :] = 0.02
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate = 48_000
            frame.pts = start
            frame.time_base = Fraction(1, 48_000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

import chronophoto.processing.motion_video as motion_video_module
from chronophoto.processing import (
    ComposeSettings,
    RenderPlan,
    RenderStage,
    StageArtifact,
    write_streaming_motion_trail_video,
)


def test_render_plan_orders_typed_stages_and_reuses_exact_cache_keys() -> None:
    calls: list[str] = []

    def decode(_dependencies, _progress):  # type: ignore[no-untyped-def]
        calls.append("decode")
        return StageArtifact("frames", [1, 2, 3], ("clip", 1))

    def encode(dependencies, _progress):  # type: ignore[no-untyped-def]
        calls.append("encode")
        return StageArtifact("video", sum(dependencies["decode"].value), ("video", 1))

    plan = RenderPlan(
        (
            RenderStage("decode", "frames", decode, cache_key=("clip", 1)),
            RenderStage("encode", "video", encode, dependencies=("decode",), mode="streaming"),
        )
    )
    cache: dict[tuple[object, ...], StageArtifact] = {}
    first = plan.execute(cache=cache)
    second = plan.execute(cache=cache)

    assert first.artifacts["encode"].value == 6
    assert calls == ["decode", "encode", "encode"]
    assert second.timings[0].reused
    assert second.timings[1].mode == "streaming"


def test_render_plan_rejects_forward_or_missing_dependencies() -> None:
    def stage(_dependencies, _progress):  # type: ignore[no-untyped-def]
        return StageArtifact("value", 1, ())

    try:
        RenderPlan((RenderStage("encode", "value", stage, dependencies=("decode",)),))
    except ValueError as exc:
        assert "unavailable stages" in str(exc)
    else:
        raise AssertionError("forward dependency must be rejected")


def test_streaming_trail_export_bounds_live_window_as_clip_grows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "long-source.mp4"
    frame_rate = 24
    frame_count = 120
    with av.open(str(source), "w") as output:
        stream = output.add_stream("libx264", rate=frame_rate)
        stream.width = 64
        stream.height = 40
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.full((40, 64, 3), (20, 35, 60), dtype=np.uint8)
            left = 2 + (index % 48)
            pixels[12:28, left : left + 10] = (220, 70, 30)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    monkeypatch.setattr(
        motion_video_module,
        "_video_encoder_candidates",
        lambda _width, _height: (
            motion_video_module._VideoEncoder(
                "libx264",
                "CPU H.264",
                {"crf": "23", "preset": "ultrafast"},
            ),
        ),
    )
    target, telemetry = write_streaming_motion_trail_video(
        tmp_path / "trail.mp4",
        source,
        0.5,
        ComposeSettings(),
        start=0.0,
        end=(frame_count - 1) / frame_rate,
        frame_rate=frame_rate,
    )

    assert target.is_file()
    assert telemetry.rendered_frames >= frame_count - 2
    assert telemetry.active_window_peak <= 14
    materialized_rgb_bytes = frame_count * 64 * 40 * 3
    assert telemetry.estimated_peak_bytes < materialized_rgb_bytes
    assert not telemetry.zero_copy
    with av.open(str(target)) as container:
        assert len(list(container.decode(video=0))) == telemetry.rendered_frames

"""Render the bundled bridge jump as a real DaVinci Resolve export fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chronophoto.processing import (
    ComposeSettings,
    MediaSequence,
    align_sequence,
    available_resolve_package_directory,
    build_compose_cache,
    load_video_sequence,
    select_video_sequence,
    write_resolve_package,
)
from chronophoto.processing.sources import probe_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "sample" / "sample-01.mp4"
OUTPUT_PARENT = PROJECT_ROOT / "build" / "verification"
START = 4.35
END = 6.65
POSE_COUNT = 8
TRAIL_DURATION = 0.7


def main() -> int:
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    info = probe_video(SOURCE)
    last_reported = -10

    def progress(value: int, message: str) -> None:
        nonlocal last_reported
        if value >= last_reported + 10 or value == 100:
            print(f"{value:3d}%  {message}")
            last_reported = value

    sequence = load_video_sequence(SOURCE, START, END, None, progress=progress)
    aligned = align_sequence(sequence.frames, "off", progress=progress)
    settings = ComposeSettings()
    cache = build_compose_cache(aligned, settings, progress=progress)
    timeline_sequence = MediaSequence(
        aligned,
        sequence.labels,
        sequence.source_size,
        sequence.timestamps,
        list(range(len(aligned))),
    )
    poses = select_video_sequence(timeline_sequence, START, END, POSE_COUNT)
    effect_progress = np.linspace(0.0, 1.0, len(aligned)).tolist()
    destination = available_resolve_package_directory(OUTPUT_PARENT, SOURCE.stem)
    result = write_resolve_package(
        destination,
        source_path=SOURCE,
        frames=aligned,
        cache=cache,
        settings=settings,
        pose_indices=poses.source_indices or [],
        pose_labels=poses.labels,
        effect_progress=effect_progress,
        timestamps=sequence.timestamps,
        start=START,
        end=END,
        frame_rate=info.frame_rate,
        trail_duration=TRAIL_DURATION,
        pixel_aspect_ratio=info.pixel_aspect_ratio,
        progress=progress,
    )
    print(result.timeline.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render the compact motion-trail sample used by the README."""

from __future__ import annotations

from pathlib import Path

import av
from PIL import Image, ImageDraw

from chronophoto.processing import (
    ComposeSettings,
    build_compose_cache,
    load_video_sequence,
    write_motion_trail_video,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "sample" / "sample-01.mp4"
VIDEO_OUTPUT = PROJECT_ROOT / "docs" / "videos" / "motion-trail-bridge-jump.mp4"
POSTER_OUTPUT = PROJECT_ROOT / "docs" / "images" / "motion-trail-video-preview.jpg"
START = 4.35
END = 6.65
TRAIL_DURATION = 0.7
MAX_DIMENSION = 640


def _contact_sheet(video_path: Path, output_path: Path) -> None:
    with av.open(str(video_path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    selected = [frames[round(index * (len(frames) - 1) / 4)] for index in range(5)]
    gap = 8
    label_height = 54
    width = sum(frame.width for frame in selected) + gap * (len(selected) - 1)
    sheet = Image.new("RGB", (width, selected[0].height + label_height), "#0d1117")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for index, frame in enumerate(selected, start=1):
        sheet.paste(frame, (x, label_height))
        draw.text((x + 12, 16), f"{index}", fill="#f5f7fa", stroke_width=1)
        x += frame.width + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90, optimize=True)


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Sample source is missing: {SOURCE}")
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sequence = load_video_sequence(
        SOURCE,
        START,
        END,
        None,
        max_dimension=MAX_DIMENSION,
    )
    settings = ComposeSettings()
    cache = build_compose_cache(sequence.frames, settings)
    write_motion_trail_video(
        VIDEO_OUTPUT,
        SOURCE,
        sequence.frames,
        sequence.timestamps or [],
        TRAIL_DURATION,
        settings,
        cache,
        start=START,
        end=END,
        frame_rate=0.0,
    )
    _contact_sheet(VIDEO_OUTPUT, POSTER_OUTPUT)
    print(VIDEO_OUTPUT.resolve())
    print(POSTER_OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

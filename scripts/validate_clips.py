"""Render repeatable composite and mask evidence for real-world video clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from chronophoto.processing import ComposeSettings, compose_sequence, load_video_sequence
from chronophoto.processing.sources import probe_video


def mask_contact_sheet(frames: list[np.ndarray], masks: list[np.ndarray]) -> Image.Image:
    tiles: list[Image.Image] = []
    for frame, mask in zip(frames, masks, strict=True):
        alpha = np.clip(mask[..., None], 0.0, 1.0)
        dimmed = frame.astype(np.float32) * 0.16
        amber = np.empty_like(frame)
        amber[:] = (232, 132, 45)
        highlighted = frame.astype(np.float32) * 0.30 + amber.astype(np.float32) * 0.70
        visual = dimmed * (1.0 - alpha) + highlighted * alpha
        tiles.append(Image.fromarray(np.clip(visual, 0, 255).astype(np.uint8)))

    tile_width = min(320, tiles[0].width)
    tile_height = round(tiles[0].height * tile_width / tiles[0].width)
    resized = [tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS) for tile in tiles]
    sheet = Image.new("RGB", (tile_width * len(resized), tile_height), (18, 17, 15))
    for index, tile in enumerate(resized):
        sheet.paste(tile, (index * tile_width, 0))
    return sheet


def validate_clip(path: Path, output_dir: Path, poses: int, threshold: int) -> None:
    info = probe_video(path)
    start = info.duration * 0.05
    end = info.duration * 0.95
    sequence = load_video_sequence(path, start, end, poses, max_dimension=1600)
    result, masks = compose_sequence(
        sequence.frames,
        ComposeSettings(threshold=threshold, feather=5, background="median"),
    )

    clip_dir = output_dir / path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(clip_dir / "composite.png")
    mask_contact_sheet(sequence.frames, masks).save(clip_dir / "masks.jpg", quality=92)
    coverage = [float(np.mean(mask)) for mask in masks]
    print(
        f"{path.name}: {len(sequence.frames)} poses, "
        f"mask coverage {min(coverage):.1%}–{max(coverage):.1%}, "
        f"output {clip_dir.resolve()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render composites and mask contact sheets for real-footage evaluation."
    )
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/clip-validation"))
    parser.add_argument("--poses", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=28)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        validate_clip(clip, args.output, args.poses, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

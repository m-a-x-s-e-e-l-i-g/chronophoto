"""Render a deterministic synthetic action sequence for visual QA."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chronophoto.processing import ComposeSettings, compose_sequence


def build_frames() -> list[np.ndarray]:
    height, width = 720, 1280
    y, x = np.mgrid[:height, :width]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = 24 + (x / width * 28).astype(np.uint8)
    base[..., 1] = 27 + (y / height * 22).astype(np.uint8)
    base[..., 2] = 31 + (x / width * 18).astype(np.uint8)
    base[545:552, :, :] = (112, 91, 67)

    frames: list[np.ndarray] = []
    for index in range(9):
        frame = base.copy()
        center_x = 145 + index * 116
        center_y = 420 - int(np.sin(index / 8 * np.pi) * 185)
        cv2.circle(frame, (center_x, center_y - 90), 34, (232, 171, 97), -1)
        cv2.line(frame, (center_x, center_y - 58), (center_x, center_y + 52), (215, 102, 51), 42)
        cv2.line(
            frame, (center_x, center_y - 25), (center_x - 58, center_y + 14), (221, 120, 58), 24
        )
        cv2.line(
            frame, (center_x, center_y - 23), (center_x + 65, center_y - 2), (221, 120, 58), 24
        )
        cv2.line(
            frame, (center_x, center_y + 46), (center_x - 42, center_y + 119), (62, 78, 87), 27
        )
        cv2.line(
            frame, (center_x, center_y + 46), (center_x + 51, center_y + 105), (62, 78, 87), 27
        )
        frames.append(frame)
    return frames


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/verification/composite.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    result, _ = compose_sequence(
        build_frames(),
        ComposeSettings(threshold=16, feather=4, background="median", overlap="newest"),
    )
    Image.fromarray(result).save(output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

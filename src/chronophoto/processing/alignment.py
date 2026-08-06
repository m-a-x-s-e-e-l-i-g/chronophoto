from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
ProgressCallback = Callable[[int, str], None]


def _alignment_luma(frame: ImageArray) -> NDArray[np.float32]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray.astype(np.float32) / 255.0


def align_sequence(
    frames: Sequence[ImageArray],
    mode: str = "off",
    progress: ProgressCallback | None = None,
) -> list[ImageArray]:
    """Align a sequence to its first frame using background-dominant translation."""

    if mode == "off":
        return [np.ascontiguousarray(frame) for frame in frames]
    if mode != "translation":
        raise ValueError(f"Unsupported alignment mode: {mode}")
    if not frames:
        return []

    reference = _alignment_luma(frames[0])
    if float(np.std(reference)) < 0.02:
        return [np.ascontiguousarray(frame) for frame in frames]
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    aligned = [np.ascontiguousarray(frames[0])]
    maximum_shift = min(reference.shape) * 0.08

    for index, frame in enumerate(frames[1:], start=1):
        current = _alignment_luma(frame)
        (shift_x, shift_y), response = cv2.phaseCorrelate(reference, current, window)
        if (
            not np.isfinite(shift_x + shift_y)
            or response < 0.08
            or abs(shift_x) > maximum_shift
            or abs(shift_y) > maximum_shift
        ):
            aligned_frame = np.ascontiguousarray(frame)
        else:
            transform = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
            aligned_frame = cv2.warpAffine(
                frame,
                transform,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        aligned.append(aligned_frame)
        if progress:
            progress(
                int(((index + 1) / len(frames)) * 100),
                f"Aligning frame {index + 1} of {len(frames)}",
            )

    return aligned

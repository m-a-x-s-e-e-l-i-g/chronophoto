from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
ProgressCallback = Callable[[int, str], None]


class FrameAligner:
    """Align independently decoded frames against one stable reference frame."""

    def __init__(self, reference_frame: ImageArray, mode: str = "off") -> None:
        if mode not in {"off", "translation"}:
            raise ValueError(f"Unsupported alignment mode: {mode}")
        self.mode = mode
        self.reference_frame = np.ascontiguousarray(reference_frame)
        self.reference = _alignment_luma(reference_frame)
        self.enabled = mode == "translation" and float(np.std(self.reference)) >= 0.02
        self.window = (
            cv2.createHanningWindow(
                (self.reference.shape[1], self.reference.shape[0]),
                cv2.CV_32F,
            )
            if self.enabled
            else None
        )
        self.maximum_shift = min(self.reference.shape) * 0.08

    def align(self, frame: ImageArray) -> ImageArray:
        if not self.enabled:
            return np.ascontiguousarray(frame)
        current = _alignment_luma(frame)
        assert self.window is not None
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            self.reference,
            current,
            self.window,
        )
        if (
            not np.isfinite(shift_x + shift_y)
            or response < 0.08
            or abs(shift_x) > self.maximum_shift
            or abs(shift_y) > self.maximum_shift
        ):
            return np.ascontiguousarray(frame)
        transform = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        return cv2.warpAffine(
            frame,
            transform,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )


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

    aligner = FrameAligner(frames[0], mode)
    aligned = [aligner.align(frames[0])]

    for index, frame in enumerate(frames[1:], start=1):
        aligned.append(aligner.align(frame))
        if progress:
            progress(
                int(((index + 1) / len(frames)) * 100),
                f"Aligning frame {index + 1} of {len(frames)}",
            )

    return aligned

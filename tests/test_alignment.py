from __future__ import annotations

import cv2
import numpy as np

from chronophoto.processing.alignment import align_sequence


def test_translation_alignment_reduces_camera_shift() -> None:
    rng = np.random.default_rng(42)
    reference = rng.integers(20, 220, size=(180, 260, 3), dtype=np.uint8)
    reference = cv2.GaussianBlur(reference, (9, 9), 0)
    transform = np.float32([[1, 0, 8], [0, 1, -5]])
    shifted = cv2.warpAffine(reference, transform, (260, 180), borderMode=cv2.BORDER_REFLECT)

    aligned = align_sequence([reference, shifted], "translation")[1]
    crop = np.s_[16:-16, 16:-16]
    before = np.mean(np.abs(reference[crop].astype(float) - shifted[crop].astype(float)))
    after = np.mean(np.abs(reference[crop].astype(float) - aligned[crop].astype(float)))

    assert after < before * 0.35

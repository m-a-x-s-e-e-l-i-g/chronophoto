from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
MaskArray = NDArray[np.float32]

EFFECT_KINDS = (
    "opacity",
    "saturation",
    "blur",
    "jpeg_quality",
    "stippling",
    "dithering",
    "halftone",
)
EFFECT_LABELS = {
    "opacity": "Opacity",
    "saturation": "Saturation",
    "blur": "Blur",
    "jpeg_quality": "JPEG quality",
    "stippling": "Stippling",
    "dithering": "Dithering",
    "halftone": "Halftone",
}
EFFECT_NEUTRAL_VALUES = {
    "opacity": 100.0,
    "saturation": 100.0,
    "blur": 0.0,
    "jpeg_quality": 100.0,
    "stippling": 0.0,
    "dithering": 0.0,
    "halftone": 0.0,
}
EFFECT_DEFAULT_AMOUNTS = {
    "opacity": 0.0,
    "saturation": 0.0,
    "blur": 24.0,
    "jpeg_quality": 0.0,
    "stippling": 5.0,
    "dithering": 3.0,
    "halftone": 10.0,
}


@dataclass(slots=True, frozen=True)
class EffectKeyframe:
    """One value on a normalized 0..1 motion timeline."""

    progress: float
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.progress) or not 0.0 <= self.progress <= 1.0:
            raise ValueError("keyframe progress must be between 0 and 1")
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 100.0:
            raise ValueError("keyframe value must be between 0 and 100")


@dataclass(slots=True, frozen=True)
class EffectTrack:
    """A stackable, linearly interpolated subject effect."""

    kind: str
    keyframes: tuple[EffectKeyframe, ...]
    enabled: bool = True
    amount: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_KINDS:
            raise ValueError(f"unsupported effect: {self.kind}")
        if len(self.keyframes) < 2:
            raise ValueError("an effect track needs at least two keyframes")
        if self.keyframes[0].progress != 0.0 or self.keyframes[-1].progress != 1.0:
            raise ValueError("effect tracks must start at 0 and end at 1")
        if any(
            left.progress >= right.progress
            for left, right in zip(self.keyframes, self.keyframes[1:], strict=False)
        ):
            raise ValueError("effect keyframes must be ordered and unique")
        if not math.isfinite(self.amount) or not 0.0 <= self.amount <= 200.0:
            raise ValueError("effect amount must be between 0 and 200")

    def value_at(self, progress: float) -> float:
        position = max(0.0, min(1.0, progress))
        for left, right in zip(self.keyframes, self.keyframes[1:], strict=False):
            if position <= right.progress:
                width = right.progress - left.progress
                ratio = 0.0 if width <= 0.0 else (position - left.progress) / width
                return left.value + (right.value - left.value) * ratio
        return self.keyframes[-1].value


def neutral_effect_track(kind: str) -> EffectTrack:
    if kind not in EFFECT_KINDS:
        raise ValueError(f"unsupported effect: {kind}")
    neutral = EFFECT_NEUTRAL_VALUES[kind]
    return EffectTrack(
        kind,
        (EffectKeyframe(0.0, neutral), EffectKeyframe(1.0, neutral)),
        amount=EFFECT_DEFAULT_AMOUNTS[kind],
    )


def effect_preset(track: EffectTrack, preset: str) -> EffectTrack:
    values = {
        "rise_fall": ((0.0, 0.0), (0.5, 100.0), (1.0, 0.0)),
        "rise": ((0.0, 0.0), (1.0, 100.0)),
        "fall": ((0.0, 100.0), (1.0, 0.0)),
    }
    if preset not in values:
        raise ValueError(f"unsupported effect preset: {preset}")
    return EffectTrack(
        track.kind,
        tuple(EffectKeyframe(progress, value) for progress, value in values[preset]),
        track.enabled,
        track.amount,
    )


def _subject_bounds(mask: MaskArray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero((mask > 0.001).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def _masked_blur(rgb: NDArray[np.float32], alpha: MaskArray, sigma: float) -> NDArray[np.float32]:
    if sigma < 0.1:
        return rgb
    blurred_alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    premultiplied = rgb * alpha[..., None]
    blurred_color = cv2.GaussianBlur(premultiplied, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred_color / np.maximum(blurred_alpha[..., None], 0.001)


def _jpeg_roundtrip(rgb: NDArray[np.float32], quality: float) -> NDArray[np.float32]:
    encoded_quality = max(1, min(100, round(quality)))
    if encoded_quality >= 100:
        return rgb
    source = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg",
        source,
        (cv2.IMWRITE_JPEG_QUALITY, encoded_quality),
    )
    if not success:
        return rgb
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return rgb
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32)


def _block_reduce(image: NDArray[np.float32], scale: int) -> NDArray[np.float32]:
    if scale <= 1:
        return image
    height, width = image.shape[:2]
    small_width = max(1, math.ceil(width / scale))
    small_height = max(1, math.ceil(height / scale))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _dither(rgb: NDArray[np.float32], scale: int) -> NDArray[np.float32]:
    source = _block_reduce(rgb, scale)
    matrix = np.array(
        ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5)),
        dtype=np.float32,
    )
    height, width = source.shape[:2]
    threshold = np.tile(matrix, (math.ceil(height / 4), math.ceil(width / 4)))[:height, :width]
    threshold = (threshold - 7.5) * 8.0
    levels = 4.0
    return np.clip(np.floor((source + threshold[..., None]) / 255.0 * levels) / levels, 0, 1) * 255


def _stipple(rgb: NDArray[np.float32], cell_size: int) -> NDArray[np.float32]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cell = max(2, cell_size)
    local_luminance = _block_reduce(gray.astype(np.float32), cell) / 255.0
    y, x = np.indices((height, width), dtype=np.uint32)
    noise = ((x * 1597334677) ^ (y * 3812015801) ^ ((x // cell) * 9586891)) & 0xFFFF
    noise = noise.astype(np.float32) / 65535.0
    ink = noise < (1.0 - local_luminance) * 0.42
    return np.where(ink[..., None], rgb * 0.14, 242.0)


def _halftone(rgb: NDArray[np.float32], cell_size: int) -> NDArray[np.float32]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cell = max(3, cell_size)
    local_luminance = _block_reduce(gray.astype(np.float32), cell) / 255.0
    y, x = np.indices((height, width), dtype=np.float32)
    dx = np.mod(x, cell) - (cell - 1) / 2.0
    dy = np.mod(y, cell) - (cell - 1) / 2.0
    radius = np.sqrt(np.maximum(0.0, 1.0 - local_luminance)) * cell * 0.54
    ink = dx * dx + dy * dy <= radius * radius
    return np.where(ink[..., None], rgb * 0.12, 242.0)


def apply_effect_tracks(
    frame: ImageArray,
    mask: MaskArray,
    progress: float,
    tracks: tuple[EffectTrack, ...],
    *,
    pixel_scale: float = 1.0,
) -> tuple[ImageArray, MaskArray]:
    """Apply a stack of effects to subject pixels while preserving the background clip."""

    active = tuple(track for track in tracks if track.enabled)
    if not active:
        return frame, mask
    bounds = _subject_bounds(mask)
    if bounds is None:
        return frame, mask
    left, top, right, bottom = bounds
    result = frame.copy()
    effected_mask = mask.copy()
    rgb = result[top:bottom, left:right].astype(np.float32)
    alpha = effected_mask[top:bottom, left:right]

    for track in active:
        value = track.value_at(progress)
        if track.kind == "opacity":
            alpha = alpha * (value / 100.0)
        elif track.kind == "saturation" and value < 99.999:
            gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grayscale = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
            ratio = value / 100.0
            rgb = grayscale * (1.0 - ratio) + rgb * ratio
        elif track.kind == "blur" and value > 0.001:
            sigma = track.amount * max(0.01, pixel_scale) * value / 100.0
            rgb = _masked_blur(rgb, alpha, sigma)
        elif track.kind == "jpeg_quality" and value < 99.999:
            rgb = _jpeg_roundtrip(rgb, value)
        elif track.kind in {"stippling", "dithering", "halftone"} and value > 0.001:
            size = max(1, round(track.amount * max(0.01, pixel_scale)))
            if track.kind == "stippling":
                transformed = _stipple(rgb, size)
            elif track.kind == "dithering":
                transformed = _dither(rgb, size)
            else:
                transformed = _halftone(rgb, size)
            ratio = value / 100.0
            rgb = rgb * (1.0 - ratio) + transformed * ratio

    result[top:bottom, left:right] = np.clip(rgb, 0, 255).astype(np.uint8)
    effected_mask[top:bottom, left:right] = np.clip(alpha, 0.0, 1.0)
    return result, effected_mask

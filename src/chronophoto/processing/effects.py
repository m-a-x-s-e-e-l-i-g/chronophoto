from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
MaskArray = NDArray[np.float32]

BLEND_MODES = (
    "normal",
    "dissolve",
    "darken",
    "multiply",
    "color_burn",
    "linear_burn",
    "darker_color",
    "lighten",
    "screen",
    "color_dodge",
    "linear_dodge",
    "lighter_color",
    "overlay",
    "soft_light",
    "hard_light",
    "vivid_light",
    "linear_light",
    "pin_light",
    "hard_mix",
    "difference",
    "exclusion",
    "subtract",
    "divide",
    "hue",
    "color_saturation",
    "color",
    "luminosity",
)
BLEND_MODE_LABELS = {
    "normal": "Normal",
    "dissolve": "Dissolve",
    "darken": "Darken",
    "multiply": "Multiply",
    "color_burn": "Color Burn",
    "linear_burn": "Linear Burn",
    "darker_color": "Darker Color",
    "lighten": "Lighten",
    "screen": "Screen",
    "color_dodge": "Color Dodge",
    "linear_dodge": "Linear Dodge (Add)",
    "lighter_color": "Lighter Color",
    "overlay": "Overlay",
    "soft_light": "Soft Light",
    "hard_light": "Hard Light",
    "vivid_light": "Vivid Light",
    "linear_light": "Linear Light",
    "pin_light": "Pin Light",
    "hard_mix": "Hard Mix",
    "difference": "Difference",
    "exclusion": "Exclusion",
    "subtract": "Subtract",
    "divide": "Divide",
    "hue": "Hue",
    "color_saturation": "Saturation",
    "color": "Color",
    "luminosity": "Luminosity",
}

EFFECT_KINDS = (
    "opacity",
    "blend_mode",
    "saturation",
    "blur",
    "jpeg_quality",
    "stippling",
    "dithering",
    "halftone",
)
EFFECT_LABELS = {
    "opacity": "Opacity",
    "blend_mode": "Blend mode",
    "saturation": "Saturation",
    "blur": "Blur",
    "jpeg_quality": "JPEG quality",
    "stippling": "Stippling",
    "dithering": "Dithering",
    "halftone": "Halftone",
}
EFFECT_NEUTRAL_VALUES = {
    "opacity": 100.0,
    "blend_mode": 0.0,
    "saturation": 100.0,
    "blur": 0.0,
    "jpeg_quality": 100.0,
    "stippling": 0.0,
    "dithering": 0.0,
    "halftone": 0.0,
}
EFFECT_DEFAULT_AMOUNTS = {
    "opacity": 0.0,
    "blend_mode": 0.0,
    "saturation": 0.0,
    "blur": 24.0,
    "jpeg_quality": 0.0,
    "stippling": 5.0,
    "dithering": 3.0,
    "halftone": 10.0,
}

EFFECT_TIMING_BASES = ("movement", "trail")


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
    option: str = ""
    timing_basis: str = "movement"

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
        if self.timing_basis not in EFFECT_TIMING_BASES:
            raise ValueError(f"unsupported effect timing: {self.timing_basis}")
        if self.kind == "blend_mode":
            option = self.option or "normal"
            if option not in BLEND_MODES:
                raise ValueError(f"unsupported blend mode: {option}")
            object.__setattr__(self, "option", option)
        elif self.option:
            raise ValueError(f"{self.kind} does not support an option")

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
        option="multiply" if kind == "blend_mode" else "",
    )


def effect_preset(track: EffectTrack, preset: str) -> EffectTrack:
    values = {
        "full": ((0.0, 100.0), (1.0, 100.0)),
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
        track.option,
        track.timing_basis,
    )


def effect_track_progress(
    track: EffectTrack,
    movement_progress: float,
    trail_progress: float | None = None,
) -> float:
    """Select the timeline clock configured for one effect track."""

    if track.timing_basis == "trail" and trail_progress is not None:
        return trail_progress
    return movement_progress


def _color_dodge(backdrop: NDArray[np.float32], source: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.where(source >= 1.0, 1.0, np.minimum(1.0, backdrop / np.maximum(1.0 - source, 1e-6)))


def _color_burn(backdrop: NDArray[np.float32], source: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.where(
        source <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - backdrop) / np.maximum(source, 1e-6))
    )


def _soft_light(backdrop: NDArray[np.float32], source: NDArray[np.float32]) -> NDArray[np.float32]:
    low = backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)
    curve = np.where(
        backdrop <= 0.25,
        ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop,
        np.sqrt(backdrop),
    )
    high = backdrop + (2.0 * source - 1.0) * (curve - backdrop)
    return np.where(source <= 0.5, low, high)


def _luminosity(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    return rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11


def _saturation(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.max(rgb, axis=2) - np.min(rgb, axis=2)


def _clip_color(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    luminosity = _luminosity(rgb)
    minimum = np.min(rgb, axis=2)
    maximum = np.max(rgb, axis=2)
    low_scale = luminosity / np.maximum(luminosity - minimum, 1e-6)
    clipped = np.where(
        (minimum < 0.0)[..., None],
        luminosity[..., None] + (rgb - luminosity[..., None]) * low_scale[..., None],
        rgb,
    )
    high_scale = (1.0 - luminosity) / np.maximum(maximum - luminosity, 1e-6)
    return np.where(
        (maximum > 1.0)[..., None],
        luminosity[..., None] + (clipped - luminosity[..., None]) * high_scale[..., None],
        clipped,
    )


def _set_luminosity(
    rgb: NDArray[np.float32], luminosity: NDArray[np.float32]
) -> NDArray[np.float32]:
    shifted = rgb + (luminosity - _luminosity(rgb))[..., None]
    return _clip_color(shifted)


def _set_saturation(
    rgb: NDArray[np.float32], saturation: NDArray[np.float32]
) -> NDArray[np.float32]:
    order = np.argsort(rgb, axis=2)
    ordered = np.take_along_axis(rgb, order, axis=2)
    span = ordered[..., 2] - ordered[..., 0]
    middle = np.where(
        span > 1e-6,
        (ordered[..., 1] - ordered[..., 0]) * saturation / np.maximum(span, 1e-6),
        0.0,
    )
    adjusted = np.stack(
        (np.zeros_like(middle), middle, np.where(span > 1e-6, saturation, 0.0)), axis=2
    )
    result = np.empty_like(adjusted)
    np.put_along_axis(result, order, adjusted, axis=2)
    return result


def _component_blend(
    backdrop: NDArray[np.float32], source: NDArray[np.float32], mode: str
) -> NDArray[np.float32]:
    backdrop_luminosity = _luminosity(backdrop)
    if mode == "hue":
        return _set_luminosity(
            _set_saturation(source, _saturation(backdrop)),
            backdrop_luminosity,
        )
    if mode == "color_saturation":
        return _set_luminosity(
            _set_saturation(backdrop, _saturation(source)),
            backdrop_luminosity,
        )
    if mode == "color":
        return _set_luminosity(source, backdrop_luminosity)
    return _set_luminosity(backdrop, _luminosity(source))


def blend_mode_rgb(
    backdrop: NDArray[np.float32],
    source: NDArray[np.float32],
    mode: str,
) -> NDArray[np.float32]:
    """Return the Photoshop-style blend of source over backdrop in 0..255 RGB."""

    if mode not in BLEND_MODES:
        raise ValueError(f"unsupported blend mode: {mode}")
    base = np.clip(backdrop.astype(np.float32) / 255.0, 0.0, 1.0)
    blend = np.clip(source.astype(np.float32) / 255.0, 0.0, 1.0)
    if mode in {"normal", "dissolve"}:
        result = blend
    elif mode == "darken":
        result = np.minimum(base, blend)
    elif mode == "multiply":
        result = base * blend
    elif mode == "color_burn":
        result = _color_burn(base, blend)
    elif mode == "linear_burn":
        result = base + blend - 1.0
    elif mode == "darker_color":
        use_blend = np.sum(blend, axis=2) < np.sum(base, axis=2)
        result = np.where(use_blend[..., None], blend, base)
    elif mode == "lighten":
        result = np.maximum(base, blend)
    elif mode == "screen":
        result = 1.0 - (1.0 - base) * (1.0 - blend)
    elif mode == "color_dodge":
        result = _color_dodge(base, blend)
    elif mode == "linear_dodge":
        result = base + blend
    elif mode == "lighter_color":
        use_blend = np.sum(blend, axis=2) > np.sum(base, axis=2)
        result = np.where(use_blend[..., None], blend, base)
    elif mode == "overlay":
        result = np.where(
            base <= 0.5,
            2.0 * base * blend,
            1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        )
    elif mode == "soft_light":
        result = _soft_light(base, blend)
    elif mode == "hard_light":
        result = np.where(
            blend <= 0.5,
            2.0 * base * blend,
            1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        )
    elif mode == "vivid_light":
        result = np.where(
            blend <= 0.5,
            _color_burn(base, 2.0 * blend),
            _color_dodge(base, 2.0 * blend - 1.0),
        )
    elif mode == "linear_light":
        result = base + 2.0 * blend - 1.0
    elif mode == "pin_light":
        result = np.where(
            blend <= 0.5,
            np.minimum(base, 2.0 * blend),
            np.maximum(base, 2.0 * blend - 1.0),
        )
    elif mode == "hard_mix":
        vivid = np.where(
            blend <= 0.5,
            _color_burn(base, 2.0 * blend),
            _color_dodge(base, 2.0 * blend - 1.0),
        )
        result = (vivid >= 0.5).astype(np.float32)
    elif mode == "difference":
        result = np.abs(base - blend)
    elif mode == "exclusion":
        result = base + blend - 2.0 * base * blend
    elif mode == "subtract":
        result = base - blend
    elif mode == "divide":
        result = np.where(blend <= 0.0, 1.0, base / np.maximum(blend, 1e-6))
    else:
        result = _component_blend(base, blend, mode)
    return np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)


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
    trail_progress: float | None = None,
) -> tuple[ImageArray, MaskArray]:
    """Apply a stack of effects to subject pixels while preserving the background clip."""

    active = tuple(track for track in tracks if track.enabled and track.kind != "blend_mode")
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
        value = track.value_at(effect_track_progress(track, progress, trail_progress))
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


def apply_background_effect_tracks(
    background: ImageArray,
    tracks: tuple[EffectTrack, ...],
    *,
    pixel_scale: float = 1.0,
) -> ImageArray:
    """Process a clean plate as a full-frame layer over its untouched original."""

    active = tuple(track for track in tracks if track.enabled)
    if not active:
        return background
    full_mask = np.ones(background.shape[:2], dtype=np.float32)
    pixel_tracks = tuple(track for track in active if track.kind != "blend_mode")
    processed, alpha = apply_effect_tracks(
        background,
        full_mask,
        0.5,
        pixel_tracks,
        pixel_scale=pixel_scale,
    )
    backdrop = background.astype(np.float32)
    source = processed.astype(np.float32)
    alpha_3d = alpha[..., None]
    composite = source * alpha_3d + backdrop * (1.0 - alpha_3d)
    stacked_mode = False
    for track in active:
        if track.kind != "blend_mode":
            continue
        strength = track.value_at(0.5) / 100.0
        if strength <= 0.00001 or track.option == "normal":
            continue
        mode_backdrop = composite if stacked_mode else backdrop
        if track.option == "dissolve":
            height, width = alpha.shape
            y, x = np.indices((height, width), dtype=np.uint32)
            noise = ((x * 1597334677) ^ (y * 3812015801)) & 0xFFFF
            visible = noise.astype(np.float32) / 65535.0 < alpha
            mode_composite = np.where(visible[..., None], source, mode_backdrop)
        else:
            mode_source = blend_mode_rgb(mode_backdrop, source, track.option)
            mode_composite = mode_source * alpha_3d + mode_backdrop * (1.0 - alpha_3d)
        composite = composite * (1.0 - strength) + mode_composite * strength
        stacked_mode = True
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)

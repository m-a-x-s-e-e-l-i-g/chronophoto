from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from chronophoto.processing.effects import (
    EffectTrack,
    apply_background_effect_tracks,
    apply_effect_tracks,
    blend_mode_rgb,
    effect_track_progress,
)

ImageArray = NDArray[np.uint8]
ProgressCallback = Callable[[int, str], None]
CLEAN_PLATE_MAX_FRAMES = 21
CLEAN_PLATE_TILE_ROWS = 64


@dataclass(slots=True, frozen=True)
class _MaskComponent:
    label: int
    area: int
    x: float
    y: float


@dataclass(slots=True)
class ComposeSettings:
    """Parameters for motion-difference compositing."""

    threshold: int = 17
    feather: int = 1
    overlap: str = "newest"
    trail_style: str = "solid"
    smear_style: str = "none"
    background: str = "automatic"
    min_component_ratio: float = 0.00035
    # Kept as an input alias for projects created before trail/background scopes existed.
    effect_tracks: tuple[EffectTrack, ...] = ()
    trail_effect_tracks: tuple[EffectTrack, ...] = ()
    background_effect_tracks: tuple[EffectTrack, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.threshold <= 255:
            raise ValueError("threshold must be between 1 and 255")
        if not 0 <= self.feather <= 50:
            raise ValueError("feather must be between 0 and 50")
        if self.overlap not in {"newest", "oldest"}:
            raise ValueError("overlap must be 'newest' or 'oldest'")
        if self.trail_style != "solid":
            raise ValueError("trail style must be 'solid'")
        if self.smear_style not in {
            "none",
            "photographic",
            "dense_clones",
        }:
            raise ValueError("unsupported smear style")
        if self.background not in {"automatic", "median", "first", "last"}:
            raise ValueError("unsupported background mode")
        if self.effect_tracks and self.trail_effect_tracks:
            raise ValueError("use effect_tracks or trail_effect_tracks, not both")
        if self.effect_tracks:
            self.trail_effect_tracks = self.effect_tracks
        for name, tracks in (
            ("trail_effect_tracks", self.trail_effect_tracks),
            ("background_effect_tracks", self.background_effect_tracks),
        ):
            if any(not isinstance(track, EffectTrack) for track in tracks):
                raise ValueError(f"{name} must contain EffectTrack values")


@dataclass(slots=True)
class ComposeCache:
    """Reusable clean plate and compact 8-bit masks for an analyzed sequence."""

    background: ImageArray
    masks: list[NDArray[np.uint8]]

    def select(self, indices: Sequence[int]) -> ComposeCache:
        return ComposeCache(self.background, [self.masks[index] for index in indices])


def _notify(callback: ProgressCallback | None, value: int, message: str) -> None:
    if callback is not None:
        callback(value, message)


def _validate_frames(frames: Sequence[ImageArray]) -> list[ImageArray]:
    if len(frames) < 2:
        raise ValueError("At least two frames are required")

    normalized: list[ImageArray] = []
    first_shape = frames[0].shape
    for frame in frames:
        if frame.shape != first_shape:
            raise ValueError("All frames must have identical dimensions")
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Frames must be RGB or RGBA images")
        normalized.append(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8))
    return normalized


def _clean_plate_frames(frames: Sequence[ImageArray]) -> list[ImageArray]:
    """Evenly sample long sequences without changing their temporal coverage."""

    if len(frames) <= CLEAN_PLATE_MAX_FRAMES:
        return list(frames)
    indices = np.rint(np.linspace(0, len(frames) - 1, CLEAN_PLATE_MAX_FRAMES)).astype(np.intp)
    return [frames[int(index)] for index in indices]


def _clean_plate_frame_count(frame_count: int, mode: str) -> int:
    if mode in {"first", "last"}:
        return 1
    return min(frame_count, CLEAN_PLATE_MAX_FRAMES)


def build_background(frames: Sequence[ImageArray], mode: str) -> ImageArray:
    """Create the clean plate used underneath the detected subjects."""

    if mode == "first":
        return frames[0].copy()
    if mode == "last":
        return frames[-1].copy()

    # "automatic" currently favours a robust temporal median. An evenly spaced
    # sample preserves the full time range while avoiding redundant work for
    # high-frame-rate clips. Small row tiles also keep NumPy's median working set
    # CPU-cache friendly and bound memory use during 4K export.
    plate_frames = _clean_plate_frames(frames)
    background = np.empty_like(frames[0])
    for top in range(0, frames[0].shape[0], CLEAN_PLATE_TILE_ROWS):
        bottom = min(frames[0].shape[0], top + CLEAN_PLATE_TILE_ROWS)
        tile = np.stack([frame[top:bottom] for frame in plate_frames], axis=0)
        background[top:bottom] = np.median(tile, axis=0).astype(np.uint8)
    return background


def create_motion_mask(
    frame: ImageArray,
    background: ImageArray,
    settings: ComposeSettings,
    *,
    background_blur: ImageArray | None = None,
) -> NDArray[np.float32]:
    """Return a feathered 0..1 alpha mask for changed image regions."""

    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    if background_blur is None:
        background_blur = cv2.GaussianBlur(background, (5, 5), 0)
    difference = cv2.absdiff(frame_blur, background_blur)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(difference_gray, settings.threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum_area = max(24, int(mask.shape[0] * mask.shape[1] * settings.min_component_ratio))
    cleaned = np.zeros_like(mask)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == component] = 255

    if settings.feather:
        radius = settings.feather * 2 + 1
        cleaned = cv2.GaussianBlur(cleaned, (radius, radius), 0)

    return cleaned.astype(np.float32) / 255.0


def build_compose_cache(
    frames: Sequence[ImageArray],
    settings: ComposeSettings | None = None,
    progress: ProgressCallback | None = None,
) -> ComposeCache:
    """Analyze a sequence once so later frame selections can reuse its masks."""

    settings = settings or ComposeSettings()
    source_frames = _validate_frames(frames)
    plate_count = _clean_plate_frame_count(len(source_frames), settings.background)
    _notify(
        progress,
        8,
        f"Building reusable clean plate from {plate_count} of {len(source_frames)} frames",
    )
    background = build_background(source_frames, settings.background)
    background_blur = cv2.GaussianBlur(background, (5, 5), 0)
    masks: list[NDArray[np.uint8]] = []
    for index, frame in enumerate(source_frames):
        value = 12 + int((index / len(source_frames)) * 84)
        _notify(progress, value, f"Analyzing pose {index + 1} of {len(source_frames)}")
        mask = create_motion_mask(
            frame,
            background,
            settings,
            background_blur=background_blur,
        )
        masks.append(np.rint(mask * 255.0).astype(np.uint8))
    _notify(progress, 100, f"Cached {len(masks)} pose masks")
    return ComposeCache(background, masks)


def _mask_components(mask: NDArray[np.float32]) -> list[_MaskComponent]:
    minimum_area = max(24, int(mask.size * 0.0001))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0.15).astype(np.uint8),
        8,
    )
    return [
        _MaskComponent(
            component,
            int(stats[component, cv2.CC_STAT_AREA]),
            float(centroids[component, 0]),
            float(centroids[component, 1]),
        )
        for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area
    ]


def _track_score(track: Sequence[_MaskComponent]) -> tuple[float, float]:
    xs = [component.x for component in track]
    ys = [component.y for component in track]
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    distance = sum(
        math.hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(track, track[1:], strict=False)
    )
    return span * span / max(distance, 1.0), span


def _maximum_tracking_step(mask_shape: tuple[int, int]) -> float:
    return max(40.0, math.hypot(*mask_shape) * 0.06)


def _isolate_primary_motion(
    masks: Sequence[NDArray[np.float32]],
) -> list[NDArray[np.float32]]:
    """Keep a confident, smoothly moving component and reject ambient motion."""

    if len(masks) < 3:
        return list(masks)
    components = [_mask_components(mask) for mask in masks]
    if not components[0] or any(not frame_components for frame_components in components):
        return list(masks)

    diagonal = math.hypot(*masks[0].shape)
    maximum_step = _maximum_tracking_step(masks[0].shape)
    tracks: list[list[_MaskComponent]] = []
    for start in components[0]:
        track = [start]
        previous = start
        for frame_components in components[1:]:
            matches: list[tuple[float, _MaskComponent]] = []
            for candidate in frame_components:
                distance = math.hypot(candidate.x - previous.x, candidate.y - previous.y)
                area_ratio = max(candidate.area, previous.area) / max(
                    1,
                    min(candidate.area, previous.area),
                )
                if distance <= maximum_step and area_ratio <= 12.0:
                    score = distance + maximum_step * 0.08 * abs(math.log(area_ratio))
                    matches.append((score, candidate))
            if not matches:
                break
            previous = min(matches, key=lambda match: match[0])[1]
            track.append(previous)
        if len(track) == len(masks):
            tracks.append(track)

    if not tracks:
        return list(masks)
    ranked = sorted(
        ((_track_score(track), track) for track in tracks),
        key=lambda item: item[0][0],
        reverse=True,
    )
    (best_score, best_span), best_track = ranked[0]
    second_score = ranked[1][0][0] if len(ranked) > 1 else 0.0
    if (
        best_span < max(16.0, diagonal * 0.04)
        or best_score < diagonal * 0.04
        or (second_score > 0.0 and best_score < second_score * 1.6)
    ):
        return list(masks)

    radius = max(2, round(min(masks[0].shape) * 0.003))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    isolated: list[NDArray[np.float32]] = []
    for mask, component in zip(masks, best_track, strict=True):
        _, labels, _, _ = cv2.connectedComponentsWithStats(
            (mask > 0.15).astype(np.uint8),
            8,
        )
        keep = cv2.dilate((labels == component.label).astype(np.uint8), kernel)
        isolated.append(mask * keep)
    return isolated


def _blend_pose(
    result: NDArray[np.float32],
    frame: ImageArray,
    mask: NDArray[np.float32],
    blend_tracks: Sequence[EffectTrack] = (),
    effect_progress: float = 0.0,
    trail_progress: float | None = None,
) -> NDArray[np.float32]:
    points = cv2.findNonZero((mask > 0.001).astype(np.uint8))
    if points is None:
        return result
    x, y, width, height = cv2.boundingRect(points)
    target = result[y : y + height, x : x + width]
    source = frame[y : y + height, x : x + width].astype(np.float32)
    alpha = mask[y : y + height, x : x + width]
    target[:] = _blend_region(
        target,
        source,
        alpha,
        blend_tracks,
        effect_progress,
        trail_progress,
        origin=(x, y),
    )
    return result


def _blend_region(
    target: NDArray[np.float32],
    source: NDArray[np.float32],
    alpha: NDArray[np.float32],
    blend_tracks: Sequence[EffectTrack],
    effect_progress: float,
    trail_progress: float | None = None,
    *,
    origin: tuple[int, int] = (0, 0),
) -> NDArray[np.float32]:
    """Composite one masked layer, optionally mixing in true backdrop blend modes."""

    alpha_3d = np.clip(alpha, 0.0, 1.0)[..., None]
    normal = source * alpha_3d + target * (1.0 - alpha_3d)
    composite = normal
    stacked_mode = False
    for track in blend_tracks:
        if not track.enabled or track.kind != "blend_mode":
            continue
        strength = (
            track.value_at(effect_track_progress(track, effect_progress, trail_progress)) / 100.0
        )
        if strength <= 0.00001 or track.option == "normal":
            continue
        mode_backdrop = composite if stacked_mode else target
        if track.option == "dissolve":
            height, width = alpha.shape
            y, x = np.indices((height, width), dtype=np.uint32)
            x += np.uint32(max(0, origin[0]))
            y += np.uint32(max(0, origin[1]))
            noise = ((x * 1597334677) ^ (y * 3812015801)) & 0xFFFF
            visible = noise.astype(np.float32) / 65535.0 < alpha
            mode_composite = np.where(visible[..., None], source, mode_backdrop)
        else:
            mode_source = blend_mode_rgb(mode_backdrop, source, track.option)
            mode_composite = mode_source * alpha_3d + mode_backdrop * (1.0 - alpha_3d)
        composite = composite * (1.0 - strength) + mode_composite * strength
        stacked_mode = True
    return composite


def _mask_centroid(mask: NDArray[np.float32]) -> tuple[float, float] | None:
    moments = cv2.moments(mask)
    if moments["m00"] <= 0.001:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _translate(
    source: NDArray[np.float32],
    dx: float,
    dy: float,
) -> NDArray[np.float32]:
    height, width = source.shape[:2]
    matrix = np.array(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32)
    return cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _dense_clone_step_count(dx: float, dy: float) -> int:
    """Return enough sub-frame steps to move no more than one pixel per copy."""

    return max(1, math.ceil(max(abs(dx), abs(dy))))


def _pair_bounds(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
) -> tuple[int, int, int, int] | None:
    active = np.maximum(first, second) > 0.01
    points = cv2.findNonZero(active.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    padding = 4
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(first.shape[1], x + width + padding),
        min(first.shape[0], y + height + padding),
    )


def _swept_pair(
    first_frame: ImageArray,
    second_frame: ImageArray,
    first_mask: NDArray[np.float32],
    second_mask: NDArray[np.float32],
    maximum_distance: float | None = None,
    first_tracking_mask: NDArray[np.float32] | None = None,
    second_tracking_mask: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    first_center = _mask_centroid(
        first_mask if first_tracking_mask is None else first_tracking_mask
    )
    second_center = _mask_centroid(
        second_mask if second_tracking_mask is None else second_tracking_mask
    )
    if first_center is None or second_center is None:
        return None
    dx = second_center[0] - first_center[0]
    dy = second_center[1] - first_center[1]
    distance = math.hypot(dx, dy)
    if maximum_distance is not None and distance > maximum_distance:
        return None
    step_spacing = max(3.0, min(first_mask.shape) * 0.012)
    intermediate_count = max(6, min(18, math.ceil(distance / step_spacing)))
    positions = np.linspace(0.0, 1.0, intermediate_count + 2, dtype=np.float32)

    alpha = np.maximum(first_mask, second_mask)
    weight_sum = first_mask + second_mask
    first_premultiplied = first_frame.astype(np.float32) * first_mask[..., None]
    second_premultiplied = second_frame.astype(np.float32) * second_mask[..., None]
    color_sum = first_premultiplied + second_premultiplied

    for position in positions[1:-1]:
        first_weight = 1.0 - float(position)
        second_weight = float(position)
        moved_first_mask = _translate(first_mask, dx * position, dy * position)
        moved_second_mask = _translate(
            second_mask,
            -dx * (1.0 - position),
            -dy * (1.0 - position),
        )
        intermediate_mask = np.clip(
            moved_first_mask * first_weight + moved_second_mask * second_weight,
            0.0,
            1.0,
        )
        np.maximum(alpha, intermediate_mask, out=alpha)
        weight_sum += intermediate_mask
        color_sum += (
            _translate(first_premultiplied, dx * position, dy * position) * first_weight
            + _translate(
                second_premultiplied,
                -dx * (1.0 - position),
                -dy * (1.0 - position),
            )
            * second_weight
        )

    color = color_sum / np.maximum(weight_sum[..., None], 0.001)
    return color, alpha


def _interpolated_optional_progress(
    positions: Sequence[float | None],
    index: int,
    amount: float,
) -> float | None:
    if not positions:
        return None
    first = positions[index]
    second = positions[index + 1]
    if first is None or second is None:
        return None
    return first + amount * (second - first)


def _apply_motion_ribbon(
    background: ImageArray,
    frames: Sequence[ImageArray],
    masks: Sequence[NDArray[np.float32]],
    progress: ProgressCallback | None,
    blend_tracks: Sequence[EffectTrack] = (),
    effect_progress: Sequence[float] = (),
    trail_progress: Sequence[float | None] = (),
    tracking_masks: Sequence[NDArray[np.float32]] = (),
    maximum_pair_distance: float | None = None,
) -> NDArray[np.float32]:
    """Build pairwise silhouette connectors underneath the original poses."""

    result = background.astype(np.float32)
    color_sum = np.zeros_like(result)
    weight_sum = np.zeros(result.shape[:2], dtype=np.float32)
    alpha_union = np.zeros(result.shape[:2], dtype=np.float32)
    pair_total = max(1, len(frames) - 1)
    geometry_masks = tracking_masks or masks

    for index in range(len(frames) - 1):
        bounds = _pair_bounds(geometry_masks[index], geometry_masks[index + 1])
        if bounds is not None:
            left, top, right, bottom = bounds
            pair = _swept_pair(
                frames[index][top:bottom, left:right],
                frames[index + 1][top:bottom, left:right],
                masks[index][top:bottom, left:right],
                masks[index + 1][top:bottom, left:right],
                maximum_pair_distance,
                geometry_masks[index][top:bottom, left:right],
                geometry_masks[index + 1][top:bottom, left:right],
            )
            if pair is not None:
                pair_color, pair_alpha = pair
                if blend_tracks:
                    ribbon_alpha = cv2.GaussianBlur(pair_alpha, (3, 3), 0)
                    ribbon_alpha = np.clip(ribbon_alpha * 0.90, 0.0, 0.90)
                    pair_progress = (
                        (effect_progress[index] + effect_progress[index + 1]) / 2.0
                        if effect_progress
                        else (index + 0.5) / max(1, len(frames) - 1)
                    )
                    pair_trail_progress = _interpolated_optional_progress(
                        trail_progress, index, 0.5
                    )
                    target = result[top:bottom, left:right]
                    target[:] = _blend_region(
                        target,
                        pair_color,
                        ribbon_alpha,
                        blend_tracks,
                        pair_progress,
                        pair_trail_progress,
                        origin=(left, top),
                    )
                else:
                    color_sum[top:bottom, left:right] += pair_color * pair_alpha[..., None]
                    weight_sum[top:bottom, left:right] += pair_alpha
                    np.maximum(
                        alpha_union[top:bottom, left:right],
                        pair_alpha,
                        out=alpha_union[top:bottom, left:right],
                    )
        value = 60 + int(((index + 1) / pair_total) * 20)
        _notify(progress, value, f"Connecting silhouette {index + 1} of {pair_total}")

    if blend_tracks or not np.any(alpha_union):
        return result
    ribbon_source = color_sum / np.maximum(weight_sum[..., None], 0.001)
    ribbon_alpha = cv2.GaussianBlur(alpha_union, (3, 3), 0)
    ribbon_alpha = np.clip(ribbon_alpha * 0.90, 0.0, 0.90)[..., None]
    return ribbon_source * ribbon_alpha + result * (1.0 - ribbon_alpha)


def _apply_dense_clone_trail(
    background: ImageArray,
    frames: Sequence[ImageArray],
    masks: Sequence[NDArray[np.float32]],
    overlap: str,
    progress: ProgressCallback | None,
    blend_tracks: Sequence[EffectTrack] = (),
    effect_progress: Sequence[float] = (),
    trail_progress: Sequence[float | None] = (),
    analysis_background: ImageArray | None = None,
    tracking_masks: Sequence[NDArray[np.float32]] = (),
    maximum_pair_distance: float | None = None,
) -> NDArray[np.float32]:
    """Fill motion with overlapping photographic copies at pixel-spaced positions."""

    result = background.astype(np.float32)
    pair_total = max(1, len(frames) - 1)
    newest_on_top = overlap == "newest"
    geometry_masks = tracking_masks or masks

    if not newest_on_top:
        result = _blend_pose(
            result,
            frames[-1],
            masks[-1],
            blend_tracks,
            effect_progress[-1] if effect_progress else 1.0,
            trail_progress[-1] if trail_progress else None,
        )

    pair_indices = range(len(frames) - 1)
    if not newest_on_top:
        pair_indices = range(len(frames) - 2, -1, -1)

    for completed, index in enumerate(pair_indices, start=1):
        bounds = _pair_bounds(geometry_masks[index], geometry_masks[index + 1])
        if bounds is not None:
            left, top, right, bottom = bounds
            first_frame = frames[index][top:bottom, left:right].astype(np.float32)
            second_frame = frames[index + 1][top:bottom, left:right].astype(np.float32)
            first_mask = masks[index][top:bottom, left:right]
            second_mask = masks[index + 1][top:bottom, left:right]
            first_center = _mask_centroid(geometry_masks[index][top:bottom, left:right])
            second_center = _mask_centroid(geometry_masks[index + 1][top:bottom, left:right])
            if first_center is not None and second_center is not None:
                dx = second_center[0] - first_center[0]
                dy = second_center[1] - first_center[1]
                if maximum_pair_distance is not None and math.hypot(dx, dy) > maximum_pair_distance:
                    continue
                step_count = _dense_clone_step_count(dx, dy)
                steps = range(step_count)
                if not newest_on_top:
                    steps = range(step_count - 1, -1, -1)

                clean_plate = background if analysis_background is None else analysis_background
                pair_plate = clean_plate[top:bottom, left:right].astype(np.float32)
                first_detail = np.max(np.abs(first_frame - pair_plate), axis=2)
                second_detail = np.max(np.abs(second_frame - pair_plate), axis=2)
                first_alpha = first_mask * np.clip(first_detail / 12.0, 0.0, 1.0)
                second_alpha = second_mask * np.clip(second_detail / 12.0, 0.0, 1.0)
                first_source = first_frame * first_alpha[..., None]
                second_source = second_frame * second_alpha[..., None]
                target = result[top:bottom, left:right]
                for step in steps:
                    position = step / step_count
                    if position < 0.5:
                        alpha = _translate(first_alpha, dx * position, dy * position)
                        source = _translate(first_source, dx * position, dy * position)
                    else:
                        remaining = 1.0 - position
                        alpha = _translate(second_alpha, -dx * remaining, -dy * remaining)
                        source = _translate(second_source, -dx * remaining, -dy * remaining)
                    if blend_tracks:
                        source_rgb = source / np.maximum(alpha[..., None], 0.001)
                        pair_progress = (
                            effect_progress[index]
                            + position * (effect_progress[index + 1] - effect_progress[index])
                            if effect_progress
                            else (index + position) / max(1, len(frames) - 1)
                        )
                        pair_trail_progress = _interpolated_optional_progress(
                            trail_progress, index, position
                        )
                        target[:] = _blend_region(
                            target,
                            source_rgb,
                            alpha,
                            blend_tracks,
                            pair_progress,
                            pair_trail_progress,
                            origin=(left, top),
                        )
                    else:
                        target[:] = source + target * (1.0 - alpha[..., None])

        value = 60 + int((completed / pair_total) * 20)
        _notify(progress, value, f"Cloning silhouette {completed} of {pair_total}")

    if newest_on_top:
        result = _blend_pose(
            result,
            frames[-1],
            masks[-1],
            blend_tracks,
            effect_progress[-1] if effect_progress else 1.0,
            trail_progress[-1] if trail_progress else None,
        )
    return result


def _pose_stack_order(
    frame_count: int,
    overlap: str,
    top_pose_index: int | None = None,
) -> list[int]:
    if top_pose_index is not None and not 0 <= top_pose_index < frame_count:
        raise IndexError("top pose index is outside the selected frames")
    order = list(range(frame_count))
    if overlap == "oldest":
        order.reverse()
    if top_pose_index is not None:
        order.remove(top_pose_index)
        order.append(top_pose_index)
    return order


def compose_sequence(
    frames: Sequence[ImageArray],
    settings: ComposeSettings | None = None,
    progress: ProgressCallback | None = None,
    *,
    return_masks: bool = True,
    cache: ComposeCache | None = None,
    effect_progress: Sequence[float] | None = None,
    trail_progress: Sequence[float] | None = None,
    effect_pixel_scale: float = 1.0,
    frame_contiguous: bool = False,
    top_pose_index: int | None = None,
) -> tuple[ImageArray, list[NDArray[np.float32]]]:
    """Composite a chronological sequence and return the result plus pose masks."""

    settings = settings or ComposeSettings()
    source_frames = _validate_frames(frames)
    if effect_progress is None:
        progress_positions = np.linspace(0.0, 1.0, len(source_frames)).tolist()
    else:
        progress_positions = [float(position) for position in effect_progress]
        if len(progress_positions) != len(source_frames):
            raise ValueError("Effect progress must match the selected frames")
        if any(
            not math.isfinite(position) or not 0.0 <= position <= 1.0
            for position in progress_positions
        ):
            raise ValueError("Effect progress values must be between 0 and 1")
        if any(
            left > right
            for left, right in zip(progress_positions, progress_positions[1:], strict=False)
        ):
            raise ValueError("Effect progress values must be chronological")
    if trail_progress is None:
        trail_positions: list[float | None] = [None] * len(source_frames)
    else:
        trail_positions = [float(position) for position in trail_progress]
        if len(trail_positions) != len(source_frames):
            raise ValueError("Trail progress must match the selected frames")
        if any(
            not math.isfinite(position) or not 0.0 <= position <= 1.0
            for position in trail_positions
        ):
            raise ValueError("Trail progress values must be between 0 and 1")
        if any(
            left > right for left, right in zip(trail_positions, trail_positions[1:], strict=False)
        ):
            raise ValueError("Trail progress values must be chronological")
    if not math.isfinite(effect_pixel_scale) or effect_pixel_scale <= 0.0:
        raise ValueError("effect_pixel_scale must be positive")
    if cache is not None:
        if cache.background.shape != source_frames[0].shape:
            raise ValueError("Cached clean plate does not match the selected frames")
        if len(cache.masks) != len(source_frames):
            raise ValueError("Cached masks do not match the selected frames")
        if any(mask.shape != source_frames[0].shape[:2] for mask in cache.masks):
            raise ValueError("Cached mask dimensions do not match the selected frames")
        background = cache.background
        masks = [mask.astype(np.float32) / 255.0 for mask in cache.masks]
        _notify(progress, 8, "Using cached clean plate")
        _notify(progress, 58, f"Using {len(masks)} cached pose masks")
    else:
        plate_count = _clean_plate_frame_count(len(source_frames), settings.background)
        _notify(
            progress,
            8,
            f"Building clean plate from {plate_count} of {len(source_frames)} frames",
        )
        background = build_background(source_frames, settings.background)
        background_blur = cv2.GaussianBlur(background, (5, 5), 0)

    order = _pose_stack_order(len(source_frames), settings.overlap, top_pose_index)
    smear_enabled = settings.smear_style != "none"
    active_trail_effects = tuple(track for track in settings.trail_effect_tracks if track.enabled)
    active_background_effects = tuple(
        track for track in settings.background_effect_tracks if track.enabled
    )
    pixel_effects = tuple(track for track in active_trail_effects if track.kind != "blend_mode")
    blend_effects = tuple(track for track in active_trail_effects if track.kind == "blend_mode")
    if (
        cache is None
        and not smear_enabled
        and not return_masks
        and not active_trail_effects
        and not active_background_effects
    ):
        result = background.astype(np.float32)
        for position, index in enumerate(order):
            value = 12 + int((position / len(order)) * 84)
            _notify(progress, value, f"Finding pose {index + 1} of {len(source_frames)}")
            mask = create_motion_mask(
                source_frames[index],
                background,
                settings,
                background_blur=background_blur,
            )
            result = _blend_pose(result, source_frames[index], mask)
            value = 12 + int(((position + 1) / len(order)) * 86)
            _notify(progress, value, "Compositing sharp poses")
        _notify(progress, 100, "Composite ready")
        return np.clip(result, 0, 255).astype(np.uint8), []

    if cache is None:
        masks_by_index: list[NDArray[np.float32] | None] = [None] * len(source_frames)
        for index in range(len(source_frames)):
            value = 12 + int((index / len(source_frames)) * 46)
            _notify(progress, value, f"Finding pose {index + 1} of {len(source_frames)}")
            mask = create_motion_mask(
                source_frames[index],
                background,
                settings,
                background_blur=background_blur,
            )
            masks_by_index[index] = mask
        masks = [mask for mask in masks_by_index if mask is not None]
    if smear_enabled:
        _notify(progress, 59, "Tracking the primary moving subject")
        masks = _isolate_primary_motion(masks)
    render_background = apply_background_effect_tracks(
        background,
        active_background_effects,
        pixel_scale=effect_pixel_scale,
    )
    returned_masks = masks
    effected_frames: list[ImageArray] = []
    effected_masks: list[NDArray[np.float32]] = []
    if pixel_effects:
        effect_total = len(source_frames)
        for index, (frame, mask, position, trail_position) in enumerate(
            zip(source_frames, masks, progress_positions, trail_positions, strict=True)
        ):
            effected_frame, effected_mask = apply_effect_tracks(
                frame,
                mask,
                position,
                pixel_effects,
                pixel_scale=effect_pixel_scale,
                trail_progress=trail_position,
            )
            effected_frames.append(effected_frame)
            effected_masks.append(effected_mask)
            _notify(progress, 59, f"Applying effects to pose {index + 1} of {effect_total}")
    else:
        effected_frames = source_frames
        effected_masks = masks

    maximum_pair_distance = (
        _maximum_tracking_step(effected_masks[0].shape) if frame_contiguous else None
    )
    if smear_enabled:
        if settings.smear_style == "dense_clones":
            _notify(progress, 60, "Filling motion with pixel-spaced clones")
            result = _apply_dense_clone_trail(
                render_background,
                effected_frames,
                effected_masks,
                settings.overlap,
                progress,
                blend_effects,
                progress_positions,
                trail_positions,
                background,
                masks,
                maximum_pair_distance,
            )
        else:
            _notify(progress, 60, "Building silhouette ribbon")
            result = _apply_motion_ribbon(
                render_background,
                effected_frames,
                effected_masks,
                progress,
                blend_effects,
                progress_positions,
                trail_positions,
                masks,
                maximum_pair_distance,
            )
    else:
        result = render_background.astype(np.float32)

    pose_order = order
    if smear_enabled:
        if frame_contiguous:
            top_endpoint = len(source_frames) - 1 if settings.overlap == "newest" else 0
            pose_order = [top_endpoint]
        else:
            endpoints = {0, len(source_frames) - 1}
            if top_pose_index is not None:
                endpoints.add(top_pose_index)
            pose_order = [index for index in order if index in endpoints]
    for position, index in enumerate(pose_order):
        result = _blend_pose(
            result,
            effected_frames[index],
            effected_masks[index],
            blend_effects,
            progress_positions[index],
            trail_positions[index],
        )
        value = 82 + int(((position + 1) / len(pose_order)) * 16)
        if smear_enabled and frame_contiguous:
            message = "Compositing sharp trail endpoint"
        else:
            message = "Compositing sharp endpoints" if smear_enabled else "Compositing sharp poses"
        _notify(progress, value, message)

    _notify(progress, 100, "Composite ready")
    output_masks = returned_masks if return_masks else []
    return np.clip(result, 0, 255).astype(np.uint8), output_masks

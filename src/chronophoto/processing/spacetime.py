from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

StretchDirection = Literal["left", "right", "up", "down", "both-horizontal", "both-vertical"]
SliceKind = Literal["xy", "xt", "yt", "diagonal", "surface"]


def silhouette_edge_stretch(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    direction: StretchDirection = "right",
    distance: int | None = None,
    fade: float = 0.0,
) -> np.ndarray:
    """Extrude silhouette boundary pixels along scanlines.

    The source silhouette is preserved. Only pixels outside the mask are replaced,
    using the first/last mask pixel on each scanline as the sampled colour.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must have shape (height, width, 3|4)")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask dimensions must match image")
    if distance is not None and distance < 0:
        raise ValueError("distance must be non-negative")
    fade = float(np.clip(fade, 0.0, 1.0))

    if direction in {"up", "down", "both-vertical"}:
        mapping = {"up": "left", "down": "right", "both-vertical": "both-horizontal"}
        transposed = silhouette_edge_stretch(
            np.swapaxes(image, 0, 1),
            np.swapaxes(mask, 0, 1),
            direction=mapping[direction],  # type: ignore[arg-type]
            distance=distance,
            fade=fade,
        )
        return np.swapaxes(transposed, 0, 1)

    source = image.astype(np.float32)
    output = source.copy()
    active = mask > 0
    width = image.shape[1]
    for y in range(image.shape[0]):
        hits = np.flatnonzero(active[y])
        if hits.size == 0:
            continue
        left, right = int(hits[0]), int(hits[-1])
        sides = ()
        if direction in {"left", "both-horizontal"}:
            sides += ((left, -1),)
        if direction in {"right", "both-horizontal"}:
            sides += ((right, 1),)
        for edge, step in sides:
            available = edge if step < 0 else width - edge - 1
            length = available if distance is None else min(distance, available)
            if length == 0:
                continue
            positions = edge + step * np.arange(1, length + 1)
            if fade:
                alpha = 1.0 - fade * np.arange(1, length + 1) / length
                alpha = alpha[:, None]
                output[y, positions] = (
                    source[y, edge][None, :] * alpha
                    + source[y, positions] * (1.0 - alpha)
                )
            else:
                output[y, positions] = source[y, edge]
    return np.clip(output, 0, 255).astype(image.dtype)


@dataclass(frozen=True, slots=True)
class TimeSurface:
    """A normalized time surface: t = base + x*x_slope + y*y_slope."""

    base: float = 0.5
    x_slope: float = 0.0
    y_slope: float = 0.0

    def map(self, height: int, width: int, *, phase: float = 0.0) -> np.ndarray:
        x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
        y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
        return np.clip(self.base + phase + x * self.x_slope + y * self.y_slope, 0.0, 1.0)


def sample_time_surface(frames: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Sample an X × Y × Time video volume with linear interpolation in time."""
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError("frames must have shape (time, height, width, 3|4)")
    if surface.shape != frames.shape[1:3]:
        raise ValueError("surface dimensions must match frame dimensions")
    scaled = np.clip(surface, 0.0, 1.0) * (len(frames) - 1)
    before = np.floor(scaled).astype(np.intp)
    after = np.minimum(before + 1, len(frames) - 1)
    amount = (scaled - before)[..., None]
    yy, xx = np.indices(surface.shape)
    result = frames[before, yy, xx] * (1.0 - amount) + frames[after, yy, xx] * amount
    return np.clip(result, 0, 255).astype(frames.dtype)


def slice_video_volume(
    frames: np.ndarray,
    kind: SliceKind,
    *,
    position: float = 0.5,
    surface: TimeSurface | None = None,
    phase: float = 0.0,
) -> np.ndarray:
    """Extract an XY, XT, YT, diagonal, or arbitrary surface from a video volume."""
    if frames.ndim != 4 or len(frames) == 0:
        raise ValueError("frames must be a non-empty 4D array")
    position = float(np.clip(position, 0.0, 1.0))
    if kind == "xy":
        return frames[round(position * (len(frames) - 1))].copy()
    if kind == "xt":
        y = round(position * (frames.shape[1] - 1))
        return frames[:, y, :, :].copy()
    if kind == "yt":
        x = round(position * (frames.shape[2] - 1))
        return np.swapaxes(frames[:, :, x, :], 0, 1).copy()
    if kind == "diagonal":
        count = max(frames.shape[1], frames.shape[2])
        ys = np.rint(np.linspace(0, frames.shape[1] - 1, count)).astype(np.intp)
        xs = np.rint(np.linspace(0, frames.shape[2] - 1, count)).astype(np.intp)
        return frames[:, ys, xs, :].copy()
    if kind == "surface":
        selected = surface or TimeSurface()
        return sample_time_surface(
            frames, selected.map(frames.shape[1], frames.shape[2], phase=phase)
        )
    raise ValueError(f"unknown slice kind: {kind}")

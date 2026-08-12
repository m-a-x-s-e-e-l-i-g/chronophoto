from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from chronophoto.processing.compositor import ComposeSettings, _pose_stack_order
from chronophoto.processing.effects import apply_background_effect_tracks, apply_effect_tracks
from chronophoto.processing.parallel import (
    export_io_workers,
    export_render_workers,
    ordered_parallel_map,
)

ImageArray = NDArray[np.uint8]
MaskArray = NDArray[np.float32]
ExportKind = Literal[
    "composite",
    "combined_poses",
    "individual_poses",
    "background",
    "trail_video",
    "resolve_timeline",
]


@dataclass(slots=True)
class ExportLayers:
    background: ImageArray
    combined_poses: ImageArray
    poses: list[ImageArray]


class _IncrementalAlphaCompositor:
    """Reuse unchanged RGBA pixels across a chronological moving pose window."""

    def __init__(self, overlap: str) -> None:
        self.overlap = overlap
        self.combined: Image.Image | None = None
        self.previous_indices: tuple[int, ...] = ()
        self.known_poses: dict[int, Image.Image] = {}
        self.pose_regions: dict[int, tuple[tuple[int, int, int, int], ...]] = {}

    def compose(self, current: Sequence[tuple[int, Image.Image]]) -> Image.Image:
        if not current:
            raise ValueError("an incremental alpha composite requires at least one pose")
        current_indices = tuple(index for index, _pose in current)
        current_set = set(current_indices)
        previous_set = set(self.previous_indices)
        for index, pose in current:
            self.known_poses[index] = pose
            if index not in self.pose_regions:
                self.pose_regions[index] = _alpha_regions(pose)

        if self.combined is None:
            self.combined = Image.new("RGBA", current[0][1].size)
            order = current if self.overlap == "newest" else tuple(reversed(current))
            for _index, pose in order:
                self.combined.alpha_composite(pose)
        else:
            added = [index for index in current_indices if index not in previous_set]
            removed = [index for index in self.previous_indices if index not in current_set]
            if self.overlap == "newest":
                retained = [index for index in current_indices if index not in added]
                _recompose_alpha_regions(
                    self.combined,
                    _merge_alpha_regions(
                        [box for index in removed for box in self.pose_regions[index]]
                    ),
                    retained,
                    self.known_poses,
                )
                for index in added:
                    for box in self.pose_regions[index]:
                        self.combined.alpha_composite(
                            self.known_poses[index],
                            dest=box[:2],
                            source=box,
                        )
            else:
                dirty = [box for index in (*removed, *added) for box in self.pose_regions[index]]
                _recompose_alpha_regions(
                    self.combined,
                    _merge_alpha_regions(dirty),
                    list(reversed(current_indices)),
                    self.known_poses,
                )

        result = self.combined.copy()
        self.previous_indices = current_indices
        for index in tuple(self.known_poses):
            if index not in current_set:
                del self.known_poses[index]
                self.pose_regions.pop(index, None)
        return result


def _recompose_alpha_regions(
    target: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    stack_order: Sequence[int],
    poses: dict[int, Image.Image],
) -> None:
    for left, top, right, bottom in boxes:
        region = Image.new("RGBA", (right - left, bottom - top))
        source_box = (left, top, right, bottom)
        for index in stack_order:
            region.alpha_composite(poses[index], dest=(0, 0), source=source_box)
        target.paste(region, (left, top))


def _alpha_regions(
    pose: Image.Image,
    *,
    tile_size: int = 256,
) -> tuple[tuple[int, int, int, int], ...]:
    alpha = np.asarray(pose.getchannel("A"))
    height, width = alpha.shape
    regions: list[tuple[int, int, int, int]] = []
    for top in range(0, height, tile_size):
        bottom = min(height, top + tile_size)
        run_left: int | None = None
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            if np.any(alpha[top:bottom, left:right]):
                if run_left is None:
                    run_left = left
            elif run_left is not None:
                regions.append((run_left, top, left, bottom))
                run_left = None
        if run_left is not None:
            regions.append((run_left, top, width, bottom))
    return tuple(_merge_alpha_regions(regions))


def _merge_alpha_regions(
    boxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
        left, top, right, bottom = candidate
        position = 0
        while position < len(merged):
            other = merged[position]
            if right < other[0] or other[2] < left or bottom < other[1] or other[3] < top:
                position += 1
                continue
            merged_left = min(left, other[0])
            merged_top = min(top, other[1])
            merged_right = max(right, other[2])
            merged_bottom = max(bottom, other[3])
            overlap_width = max(0, min(right, other[2]) - max(left, other[0]))
            overlap_height = max(0, min(bottom, other[3]) - max(top, other[1]))
            union_area = (
                (right - left) * (bottom - top)
                + ((other[2] - other[0]) * (other[3] - other[1]))
                - overlap_width * overlap_height
            )
            merged_area = (merged_right - merged_left) * (merged_bottom - merged_top)
            if merged_area != union_area:
                position += 1
                continue
            left, top, right, bottom = (
                merged_left,
                merged_top,
                merged_right,
                merged_bottom,
            )
            merged.pop(position)
            position = 0
        merged.append((left, top, right, bottom))
    return merged


def build_export_layers(
    frames: list[ImageArray],
    masks: list[MaskArray],
    background: ImageArray,
    settings: ComposeSettings,
    effect_progress: list[float],
    *,
    pixel_scale: float = 1.0,
    top_pose_index: int | None = None,
    trail_progress: list[float] | None = None,
) -> ExportLayers:
    """Build clean transparent pose cutouts plus the processed clean plate."""

    poses = build_transparent_poses(
        frames,
        masks,
        settings,
        effect_progress,
        pixel_scale=pixel_scale,
        trail_progress=trail_progress,
    )

    combined = np.zeros_like(poses[0])
    order = _pose_stack_order(len(poses), settings.overlap, top_pose_index)
    for index in order:
        combined = _alpha_over(combined, poses[index])

    processed_background = apply_background_effect_tracks(
        background,
        settings.background_effect_tracks,
        pixel_scale=pixel_scale,
    )
    return ExportLayers(processed_background, combined, poses)


def build_transparent_poses(
    frames: list[ImageArray],
    masks: list[MaskArray],
    settings: ComposeSettings,
    effect_progress: list[float],
    *,
    pixel_scale: float = 1.0,
    trail_progress: list[float] | None = None,
) -> list[ImageArray]:
    """Bake pixel-local effects into reusable straight-alpha pose cutouts."""

    return list(
        _iter_transparent_poses(
            frames,
            masks,
            settings,
            effect_progress,
            pixel_scale=pixel_scale,
            trail_progress=trail_progress,
        )
    )


def _iter_transparent_poses(
    frames: list[ImageArray],
    masks: list[MaskArray],
    settings: ComposeSettings,
    effect_progress: list[float],
    *,
    pixel_scale: float = 1.0,
    trail_progress: list[float] | None = None,
) -> Iterator[ImageArray]:
    """Yield pose cutouts through one bounded worker pool instead of one pool per pose."""

    if not frames or len(frames) != len(masks) or len(frames) != len(effect_progress):
        raise ValueError("Frames, masks, and effect positions must have equal non-zero lengths")
    if trail_progress is not None and len(trail_progress) != len(frames):
        raise ValueError("Trail positions must match the selected frames")
    trail_tracks = tuple(
        track
        for track in settings.trail_effect_tracks
        if track.enabled and track.kind != "blend_mode"
    )
    trail_positions: list[float | None] = (
        [None] * len(frames) if trail_progress is None else list(trail_progress)
    )

    def build_pose(
        item: tuple[ImageArray, MaskArray, float, float | None],
    ) -> ImageArray:
        frame, mask, progress, trail_position = item
        if not trail_tracks and mask.dtype == np.uint8 and np.max(mask) > 1:
            return _rgba_pose(frame, mask)
        normalized_mask = mask.astype(np.float32)
        if np.max(normalized_mask) > 1.0:
            normalized_mask /= 255.0
        effected_frame, effected_mask = apply_effect_tracks(
            frame,
            normalized_mask,
            progress,
            trail_tracks,
            pixel_scale=pixel_scale,
            trail_progress=trail_position,
        )
        alpha = np.rint(np.clip(effected_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        return _rgba_pose(effected_frame, alpha)

    items = zip(frames, masks, effect_progress, trail_positions, strict=True)
    height, width = frames[0].shape[:2]
    yield from ordered_parallel_map(
        build_pose,
        items,
        workers=export_render_workers(width, height),
    )


def _rgba_pose(frame: ImageArray, alpha: NDArray[np.uint8]) -> ImageArray:
    pose = np.empty((*frame.shape[:2], 4), dtype=np.uint8)
    pose[..., :3] = frame
    pose[..., 3] = alpha
    pose[alpha == 0, :3] = 0
    return pose


def _alpha_over(backdrop: ImageArray, source: ImageArray) -> ImageArray:
    source_alpha = source[..., 3:4].astype(np.float32) / 255.0
    backdrop_alpha = backdrop[..., 3:4].astype(np.float32) / 255.0
    output_alpha = source_alpha + backdrop_alpha * (1.0 - source_alpha)
    premultiplied = source[..., :3].astype(np.float32) * source_alpha + backdrop[..., :3].astype(
        np.float32
    ) * backdrop_alpha * (1.0 - source_alpha)
    output_rgb = np.divide(
        premultiplied,
        np.maximum(output_alpha, 1e-6),
        out=np.zeros_like(premultiplied),
        where=output_alpha > 1e-6,
    )
    return np.dstack(
        (
            np.clip(output_rgb, 0.0, 255.0).astype(np.uint8),
            np.rint(np.clip(output_alpha[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8),
        )
    )


def available_package_directory(parent: Path, source_stem: str) -> Path:
    base = parent / f"{source_stem}-chronophoto-layers"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base.name}-{suffix}"
        suffix += 1
    return candidate


def write_export_package(
    directory: Path,
    selections: tuple[ExportKind, ...],
    *,
    composite: ImageArray,
    layers: ExportLayers,
    labels: list[str],
) -> list[Path]:
    """Write a selected layer package without overwriting an existing directory."""

    if not selections:
        raise ValueError("Select at least one export output")
    if "resolve_timeline" in selections:
        raise ValueError("Resolve timeline packages use write_resolve_package")
    if directory.exists():
        raise FileExistsError(f"Export directory already exists: {directory}")
    directory.mkdir(parents=True)
    jobs: list[tuple[Path, ImageArray | Image.Image]] = []
    if "composite" in selections:
        jobs.append((directory / "composite.png", composite))
    if "combined_poses" in selections:
        jobs.append((directory / "poses.png", layers.combined_poses))
    if "background" in selections:
        jobs.append((directory / "background.png", layers.background))
    if "individual_poses" in selections:
        pose_directory = directory / "poses"
        pose_directory.mkdir()
        for index, pose in enumerate(layers.poses):
            label = labels[index] if index < len(labels) else ""
            filename = f"pose-{index + 1:03d}{_safe_label_suffix(label)}.png"
            jobs.append((pose_directory / filename, pose))
    return list(_write_png_jobs(jobs))


def _safe_label_suffix(label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower()
    return f"_{safe[:48]}" if safe else ""


def _save_png(path: Path, pixels: ImageArray) -> Path:
    return _save_png_job((path, pixels))


def _save_png_job(
    job: tuple[Path, ImageArray | Image.Image],
    *,
    compress_level: int = 6,
) -> Path:
    path, pixels = job
    image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
    image.save(path, compress_level=compress_level)
    return path


def _write_png_jobs(
    jobs: Iterable[tuple[Path, ImageArray | Image.Image]],
    *,
    workers: int | None = None,
    compress_level: int = 6,
) -> Iterator[Path]:
    yield from ordered_parallel_map(
        partial(_save_png_job, compress_level=compress_level),
        jobs,
        workers=export_io_workers() if workers is None else workers,
    )

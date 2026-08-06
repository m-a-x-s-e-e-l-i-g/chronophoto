from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from chronophoto.processing.compositor import ComposeSettings
from chronophoto.processing.effects import apply_background_effect_tracks, apply_effect_tracks

ImageArray = NDArray[np.uint8]
MaskArray = NDArray[np.float32]
ExportKind = Literal["composite", "combined_poses", "individual_poses", "background"]


@dataclass(slots=True)
class ExportLayers:
    background: ImageArray
    combined_poses: ImageArray
    poses: list[ImageArray]


def build_export_layers(
    frames: list[ImageArray],
    masks: list[MaskArray],
    background: ImageArray,
    settings: ComposeSettings,
    effect_progress: list[float],
    *,
    pixel_scale: float = 1.0,
) -> ExportLayers:
    """Build clean transparent pose cutouts plus the processed clean plate."""

    if not frames or len(frames) != len(masks) or len(frames) != len(effect_progress):
        raise ValueError("Frames, masks, and effect positions must have equal non-zero lengths")
    trail_tracks = tuple(
        track
        for track in settings.trail_effect_tracks
        if track.enabled and track.kind != "blend_mode"
    )
    poses: list[ImageArray] = []
    for frame, mask, progress in zip(frames, masks, effect_progress, strict=True):
        normalized_mask = mask.astype(np.float32)
        if np.max(normalized_mask) > 1.0:
            normalized_mask /= 255.0
        effected_frame, effected_mask = apply_effect_tracks(
            frame,
            normalized_mask,
            progress,
            trail_tracks,
            pixel_scale=pixel_scale,
        )
        alpha = np.rint(np.clip(effected_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        pose_rgb = effected_frame.copy()
        pose_rgb[alpha == 0] = 0
        poses.append(np.dstack((pose_rgb, alpha)))

    combined = np.zeros_like(poses[0])
    order = range(len(poses))
    if settings.overlap == "oldest":
        order = reversed(range(len(poses)))
    for index in order:
        combined = _alpha_over(combined, poses[index])

    processed_background = apply_background_effect_tracks(
        background,
        settings.background_effect_tracks,
        pixel_scale=pixel_scale,
    )
    return ExportLayers(processed_background, combined, poses)


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
    if directory.exists():
        raise FileExistsError(f"Export directory already exists: {directory}")
    directory.mkdir(parents=True)
    written: list[Path] = []
    if "composite" in selections:
        written.append(_save_png(directory / "composite.png", composite))
    if "combined_poses" in selections:
        written.append(_save_png(directory / "poses.png", layers.combined_poses))
    if "background" in selections:
        written.append(_save_png(directory / "background.png", layers.background))
    if "individual_poses" in selections:
        pose_directory = directory / "poses"
        pose_directory.mkdir()
        for index, pose in enumerate(layers.poses):
            label = labels[index] if index < len(labels) else ""
            filename = f"pose-{index + 1:03d}{_safe_label_suffix(label)}.png"
            written.append(_save_png(pose_directory / filename, pose))
    return written


def _safe_label_suffix(label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower()
    return f"_{safe[:48]}" if safe else ""


def _save_png(path: Path, pixels: ImageArray) -> Path:
    Image.fromarray(pixels).save(path, compress_level=6)
    return path

from __future__ import annotations

import numpy as np
from PIL import Image

from chronophoto.processing import (
    ComposeSettings,
    EffectKeyframe,
    EffectTrack,
    available_package_directory,
    build_export_layers,
    write_export_package,
)


def _fixture() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    background = np.full((24, 36, 3), (30, 90, 180), dtype=np.uint8)
    frames = [background.copy(), background.copy()]
    frames[0][6:18, 5:17] = (220, 50, 30)
    frames[1][6:18, 12:24] = (30, 220, 70)
    masks = [np.zeros((24, 36), dtype=np.float32) for _ in frames]
    masks[0][6:18, 5:17] = 1.0
    masks[1][6:18, 12:24] = 1.0
    return frames, masks, background


def test_export_layers_create_transparent_combined_and_individual_poses() -> None:
    frames, masks, background = _fixture()
    half = (EffectKeyframe(0.0, 50.0), EffectKeyframe(1.0, 50.0))
    settings = ComposeSettings(trail_effect_tracks=(EffectTrack("opacity", half),))

    layers = build_export_layers(frames, masks, background, settings, [0.0, 1.0])

    assert layers.combined_poses.shape == (24, 36, 4)
    assert len(layers.poses) == 2
    assert layers.poses[0][10, 8, 3] in {127, 128}
    assert layers.poses[0][2, 2, 3] == 0
    assert np.array_equal(layers.poses[0][2, 2, :3], (0, 0, 0))
    assert layers.combined_poses[10, 14, 1] > layers.combined_poses[10, 14, 0]
    assert np.array_equal(layers.background, background)


def test_export_layer_overlap_respects_the_selected_pose_order() -> None:
    frames, masks, background = _fixture()

    newest = build_export_layers(
        frames, masks, background, ComposeSettings(overlap="newest"), [0.0, 1.0]
    )
    oldest = build_export_layers(
        frames, masks, background, ComposeSettings(overlap="oldest"), [0.0, 1.0]
    )

    assert np.array_equal(newest.combined_poses[10, 14, :3], (30, 220, 70))
    assert np.array_equal(oldest.combined_poses[10, 14, :3], (220, 50, 30))


def test_export_layers_place_the_selected_focus_pose_on_top() -> None:
    frames, masks, background = _fixture()

    layers = build_export_layers(
        frames,
        masks,
        background,
        ComposeSettings(overlap="newest"),
        [0.0, 1.0],
        top_pose_index=0,
    )

    assert np.array_equal(layers.combined_poses[10, 14, :3], (220, 50, 30))


def test_export_package_writes_selected_layers_without_overwriting(tmp_path) -> None:
    frames, masks, background = _fixture()
    layers = build_export_layers(frames, masks, background, ComposeSettings(), [0.0, 1.0])
    destination = available_package_directory(tmp_path, "jump")

    written = write_export_package(
        destination,
        ("composite", "combined_poses", "individual_poses", "background"),
        composite=background,
        layers=layers,
        labels=["00:03.80", "Frame / 2"],
    )

    assert destination.is_dir()
    assert {path.relative_to(destination).as_posix() for path in written} == {
        "composite.png",
        "poses.png",
        "background.png",
        "poses/pose-001_00-03-80.png",
        "poses/pose-002_frame-2.png",
    }
    assert Image.open(destination / "poses.png").mode == "RGBA"
    assert Image.open(destination / "background.png").mode == "RGB"
    assert available_package_directory(tmp_path, "jump").name == "jump-chronophoto-layers-2"

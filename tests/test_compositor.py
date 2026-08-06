from __future__ import annotations

import cv2
import numpy as np
import pytest

from chronophoto.processing.compositor import (
    CLEAN_PLATE_MAX_FRAMES,
    ComposeCache,
    ComposeSettings,
    _clean_plate_frames,
    _dense_clone_step_count,
    _isolate_primary_motion,
    build_background,
    build_compose_cache,
    compose_sequence,
)
from chronophoto.processing.effects import EffectKeyframe, EffectTrack, neutral_effect_track


def moving_subject_frames(count: int = 7) -> list[np.ndarray]:
    height, width = 180, 300
    y, x = np.mgrid[:height, :width]
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[..., 0] = 34 + (x // 28) % 2 * 5
    background[..., 1] = 38 + (y // 24) % 2 * 4
    background[..., 2] = 42

    frames: list[np.ndarray] = []
    for index in range(count):
        frame = background.copy()
        left = 24 + index * 34
        cv2.rectangle(frame, (left, 58), (left + 24, 132), (224, 121, 47), -1)
        cv2.circle(frame, (left + 12, 47), 12, (238, 174, 92), -1)
        frames.append(frame)
    return frames


def test_compose_keeps_each_detected_pose() -> None:
    frames = moving_subject_frames()
    result, masks = compose_sequence(
        frames,
        ComposeSettings(threshold=18, feather=2, background="median"),
    )

    assert result.shape == frames[0].shape
    assert len(masks) == len(frames)
    for index, mask in enumerate(masks):
        left = 24 + index * 34
        assert mask[90, left + 12] > 0.75
        assert result[90, left + 12, 0] > 170


def test_progress_reaches_completion() -> None:
    updates: list[tuple[int, str]] = []
    compose_sequence(
        moving_subject_frames(5),
        ComposeSettings(threshold=18),
        progress=lambda value, message: updates.append((value, message)),
    )

    assert updates[0][0] == 8
    assert updates[-1] == (100, "Composite ready")
    assert all(left <= right for (left, _), (right, _) in zip(updates, updates[1:], strict=False))


def test_clean_plate_evenly_samples_long_sequences() -> None:
    frames = [np.full((4, 5, 3), index, dtype=np.uint8) for index in range(90)]

    sampled = _clean_plate_frames(frames)
    background = build_background(frames, "median")
    expected = np.median(np.stack(sampled), axis=0).astype(np.uint8)

    assert len(sampled) == CLEAN_PLATE_MAX_FRAMES
    assert sampled[0] is frames[0]
    assert sampled[-1] is frames[-1]
    assert np.array_equal(background, expected)


def test_clean_plate_progress_reports_sample_size() -> None:
    updates: list[str] = []
    compose_sequence(
        [np.full((12, 16, 3), index, dtype=np.uint8) for index in range(30)],
        ComposeSettings(background="median"),
        progress=lambda _value, message: updates.append(message),
        return_masks=False,
    )

    assert f"from {CLEAN_PLATE_MAX_FRAMES} of 30 frames" in updates[0]


def test_export_mode_does_not_retain_float_masks() -> None:
    result, masks = compose_sequence(
        moving_subject_frames(5),
        ComposeSettings(threshold=18),
        return_masks=False,
    )

    assert result.shape == (180, 300, 3)
    assert masks == []


def test_cached_composition_does_not_find_poses_again(monkeypatch) -> None:
    frames = moving_subject_frames(5)
    settings = ComposeSettings(threshold=18, feather=2, background="median")
    cache = build_compose_cache(frames, settings)
    updates: list[str] = []

    def unexpected_mask(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Cached composition recalculated a pose mask")

    monkeypatch.setattr(
        "chronophoto.processing.compositor.create_motion_mask",
        unexpected_mask,
    )
    result, masks = compose_sequence(
        frames,
        settings,
        progress=lambda _value, message: updates.append(message),
        cache=cache,
    )

    assert result.shape == frames[0].shape
    assert len(masks) == len(frames)
    assert any("cached pose masks" in message for message in updates)
    assert not any("Finding pose" in message for message in updates)


def test_photographic_smear_connects_adjacent_poses() -> None:
    frames = moving_subject_frames()
    solid, _ = compose_sequence(
        frames,
        ComposeSettings(threshold=18, feather=0, background="median"),
    )
    blurred, _ = compose_sequence(
        frames,
        ComposeSettings(
            threshold=18,
            feather=0,
            background="median",
            smear_style="photographic",
        ),
    )

    gap_x = 53
    assert blurred[90, gap_x, 0] > solid[90, gap_x, 0] + 35
    assert not np.array_equal(blurred, solid)


def test_primary_motion_tracking_rejects_a_persistent_background_component() -> None:
    masks = []
    for index in range(5):
        mask = np.zeros((120, 200), dtype=np.float32)
        left = 20 + index * 20
        mask[20:50, left : left + 20] = 1.0
        mask[92:106, 45:165] = 1.0
        masks.append(mask)

    isolated = _isolate_primary_motion(masks)

    for index, mask in enumerate(isolated):
        assert mask[35, 30 + index * 20] == 1.0
        assert mask[98, 100] == 0.0


def test_smear_options_build_distinct_results_without_softening_poses() -> None:
    frames = moving_subject_frames()
    solid, _ = compose_sequence(
        frames,
        ComposeSettings(threshold=18, feather=0, background="median"),
    )
    variants = {
        style: compose_sequence(
            frames,
            ComposeSettings(
                threshold=18,
                feather=0,
                background="median",
                smear_style=style,
            ),
        )[0]
        for style in ("photographic", "dense_clones")
    }

    gap_x = 53
    pose_x = 36
    assert all(result[90, gap_x, 0] > 150 for result in variants.values())
    assert all(
        np.array_equal(result[90, pose_x], solid[90, pose_x]) for result in variants.values()
    )
    assert not np.array_equal(variants["dense_clones"], variants["photographic"])


def test_dense_clone_spacing_fills_every_pixel_between_frames() -> None:
    step_count = _dense_clone_step_count(34.0, -12.0)
    positions = np.linspace(0.0, 1.0, step_count + 1)

    assert step_count == 34
    assert np.max(np.diff(positions) * 34.0) <= 1.0 + 1e-9


def test_settings_reject_unknown_modes() -> None:
    try:
        ComposeSettings(overlap="middle")
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Invalid overlap mode was accepted")

    try:
        ComposeSettings(trail_style="opacity_fade")
    except ValueError as error:
        assert "trail style" in str(error)
    else:
        raise AssertionError("Invalid trail style was accepted")

    try:
        ComposeSettings(smear_style="cloud")
    except ValueError as error:
        assert "smear style" in str(error)
    else:
        raise AssertionError("Invalid smear style was accepted")


def test_settings_use_low_sensitivity_and_soft_edge_defaults() -> None:
    settings = ComposeSettings()

    assert settings.threshold == 17
    assert settings.feather == 1
    assert settings.smear_style == "none"


def test_opacity_effect_uses_explicit_normalized_frame_progress() -> None:
    frames = moving_subject_frames(3)
    background = np.full_like(frames[0], (34, 38, 42))
    masks = []
    for index in range(3):
        mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        left = 24 + index * 34
        mask[35:133, left : left + 25] = 255
        masks.append(mask)
    cache = ComposeCache(background, masks)
    track = EffectTrack(
        "opacity",
        (
            EffectKeyframe(0.0, 0.0),
            EffectKeyframe(0.5, 100.0),
            EffectKeyframe(1.0, 0.0),
        ),
    )

    result, returned_masks = compose_sequence(
        frames,
        ComposeSettings(effect_tracks=(track,)),
        cache=cache,
        effect_progress=(0.0, 0.5, 1.0),
    )

    assert np.array_equal(result[90, 36], background[90, 36])
    assert result[90, 70, 0] > 180
    assert np.array_equal(result[90, 104], background[90, 104])
    assert returned_masks[0][90, 36] == 1.0


def test_blend_mode_uses_the_existing_composite_as_its_backdrop() -> None:
    background = np.full((48, 80, 3), (80, 120, 160), dtype=np.uint8)
    frames = [background.copy(), background.copy()]
    frames[0][12:32, 10:26] = (210, 70, 40)
    frames[1][12:32, 48:64] = (40, 200, 100)
    masks = [np.zeros((48, 80), dtype=np.uint8) for _ in frames]
    masks[0][12:32, 10:26] = 255
    masks[1][12:32, 48:64] = 255
    full = (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0))
    track = EffectTrack("blend_mode", full, option="multiply")

    result, _ = compose_sequence(
        frames,
        ComposeSettings(effect_tracks=(track,)),
        cache=ComposeCache(background, masks),
    )

    expected_first = np.array((80, 120, 160)) * np.array((210, 70, 40)) / 255.0
    expected_second = np.array((80, 120, 160)) * np.array((40, 200, 100)) / 255.0
    assert result[20, 18] == pytest.approx(expected_first, abs=1.0)
    assert result[20, 56] == pytest.approx(expected_second, abs=1.0)
    assert np.array_equal(result[2, 2], background[2, 2])


def test_zero_blend_strength_is_identical_to_normal_compositing() -> None:
    frames = moving_subject_frames(3)
    cache = build_compose_cache(frames, ComposeSettings())
    zero = (EffectKeyframe(0.0, 0.0), EffectKeyframe(1.0, 0.0))

    normal, _ = compose_sequence(frames, ComposeSettings(), cache=cache)
    blended, _ = compose_sequence(
        frames,
        ComposeSettings(effect_tracks=(EffectTrack("blend_mode", zero, option="screen"),)),
        cache=cache,
    )

    assert np.array_equal(blended, normal)


@pytest.mark.parametrize("smear_style", ["photographic", "dense_clones"])
def test_effects_apply_to_generated_smear_pixels(smear_style: str) -> None:
    frames = moving_subject_frames(4)
    settings = ComposeSettings(
        threshold=18,
        feather=0,
        background="median",
        smear_style=smear_style,
        effect_tracks=(
            EffectTrack(
                "opacity",
                (EffectKeyframe(0.0, 0.0), EffectKeyframe(1.0, 0.0)),
            ),
        ),
    )

    result, _ = compose_sequence(frames, settings)
    background = build_background(frames, "median")

    assert np.array_equal(result, background)


@pytest.mark.parametrize("smear_style", ["photographic", "dense_clones"])
def test_blend_modes_apply_to_generated_smear_layers(smear_style: str) -> None:
    frames = moving_subject_frames(4)
    base_settings = ComposeSettings(
        threshold=18,
        feather=0,
        background="median",
        smear_style=smear_style,
    )
    cache = build_compose_cache(frames, base_settings)
    full = (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0))

    normal, _ = compose_sequence(frames, base_settings, cache=cache)
    difference, _ = compose_sequence(
        frames,
        ComposeSettings(
            threshold=18,
            feather=0,
            background="median",
            smear_style=smear_style,
            effect_tracks=(EffectTrack("blend_mode", full, option="difference"),),
        ),
        cache=cache,
    )

    assert not np.array_equal(difference, normal)


def test_effect_progress_requires_one_chronological_value_per_frame() -> None:
    frames = moving_subject_frames(3)
    settings = ComposeSettings(effect_tracks=(neutral_effect_track("opacity"),))

    with pytest.raises(ValueError, match="match"):
        compose_sequence(frames, settings, effect_progress=(0.0, 1.0))
    with pytest.raises(ValueError, match="chronological"):
        compose_sequence(frames, settings, effect_progress=(0.0, 0.8, 0.5))

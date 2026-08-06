from __future__ import annotations

import cv2
import numpy as np
import pytest

from chronophoto.processing.effects import (
    EFFECT_KINDS,
    EffectKeyframe,
    EffectTrack,
    apply_effect_tracks,
    effect_preset,
    neutral_effect_track,
)


def subject_fixture() -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((72, 96, 3), (38, 76, 180), dtype=np.uint8)
    frame[18:56, 24:72] = (230, 90, 36)
    mask = np.zeros((72, 96), dtype=np.float32)
    mask[18:56, 24:72] = 1.0
    return frame, mask


def test_effect_track_interpolates_normalized_progress() -> None:
    track = EffectTrack(
        "opacity",
        (
            EffectKeyframe(0.0, 0.0),
            EffectKeyframe(0.5, 100.0),
            EffectKeyframe(1.0, 0.0),
        ),
    )

    assert track.value_at(-1.0) == 0.0
    assert track.value_at(0.25) == pytest.approx(50.0)
    assert track.value_at(0.5) == 100.0
    assert track.value_at(0.75) == pytest.approx(50.0)
    assert track.value_at(2.0) == 0.0


def test_requested_presets_are_available_for_every_effect() -> None:
    for kind in EFFECT_KINDS:
        neutral = neutral_effect_track(kind)
        rise_fall = effect_preset(neutral, "rise_fall")
        rise = effect_preset(neutral, "rise")
        fall = effect_preset(neutral, "fall")

        assert [point.value for point in rise_fall.keyframes] == [0.0, 100.0, 0.0]
        assert [point.value for point in rise.keyframes] == [0.0, 100.0]
        assert [point.value for point in fall.keyframes] == [100.0, 0.0]


def test_neutral_effect_stack_does_not_change_subject() -> None:
    frame, mask = subject_fixture()
    tracks = tuple(neutral_effect_track(kind) for kind in EFFECT_KINDS)

    effected, effected_mask = apply_effect_tracks(frame, mask, 0.4, tracks)

    assert np.array_equal(effected, frame)
    assert np.array_equal(effected_mask, mask)


def test_opacity_only_changes_the_subject_alpha() -> None:
    frame, mask = subject_fixture()
    track = EffectTrack(
        "opacity",
        (EffectKeyframe(0.0, 0.0), EffectKeyframe(1.0, 100.0)),
    )

    effected, effected_mask = apply_effect_tracks(frame, mask, 0.25, (track,))

    assert np.array_equal(effected, frame)
    assert effected_mask[30, 40] == pytest.approx(0.25)
    assert effected_mask[0, 0] == 0.0


def test_saturation_blur_and_jpeg_quality_have_distinct_results() -> None:
    frame, mask = subject_fixture()
    cv2.line(frame, (24, 18), (71, 55), (10, 240, 80), 3)
    full = (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0))
    zero = (EffectKeyframe(0.0, 0.0), EffectKeyframe(1.0, 0.0))

    desaturated, _ = apply_effect_tracks(frame, mask, 0.5, (EffectTrack("saturation", zero),))
    blurred, _ = apply_effect_tracks(frame, mask, 0.5, (EffectTrack("blur", full, amount=8),))
    compressed, _ = apply_effect_tracks(frame, mask, 0.5, (EffectTrack("jpeg_quality", zero),))

    assert np.array_equal(desaturated[30, 40, 0], desaturated[30, 40, 1])
    assert not np.array_equal(blurred, frame)
    assert not np.array_equal(compressed, frame)
    assert not np.array_equal(blurred, compressed)


def test_independent_texture_effects_can_stack() -> None:
    frame, mask = subject_fixture()
    full = (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0))
    tracks = (
        EffectTrack("stippling", full, amount=4),
        EffectTrack("dithering", full, amount=2),
        EffectTrack("halftone", full, amount=8),
    )

    stippled, _ = apply_effect_tracks(frame, mask, 0.5, tracks[:1])
    dithered, _ = apply_effect_tracks(frame, mask, 0.5, tracks[1:2])
    halftoned, _ = apply_effect_tracks(frame, mask, 0.5, tracks[2:])
    stacked, _ = apply_effect_tracks(frame, mask, 0.5, tracks)

    assert not np.array_equal(stippled, dithered)
    assert not np.array_equal(dithered, halftoned)
    assert not np.array_equal(stacked, stippled)
    assert np.array_equal(stacked[:18], frame[:18])


def test_effect_tracks_reject_invalid_keyframes() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        EffectTrack(
            "opacity",
            (EffectKeyframe(0.2, 0.0), EffectKeyframe(1.0, 100.0)),
        )
    with pytest.raises(ValueError, match="ordered"):
        EffectTrack(
            "opacity",
            (
                EffectKeyframe(0.0, 0.0),
                EffectKeyframe(0.7, 30.0),
                EffectKeyframe(0.7, 80.0),
                EffectKeyframe(1.0, 100.0),
            ),
        )

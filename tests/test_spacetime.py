from __future__ import annotations

import numpy as np
import pytest

from chronophoto.processing.spacetime import (
    TimeSurface,
    sample_time_surface,
    silhouette_edge_stretch,
    slice_video_volume,
)


def test_edge_stretch_samples_each_scanline_boundary() -> None:
    image = np.zeros((3, 6, 3), dtype=np.uint8)
    image[0, 2] = (10, 20, 30)
    image[1, 3] = (40, 50, 60)
    mask = np.zeros((3, 6), dtype=bool)
    mask[0, 2] = True
    mask[1, 3] = True

    result = silhouette_edge_stretch(image, mask, direction="right", distance=2)

    np.testing.assert_array_equal(result[0, 3:5], [[10, 20, 30], [10, 20, 30]])
    np.testing.assert_array_equal(result[1, 4:6], [[40, 50, 60], [40, 50, 60]])
    assert not result[2].any()


def test_edge_stretch_supports_vertical_directions_and_preserves_subject() -> None:
    image = np.zeros((5, 3, 3), dtype=np.uint8)
    image[2, 1] = 180
    mask = np.zeros((5, 3), dtype=bool)
    mask[2, 1] = True

    result = silhouette_edge_stretch(image, mask, direction="both-vertical")

    np.testing.assert_array_equal(result[:, 1], np.full((5, 3), 180))
    np.testing.assert_array_equal(result[2, 1], image[2, 1])


def test_time_surface_interpolates_between_video_frames() -> None:
    frames = np.stack(
        (np.zeros((2, 3, 3), dtype=np.uint8), np.full((2, 3, 3), 100, dtype=np.uint8))
    )
    result = sample_time_surface(frames, np.full((2, 3), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(result, np.full((2, 3, 3), 25, dtype=np.uint8))


def test_video_volume_exposes_orthogonal_diagonal_and_surface_slices() -> None:
    frames = np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3)

    assert slice_video_volume(frames, "xt").shape == (4, 5, 3)
    assert slice_video_volume(frames, "yt").shape == (3, 4, 3)
    assert slice_video_volume(frames, "diagonal").shape == (4, 5, 3)
    assert slice_video_volume(frames, "surface", surface=TimeSurface()).shape == (3, 5, 3)


def test_spacetime_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="mask dimensions"):
        silhouette_edge_stretch(np.zeros((2, 2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="surface dimensions"):
        sample_time_surface(np.zeros((2, 2, 2, 3)), np.zeros((3, 3)))

import numpy as np
import pytest

from app.inference.stitching import stitch_probability_tiles
from varun_phase1_core import Tile


def test_stitch_probability_tiles_preserves_scene_shape():
    tiles = [
        Tile(x=0, y=0, width=4, height=4),
        Tile(x=2, y=0, width=4, height=4),
    ]

    probabilities = [
        np.zeros((4, 4), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
    ]

    result = stitch_probability_tiles(
        probabilities=probabilities,
        tiles=tiles,
        image_height=4,
        image_width=6,
    )

    assert result.shape == (4, 6)
    assert result.dtype == np.float32
    assert np.all(result[:, :2] == 0.0)
    assert np.all(result[:, 4:] == 1.0)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_stitch_discards_padded_edge_region():
    tiles = [
        Tile(x=0, y=0, width=3, height=2),
    ]

    # Model produced a padded 4x4 tile, but only 2x3 is valid.
    probabilities = [
        np.ones((4, 4), dtype=np.float32),
    ]

    result = stitch_probability_tiles(
        probabilities=probabilities,
        tiles=tiles,
        image_height=2,
        image_width=3,
    )

    assert result.shape == (2, 3)
    assert np.allclose(result, 1.0)


def test_stitch_rejects_count_mismatch():
    with pytest.raises(
        ValueError,
        match="number of probability rasters",
    ):
        stitch_probability_tiles(
            probabilities=[],
            tiles=[Tile(x=0, y=0, width=2, height=2)],
            image_height=2,
            image_width=2,
        )
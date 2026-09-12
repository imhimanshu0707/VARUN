import numpy as np
import pytest

from app.postprocessing.mask import remove_small_components


def test_remove_small_components():
    mask = np.zeros((10, 10), dtype=np.uint8)

    # One-pixel false alarm
    mask[1, 1] = 1

    # Valid 3x3 oil region: 9 pixels
    mask[5:8, 5:8] = 1

    cleaned = remove_small_components(
        mask=mask,
        min_area_pixels=5,
    )

    assert cleaned[1, 1] == 0
    assert int(cleaned.sum()) == 9
    assert np.all(cleaned[5:8, 5:8] == 1)


def test_minimum_area_one_preserves_mask():
    mask = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    cleaned = remove_small_components(
        mask=mask,
        min_area_pixels=1,
    )

    assert np.array_equal(cleaned, mask)


def test_cleanup_rejects_non_binary_mask():
    mask = np.array(
        [[0, 2]],
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        remove_small_components(
            mask=mask,
            min_area_pixels=1,
        )
import numpy as np
import pytest
import torch

from app.preprocessing import preprocess_image


def test_preprocess_returns_two_channel_float_tensor():
    image = np.stack(
        [
            np.arange(100, dtype=np.float32).reshape(10, 10),
            np.arange(100, 200, dtype=np.float32).reshape(10, 10),
        ]
    )

    result = preprocess_image(image)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 10, 10)
    assert result.dtype == torch.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_preprocess_supports_channel_last_input():
    image = np.zeros((10, 12, 2), dtype=np.float32)

    result = preprocess_image(image)

    assert result.shape == (2, 10, 12)


def test_preprocess_rejects_single_band_input():
    image = np.zeros((1, 10, 10), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="requires exactly two SAR bands",
    ):
        preprocess_image(image)


def test_preprocess_handles_nan_and_infinity():
    image = np.stack(
        [
            np.arange(100, dtype=np.float32).reshape(10, 10),
            np.arange(100, 200, dtype=np.float32).reshape(10, 10),
        ]
    )

    image[0, 0, 0] = np.nan
    image[1, 0, 0] = np.inf

    result = preprocess_image(image)

    assert torch.isfinite(result).all()
    assert result[0, 0, 0].item() == 0.0
    assert result[1, 0, 0].item() == 0.0

def test_preprocess_excludes_invalid_pixels_from_percentiles():
    image = np.stack(
        [
            np.array(
                [
                    [1.0, 2.0],
                    [-9999.0, 4.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [10.0, 20.0],
                    [-9999.0, 40.0],
                ],
                dtype=np.float32,
            ),
        ]
    )

    valid_mask = np.array(
        [
            [True, True],
            [False, True],
        ]
    )

    result = preprocess_image(
        image,
        lower_percentile=0.0,
        upper_percentile=100.0,
        valid_mask=valid_mask,
    )

    assert result.shape == (2, 2, 2)
    assert result[0, 1, 0].item() == 0.0
    assert result[1, 1, 0].item() == 0.0

    assert result[0, 0, 0].item() == pytest.approx(0.0)
    assert result[0, 1, 1].item() == pytest.approx(1.0)

    assert result[1, 0, 0].item() == pytest.approx(0.0)
    assert result[1, 1, 1].item() == pytest.approx(1.0)


def test_preprocess_rejects_mismatched_validity_mask():
    image = np.zeros((2, 10, 12), dtype=np.float32)
    valid_mask = np.ones((9, 12), dtype=bool)

    with pytest.raises(
        ValueError,
        match="Validity mask dimensions do not match",
    ):
        preprocess_image(
            image,
            valid_mask=valid_mask,
        )
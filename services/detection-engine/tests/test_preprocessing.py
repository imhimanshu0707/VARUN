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
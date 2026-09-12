from typing import Any

import numpy as np
import torch


LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0
EXPECTED_CHANNELS = 2


def _to_channel_first(image: Any) -> np.ndarray:
    """
    Convert the input into a strict (2, height, width) float32 array.
    """

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(
            "Expected a 3D two-band SAR image with shape "
            f"(2,H,W) or (H,W,2), got {array.shape}"
        )

    if array.shape[0] == EXPECTED_CHANNELS:
        channel_first = array
    elif array.shape[-1] == EXPECTED_CHANNELS:
        channel_first = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            "Phase-1 U-Net requires exactly two SAR bands "
            f"(VV and VH), got shape {array.shape}"
        )

    return np.ascontiguousarray(
        channel_first,
        dtype=np.float32,
    )


def preprocess_image(
    image: Any,
    lower_percentile: float = LOWER_PERCENTILE,
    upper_percentile: float = UPPER_PERCENTILE,
) -> torch.Tensor:
    """
    Apply per-band global percentile normalization.

    This matches the Phase-1 training preprocessing profile:
    sar-percentile-v1.

    Each SAR band is independently clipped between its 2nd and
    98th percentiles and scaled into the [0, 1] range.
    """

    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100"
        )

    array = _to_channel_first(image)
    normalized = np.zeros_like(array, dtype=np.float32)

    for band_index in range(EXPECTED_CHANNELS):
        band = array[band_index]
        valid = np.isfinite(band)

        if not np.any(valid):
            raise ValueError(
                f"SAR band {band_index + 1} contains no finite values"
            )

        low, high = np.percentile(
            band[valid],
            [lower_percentile, upper_percentile],
        )

        if high <= low:
            normalized[band_index] = 0.0
            continue

        scaled = (band - low) / (high - low)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~valid] = 0.0

        normalized[band_index] = scaled.astype(
            np.float32,
            copy=False,
        )

    return torch.from_numpy(normalized)
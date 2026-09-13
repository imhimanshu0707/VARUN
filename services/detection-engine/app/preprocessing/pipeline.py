from typing import Any

import numpy as np
import torch


LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0
EXPECTED_CHANNELS = 2


def _to_channel_first(image: Any) -> np.ndarray:
    """
    Convert input into a strict (2, height, width) float32 array.
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
            f"(VH and VV), got shape {array.shape}"
        )

    return np.ascontiguousarray(
        channel_first,
        dtype=np.float32,
    )


def _prepare_valid_mask(
    valid_mask: Any | None,
    height: int,
    width: int,
) -> np.ndarray | None:
    if valid_mask is None:
        return None

    mask = np.asarray(valid_mask)

    if mask.shape != (height, width):
        raise ValueError(
            "Validity mask dimensions do not match SAR image: "
            f"mask={mask.shape}, image={(height, width)}"
        )

    return mask.astype(bool, copy=False)


def preprocess_image(
    image: Any,
    lower_percentile: float = LOWER_PERCENTILE,
    upper_percentile: float = UPPER_PERCENTILE,
    valid_mask: Any | None = None,
) -> torch.Tensor:
    """
    Apply per-band global percentile normalization.

    Percentiles are calculated only from finite pixels allowed by the
    optional geospatial validity mask. Invalid output pixels are zero.
    """

    if not (
        0.0
        <= lower_percentile
        < upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100"
        )

    array = _to_channel_first(image)
    height, width = array.shape[-2:]

    spatial_valid = _prepare_valid_mask(
        valid_mask=valid_mask,
        height=height,
        width=width,
    )

    normalized = np.zeros_like(
        array,
        dtype=np.float32,
    )

    for band_index in range(EXPECTED_CHANNELS):
        band = array[band_index]
        valid = np.isfinite(band)

        if spatial_valid is not None:
            valid &= spatial_valid

        if not np.any(valid):
            raise ValueError(
                f"SAR band {band_index + 1} "
                "contains no valid finite pixels"
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
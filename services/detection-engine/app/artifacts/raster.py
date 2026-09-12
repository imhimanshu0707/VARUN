from pathlib import Path
from typing import Any

import numpy as np
import rasterio


CONTRACT_VERSION = "phase1-to-phase2-v1"
PROBABILITY_NODATA = -9999.0
MASK_NODATA = 255


def _validate_georeferencing(
    crs: Any,
    transform: Any,
) -> None:
    if crs is None:
        raise ValueError(
            "Input GeoTIFF must contain a CRS"
        )

    if transform is None:
        raise ValueError(
            "Input GeoTIFF must contain an affine transform"
        )


def write_probability_geotiff(
    probability_map: np.ndarray,
    output_path: str | Path,
    crs: Any,
    transform: Any,
) -> str:
    """
    Write the full-scene oil probability map as a georeferenced
    single-band float32 GeoTIFF.
    """

    _validate_georeferencing(crs, transform)

    probability = np.asarray(
        probability_map,
        dtype=np.float32,
    )

    if probability.ndim != 2:
        raise ValueError(
            "Probability map must be 2D, "
            f"got shape {probability.shape}"
        )

    if not np.isfinite(probability).all():
        raise ValueError(
            "Probability map contains NaN or infinity"
        )

    if (
        np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError(
            "Probability values must remain within [0, 1]"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    height, width = probability.shape

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=PROBABILITY_NODATA,
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(probability, 1)
        destination.set_band_description(
            1,
            "oil_probability",
        )
        destination.update_tags(
            contract_version=CONTRACT_VERSION,
            artifact_type="oil_probability",
            value_range="0.0-1.0",
        )

    return str(output)


def write_mask_geotiff(
    mask: np.ndarray,
    output_path: str | Path,
    crs: Any,
    transform: Any,
) -> str:
    """
    Write the thresholded oil mask as a georeferenced uint8
    GeoTIFF: background=0, oil=1, nodata=255.
    """

    _validate_georeferencing(crs, transform)

    binary_mask = np.asarray(mask)

    if binary_mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D, got shape {binary_mask.shape}"
        )

    unique_values = set(
        np.unique(binary_mask).tolist()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Mask must contain only 0 and 1, "
            f"got {sorted(unique_values)}"
        )

    binary_mask = binary_mask.astype(
        np.uint8,
        copy=False,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    height, width = binary_mask.shape

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=MASK_NODATA,
        compress="lzw",
    ) as destination:
        destination.write(binary_mask, 1)
        destination.set_band_description(
            1,
            "oil_mask",
        )
        destination.update_tags(
            contract_version=CONTRACT_VERSION,
            artifact_type="oil_mask",
            background_value="0",
            oil_value="1",
        )

    return str(output)
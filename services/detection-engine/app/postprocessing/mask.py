import numpy as np
from rasterio.features import sieve


def remove_small_components(
    mask: np.ndarray,
    min_area_pixels: int,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Remove connected oil regions smaller than min_area_pixels.

    Input/output values:
    background = 0
    oil = 1
    """

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

    if min_area_pixels < 1:
        raise ValueError(
            "min_area_pixels must be at least 1"
        )

    if connectivity not in {4, 8}:
        raise ValueError(
            "connectivity must be either 4 or 8"
        )

    binary_mask = binary_mask.astype(
        np.uint8,
        copy=False,
    )

    if min_area_pixels == 1:
        return binary_mask.copy()

    cleaned = sieve(
        binary_mask,
        size=min_area_pixels,
        connectivity=connectivity,
    )

    return (cleaned == 1).astype(np.uint8)
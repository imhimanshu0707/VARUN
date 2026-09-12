from typing import Any, Sequence

import numpy as np

from varun_phase1_core import Detection


def _blend_weights(
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create centre-weighted blending weights.

    Central tile pixels receive higher weight while tile borders
    receive a small non-zero weight. This reduces visible seams
    between overlapping predictions.
    """

    if height <= 0 or width <= 0:
        raise ValueError("Weight dimensions must be positive")

    y_weights = (
        np.hanning(height)
        if height > 1
        else np.ones(1, dtype=np.float32)
    )
    x_weights = (
        np.hanning(width)
        if width > 1
        else np.ones(1, dtype=np.float32)
    )

    weights = np.outer(y_weights, x_weights)
    weights = np.maximum(weights, 1e-3)

    return weights.astype(np.float32)


def stitch_probability_tiles(
    probabilities: Sequence[np.ndarray],
    tiles: Sequence[Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Stitch tile probability rasters into one full-scene raster.

    Padded prediction regions outside each tile's valid width and
    height are discarded. Overlapping regions are blended using
    weighted averaging.
    """

    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive")

    if len(probabilities) != len(tiles):
        raise ValueError(
            "The number of probability rasters must match "
            "the number of tiles"
        )

    probability_sum = np.zeros(
        (image_height, image_width),
        dtype=np.float32,
    )
    weight_sum = np.zeros(
        (image_height, image_width),
        dtype=np.float32,
    )

    for probability, tile in zip(probabilities, tiles):
        probability = np.asarray(
            probability,
            dtype=np.float32,
        )

        if probability.ndim != 2:
            raise ValueError(
                "Each probability tile must be 2D, "
                f"got shape {probability.shape}"
            )

        if (
            probability.shape[0] < tile.height
            or probability.shape[1] < tile.width
        ):
            raise ValueError(
                "Probability tile is smaller than its valid "
                f"region: probability={probability.shape}, "
                f"valid={tile.height}x{tile.width}"
            )

        valid_probability = probability[
            :tile.height,
            :tile.width,
        ]

        if not np.isfinite(valid_probability).all():
            raise ValueError(
                "Probability tile contains NaN or infinity"
            )

        valid_probability = np.clip(
            valid_probability,
            0.0,
            1.0,
        )

        weights = _blend_weights(
            tile.height,
            tile.width,
        )

        y_start = int(tile.y)
        y_end = y_start + int(tile.height)
        x_start = int(tile.x)
        x_end = x_start + int(tile.width)

        if (
            x_start < 0
            or y_start < 0
            or x_end > image_width
            or y_end > image_height
        ):
            raise ValueError(
                "Tile extends outside the target image: "
                f"x={x_start}:{x_end}, y={y_start}:{y_end}"
            )

        probability_sum[
            y_start:y_end,
            x_start:x_end,
        ] += valid_probability * weights

        weight_sum[
            y_start:y_end,
            x_start:x_end,
        ] += weights

    if np.any(weight_sum == 0):
        raise ValueError(
            "Tile layout does not cover the complete image"
        )

    stitched = probability_sum / weight_sum

    return np.clip(
        stitched,
        0.0,
        1.0,
    ).astype(np.float32)


def stitch_predictions(
    predictions: Sequence[Detection],
    tiles: Sequence[Any],
) -> list[Detection]:
    """
    Legacy bounding-box stitching retained temporarily.
    """

    results: list[Detection] = []

    for prediction, tile in zip(predictions, tiles):
        results.append(
            Detection(
                x=prediction.x + tile.x,
                y=prediction.y + tile.y,
                width=prediction.width,
                height=prediction.height,
                score=prediction.score,
            )
        )

    return results
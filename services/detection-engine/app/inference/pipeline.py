from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from app.inference.stitching import stitch_probability_tiles
from app.inference.tiling import create_tiles
from app.postprocessing.mask_to_detection import mask_to_detections
from app.predictor.unet_predictor import UNetPredictor
from app.preprocessing import preprocess_image
from app.postprocessing.mask import remove_small_components


UNET_DIVISIBILITY = 16


def _pad_tile(
    tile: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """
    Pad an edge tile to the model input size without resizing it.

    Padding is added only on the right and bottom. Replication padding
    avoids inventing extreme SAR values at the image boundary.
    """

    if tile.ndim != 3:
        raise ValueError(
            f"Expected tile shape (C,H,W), got {tuple(tile.shape)}"
        )

    height, width = tile.shape[-2:]

    if height > target_size or width > target_size:
        raise ValueError(
            f"Tile {height}x{width} exceeds target size {target_size}"
        )

    pad_height = target_size - height
    pad_width = target_size - width

    if pad_height == 0 and pad_width == 0:
        return tile

    return functional.pad(
        tile,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )


def run_inference(
    image: Any,
    image_width: int,
    image_height: int,
    model_path: str,
    tile_size: int = 512,
    overlap: int = 64,
    confidence_threshold: float = 0.5,
    min_area_pixels: int = 20,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Run full-scene oil-spill segmentation.

    Returns the stitched probability raster, binary mask, tiles and
    backward-compatible pixel bounding-box detections.
    """

    if tile_size % UNET_DIVISIBILITY != 0:
        raise ValueError(
            "tile_size must be divisible by 16 for the U-Net, "
            f"got {tile_size}"
        )

    # Retained only for old API compatibility. Segmentation output
    # does not require bounding-box IoU suppression.
    _ = iou_threshold

    prepared = preprocess_image(image)

    actual_height, actual_width = prepared.shape[-2:]

    if (
        actual_width != image_width
        or actual_height != image_height
    ):
        raise ValueError(
            "Provided image dimensions do not match image data: "
            f"provided={image_width}x{image_height}, "
            f"actual={actual_width}x{actual_height}"
        )

    tiles = create_tiles(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        overlap=overlap,
    )

    predictor = UNetPredictor(
        model_path=model_path,
        threshold=confidence_threshold,
        min_area=min_area_pixels,
    )

    tile_probabilities: list[np.ndarray] = []

    for tile in tiles:
        tile_tensor = prepared[
            :,
            tile.y:tile.y + tile.height,
            tile.x:tile.x + tile.width,
        ]

        padded_tile = _pad_tile(
            tile=tile_tensor,
            target_size=tile_size,
        )

        probability = predictor.predict_probability(
            padded_tile,
        )

        tile_probabilities.append(probability)

    probability_map = stitch_probability_tiles(
        probabilities=tile_probabilities,
        tiles=tiles,
        image_height=image_height,
        image_width=image_width,
    )

    raw_mask = (
    probability_map >= confidence_threshold
    ).astype(np.uint8)

    mask = remove_small_components(
    mask=raw_mask,
    min_area_pixels=min_area_pixels,
    connectivity=8,
)

    # Bounding boxes are retained only for old backend/debug clients.
    # Final geographic output will be generated from the full mask.
    detections = mask_to_detections(
        mask=mask,
        probability=probability_map,
        min_area=min_area_pixels,
    )

    oil_pixel_count = int(mask.sum())
    total_pixel_count = int(mask.size)

    return {
        "tiles": tiles,
        "tile_probabilities": tile_probabilities,
        "probability_map": probability_map,
        "mask": mask,
        "detections": detections,
        "oil_pixel_count": oil_pixel_count,
        "total_pixel_count": total_pixel_count,
        "oil_coverage_percent": (
            oil_pixel_count / total_pixel_count
        ) * 100.0,
        "oil_detected": oil_pixel_count > 0,
        "threshold": confidence_threshold,
        "min_area_pixels": min_area_pixels,
    }
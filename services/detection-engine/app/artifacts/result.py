import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CONTRACT_VERSION = "phase1-to-phase2-v1"


def write_detection_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> str:
    """
    Write the backend/dashboard and Phase-2 handoff summary.
    """

    required_fields = {
        "contract_version",
        "case_id",
        "scene_id",
        "status",
        "oil_detected",
        "acquisition_time_utc",
        "crs",
        "source_image",
    }

    missing = sorted(
        field
        for field in required_fields
        if summary.get(field) is None
    )

    if missing:
        raise ValueError(
            "Detection summary is missing required fields: "
            + ", ".join(missing)
        )

    if summary["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Unsupported Phase-1 handoff contract version"
        )

    if summary["oil_detected"]:
        oil_required = {
            "confidence",
            "centroid",
            "area_km2",
            "perimeter_km",
            "polygon_file",
        }

        oil_missing = sorted(
            field
            for field in oil_required
            if summary.get(field) is None
        )

        if oil_missing:
            raise ValueError(
                "Oil detection summary is missing fields: "
                + ", ".join(oil_missing)
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
        )
        destination.write("\n")

    return str(output)


def write_preview_png(
    sar_image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> str:
    """
    Create a dashboard preview using SAR band 1 with a red
    oil-mask overlay.
    """

    image = np.asarray(sar_image)
    binary_mask = np.asarray(mask)

    if image.ndim != 3:
        raise ValueError(
            "SAR image must have shape (C,H,W) or (H,W,C)"
        )

    if image.shape[0] == 2:
        band = image[0]
    elif image.shape[-1] == 2:
        band = image[..., 0]
    else:
        raise ValueError(
            "Preview requires exactly two SAR bands"
        )

    if binary_mask.shape != band.shape:
        raise ValueError(
            "Preview mask shape must match SAR image dimensions"
        )

    valid = np.isfinite(band)

    if not np.any(valid):
        raise ValueError(
            "SAR preview band contains no finite values"
        )

    low, high = np.percentile(
        band[valid],
        [2.0, 98.0],
    )

    if high > low:
        normalized = np.clip(
            (band - low) / (high - low),
            0.0,
            1.0,
        )
    else:
        normalized = np.zeros_like(
            band,
            dtype=np.float32,
        )

    normalized[~valid] = 0.0

    grayscale = (
        normalized * 255.0
    ).astype(np.uint8)

    rgb = np.stack(
        [grayscale, grayscale, grayscale],
        axis=-1,
    ).astype(np.float32)

    oil_pixels = binary_mask == 1

    overlay_colour = np.array(
        [255.0, 55.0, 25.0],
        dtype=np.float32,
    )

    rgb[oil_pixels] = (
        0.45 * rgb[oil_pixels]
        + 0.55 * overlay_colour
    )

    preview = Image.fromarray(
        np.clip(rgb, 0, 255).astype(np.uint8),
        mode="RGB",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    preview.save(
        output,
        format="PNG",
        optimize=True,
    )

    return str(output)
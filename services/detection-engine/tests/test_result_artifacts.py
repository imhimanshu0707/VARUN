import json

import numpy as np
import pytest
from PIL import Image

from app.artifacts.result import (
    write_detection_summary,
    write_preview_png,
)


def test_write_detection_summary(tmp_path):
    summary = {
        "contract_version": "phase1-to-phase2-v1",
        "case_id": "CASE_DEMO_001",
        "scene_id": "SCENE_DEMO_001",
        "status": "SUCCESS",
        "oil_detected": True,
        "acquisition_time_utc": "2026-09-02T12:00:00Z",
        "crs": "EPSG:4326",
        "confidence": 0.91,
        "centroid": {
            "longitude": 72.75,
            "latitude": 19.25,
        },
        "area_km2": 2.4,
        "perimeter_km": 8.1,
        "source_image": "synthetic_spill.tif",
        "polygon_file": "spill_detection.geojson",
    }

    output = tmp_path / "detection_summary.json"

    write_detection_summary(summary, output)

    saved = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert saved == summary


def test_summary_rejects_missing_required_field(
    tmp_path,
):
    summary = {
        "contract_version": "phase1-to-phase2-v1",
    }

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        write_detection_summary(
            summary,
            tmp_path / "invalid.json",
        )


def test_write_preview_png(tmp_path):
    sar_image = np.stack(
        [
            np.arange(100, dtype=np.float32).reshape(10, 10),
            np.arange(100, 200, dtype=np.float32).reshape(10, 10),
        ]
    )

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1

    output = tmp_path / "preview.png"

    write_preview_png(
        sar_image=sar_image,
        mask=mask,
        output_path=output,
    )

    with Image.open(output) as preview:
        assert preview.format == "PNG"
        assert preview.mode == "RGB"
        assert preview.size == (10, 10)
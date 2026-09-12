import json

import numpy as np
from rasterio.transform import from_origin

from app.postprocessing.geometry import (
    build_spill_geometry,
    write_spill_geojson,
)


def test_build_spill_geometry_in_epsg4326():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1

    probability = np.zeros((10, 10), dtype=np.float32)
    probability[2:6, 3:8] = 0.8

    result = build_spill_geometry(
        mask=mask,
        transform=from_origin(
            72.74,
            19.26,
            0.001,
            0.001,
        ),
        source_crs="EPSG:4326",
        probability_map=probability,
        properties={
            "case_id": "CASE_DEMO_001",
            "scene_id": "SCENE_DEMO_001",
            "acquisition_time_utc": "2026-09-02T12:00:00Z",
            "source_image": "synthetic_spill.tif",
        },
    )

    feature = result["feature"]

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] in {
        "Polygon",
        "MultiPolygon",
    }
    assert feature["properties"]["case_id"] == "CASE_DEMO_001"
    assert feature["properties"]["crs"] == "EPSG:4326"

    assert 72.74 <= result["centroid"]["longitude"] <= 72.75
    assert 19.25 <= result["centroid"]["latitude"] <= 19.26

    assert result["area_km2"] > 0
    assert result["perimeter_km"] > 0
    assert abs(result["confidence"] - 0.8) < 1e-5


def test_empty_mask_returns_no_geometry():
    result = build_spill_geometry(
        mask=np.zeros((5, 5), dtype=np.uint8),
        transform=from_origin(
            72.74,
            19.26,
            0.001,
            0.001,
        ),
        source_crs="EPSG:4326",
    )

    assert result["feature"] is None
    assert result["geometry"] is None
    assert result["area_km2"] == 0.0
    assert result["perimeter_km"] == 0.0


def test_write_spill_geojson(tmp_path):
    mask = np.ones((2, 2), dtype=np.uint8)

    result = build_spill_geometry(
        mask=mask,
        transform=from_origin(
            72.74,
            19.26,
            0.001,
            0.001,
        ),
        source_crs="EPSG:4326",
    )

    output = tmp_path / "spill_detection.geojson"

    write_spill_geojson(
        feature=result["feature"],
        output_path=output,
    )

    saved = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert saved["type"] == "Feature"
    assert saved["geometry"]["type"] in {
        "Polygon",
        "MultiPolygon",
    }
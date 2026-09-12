import importlib
import json

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.main import app
from varun_phase1_core import Detection, Tile


inference_api = importlib.import_module(
    "app.api.inference"
)

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inference_generates_tiles():
    response = client.post(
        "/v1/inference",
        json={
            "image_width": 2048,
            "image_height": 2048,
            "tile_size": 1024,
            "overlap": 128,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["tile_count"] > 1
    assert data["tiles"][0]["x"] == 0
    assert data["tiles"][0]["y"] == 0


def test_real_inference_requires_handoff_metadata(
    tmp_path,
):
    image_path = tmp_path / "input.tif"
    image_path.write_bytes(b"not-opened")

    response = client.post(
        "/v1/inference",
        json={
            "image_path": str(image_path),
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert detail["error"] == "MISSING_PHASE1_METADATA"
    assert "case_id" in detail["missing_fields"]
    assert "scene_id" in detail["missing_fields"]
    assert (
        "acquisition_time_utc"
        in detail["missing_fields"]
    )


def test_real_inference_generates_complete_artifact_bundle(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "synthetic_spill.tif"

    image = np.stack(
        [
            np.arange(
                360,
                dtype=np.float32,
            ).reshape(18, 20),
            np.arange(
                360,
                720,
                dtype=np.float32,
            ).reshape(18, 20),
        ]
    )

    transform = from_origin(
        72.74,
        19.26,
        0.001,
        0.001,
    )

    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=18,
        width=20,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as destination:
        destination.write(image)

    fake_model_path = tmp_path / "model.pth"
    fake_model_path.write_bytes(b"fake-checkpoint")

    probability = np.zeros(
        (18, 20),
        dtype=np.float32,
    )
    probability[5:9, 7:12] = 0.8

    mask = np.zeros(
        (18, 20),
        dtype=np.uint8,
    )
    mask[5:9, 7:12] = 1

    def fake_run_inference(**kwargs):
        return {
            "tiles": [
                Tile(
                    x=0,
                    y=0,
                    width=20,
                    height=18,
                )
            ],
            "tile_probabilities": [probability],
            "probability_map": probability,
            "mask": mask,
            "detections": [
                Detection(
                    x=7.0,
                    y=5.0,
                    width=5.0,
                    height=4.0,
                    score=0.8,
                )
            ],
            "oil_pixel_count": 20,
            "total_pixel_count": 360,
            "oil_coverage_percent": (
                20 / 360
            ) * 100.0,
            "oil_detected": True,
            "threshold": 0.5,
            "min_area_pixels": 1,
        }

    monkeypatch.setattr(
        inference_api,
        "MODEL_PATH",
        fake_model_path,
    )
    monkeypatch.setattr(
        inference_api,
        "OUTPUTS_ROOT",
        tmp_path / "outputs",
    )
    monkeypatch.setattr(
        inference_api,
        "run_inference",
        fake_run_inference,
    )

    response = client.post(
        "/v1/inference",
        json={
            "image_path": str(image_path),
            "case_id": "CASE_DEMO_001",
            "scene_id": "SCENE_DEMO_001",
            "acquisition_time_utc": (
                "2026-09-02T12:00:00Z"
            ),
            "tile_size": 16,
            "overlap": 4,
            "confidence_threshold": 0.5,
            "min_area_pixels": 1,
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["contract_version"] == (
        "phase1-to-phase2-v1"
    )
    assert data["case_id"] == "CASE_DEMO_001"
    assert data["scene_id"] == "SCENE_DEMO_001"
    assert data["oil_detected"] is True
    assert data["crs"] == "EPSG:4326"
    assert data["geometry"]["type"] in {
        "Polygon",
        "MultiPolygon",
    }
    assert data["area_km2"] > 0
    assert data["perimeter_km"] > 0

    output_directory = (
        tmp_path
        / "outputs"
        / "CASE_DEMO_001"
        / "phase1"
    )

    expected_files = {
        "spill_detection.geojson",
        "detection_summary.json",
        "oil_probability.tif",
        "oil_mask.tif",
        "preview.png",
    }

    assert {
        path.name
        for path in output_directory.iterdir()
    } == expected_files

    summary = json.loads(
        (
            output_directory
            / "detection_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["case_id"] == "CASE_DEMO_001"
    assert summary["polygon_file"] == (
        "spill_detection.geojson"
    )

    geojson = json.loads(
        (
            output_directory
            / "spill_detection.geojson"
        ).read_text(encoding="utf-8")
    )

    assert geojson["type"] == "Feature"
    assert geojson["properties"]["case_id"] == (
        "CASE_DEMO_001"
    )

    with rasterio.open(
        output_directory / "oil_probability.tif"
    ) as source:
        assert source.crs.to_string() == "EPSG:4326"
        assert source.transform == transform
        assert source.dtypes[0] == "float32"

    with rasterio.open(
        output_directory / "oil_mask.tif"
    ) as source:
        assert source.crs.to_string() == "EPSG:4326"
        assert source.transform == transform
        assert source.dtypes[0] == "uint8"
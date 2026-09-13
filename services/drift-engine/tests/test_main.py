import app.main as main_module
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "phase2-drift"
    assert data["contractVersion"] == "phase2-to-phase3-v1"


def test_drift_run(
    monkeypatch,
    tmp_path,
):
    class FakeForcingConfig:
        def get_files(self):
            return []

    class FakeRunner:
        def run(self, **kwargs):
            assert kwargs["case_id"] == "CASE_TEST_001"
            assert kwargs["scene_id"] == "SCENE_TEST_001"
            assert (
                kwargs["run_id"]
                == "DRIFT_RUN_TEST"
            )
            assert kwargs["spill_polygon_geojson"][
                "type"
            ] == "Polygon"

            return {
                "status": "SUCCESS",
                "run_id": "DRIFT_RUN_TEST",
                "case_id": "CASE_TEST_001",
                "output_dir": str(tmp_path),
                "summary": {
                    "particle_count": 50,
                    "release_ages": [
                        6,
                        12,
                        24,
                        36,
                    ],
                    "successful_release_ages": [
                        6,
                        12,
                        24,
                        36,
                    ],
                    "best_release_age_hours": 12,
                    "forecast_hours": 6.0,
                },
                "validation": {
                    "status": "PASS",
                    "checks": {},
                },
            }

    monkeypatch.setattr(
        main_module,
        "get_forcing_config",
        lambda: FakeForcingConfig(),
    )
    monkeypatch.setattr(
        main_module,
        "Phase2CompleteRunner",
        FakeRunner,
    )

    response = client.post(
        "/internal/v1/drift-runs",
        json={
            "case_id": "CASE_TEST_001",
            "phase2_run_id": "DRIFT_RUN_TEST",
            "phase1_handoff_ref": "P1_RUN_001",
            "scene_id": "SCENE_TEST_001",
            "observation_time_utc":
                "2026-09-02T12:00:00Z",
            "spill_geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [72.74, 19.24],
                        [72.76, 19.24],
                        [72.76, 19.26],
                        [72.74, 19.26],
                        [72.74, 19.24],
                    ]
                ],
            },
            "mode": "HINDCAST_AND_FORECAST",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "COMPLETED"
    assert (
        data["contract_version"]
        == "phase2-to-phase3-v1"
    )
    assert (
        data["phase2_run_id"]
        == "DRIFT_RUN_TEST"
    )
    assert (
        data["data_origin"]
        == "PHASE1_HANDOFF_OPENOIL"
    )
    assert data["hindcast"][
        "bestReleaseAgeHours"
    ] == 12
    assert data["trajectories"] == []

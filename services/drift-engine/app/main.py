from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.forcing.models import ForcingConfig
from app.simulation.runner import Phase2CompleteRunner

app = FastAPI(
    title="VARUN Phase 2 Drift",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

class DriftRequest(BaseModel):
    case_id: str
    phase2_run_id: str
    phase1_handoff_ref: str
    scene_id: str
    observation_time_utc: datetime
    spill_geometry: dict
    mode: Literal["HINDCAST_AND_FORECAST", "HINDCAST", "FORECAST"] = (
        "HINDCAST_AND_FORECAST"
    )


class Point(BaseModel):
    latitude: float
    longitude: float
    timestamp_utc: str


class DriftTrajectory(BaseModel):
    trajectory_id: str
    kind: Literal["HINDCAST", "RECONSTRUCTION", "FORECAST"]
    seed_index: int
    points: list[Point]


class DriftResponse(BaseModel):
    status: Literal["COMPLETED", "COMPLETED_WITH_WARNINGS"]
    contract_version: str
    phase2_run_id: str
    case_id: str

    data_origin: str
    forcing: dict
    seeding: dict

    hindcast: dict
    reconstruction: dict
    forecast: dict

    trajectories: list[DriftTrajectory]

    artifacts: list[dict]
    provenance: dict
    warnings: list[str]


CONTRACT_VERSION = "phase2-to-phase3-v1"
ENGINE_VERSION = "VARUN-PHASE2-DRIFT-V2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_forcing_config() -> ForcingConfig:
    combined_value = os.getenv(
        "PHASE2_COMBINED_FORCING_FILE"
    )
    current_value = os.getenv(
        "PHASE2_CURRENT_FORCING_FILE"
    )
    wind_value = os.getenv(
        "PHASE2_WIND_FORCING_FILE"
    )

    if combined_value:
        if current_value or wind_value:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Configure either combined forcing or "
                    "separate current/wind forcing files"
                ),
            )

        combined_path = Path(
            combined_value
        ).expanduser().resolve()

        if not combined_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Configured combined forcing file "
                    "does not exist"
                ),
            )

        return ForcingConfig(
            combined_file=combined_path
        )

    if current_value and wind_value:
        current_path = Path(
            current_value
        ).expanduser().resolve()
        wind_path = Path(
            wind_value
        ).expanduser().resolve()

        if (
            not current_path.is_file()
            or not wind_path.is_file()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Configured current or wind forcing "
                    "file does not exist"
                ),
            )

        return ForcingConfig(
            current_file=current_path,
            wind_file=wind_path,
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "Phase 2 forcing is not configured. Set "
            "PHASE2_COMBINED_FORCING_FILE or both "
            "PHASE2_CURRENT_FORCING_FILE and "
            "PHASE2_WIND_FORCING_FILE"
        ),
    )

def get_particle_count() -> int:
    raw_value = os.getenv(
        "PHASE2_PARTICLE_COUNT",
        "1500",
    )

    try:
        particle_count = int(raw_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "PHASE2_PARTICLE_COUNT must be an integer"
            ),
        ) from exc

    if not 1 <= particle_count <= 10_000:
        raise HTTPException(
            status_code=500,
            detail=(
                "PHASE2_PARTICLE_COUNT must be "
                "between 1 and 10000"
            ),
        )

    return particle_count

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")



def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def artifact_media_type(path: Path) -> str:
    media_types = {
        ".json": "application/json",
        ".geojson": "application/geo+json",
        ".csv": "text/csv",
        ".nc": "application/x-netcdf",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }

    return media_types.get(
        path.suffix.lower(),
        "application/octet-stream",
    )


def collect_artifacts(
    output_dir: Path,
) -> list[dict]:
    artifacts = []

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_name = path.relative_to(
            output_dir
        ).as_posix()

        logical_name = relative_name.replace(
            "/",
            "__",
        )

        artifacts.append(
            {
                "logicalName": logical_name,
                "relativePath": relative_name,
                "uri": str(path.resolve()),
                "mediaType": artifact_media_type(path),
                "checksumSha256": sha256_file(path),
                "sizeBytes": path.stat().st_size,
            }
        )

    return artifacts


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "phase2-drift",
        "version": "0.1.0",
    }


@app.get("/version")
def version():
    return {
        "service": "phase2-drift",
        "engineVersion": ENGINE_VERSION,
        "contractVersion": CONTRACT_VERSION,
        "scientificBackend": "OpenDrift/OpenOil",
    }


@app.post(
    "/internal/v1/drift-runs",
    response_model=DriftResponse,
)
def run(request: DriftRequest) -> DriftResponse:
    if request.mode != "HINDCAST_AND_FORECAST":
        raise HTTPException(
            status_code=422,
            detail=(
                "The real Phase 2 pipeline currently requires "
                "HINDCAST_AND_FORECAST mode"
            ),
        )

    forcing_config = get_forcing_config()
    started = utc_now()

    handoff_digest = sha256_json(
        {
            "case_id": request.case_id,
            "scene_id": request.scene_id,
            "observation_time_utc": iso(
                request.observation_time_utc
            ),
            "spill_geometry": request.spill_geometry,
            "phase1_handoff_ref":
                request.phase1_handoff_ref,
        }
    )

    runner = Phase2CompleteRunner()

    result = runner.run(
        case_id=request.case_id,
        scene_id=request.scene_id,
        observation_time=request.observation_time_utc,
        spill_polygon_geojson=request.spill_geometry,
        forcing_config=forcing_config,
        run_id=request.phase2_run_id,
        particle_count=get_particle_count(),
    )

    if result.get("status") != "SUCCESS":
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PHASE2_SIMULATION_FAILED",
                "message": result.get(
                    "error",
                    "Real Phase 2 simulation failed",
                ),
            },
        )

    output_dir_value = result.get("output_dir")

    if not output_dir_value:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PHASE2_OUTPUT_MISSING",
                "message": (
                    "Phase 2 completed without an output directory"
                ),
            },
        )

    output_dir = Path(output_dir_value).resolve()
    artifacts = collect_artifacts(output_dir)

    summary = result.get("summary", {})
    validation = result.get("validation", {})

    metrics_path = (
        output_dir / "reconstruction_metrics.json"
    )
    reconstruction_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )

    search_window_path = (
        output_dir / "search_window.json"
    )
    search_window = (
        json.loads(
            search_window_path.read_text(
                encoding="utf-8"
            )
        )
        if search_window_path.is_file()
        else {}
    )

    successful_release_ages = summary.get(
        "successful_release_ages",
        [],
    )

    forcing_files = [
        str(path.resolve())
        for path in forcing_config.get_files()
    ]

    warnings: list[str] = []

    if validation.get("status") != "PASS":
        warnings.append(
            "Phase 2 completed, but one or more "
            "validation checks did not pass."
        )

    response_status = (
        "COMPLETED"
        if not warnings
        else "COMPLETED_WITH_WARNINGS"
    )

    finished = utc_now()

    return DriftResponse(
        status=response_status,
        contract_version=CONTRACT_VERSION,
        phase2_run_id=request.phase2_run_id,
        case_id=request.case_id,
        data_origin="PHASE1_HANDOFF_OPENOIL",
        forcing={
            "source": "CONFIGURED_NETCDF",
            "files": forcing_files,
            "validated": True,
        },
        seeding={
            "strategy": "PHASE1_SPILL_GEOMETRY",
            "particleCount": summary.get(
                "particle_count"
            ),
            "sourceRunId":
                request.phase1_handoff_ref,
        },
        hindcast={
            "enabled": True,
            "releaseAgesHours": summary.get(
                "release_ages",
                [],
            ),
            "successfulReleaseAgesHours":
                successful_release_ages,
            "bestReleaseAgeHours": summary.get(
                "best_release_age_hours"
            ),
        },
        reconstruction={
            "enabled": True,
            "metrics": reconstruction_metrics,
        },
        forecast={
            "enabled": True,
            "horizonHours": summary.get(
                "forecast_hours"
            ),
        },
        trajectories=[],
        artifacts=artifacts,
        provenance={
            "engineVersion": ENGINE_VERSION,
            "contractVersion": CONTRACT_VERSION,
            "phase1HandoffRef":
                request.phase1_handoff_ref,
            "phase1HandoffDigest":
                handoff_digest,
            "sceneId": request.scene_id,
            "observationTimeUtc": iso(
                request.observation_time_utc
            ),
            "executionStartedAt": iso(started),
            "executionFinishedAt": iso(finished),
            "scientificBackend":
                "OpenDrift/OpenOil",
            "executionMode": request.mode,
            "dataOrigin":
                "PHASE1_HANDOFF_OPENOIL",
            "outputDirectory": str(output_dir),
            "searchWindow": search_window,
            "validation": validation,
            "trajectoryDelivery":
                "NETCDF_AND_GEOJSON_ARTIFACTS",
        },
        warnings=warnings,
    )

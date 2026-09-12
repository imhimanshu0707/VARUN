from typing import Any

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    image_path: str | None = None

    contract_version: str = "phase1-to-phase2-v1"
    case_id: str | None = None
    scene_id: str | None = None
    acquisition_time_utc: str | None = None
    output_dir: str | None = None

    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)

    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    min_area_pixels: int = Field(default=20, ge=1)

    # Temporary backward compatibility
    iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


class TileResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectionResponse(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str


class CentroidResponse(BaseModel):
    longitude: float
    latitude: float


class ArtifactResponse(BaseModel):
    spill_detection_geojson: str | None = None
    detection_summary_json: str | None = None
    oil_probability_tif: str | None = None
    oil_mask_tif: str | None = None
    preview_png: str | None = None


class InferenceResponse(BaseModel):
    contract_version: str = "phase1-to-phase2-v1"
    status: str

    case_id: str | None = None
    scene_id: str | None = None
    run_id: str | None = None
    acquisition_time_utc: str | None = None

    oil_detected: bool = False
    confidence: float | None = None

    centroid: CentroidResponse | None = None
    area_km2: float | None = None
    perimeter_km: float | None = None

    crs: str | None = None
    source_image: str | None = None
    geometry: dict[str, Any] | None = None

    model_version: str | None = None
    preprocessing_version: str | None = None
    threshold: float | None = None
    min_area_pixels: int | None = None

    tile_count: int = 0
    detection_count: int = 0
    oil_pixel_count: int = 0
    total_pixel_count: int = 0
    oil_coverage_percent: float = 0.0

    image_width: int | None = None
    image_height: int | None = None

    artifacts: ArtifactResponse = Field(
        default_factory=ArtifactResponse
    )

    # Retained for debugging/backward compatibility
    detections: list[DetectionResponse] = Field(
        default_factory=list
    )
    tiles: list[TileResponse] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)
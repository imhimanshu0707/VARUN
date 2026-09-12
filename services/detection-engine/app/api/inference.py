from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException

from app.artifacts.raster import (
    write_mask_geotiff,
    write_probability_geotiff,
)
from app.artifacts.result import (
    write_detection_summary,
    write_preview_png,
)
from app.contracts.inference import (
    ArtifactResponse,
    DetectionResponse,
    InferenceRequest,
    InferenceResponse,
    TileResponse,
)
from app.inference.pipeline import run_inference
from app.inference.tiling import create_tiles
from app.postprocessing.geometry import (
    build_spill_geometry,
    write_spill_geojson,
)


router = APIRouter()

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = (
    REPOSITORY_ROOT
    / "models"
    / "phase1"
    / "unet_oil_spill_v0.1.pth"
)
OUTPUTS_ROOT = REPOSITORY_ROOT / "outputs"

MODEL_VERSION = "unet-oil-spill-v0.1"
PREPROCESSING_VERSION = "sar-percentile-v1"


def _require_real_inference_metadata(
    request: InferenceRequest,
) -> None:
    missing = []

    if not request.case_id:
        missing.append("case_id")

    if not request.scene_id:
        missing.append("scene_id")

    if not request.acquisition_time_utc:
        missing.append("acquisition_time_utc")

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "MISSING_PHASE1_METADATA",
                "missing_fields": missing,
            },
        )


@router.post(
    "/inference",
    response_model=InferenceResponse,
)
def inference(request: InferenceRequest):
    # Tile-layout validation mode: no model execution.
    if request.image_path is None:
        if (
            request.image_width is None
            or request.image_height is None
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "image_width and image_height are required "
                    "when image_path is not provided"
                ),
            )

        tiles = create_tiles(
            image_width=request.image_width,
            image_height=request.image_height,
            tile_size=request.tile_size,
            overlap=request.overlap,
        )

        return InferenceResponse(
            status="SUCCESS",
            tile_count=len(tiles),
            tiles=[
                TileResponse(
                    x=int(tile.x),
                    y=int(tile.y),
                    width=int(tile.width),
                    height=int(tile.height),
                )
                for tile in tiles
            ],
            model_version=MODEL_VERSION,
            preprocessing_version=PREPROCESSING_VERSION,
            threshold=request.confidence_threshold,
            min_area_pixels=request.min_area_pixels,
            image_width=request.image_width,
            image_height=request.image_height,
        )

    _require_real_inference_metadata(request)

    image_path = Path(request.image_path)

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Input GeoTIFF not found: {image_path}",
        )

    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(
            status_code=422,
            detail="Phase-1 input must be a GeoTIFF",
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model weights not found: {MODEL_PATH}",
        )

    run_id = f"PHASE1_{uuid4().hex[:12].upper()}"

    try:
        with rasterio.open(image_path) as source:
            if source.count != 2:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Phase-1 U-Net requires exactly two "
                        f"VV/VH bands, got {source.count}"
                    ),
                )

            if source.crs is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Input GeoTIFF has no CRS. Geographic "
                        "polygon generation is not possible."
                    ),
                )

            image = source.read().astype(np.float32)
            image_width = source.width
            image_height = source.height
            source_crs = source.crs
            source_transform = source.transform

        if (
            request.image_width is not None
            and request.image_width != image_width
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Provided image_width does not match "
                    f"GeoTIFF width {image_width}"
                ),
            )

        if (
            request.image_height is not None
            and request.image_height != image_height
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Provided image_height does not match "
                    f"GeoTIFF height {image_height}"
                ),
            )

        result = run_inference(
            image=image,
            image_width=image_width,
            image_height=image_height,
            model_path=str(MODEL_PATH),
            tile_size=request.tile_size,
            overlap=request.overlap,
            confidence_threshold=(
                request.confidence_threshold
            ),
            min_area_pixels=request.min_area_pixels,
            iou_threshold=request.iou_threshold,
        )

        output_directory = (
            OUTPUTS_ROOT
            / request.case_id
            / "phase1"
        )
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        probability_path = (
            output_directory / "oil_probability.tif"
        )
        mask_path = output_directory / "oil_mask.tif"
        geojson_path = (
            output_directory / "spill_detection.geojson"
        )
        summary_path = (
            output_directory / "detection_summary.json"
        )
        preview_path = output_directory / "preview.png"

        write_probability_geotiff(
            probability_map=result["probability_map"],
            output_path=probability_path,
            crs=source_crs,
            transform=source_transform,
        )

        write_mask_geotiff(
            mask=result["mask"],
            output_path=mask_path,
            crs=source_crs,
            transform=source_transform,
        )

        write_preview_png(
            sar_image=image,
            mask=result["mask"],
            output_path=preview_path,
        )

        geometry_result = build_spill_geometry(
            mask=result["mask"],
            transform=source_transform,
            source_crs=source_crs,
            probability_map=result["probability_map"],
            properties={
                "case_id": request.case_id,
                "scene_id": request.scene_id,
                "acquisition_time_utc": (
                    request.acquisition_time_utc
                ),
                "source_image": image_path.name,
            },
        )

        polygon_file = None
        warnings = []

        if geometry_result["feature"] is not None:
            write_spill_geojson(
                feature=geometry_result["feature"],
                output_path=geojson_path,
            )
            polygon_file = geojson_path.name
        else:
            warnings.append(
                "No oil polygon was generated because the "
                "cleaned mask contains no oil pixels."
            )

        summary = {
            "contract_version": (
                "phase1-to-phase2-v1"
            ),
            "case_id": request.case_id,
            "scene_id": request.scene_id,
            "run_id": run_id,
            "status": "SUCCESS",
            "oil_detected": result["oil_detected"],
            "acquisition_time_utc": (
                request.acquisition_time_utc
            ),
            "crs": "EPSG:4326",
            "confidence": geometry_result["confidence"],
            "centroid": geometry_result["centroid"],
            "area_km2": geometry_result["area_km2"],
            "perimeter_km": (
                geometry_result["perimeter_km"]
            ),
            "source_image": image_path.name,
            "polygon_file": polygon_file,
            "model_version": MODEL_VERSION,
            "preprocessing_version": (
                PREPROCESSING_VERSION
            ),
            "threshold": result["threshold"],
            "min_area_pixels": (
                result["min_area_pixels"]
            ),
            "oil_pixel_count": (
                result["oil_pixel_count"]
            ),
            "total_pixel_count": (
                result["total_pixel_count"]
            ),
            "oil_coverage_percent": (
                result["oil_coverage_percent"]
            ),
        }

        write_detection_summary(
            summary=summary,
            output_path=summary_path,
        )

        detections = [
            DetectionResponse(
                x=float(detection.x),
                y=float(detection.y),
                width=float(detection.width),
                height=float(detection.height),
                confidence=float(detection.score),
                label="OIL_LIKELIHOOD",
            )
            for detection in result["detections"]
        ]

        tiles = [
            TileResponse(
                x=int(tile.x),
                y=int(tile.y),
                width=int(tile.width),
                height=int(tile.height),
            )
            for tile in result["tiles"]
        ]

        return InferenceResponse(
            status="SUCCESS",
            case_id=request.case_id,
            scene_id=request.scene_id,
            run_id=run_id,
            acquisition_time_utc=(
                request.acquisition_time_utc
            ),
            oil_detected=result["oil_detected"],
            confidence=geometry_result["confidence"],
            centroid=geometry_result["centroid"],
            area_km2=geometry_result["area_km2"],
            perimeter_km=(
                geometry_result["perimeter_km"]
            ),
            crs="EPSG:4326",
            source_image=image_path.name,
            geometry=geometry_result["geometry"],
            model_version=MODEL_VERSION,
            preprocessing_version=(
                PREPROCESSING_VERSION
            ),
            threshold=result["threshold"],
            min_area_pixels=result["min_area_pixels"],
            tile_count=len(tiles),
            detection_count=len(detections),
            oil_pixel_count=result["oil_pixel_count"],
            total_pixel_count=(
                result["total_pixel_count"]
            ),
            oil_coverage_percent=(
                result["oil_coverage_percent"]
            ),
            image_width=image_width,
            image_height=image_height,
            artifacts=ArtifactResponse(
                spill_detection_geojson=(
                    str(geojson_path)
                    if polygon_file is not None
                    else None
                ),
                detection_summary_json=str(
                    summary_path
                ),
                oil_probability_tif=str(
                    probability_path
                ),
                oil_mask_tif=str(mask_path),
                preview_png=str(preview_path),
            ),
            detections=detections,
            tiles=tiles,
            warnings=warnings,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PHASE1_INFERENCE_FAILED",
                "message": str(error),
                "run_id": run_id,
            },
        ) from error
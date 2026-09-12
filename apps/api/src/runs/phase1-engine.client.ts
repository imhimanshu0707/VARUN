import { Injectable } from '@nestjs/common';

export interface Phase1EngineDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  label: string;
}

export interface Phase1Centroid {
  longitude: number;
  latitude: number;
}

export interface Phase1Geometry {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: unknown;
}

export interface Phase1Artifacts {
  spill_detection_geojson: string | null;
  detection_summary_json: string | null;
  oil_probability_tif: string | null;
  oil_mask_tif: string | null;
  preview_png: string | null;
}

export interface Phase1EngineResult {
  contract_version: 'phase1-to-phase2-v1';
  status: string;

  case_id: string | null;
  scene_id: string | null;
  run_id: string | null;
  acquisition_time_utc: string | null;

  oil_detected: boolean;
  confidence: number | null;
  centroid: Phase1Centroid | null;
  area_km2: number | null;
  perimeter_km: number | null;

  crs: string | null;
  source_image: string | null;
  geometry: Phase1Geometry | null;

  model_version: string | null;
  preprocessing_version: string | null;
  threshold: number | null;
  min_area_pixels: number | null;

  tile_count: number;
  detection_count: number;
  oil_pixel_count: number;
  total_pixel_count: number;
  oil_coverage_percent: number;

  image_width: number | null;
  image_height: number | null;

  artifacts: Phase1Artifacts;
  detections: Phase1EngineDetection[];
  tiles: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
  warnings: string[];
}

@Injectable()
export class Phase1EngineClient {
  private readonly baseUrl =
    process.env.DETECTION_ENGINE_URL ??
    'http://127.0.0.1:8000';

  async infer(input: {
    imagePath: string;
    imageWidth?: number;
    imageHeight?: number;

    caseId: string;
    sceneId: string;
    acquisitionTimeUtc: string;

    tileSize?: number;
    overlap?: number;
    confidenceThreshold?: number;
    minAreaPixels?: number;
    iouThreshold?: number;
  }): Promise<Phase1EngineResult> {
    const response = await fetch(
      `${this.baseUrl}/v1/inference`,
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          image_path: input.imagePath,
          image_width: input.imageWidth,
          image_height: input.imageHeight,

          contract_version:
            'phase1-to-phase2-v1',
          case_id: input.caseId,
          scene_id: input.sceneId,
          acquisition_time_utc:
            input.acquisitionTimeUtc,

          tile_size: input.tileSize ?? 512,
          overlap: input.overlap ?? 64,
          confidence_threshold:
            input.confidenceThreshold ?? 0.5,
          min_area_pixels:
            input.minAreaPixels ?? 20,
          iou_threshold:
            input.iouThreshold ?? 0.5,
        }),
      },
    );

    const text = await response.text();

    if (!response.ok) {
      throw new Error(
        `DETECTION_ENGINE_${response.status}: ${text}`,
      );
    }

    const result =
      JSON.parse(text) as Phase1EngineResult;

    if (
      result.contract_version !==
      'phase1-to-phase2-v1'
    ) {
      throw new Error(
        `UNSUPPORTED_PHASE1_CONTRACT: ${result.contract_version}`,
      );
    }

    return result;
  }
}
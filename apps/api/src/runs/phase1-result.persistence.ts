import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import {
  createHash,
  randomUUID,
} from 'crypto';
import {
  createReadStream,
} from 'fs';
import { stat } from 'fs/promises';

import { PrismaService } from '../database/prisma.service';
import {
  Phase1Artifacts,
  Phase1EngineResult,
} from './phase1-engine.client';


interface ArtifactDefinition {
  field: keyof Phase1Artifacts;
  logicalName: string;
  artifactVersion: string;
  role:
    | 'SCIENTIFIC'
    | 'DASHBOARD'
    | 'HANDOFF';
  mediaType: string;
  layer?: {
    layerKey: string;
    kind: string;
    role: string;
    visibleByDefault: boolean;
    displayOrder: number;
    style: Prisma.InputJsonObject;
  };
}


const ARTIFACT_DEFINITIONS:
  ArtifactDefinition[] = [
    {
      field: 'spill_detection_geojson',
      logicalName: 'spill_detection.geojson',
      artifactVersion: 'phase1-to-phase2-v1',
      role: 'HANDOFF',
      mediaType: 'application/geo+json',
      layer: {
        layerKey: 'phase1-spill-detection',
        kind: 'GEOJSON',
        role: 'SPILL_DETECTION',
        visibleByDefault: true,
        displayOrder: 10,
        style: {
          fillColor: '#ef4444',
          fillOpacity: 0.35,
          lineColor: '#dc2626',
          lineWidth: 2,
        },
      },
    },
    {
      field: 'detection_summary_json',
      logicalName: 'detection_summary.json',
      artifactVersion: 'phase1-to-phase2-v1',
      role: 'HANDOFF',
      mediaType: 'application/json',
    },
    {
      field: 'oil_probability_tif',
      logicalName: 'oil_probability.tif',
      artifactVersion: 'phase1-to-phase2-v1',
      role: 'SCIENTIFIC',
      mediaType: 'image/tiff; application=geotiff',
      layer: {
        layerKey: 'phase1-oil-probability',
        kind: 'GEOTIFF',
        role: 'OIL_PROBABILITY',
        visibleByDefault: false,
        displayOrder: 20,
        style: {
          colourMap: 'inferno',
          minimum: 0,
          maximum: 1,
          opacity: 0.7,
        },
      },
    },
    {
      field: 'oil_mask_tif',
      logicalName: 'oil_mask.tif',
      artifactVersion: 'phase1-to-phase2-v1',
      role: 'SCIENTIFIC',
      mediaType: 'image/tiff; application=geotiff',
      layer: {
        layerKey: 'phase1-oil-mask',
        kind: 'GEOTIFF',
        role: 'OIL_MASK',
        visibleByDefault: false,
        displayOrder: 30,
        style: {
          backgroundValue: 0,
          oilValue: 1,
          oilColor: '#ef4444',
          opacity: 0.55,
        },
      },
    },
    {
      field: 'preview_png',
      logicalName: 'preview.png',
      artifactVersion: 'phase1-to-phase2-v1',
      role: 'DASHBOARD',
      mediaType: 'image/png',
      layer: {
        layerKey: 'phase1-preview',
        kind: 'IMAGE',
        role: 'PREVIEW',
        visibleByDefault: true,
        displayOrder: 40,
        style: {
          description:
            'SAR image with oil-mask overlay',
        },
      },
    },
  ];


async function calculateChecksum(
  filePath: string,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);

    stream.on('error', reject);
    stream.on('data', (chunk) => {
      hash.update(chunk);
    });
    stream.on('end', () => {
      resolve(hash.digest('hex'));
    });
  });
}


@Injectable()
export class Phase1ResultPersistence {
  constructor(
    private readonly prisma: PrismaService,
  ) {}

  async persist(
    analysisRunId: string,
    result: Phase1EngineResult,
  ) {
    if (
      result.contract_version !==
      'phase1-to-phase2-v1'
    ) {
      throw new Error(
        'Unsupported Phase-1 result contract',
      );
    }

    if (
      !result.case_id ||
      !result.scene_id ||
      !result.acquisition_time_utc
    ) {
      throw new Error(
        'Phase-1 result is missing case, scene or acquisition time',
      );
    }

    const run =
      await this.prisma.analysisRun.findUnique({
        where: {
          id: analysisRunId,
        },
      });

    if (!run) {
      throw new Error(
        `Analysis run ${analysisRunId} not found`,
      );
    }

    if (
      run.caseId !== result.case_id ||
      run.sceneId !== result.scene_id
    ) {
      throw new Error(
        'Phase-1 result identity does not match analysis run',
      );
    }

    const modelVersion =
      await this.prisma.modelVersion.findFirst({
        where: {
          modelName: 'VARUN U-Net Oil Spill',
          semanticVersion: '0.1.0',
        },
      });

    if (!modelVersion) {
      throw new Error(
        'Phase-1 U-Net ModelVersion is not seeded',
      );
    }

    const observationTime = new Date(
      result.acquisition_time_utc,
    );

    if (
      Number.isNaN(observationTime.getTime())
    ) {
      throw new Error(
        'Invalid Phase-1 acquisition timestamp',
      );
    }

    const storedDetectionCount =
      result.oil_detected &&
      result.geometry &&
      result.centroid
        ? 1
        : 0;

    const summary: Prisma.InputJsonObject = {
      contractVersion:
        result.contract_version,
      engineRunId:
        result.run_id,
      oilDetected:
        result.oil_detected,
      confidence:
        result.confidence,
      areaKm2:
        result.area_km2,
      perimeterKm:
        result.perimeter_km,
      centroid:
        result.centroid
    ? {
      longitude:
        result.centroid.longitude,
      latitude:
        result.centroid.latitude,
      }
    : null,
      crs:
        result.crs,
      sourceImage:
        result.source_image,
      threshold:
        result.threshold,
      minAreaPixels:
        result.min_area_pixels,
      oilPixelCount:
        result.oil_pixel_count,
      totalPixelCount:
        result.total_pixel_count,
      oilCoveragePercent:
        result.oil_coverage_percent,
      componentCount:
        result.detection_count,
      warnings:
        result.warnings,
    };

    await this.prisma.phase1Result.upsert({
      where: {
        analysisRunId,
      },
      create: {
        analysisRunId,
        modelVersionId:
          modelVersion.id,
        observationTimeUtc:
          observationTime,
        sceneMetricAvailability:
          'AVAILABLE',
        detectionCount:
          storedDetectionCount,
        summary,
      },
      update: {
        modelVersionId:
          modelVersion.id,
        observationTimeUtc:
          observationTime,
        sceneMetricAvailability:
          'AVAILABLE',
        detectionCount:
          storedDetectionCount,
        summary,
      },
    });

    await this.prisma.detectionRegion.deleteMany({
      where: {
        analysisRunId,
      },
    });

    if (
      storedDetectionCount === 1 &&
      result.geometry &&
      result.centroid
    ) {
      const geometryJson =
        JSON.stringify(result.geometry);

      const propertiesJson =
        JSON.stringify({
          contractVersion:
            result.contract_version,
          sourceImage:
            result.source_image,
          threshold:
            result.threshold,
          componentCount:
            result.detection_count,
        });

      await this.prisma.$executeRaw(
        Prisma.sql`
          INSERT INTO detection_regions (
            detection_region_id,
            analysis_run_id,
            region_no,
            classification,
            geometry,
            centroid,
            area_m2,
            perimeter_m,
            mean_likelihood,
            properties
          )
          VALUES (
            ${randomUUID()}::uuid,
            ${analysisRunId}::uuid,
            1,
            'OIL_LIKELIHOOD',
            ST_SetSRID(
              ST_GeomFromGeoJSON(${geometryJson}),
              4326
            ),
            ST_SetSRID(
              ST_MakePoint(
                ${result.centroid.longitude},
                ${result.centroid.latitude}
              ),
              4326
            ),
            ${
              result.area_km2 === null
                ? null
                : result.area_km2 * 1_000_000
            },
            ${
              result.perimeter_km === null
                ? null
                : result.perimeter_km * 1_000
            },
            ${result.confidence},
            ${propertiesJson}::jsonb
          )
        `,
      );
    }

    const artifactIds =
      new Map<string, string>();

    for (
      const definition
      of ARTIFACT_DEFINITIONS
    ) {
      const uri =
        result.artifacts[definition.field];

      if (!uri) {
        continue;
      }

      const fileInfo = await stat(uri);
      const checksum =
        await calculateChecksum(uri);

      const artifact =
        await this.prisma.artifact.upsert({
          where: {
            analysisRunId_logicalName_artifactVersion: {
              analysisRunId,
              logicalName:
                definition.logicalName,
              artifactVersion:
                definition.artifactVersion,
            },
          },
          create: {
            analysisRunId,
            role: definition.role,
            logicalName:
              definition.logicalName,
            artifactVersion:
              definition.artifactVersion,
            uri,
            mediaType:
              definition.mediaType,
            checksumSha256:
              checksum,
            sizeBytes:
              BigInt(fileInfo.size),
            timeStartUtc:
              observationTime,
            timeEndUtc:
              observationTime,
            metadata: {
              contractVersion:
                result.contract_version,
              crs:
                result.crs,
              sourceImage:
                result.source_image,
            },
          },
          update: {
            role: definition.role,
            uri,
            mediaType:
              definition.mediaType,
            checksumSha256:
              checksum,
            sizeBytes:
              BigInt(fileInfo.size),
            timeStartUtc:
              observationTime,
            timeEndUtc:
              observationTime,
            metadata: {
              contractVersion:
                result.contract_version,
              crs:
                result.crs,
              sourceImage:
                result.source_image,
            },
          },
        });

      artifactIds.set(
        definition.logicalName,
        artifact.id,
      );

      if (definition.layer) {
        await this.prisma.dashboardLayer.upsert({
          where: {
            analysisRunId_layerKey: {
              analysisRunId,
              layerKey:
                definition.layer.layerKey,
            },
          },
          create: {
            analysisRunId,
            artifactId:
              artifact.id,
            layerKey:
              definition.layer.layerKey,
            kind:
              definition.layer.kind,
            role:
              definition.layer.role,
            visibleByDefault:
              definition.layer
                .visibleByDefault,
            displayOrder:
              definition.layer.displayOrder,
            style:
              definition.layer.style,
            availability:
              'AVAILABLE',
          },
          update: {
            artifactId:
              artifact.id,
            kind:
              definition.layer.kind,
            role:
              definition.layer.role,
            visibleByDefault:
              definition.layer
                .visibleByDefault,
            displayOrder:
              definition.layer.displayOrder,
            style:
              definition.layer.style,
            availability:
              'AVAILABLE',
          },
        });
      }
    }

    return {
      phase1ResultStored: true,
      detectionCount:
        storedDetectionCount,
      artifactCount:
        artifactIds.size,
    };
  }
}
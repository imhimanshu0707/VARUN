BEGIN;

INSERT INTO cases (
    case_id,
    case_code,
    title,
    description,
    data_origin,
    region,
    status,
    metadata
)
VALUES (
    '00000000-0000-4000-8000-000000000001',
    'CASE_DEMO_001',
    'VARUN Phase-1 Demo Case',
    'Synthetic georeferenced oil-spill demonstration case.',
    'SYNTHETIC',
    ST_GeomFromText(
        'POLYGON((
            72.5 19.0,
            73.0 19.0,
            73.0 19.5,
            72.5 19.5,
            72.5 19.0
        ))',
        4326
    ),
    'OPEN',
    jsonb_build_object(
        'contract_version',
        'phase1-to-phase2-v1',
        'purpose',
        'phase1-demo'
    )
)
ON CONFLICT (case_id)
DO UPDATE SET
    case_code = EXCLUDED.case_code,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    data_origin = EXCLUDED.data_origin,
    region = EXCLUDED.region,
    status = EXCLUDED.status,
    metadata = EXCLUDED.metadata,
    updated_at = now();


INSERT INTO scenes (
    scene_id,
    case_id,
    external_scene_id,
    sensor,
    acquisition_time_utc,
    footprint,
    crs,
    data_origin,
    source_uri,
    metadata
)
VALUES (
    '00000000-0000-4000-8000-000000000002',
    '00000000-0000-4000-8000-000000000001',
    'SCENE_DEMO_001',
    'SENTINEL-1-SAR-SYNTHETIC',
    '2026-09-02T12:00:00Z',
    ST_GeomFromText(
        'POLYGON((
            72.74 19.24,
            72.76 19.24,
            72.76 19.26,
            72.74 19.26,
            72.74 19.24
        ))',
        4326
    ),
    'EPSG:4326',
    'SYNTHETIC',
    'data/fixtures/images/synthetic_spill.tif',
    jsonb_build_object(
        'bands',
        jsonb_build_array('VV', 'VH'),
        'contract_version',
        'phase1-to-phase2-v1',
        'demo_only',
        true
    )
)
ON CONFLICT (scene_id)
DO UPDATE SET
    case_id = EXCLUDED.case_id,
    external_scene_id = EXCLUDED.external_scene_id,
    sensor = EXCLUDED.sensor,
    acquisition_time_utc =
        EXCLUDED.acquisition_time_utc,
    footprint = EXCLUDED.footprint,
    crs = EXCLUDED.crs,
    data_origin = EXCLUDED.data_origin,
    source_uri = EXCLUDED.source_uri,
    metadata = EXCLUDED.metadata;


INSERT INTO model_versions (
    model_version_id,
    model_name,
    semantic_version,
    stage,
    architecture,
    framework,
    input_profile,
    preprocessing_version,
    threshold_version,
    threshold_value,
    evaluation_summary
)
VALUES (
    '00000000-0000-4000-8000-000000000003',
    'VARUN U-Net Oil Spill',
    '0.1.0',
    'PRODUCTION',
    'U-Net',
    'PyTorch',
    'sar-2ch-f32',
    'sar-percentile-v1',
    'binary-threshold-v1',
    0.5,
    jsonb_build_object(
        'status',
        'baseline-integration',
        'checkpoint',
        'models/phase1/unet_oil_spill_v0.1.pth',
        'checksum_sha256',
        'f832e1db92a9180c71b912d65b654aede0cd729d9d8626b2069bccb503d05bd3'
    )
)
ON CONFLICT (model_version_id)
DO UPDATE SET
    model_name = EXCLUDED.model_name,
    semantic_version = EXCLUDED.semantic_version,
    stage = EXCLUDED.stage,
    architecture = EXCLUDED.architecture,
    framework = EXCLUDED.framework,
    input_profile = EXCLUDED.input_profile,
    preprocessing_version =
        EXCLUDED.preprocessing_version,
    threshold_version =
        EXCLUDED.threshold_version,
    threshold_value = EXCLUDED.threshold_value,
    evaluation_summary =
        EXCLUDED.evaluation_summary;

COMMIT;
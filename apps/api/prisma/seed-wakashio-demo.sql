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
    '00000000-0000-4000-8000-000000000101',
    'WAKASHIO_20200810',
    'Wakashio Sentinel-1 Observation',
    'Real Sentinel-1 SAR observation over the Wakashio incident region.',
    'REAL',
    ST_MakeEnvelope(
        57.686119,
        -20.432804,
        57.780190,
        -20.321770,
        4326
    ),
    'OPEN',
    jsonb_build_object(
        'contract_version',
        'phase1-to-phase2-v1',
        'event',
        'MV Wakashio oil-spill incident',
        'purpose',
        'real end-to-end integration validation',
        'source',
        'Copernicus Data Space Browser'
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
    source_checksum_sha256,
    metadata
)
VALUES (
    '00000000-0000-4000-8000-000000000102',
    '00000000-0000-4000-8000-000000000101',
    'S1_WAKASHIO_20200810T0137Z',
    'SENTINEL-1-IW-GRD',
    '2020-08-10T01:37:00Z',
    ST_MakeEnvelope(
        57.686119,
        -20.432804,
        57.780190,
        -20.321770,
        4326
    ),
    'EPSG:4326',
    'REAL',
    'data/fixtures/images/wakashio_20200810_vh_vv_db.tif',
    'b4d32a77a7c6a9d1c2833814b3f0f100c13dd000be47061c0b559e14814191ef',
    jsonb_build_object(
        'platform',
        'Sentinel-1',
        'instrument_mode',
        'IW',
        'product_level',
        'GRD',
        'bands',
        jsonb_build_array('VH', 'VV'),
        'polarization_order',
        'VH,VV',
        'calibration',
        'gamma0',
        'scale',
        'decibel',
        'preprocessing',
        'linear_gamma0_to_db',
        'validity_mask',
        'embedded',
        'contract_version',
        'phase1-to-phase2-v1'
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
    source_checksum_sha256 =
        EXCLUDED.source_checksum_sha256,
    metadata = EXCLUDED.metadata;

COMMIT;
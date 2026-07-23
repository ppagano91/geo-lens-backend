-- Fase 7B.2 — Seed idempotente: escena sintética + bandas + AOIs demo
--
-- Prerrequisitos:
--   1. Migraciones aplicadas (alembic upgrade head)
--   2. GeoTIFF generados: python scripts/create_sample_rasters.py
--      (DATA_ROOT=../data → sample/scenes/test_scene/*.tif)
--
-- Uso (desde geo-lens-backend/, ajustar usuario/DB según .env):
--   psql -U postgres -d geolens -f scripts/seed_synthetic_scene.sql
--
-- UUIDs fijos para que la documentación y curls de validación sean estables.

BEGIN;

-- ---------------------------------------------------------------------------
-- Escena sintética (footprint ≈ extent del raster 50×50 @ 0.001°)
-- ---------------------------------------------------------------------------
DELETE FROM raster_scenes
WHERE id = '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4';

INSERT INTO raster_scenes (
    id,
    name,
    source,
    acquisition_date,
    cloud_cover,
    footprint,
    metadata
) VALUES (
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'Synthetic Sentinel-2 CABA Test Scene',
    'local',
    '2025-03-01',
    5,
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.45 -34.55, -58.4 -34.55, -58.4 -34.6, -58.45 -34.6, -58.45 -34.55)))'
        ),
        4326
    ),
    '{"type": "synthetic", "purpose": "local testing", "platform": "Sentinel-2"}'::jsonb
);

-- ---------------------------------------------------------------------------
-- Bandas (asset_path relativos a DATA_ROOT)
-- ---------------------------------------------------------------------------
INSERT INTO raster_bands (
    id,
    scene_id,
    band_key,
    band_name,
    description,
    resolution,
    asset_path,
    nodata,
    dtype,
    metadata
) VALUES
(
    'c26bf357-b449-4ee1-a4de-fb0ac1b33c98',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B02',
    'Blue',
    'Synthetic Sentinel-2 Blue band',
    10,
    'sample/scenes/test_scene/B02.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "blue"}'::jsonb
),
(
    '385974ce-a76e-4e2c-8483-c476845f5f20',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B03',
    'Green',
    'Synthetic Sentinel-2 Green band',
    10,
    'sample/scenes/test_scene/B03.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "green"}'::jsonb
),
(
    '5e056702-8ab7-44cb-ac4e-9f77930f9eba',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B04',
    'Red',
    'Synthetic Sentinel-2 Red band',
    10,
    'sample/scenes/test_scene/B04.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "red"}'::jsonb
),
(
    '634a9599-f776-4391-91c0-9d31d0a3b505',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B08',
    'NIR',
    'Synthetic Sentinel-2 Near Infrared band',
    10,
    'sample/scenes/test_scene/B08.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "nir"}'::jsonb
),
(
    '3e964e3f-4eb4-4643-bd49-39db66cc3718',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B11',
    'SWIR 1',
    'Synthetic Sentinel-2 SWIR 1 band',
    20,
    'sample/scenes/test_scene/B11.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "swir1"}'::jsonb
),
(
    '7ceba148-847d-45a1-bbbe-66efe5ae8661',
    '2f707fd8-c4f5-40da-92aa-6b2e7c0202c4',
    'B12',
    'SWIR 2',
    'Synthetic Sentinel-2 SWIR 2 band',
    20,
    'sample/scenes/test_scene/B12.tif',
    '0',
    'uint16',
    '{"synthetic": true, "wavelength": "swir2"}'::jsonb
);

-- ---------------------------------------------------------------------------
-- AOIs demo (útiles para cobertura espacial / futuros crops; no usados por NDVI)
-- ---------------------------------------------------------------------------
DELETE FROM aois
WHERE id IN (
    '2f98765f-3263-4219-ba60-c16918490798',
    '4ffc8de1-7164-48e0-95ea-09ea1e67caf7',
    '595c3cf8-51c2-4d74-a95c-f2f4973b47f0',
    '924da42c-6f49-454d-9893-99762479b7b5'
);

INSERT INTO aois (id, name, description, geom, properties) VALUES
(
    '2f98765f-3263-4219-ba60-c16918490798',
    'AOI Demo - Synthetic Full Raster',
    'AOI que cubre casi toda la escena sintética.',
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.449 -34.551, -58.401 -34.551, -58.401 -34.599, -58.449 -34.599, -58.449 -34.551)))'
        ),
        4326
    ),
    '{"type": "synthetic", "use_case": "full_scene"}'::jsonb
),
(
    '4ffc8de1-7164-48e0-95ea-09ea1e67caf7',
    'AOI Demo - Synthetic Vegetation',
    'AOI chica para probar NDVI.',
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.445 -34.555, -58.43 -34.555, -58.43 -34.57, -58.445 -34.57, -58.445 -34.555)))'
        ),
        4326
    ),
    '{"type": "synthetic", "use_case": "vegetation"}'::jsonb
),
(
    '595c3cf8-51c2-4d74-a95c-f2f4973b47f0',
    'AOI Demo - Synthetic Water',
    'AOI chica para futuras pruebas de NDWI.',
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.42 -34.575, -58.405 -34.575, -58.405 -34.59, -58.42 -34.59, -58.42 -34.575)))'
        ),
        4326
    ),
    '{"type": "synthetic", "use_case": "water"}'::jsonb
),
(
    '924da42c-6f49-454d-9893-99762479b7b5',
    'AOI Demo - Synthetic Dry Burn',
    'AOI chica para futuras pruebas de NBR / NDMI.',
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.44 -34.58, -58.425 -34.58, -58.425 -34.595, -58.44 -34.595, -58.44 -34.58)))'
        ),
        4326
    ),
    '{"type": "synthetic", "use_case": "dry_burn"}'::jsonb
);

COMMIT;

-- IDs de referencia (documentación / curls):
--   scene_id : 2f707fd8-c4f5-40da-92aa-6b2e7c0202c4
--   B04      : 5e056702-8ab7-44cb-ac4e-9f77930f9eba
--   B08      : 634a9599-f776-4391-91c0-9d31d0a3b505

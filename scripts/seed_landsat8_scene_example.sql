-- Fase 8B.1 — Seed idempotente: escena Landsat 8 real (recorte QGIS) + bandas SR_B*
--
-- Prerrequisitos:
--   1. Migraciones aplicadas (alembic upgrade head)
--   2. GeoTIFF bajo DATA_ROOT (ver docs/fase-8b1-resultados.md):
--        sample/scenes/landsat8_lc08_225084/SR_B2.tif … SR_B7.tif
--      Origen tipico: data/temp/band_stack/B2.tif…B7.tif (recorte CABA),
--      copiados con nombres nativos Landsat (NO renombrar a B0x Sentinel).
--
-- Uso (desde geo-lens-backend/, ajustar usuario/DB segun .env):
--   psql -U postgres -d geolens -f scripts/seed_landsat8_scene_example.sql
--
-- UUIDs fijos para documentacion y curls estables.

BEGIN;

-- ---------------------------------------------------------------------------
-- Escena Landsat 8 (footprint ≈ extent WGS84 del recorte 148×179 @ 30 m)
-- Producto: LC08_L2SP_225084_20260510_20260515_02_T1 (Collection 2 L2SP)
-- ---------------------------------------------------------------------------
DELETE FROM raster_scenes
WHERE id = '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47';

INSERT INTO raster_scenes (
    id,
    name,
    source,
    acquisition_date,
    cloud_cover,
    footprint,
    metadata
) VALUES (
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'Landsat 8 LC08 225/084 CABA crop 2026-05-10',
    'landsat-8',
    '2026-05-10',
    1.77,
    ST_SetSRID(
        ST_GeomFromText(
            'MULTIPOLYGON(((-58.4497 -34.5505, -58.4005 -34.5505, -58.4005 -34.5994, -58.4497 -34.5994, -58.4497 -34.5505)))'
        ),
        4326
    ),
    '{
        "platform": "Landsat-8",
        "sensor": "landsat-8",
        "product_id": "LC08_L2SP_225084_20260510_20260515_02_T1",
        "collection": "02",
        "processing_level": "L2SP",
        "wrs_path": 225,
        "wrs_row": 84,
        "crs": "EPSG:32621",
        "purpose": "fase_8b1_e2e",
        "notes": "QGIS crop from EarthExplorer SR bands; native SR_B* keys"
    }'::jsonb
);

-- ---------------------------------------------------------------------------
-- Bandas (asset_path relativos a DATA_ROOT) — keys nativas Landsat 8
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
    'a02b0002-e8f1-4c21-9d02-00000000b002',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B2',
    'Blue',
    'Landsat 8 OLI Surface Reflectance Blue (band 2)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B2.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 2, "wavelength": "blue"}'::jsonb
),
(
    'a02b0003-e8f1-4c21-9d03-00000000b003',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B3',
    'Green',
    'Landsat 8 OLI Surface Reflectance Green (band 3)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B3.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 3, "wavelength": "green"}'::jsonb
),
(
    'a02b0004-e8f1-4c21-9d04-00000000b004',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B4',
    'Red',
    'Landsat 8 OLI Surface Reflectance Red (band 4)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B4.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 4, "wavelength": "red"}'::jsonb
),
(
    'a02b0005-e8f1-4c21-9d05-00000000b005',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B5',
    'NIR',
    'Landsat 8 OLI Surface Reflectance NIR (band 5)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B5.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 5, "wavelength": "nir"}'::jsonb
),
(
    'a02b0006-e8f1-4c21-9d06-00000000b006',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B6',
    'SWIR1',
    'Landsat 8 OLI Surface Reflectance SWIR1 (band 6)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B6.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 6, "wavelength": "swir1"}'::jsonb
),
(
    'a02b0007-e8f1-4c21-9d07-00000000b007',
    '8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47',
    'SR_B7',
    'SWIR2',
    'Landsat 8 OLI Surface Reflectance SWIR2 (band 7)',
    30,
    'sample/scenes/landsat8_lc08_225084/SR_B7.tif',
    '0',
    'uint16',
    '{"platform": "Landsat-8", "oli_band": 7, "wavelength": "swir2"}'::jsonb
);

COMMIT;

-- IDs de referencia (documentacion / curls):
--   scene_id : 8c1a4e2f-7b3d-4a91-9e55-1f0d6c8a2b47
--   SR_B4    : a02b0004-e8f1-4c21-9d04-00000000b004  (Red)
--   SR_B5    : a02b0005-e8f1-4c21-9d05-00000000b005  (NIR)

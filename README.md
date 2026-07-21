# GeoChange Analyzer — Backend

API con FastAPI para GeoChange Analyzer.

## Requisitos

- Python 3.11+
- PostgreSQL 16+ con extensión PostGIS instalados localmente en Windows

## Instalación

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

## Base de datos (PostgreSQL + PostGIS local)

### 1. Verificar PostgreSQL

```powershell
pg_isready -h localhost -p 5432
```

### 2. Crear base `geochange` (si no existe)

```powershell
psql -U postgres -c "CREATE USER geochange WITH PASSWORD 'geochange';"
psql -U postgres -c "CREATE DATABASE geochange OWNER geochange;"
```

Omita los comandos que fallen porque el usuario o la base ya existen.

### 3. Habilitar PostGIS

```powershell
psql -U postgres -d geochange -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

### 4. Configurar `.env`

Ajuste `DATABASE_URL` en `backend/.env` según sus credenciales locales:

```env
DATABASE_URL=postgresql+psycopg://geochange:geochange@localhost:5432/geochange
APP_ENV=local
CORS_ORIGINS=http://localhost:5173
DATA_ROOT=../data
```

### 5. Aplicar migraciones

```powershell
cd backend
.venv\Scripts\activate
alembic upgrade head
```

Ver detalle en [docs/aoi_persistence.md](../docs/aoi_persistence.md).

### Alternativa: Docker

Desde la raíz del proyecto, si prefiere un contenedor en lugar de PostgreSQL local:

```powershell
docker compose up -d
```

Use las mismas credenciales en `DATABASE_URL` (`geochange:geochange@localhost:5432/geochange`).

## Ejecutar

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints principales:

- `GET /api/v1/health` — health check
- `POST /api/v1/aois` — crear AOI
- `GET /api/v1/aois` — listar AOIs
- `GET /api/v1/aois/{aoi_id}` — obtener AOI
- `DELETE /api/v1/aois/{aoi_id}` — eliminar AOI
- `POST /api/v1/scenes` — crear escena con bandas (metadata)
- `GET /api/v1/scenes` — listar escenas
- `GET /api/v1/scenes/{scene_id}` — detalle de escena con bandas
- `GET /api/v1/scenes/{scene_id}/bands` — listar bandas de escena
- `DELETE /api/v1/scenes/{scene_id}` — eliminar escena
- `GET /api/v1/indices` — listar definiciones de índices espectrales
- `GET /api/v1/indices/{index_key}` — detalle de índice por key (`ndvi`, `NDVI`, …)
- `GET /api/v1/spatial-coverage/aoi/{aoi_id}/scene/{scene_id}` — cobertura espacial AOI vs footprint
- `GET /api/v1/raster-bands/{band_id}/metadata` — metadata del GeoTIFF local de la banda
- `GET /api/v1/raster-bands/{band_id}/sample-stats` — estadísticas de muestra reducida (banda 1)

Ver [docs/scenes_metadata.md](../docs/scenes_metadata.md) para escenas satelitales.
Ver [docs/spectral_indices.md](../docs/spectral_indices.md) para el catálogo de índices (solo definiciones, sin cálculo).
Ver [docs/raster_formulas.md](../docs/raster_formulas.md) para fórmulas NumPy puras (Fase 6B; sin lectura de GeoTIFF ni endpoints de cálculo).
Ver [docs/spatial_coverage.md](../docs/spatial_coverage.md) para cobertura espacial AOI vs escena (Fase 6C; PostGIS, sin raster).
Ver [docs/raster_reading.md](../docs/raster_reading.md) para lectura local de GeoTIFF (Fase 7A; metadata / sample-stats).
Ver [docs/local_sample_rasters.md](../docs/local_sample_rasters.md) para generar GeoTIFF de prueba en `data/` (Fase 7A.1).

Documentación interactiva: `http://localhost:8000/docs`

## Tests

```powershell
pytest
```

Los tests de integración de AOIs, escenas e índices requieren PostgreSQL levantado y migraciones aplicadas.
Los tests de lectura raster generan un GeoTIFF temporal en runtime (no dependen de archivos en `data/`).

## Variables de entorno

| Variable | Descripción |
|---|---|
| `DATABASE_URL` | Conexión SQLAlchemy (`postgresql+psycopg://usuario:password@localhost:5432/geochange`) |
| `APP_ENV` | Entorno de ejecución (`local`) |
| `CORS_ORIGINS` | Orígenes permitidos para CORS (ej. `http://localhost:5173`) |
| `DATA_ROOT` | Raíz para resolver `asset_path` relativos (default `../data`) |

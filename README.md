# GeoLens — Backend

API FastAPI de GeoLens v0.1. Arranque completo, `DATA_ROOT` y flujo demo:
[README raíz](../README.md).

## Requisitos

- Python 3.11+
- PostgreSQL 16+ con PostGIS

## Instalación

```powershell
cd geo-lens-backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

Ajustá `DATABASE_URL`, `DATA_ROOT` y (opcional) `MAPTILER_API_KEY` en `.env`.

## Migraciones

```powershell
alembic upgrade head
```

## Ejecutar

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI: `http://localhost:8000/docs` · Health: `GET /api/v1/health`.

## Tests

```powershell
pytest
```

Los tests de integración requieren PostgreSQL + migraciones. Los de raster
generan GeoTIFF temporales (no usan escenas reales en `data/`).

## Variables de entorno

| Variable | Descripción |
|---|---|
| `DATABASE_URL` | SQLAlchemy (`postgresql+psycopg://…`) |
| `APP_ENV` | Entorno (`local`) |
| `CORS_ORIGINS` | Orígenes CORS (`http://localhost:5173`) |
| `DATA_ROOT` | Raíz de `asset_path` (default `../data`) |
| `MAPTILER_API_KEY` | Opcional; habilita Terrain RGB en `/map-providers/config` |

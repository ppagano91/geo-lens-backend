# GeoChange Analyzer — Backend

API con FastAPI para GeoChange Analyzer.

## Requisitos

- Python 3.11+
- Docker (para PostgreSQL + PostGIS en desarrollo local)

## Instalación

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

## Base de datos (PostgreSQL + PostGIS)

Desde la raíz del proyecto:

```powershell
docker compose up -d
```

Aplicar migraciones:

```powershell
cd backend
.venv\Scripts\activate
alembic upgrade head
```

Ver detalle en [docs/aoi_persistence.md](../docs/aoi_persistence.md).

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

Documentación interactiva: `http://localhost:8000/docs`

## Tests

```powershell
pytest
```

Los tests de integración de AOIs requieren PostgreSQL levantado y migraciones aplicadas.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `DATABASE_URL` | Conexión SQLAlchemy (`postgresql+psycopg://...`) |
| `APP_ENV` | Entorno de ejecución (`local`) |

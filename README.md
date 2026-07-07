# GeoChange Analyzer — Backend

API mínima con FastAPI para GeoChange Analyzer.

## Requisitos

- Python 3.11+

## Instalación

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
```

## Ejecutar

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La documentación interactiva estará en `http://localhost:8000/docs`.

## Tests

```bash
pytest
```

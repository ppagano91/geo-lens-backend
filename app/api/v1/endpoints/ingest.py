"""Local scene ingest endpoints (Fase 9A)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ingest import LocalSceneIngestRequest, LocalSceneIngestResult
from app.services.local_scene_ingest_service import (
    LocalIngestError,
    LocalSceneIngestService,
)
from app.services.scene_service import (
    BandKeyDuplicateError,
    GeometryValidationError,
)

router = APIRouter()


@router.post(
    "/local-scene",
    response_model=LocalSceneIngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a local GeoTIFF scene folder",
)
def ingest_local_scene(
    payload: LocalSceneIngestRequest,
    db: Session = Depends(get_db),
) -> LocalSceneIngestResult:
    """Register ``raster_scenes`` + ``raster_bands`` from a folder under DATA_ROOT.

    Initial support: Landsat 8 Collection 2 L2 Surface Reflectance
    (``SR_B2``…``SR_B7``). Optional ``MTL.txt`` enriches metadata.
    """
    service = LocalSceneIngestService(db)
    try:
        return service.ingest(payload)
    except LocalIngestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except GeometryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except BandKeyDuplicateError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

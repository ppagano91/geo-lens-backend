"""Local scene ingest endpoints (Fase 9A / 9D / 9K)."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ingest import LocalSceneIngestRequest, LocalSceneIngestResult
from app.services.local_scene_ingest_service import (
    LocalIngestError,
    LocalSceneIngestService,
    UploadedSceneFile,
)
from app.services.scene_service import (
    BandKeyDuplicateError,
    GeometryValidationError,
)

router = APIRouter()


def _map_ingest_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, LocalIngestError):
        return HTTPException(status_code=exc.status_code, detail=exc.message)
    if isinstance(exc, GeometryValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    if isinstance(exc, BandKeyDuplicateError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    raise exc


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

    Supported sensors:

    * Landsat 8 Collection 2 L2 Surface Reflectance (``SR_B2``…``SR_B7``);
      optional ``MTL.txt`` enriches metadata.
    * Sentinel-2 L2A / simplified local set at 10 m (``B02``, ``B03``, ``B04``,
      ``B08``). Optional ``B11``/``B12`` only if already aligned to the 10 m grid.

    Dev/admin mode: folder must already exist under storage.
    """
    service = LocalSceneIngestService(db)
    try:
        return service.ingest(payload)
    except (LocalIngestError, GeometryValidationError, BandKeyDuplicateError) as exc:
        raise _map_ingest_errors(exc) from exc


@router.post(
    "/upload-scene",
    response_model=LocalSceneIngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Landsat 8 or Sentinel-2 band files and register a scene",
)
async def upload_scene(
    files: list[UploadFile] = File(
        ...,
        description=(
            "GeoTIFF bands (.tif/.tiff); optional MTL (.txt) for Landsat 8"
        ),
    ),
    source: str = Form(
        ...,
        description="Sensor/source hint: landsat-8 or sentinel-2",
        examples=["landsat-8", "sentinel-2"],
    ),
    name: Optional[str] = Form(
        default=None,
        description="Optional scene display name",
    ),
    overwrite: bool = Form(
        default=False,
        description="If true, replace an existing scene from the same storage path",
    ),
    db: Session = Depends(get_db),
) -> LocalSceneIngestResult:
    """Accept multipart uploads, store under ``uploaded/scenes/{uuid}/``, then ingest.

    Same validation and response shape as ``POST /ingest/local-scene``.
    Does not require the user to know DATA_ROOT.
    """
    uploaded: list[UploadedSceneFile] = []
    try:
        for upload in files:
            content = await upload.read()
            uploaded.append(
                UploadedSceneFile(
                    filename=upload.filename or "",
                    content=content,
                )
            )
    finally:
        for upload in files:
            await upload.close()

    display_name = name.strip() if name and name.strip() else None
    service = LocalSceneIngestService(db)
    try:
        return service.ingest_upload(
            files=uploaded,
            source=source,
            name=display_name,
            overwrite=overwrite,
        )
    except (LocalIngestError, GeometryValidationError, BandKeyDuplicateError) as exc:
        raise _map_ingest_errors(exc) from exc

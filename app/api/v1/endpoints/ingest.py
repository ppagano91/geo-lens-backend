"""Local scene ingest endpoints (Fase 9A / 9D / 9K / 9M.1)."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ingest import LocalSceneIngestRequest, LocalSceneIngestResult
from app.schemas.radiometry import IngestProductLevel
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
      ``B08``). Optional ``B11``/``B12`` (typically 20 m) are resampled/aligned
      onto the 10 m grid when needed (Fase 9L). Optional SAFE metadata /
      ``product_level`` / ``source_product_id`` refine radiometry (Fase 9M.1).

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
            "GeoTIFF bands (.tif/.tiff); optional MTL (.txt) for Landsat 8; "
            "optional SAFE metadata (.xml/.safe) for Sentinel-2"
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
    product_level: Optional[IngestProductLevel] = Form(
        default=None,
        description=(
            "Optional radiometry override: sentinel_l1c / sentinel_l2a / "
            "landsat_l2 / unknown"
        ),
    ),
    source_product_id: Optional[str] = Form(
        default=None,
        description=(
            "Optional original product id (e.g. S2B_MSIL1C_..._T20JLL_...)"
        ),
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
    product_id = (
        source_product_id.strip()
        if source_product_id and source_product_id.strip()
        else None
    )

    # Mirror LocalSceneIngestRequest validation for multipart forms.
    source_norm = (source or "").strip().lower().replace("_", "-")
    if product_level is not None:
        if source_norm in {"sentinel-2", "sentinel2", "s2"} and product_level not in {
            "sentinel_l1c",
            "sentinel_l2a",
            "unknown",
        }:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "For source=sentinel-2, product_level must be "
                    "sentinel_l1c, sentinel_l2a, or unknown"
                ),
            )
        if source_norm in {"landsat-8", "landsat8", "l8"} and product_level not in {
            "landsat_l2",
            "unknown",
        }:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "For source=landsat-8, product_level must be landsat_l2 or unknown"
                ),
            )

    service = LocalSceneIngestService(db)
    try:
        return service.ingest_upload(
            files=uploaded,
            source=source,
            name=display_name,
            overwrite=overwrite,
            product_level=product_level,
            source_product_id=product_id,
        )
    except (LocalIngestError, GeometryValidationError, BandKeyDuplicateError) as exc:
        raise _map_ingest_errors(exc) from exc

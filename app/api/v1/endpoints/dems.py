"""Experimental DEM / hillshade endpoints (v0.1-P5)."""

from __future__ import annotations

from typing import NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.dem import DemAssetRead, DemHillshadeResult, DemMapOverlayResult
from app.services.dem_service import DemError, DemService

router = APIRouter()


def _raise_dem_http(exc: Exception) -> NoReturn:
    if isinstance(exc, DemError):
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    raise exc


@router.post(
    "/upload",
    response_model=DemAssetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a single-band DEM GeoTIFF",
)
async def upload_dem(
    file: UploadFile = File(..., description="Single-band DEM GeoTIFF (.tif/.tiff)"),
    name: Optional[str] = Form(
        default=None,
        description="Optional display name",
    ),
    db: Session = Depends(get_db),
) -> DemAssetRead:
    try:
        content = await file.read()
    finally:
        await file.close()

    service = DemService(db)
    try:
        return service.upload(
            content=content,
            filename=file.filename or "",
            name=name,
        )
    except DemError as exc:
        _raise_dem_http(exc)


@router.get("", response_model=list[DemAssetRead], summary="List uploaded DEMs")
def list_dems(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[DemAssetRead]:
    return DemService(db).list(limit=limit, offset=offset)


@router.get("/{dem_id}", response_model=DemAssetRead, summary="DEM detail")
def get_dem(dem_id: UUID, db: Session = Depends(get_db)) -> DemAssetRead:
    try:
        return DemService(db).get(dem_id)
    except DemError as exc:
        _raise_dem_http(exc)


@router.post(
    "/{dem_id}/hillshade",
    response_model=DemHillshadeResult,
    summary="Generate a hillshade PNG from the DEM",
)
def generate_dem_hillshade(
    dem_id: UUID,
    db: Session = Depends(get_db),
) -> DemHillshadeResult:
    try:
        return DemService(db).generate_hillshade(dem_id)
    except DemError as exc:
        _raise_dem_http(exc)


@router.get(
    "/{dem_id}/hillshade.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Hillshade PNG (inline)",
        },
        404: {"description": "DEM or hillshade PNG missing"},
    },
    summary="Serve the hillshade PNG",
)
def get_dem_hillshade_png(
    dem_id: UUID,
    db: Session = Depends(get_db),
) -> FileResponse:
    try:
        path = DemService(db).resolve_hillshade_png(dem_id)
    except DemError as exc:
        _raise_dem_http(exc)

    return FileResponse(
        path,
        media_type="image/png",
        filename="hillshade.png",
        content_disposition_type="inline",
    )


@router.get(
    "/{dem_id}/map-overlay",
    response_model=DemMapOverlayResult,
    summary="MapLibre image-overlay metadata for the hillshade",
)
def get_dem_map_overlay(
    dem_id: UUID,
    db: Session = Depends(get_db),
) -> DemMapOverlayResult:
    try:
        return DemService(db).get_map_overlay(dem_id)
    except DemError as exc:
        _raise_dem_http(exc)

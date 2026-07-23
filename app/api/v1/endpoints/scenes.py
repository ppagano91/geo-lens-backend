from typing import NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.band import BandRead
from app.schemas.index_compute import IndexComputeResult, NdviComputeResult
from app.schemas.scene import SceneCreate, SceneListItem, SceneRead
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    LocalIndexComputeService,
    MissingRequiredBandError,
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    UnsupportedIndexError,
)
from app.services.scene_service import (
    BandKeyDuplicateError,
    GeometryValidationError,
    SceneNotFoundError,
    SceneService,
)

router = APIRouter()


def _raise_index_compute_http(exc: Exception, *, scene_id: UUID) -> NoReturn:
    """Map local index compute domain errors to HTTP responses."""
    if isinstance(exc, SceneNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc
    if isinstance(exc, UnsupportedIndexError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, MissingRequiredBandError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, IncompatibleRasterBandsError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, RasterFileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (RasterPathError, RasterReadError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    raise exc


@router.post("", response_model=SceneRead, status_code=status.HTTP_201_CREATED)
def create_scene(payload: SceneCreate, db: Session = Depends(get_db)) -> SceneRead:
    service = SceneService(db)
    try:
        return service.create(payload)
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


@router.get("", response_model=list[SceneListItem])
def list_scenes(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[SceneListItem]:
    service = SceneService(db)
    return service.list(limit=limit, offset=offset)


@router.get("/{scene_id}", response_model=SceneRead)
def get_scene(scene_id: UUID, db: Session = Depends(get_db)) -> SceneRead:
    service = SceneService(db)
    try:
        return service.get(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc


@router.get("/{scene_id}/bands", response_model=list[BandRead])
def list_scene_bands(scene_id: UUID, db: Session = Depends(get_db)) -> list[BandRead]:
    service = SceneService(db)
    try:
        return service.list_bands(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc


@router.post(
    "/{scene_id}/indices/ndvi/compute",
    response_model=NdviComputeResult,
    include_in_schema=True,
)
def compute_scene_ndvi(
    scene_id: UUID,
    db: Session = Depends(get_db),
) -> NdviComputeResult:
    """Compatibility alias for NDVI local compute (Fase 7B)."""
    service = LocalIndexComputeService(db)
    try:
        return service.compute_ndvi(scene_id)
    except (
        SceneNotFoundError,
        UnsupportedIndexError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/indices/{index_key}/compute",
    response_model=IndexComputeResult,
)
def compute_scene_index(
    scene_id: UUID,
    index_key: str,
    db: Session = Depends(get_db),
) -> IndexComputeResult:
    """Compute a supported spectral index in-memory from local scene GeoTIFFs."""
    service = LocalIndexComputeService(db)
    try:
        return service.compute_index(scene_id, index_key)
    except (
        SceneNotFoundError,
        UnsupportedIndexError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scene(scene_id: UUID, db: Session = Depends(get_db)) -> None:
    service = SceneService(db)
    try:
        service.delete(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc

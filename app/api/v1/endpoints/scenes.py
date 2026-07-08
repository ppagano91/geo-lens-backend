from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.band import BandRead
from app.schemas.scene import SceneCreate, SceneListItem, SceneRead
from app.services.scene_service import (
    BandKeyDuplicateError,
    GeometryValidationError,
    SceneNotFoundError,
    SceneService,
)

router = APIRouter()


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

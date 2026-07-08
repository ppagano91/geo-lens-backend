from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.aoi import AoiCreate, AoiRead
from app.services.aoi_service import AoiNotFoundError, AoiService, GeometryValidationError

router = APIRouter()


@router.post("", response_model=AoiRead, status_code=status.HTTP_201_CREATED)
def create_aoi(payload: AoiCreate, db: Session = Depends(get_db)) -> AoiRead:
    service = AoiService(db)
    try:
        return service.create(payload)
    except GeometryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[AoiRead])
def list_aois(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[AoiRead]:
    service = AoiService(db)
    return service.list(limit=limit, offset=offset)


@router.get("/{aoi_id}", response_model=AoiRead)
def get_aoi(aoi_id: UUID, db: Session = Depends(get_db)) -> AoiRead:
    service = AoiService(db)
    try:
        return service.get(aoi_id)
    except AoiNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI {aoi_id} not found",
        ) from exc


@router.delete("/{aoi_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_aoi(aoi_id: UUID, db: Session = Depends(get_db)) -> None:
    service = AoiService(db)
    try:
        service.delete(aoi_id)
    except AoiNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI {aoi_id} not found",
        ) from exc

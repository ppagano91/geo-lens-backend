"""Derived-asset catalog endpoints (Fase 9I)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.derived_asset import DerivedAssetRead
from app.services.derived_asset_service import (
    DerivedAssetNotFoundError,
    DerivedAssetService,
)
from app.services.scene_service import SceneNotFoundError

router = APIRouter()


@router.get("/{asset_id}", response_model=DerivedAssetRead)
def get_derived_asset(
    asset_id: UUID,
    db: Session = Depends(get_db),
) -> DerivedAssetRead:
    service = DerivedAssetService(db)
    try:
        return service.get(asset_id)
    except DerivedAssetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived asset {asset_id} not found",
        ) from exc


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_derived_asset(
    asset_id: UUID,
    db: Session = Depends(get_db),
) -> None:
    """Logically deactivate a catalog row. Does not delete files from DATA_ROOT."""
    service = DerivedAssetService(db)
    try:
        service.soft_delete(asset_id)
    except DerivedAssetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived asset {asset_id} not found",
        ) from exc


__all__ = ["router"]

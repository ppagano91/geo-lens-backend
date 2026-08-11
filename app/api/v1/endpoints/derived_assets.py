"""Derived-asset catalog endpoints (Fase 9I / 9J)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.derived_asset import DerivedAssetExistsResult, DerivedAssetRead
from app.services.derived_asset_service import (
    DerivedAssetConflictError,
    DerivedAssetNotFoundError,
    DerivedAssetService,
)

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


@router.get("/{asset_id}/exists", response_model=DerivedAssetExistsResult)
def check_derived_asset_exists(
    asset_id: UUID,
    db: Session = Depends(get_db),
) -> DerivedAssetExistsResult:
    """Check whether catalog path references exist under DATA_ROOT.

    Does not create, move, or delete files. Works for active and inactive rows.
    """
    service = DerivedAssetService(db)
    try:
        return service.check_exists(asset_id)
    except DerivedAssetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived asset {asset_id} not found",
        ) from exc


@router.patch("/{asset_id}/restore", response_model=DerivedAssetRead)
def restore_derived_asset(
    asset_id: UUID,
    db: Session = Depends(get_db),
) -> DerivedAssetRead:
    """Reactivate a soft-deleted catalog row. Does not touch physical files."""
    service = DerivedAssetService(db)
    try:
        return service.restore(asset_id)
    except DerivedAssetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived asset {asset_id} not found",
        ) from exc
    except DerivedAssetConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
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

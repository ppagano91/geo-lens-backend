from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.spectral_index import SpectralIndexDefinitionListItem, SpectralIndexDefinitionRead
from app.services.spectral_index_service import SpectralIndexNotFoundError, SpectralIndexService

router = APIRouter()


@router.get("", response_model=list[SpectralIndexDefinitionListItem])
def list_indices(
    category: Optional[str] = Query(default=None),
    is_active: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
) -> list[SpectralIndexDefinitionListItem]:
    service = SpectralIndexService(db)
    return service.list_indices(category=category, is_active=is_active)


@router.get("/{index_key}", response_model=SpectralIndexDefinitionRead)
def get_index(index_key: str, db: Session = Depends(get_db)) -> SpectralIndexDefinitionRead:
    service = SpectralIndexService(db)
    try:
        return service.get_index_by_key(index_key)
    except SpectralIndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Spectral index '{exc.args[0]}' not found",
        ) from exc

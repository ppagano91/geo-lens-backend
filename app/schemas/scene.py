from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.band import BandCreate, BandRead


class SceneCreate(BaseModel):
    # Optional pre-allocated id (e.g. Fase 9L aligned assets written before insert).
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    source: str = Field(..., min_length=1, max_length=100)
    acquisition_date: date
    cloud_cover: Optional[Decimal] = None
    footprint: dict[str, Any]
    metadata: Optional[dict[str, Any]] = None
    bands: list[BandCreate] = Field(default_factory=list)


class SceneListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    source: str
    acquisition_date: date
    cloud_cover: Optional[Decimal]
    footprint: dict[str, Any]
    metadata: Optional[dict[str, Any]]
    is_active: bool = True
    deleted_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class SceneRead(SceneListItem):
    bands: list[BandRead]

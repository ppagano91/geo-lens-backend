from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BandCreate(BaseModel):
    band_key: str = Field(..., min_length=1, max_length=20)
    band_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=512)
    resolution: Optional[Decimal] = None
    asset_path: str = Field(..., min_length=1)
    nodata: Optional[str] = Field(default=None, max_length=50)
    dtype: Optional[str] = Field(default=None, max_length=50)
    metadata: Optional[dict[str, Any]] = None


class BandRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    scene_id: UUID
    band_key: str
    band_name: str
    description: Optional[str]
    resolution: Optional[Decimal]
    asset_path: str
    nodata: Optional[str]
    dtype: Optional[str]
    metadata: Optional[dict[str, Any]]
    created_at: datetime

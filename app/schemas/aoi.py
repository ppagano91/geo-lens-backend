from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AoiCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1024)
    geometry: dict[str, Any]
    properties: Optional[dict[str, Any]] = None


class AoiRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: Optional[str]
    geometry: dict[str, Any]
    properties: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


AoiListItem = AoiRead

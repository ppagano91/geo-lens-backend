from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SpectralIndexDefinitionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    key: str
    name: str
    description: str
    formula: str
    required_bands: dict[str, Any]
    category: str
    output_range: Optional[dict[str, Any]] = None
    interpretation: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SpectralIndexDefinitionListItem(SpectralIndexDefinitionRead):
    pass

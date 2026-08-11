"""Pydantic schemas for the derived-asset catalog (Fase 9I)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

DerivedAssetType = Literal[
    "index",
    "index_aoi_crop",
    "rgb_composite",
    "rgb_composite_aoi",
]


class DerivedAssetRead(BaseModel):
    """Catalog entry for a product generated under DATA_ROOT."""

    id: UUID
    scene_id: UUID
    aoi_id: Optional[UUID] = None
    asset_type: str
    product_key: str
    asset_path: str = Field(description="Path relative to DATA_ROOT (primary product)")
    preview_path: Optional[str] = Field(
        default=None,
        description="Optional PNG path relative to DATA_ROOT",
    )
    georef_path: Optional[str] = Field(
        default=None,
        description="Optional georef sidecar path relative to DATA_ROOT",
    )
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    nodata: Optional[str] = None
    dtype: Optional[str] = None
    stats: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    is_active: bool = True
    deleted_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class DerivedAssetExistsResult(BaseModel):
    """Physical presence of catalog path references under DATA_ROOT."""

    asset_id: UUID
    asset_exists: bool = Field(description="Primary product file exists")
    preview_exists: bool = Field(
        description="Preview file exists (False if no preview_path)",
    )
    georef_exists: bool = Field(
        description="Georef sidecar exists (False if no georef_path)",
    )
    missing_paths: list[str] = Field(
        default_factory=list,
        description="Relative paths registered in the catalog but missing on disk",
    )


__all__ = [
    "DerivedAssetExistsResult",
    "DerivedAssetRead",
    "DerivedAssetType",
]

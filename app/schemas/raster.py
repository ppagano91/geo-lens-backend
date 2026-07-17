"""Pydantic schemas for local raster metadata and sample stats."""

from __future__ import annotations

from typing import List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field


class RasterBounds(BaseModel):
    left: float
    bottom: float
    right: float
    top: float


class RasterMetadataRead(BaseModel):
    band_id: UUID
    asset_path: str
    resolved_path: str
    exists: bool
    driver: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    count: Optional[int] = None
    dtypes: Optional[List[str]] = None
    crs: Optional[str] = None
    bounds: Optional[RasterBounds] = None
    nodata: Optional[float] = None
    resolution: Optional[Tuple[float, float]] = None
    transform: Optional[List[float]] = None
    indexes: Optional[List[int]] = None
    is_readable: bool


class RasterSampleStatsRead(BaseModel):
    band_id: UUID
    asset_path: str
    resolved_path: str
    sample_shape: Tuple[int, int]
    min: Optional[float] = Field(default=None, description="Minimum of valid sample values")
    max: Optional[float] = Field(default=None, description="Maximum of valid sample values")
    mean: Optional[float] = Field(default=None, description="Mean of valid sample values")
    valid_count: int
    sample_has_nan: bool = False

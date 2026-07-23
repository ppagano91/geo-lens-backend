"""Pydantic schemas for local spectral index compute responses (Fase 7B/7C)."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IndexBandUsed(BaseModel):
    band_key: str
    band_id: UUID


class IndexBandsUsed(BaseModel):
    """NDVI-shaped bands_used (Fase 7B). Prefer role→band maps on IndexComputeResult."""

    red: IndexBandUsed
    nir: IndexBandUsed


class IndexRasterInfo(BaseModel):
    width: int
    height: int
    crs: Optional[str] = None
    dtype: str = "float32"


class IndexStats(BaseModel):
    min: Optional[float] = Field(default=None, description="Minimum of valid index pixels")
    max: Optional[float] = Field(default=None, description="Maximum of valid index pixels")
    mean: Optional[float] = Field(default=None, description="Mean of valid index pixels")
    valid_pixels: int
    nodata_pixels: int


class IndexComputeResult(BaseModel):
    """In-memory local index compute summary (no GeoTIFF write-back)."""

    scene_id: UUID
    index: str
    status: str = "computed"
    bands_used: dict[str, IndexBandUsed]
    raster: IndexRasterInfo
    stats: IndexStats


# Backward-compatible alias for Fase 7B callers / OpenAPI.
NdviComputeResult = IndexComputeResult

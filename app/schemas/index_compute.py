"""Pydantic schemas for local spectral index compute responses (Fase 7B)."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IndexBandUsed(BaseModel):
    band_key: str
    band_id: UUID


class IndexBandsUsed(BaseModel):
    red: IndexBandUsed
    nir: IndexBandUsed


class IndexRasterInfo(BaseModel):
    width: int
    height: int
    crs: Optional[str] = None
    dtype: str = "float32"


class IndexStats(BaseModel):
    min: Optional[float] = Field(default=None, description="Minimum of valid NDVI pixels")
    max: Optional[float] = Field(default=None, description="Maximum of valid NDVI pixels")
    mean: Optional[float] = Field(default=None, description="Mean of valid NDVI pixels")
    valid_pixels: int
    nodata_pixels: int


class NdviComputeResult(BaseModel):
    scene_id: UUID
    index: str = "NDVI"
    status: str = "computed"
    bands_used: IndexBandsUsed
    raster: IndexRasterInfo
    stats: IndexStats

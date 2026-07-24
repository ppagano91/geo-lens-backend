"""Pydantic schemas for local spectral index compute / preview responses."""

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


class IndexOutputInfo(BaseModel):
    """Derived GeoTIFF written under DATA_ROOT (Fase 7D)."""

    asset_path: str = Field(description="Path relative to DATA_ROOT")
    resolved_path: str = Field(description="Absolute filesystem path of the written file")
    nodata: float = Field(description="Nodata sentinel written to the GeoTIFF")


class IndexComputeSaveResult(BaseModel):
    """Local index compute with GeoTIFF persistence."""

    scene_id: UUID
    index: str
    status: str = "saved"
    bands_used: dict[str, IndexBandUsed]
    raster: IndexRasterInfo
    stats: IndexStats
    output: IndexOutputInfo


class IndexPreviewInputInfo(BaseModel):
    """Derived GeoTIFF used as preview input (Fase 7E)."""

    asset_path: str = Field(description="Path relative to DATA_ROOT")


class IndexPreviewOutputInfo(BaseModel):
    """PNG preview written under DATA_ROOT (Fase 7E)."""

    asset_path: str = Field(description="Path relative to DATA_ROOT")
    resolved_path: str = Field(description="Absolute filesystem path of the written PNG")


class IndexPreviewResult(BaseModel):
    """PNG preview generated from an existing derived index GeoTIFF."""

    scene_id: UUID
    index: str
    status: str = "preview_created"
    input: IndexPreviewInputInfo
    output: IndexPreviewOutputInfo
    width: int
    height: int


# Backward-compatible alias for Fase 7B callers / OpenAPI.
NdviComputeResult = IndexComputeResult

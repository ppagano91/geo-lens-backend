"""Pydantic schemas for local spectral index compute / preview responses."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.radiometry import RadiometryInfo


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
    radiometry: Optional[RadiometryInfo] = None


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
    radiometry: Optional[RadiometryInfo] = None


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


class IndexMapOverlayBounds(BaseModel):
    """Raster bounds in the original CRS (rasterio order)."""

    left: float
    bottom: float
    right: float
    top: float


class IndexMapOverlayResult(BaseModel):
    """Metadata to paint a derived index PNG as a MapLibre image overlay (Fase 9E)."""

    scene_id: UUID
    index_key: str
    image_url: str = Field(
        description="API path of the existing preview PNG (relative to API host)"
    )
    width: int
    height: int
    crs_original: str
    bounds_original: IndexMapOverlayBounds
    coordinates_wgs84: list[list[float]] = Field(
        description=(
            "Four [lng, lat] corners in MapLibre image-source order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
        min_length=4,
        max_length=4,
    )


class IndexAoiCropRequest(BaseModel):
    """Body for cropping a derived index GeoTIFF by a saved AOI (Fase 9F)."""

    aoi_id: UUID
    overwrite: bool = False
    generate_preview: bool = True


class IndexAoiCropRasterInfo(BaseModel):
    """Cropped raster metadata."""

    width: int
    height: int
    crs: Optional[str] = None
    dtype: str = "float32"
    nodata: float = Field(description="Nodata sentinel of the cropped GeoTIFF")


class IndexAoiCropOutputInfo(BaseModel):
    """Paths of cropped derived products (relative to DATA_ROOT)."""

    geotiff_asset_path: str
    png_asset_path: Optional[str] = None


class IndexAoiCropResult(BaseModel):
    """Result of cropping a derived index by AOI."""

    scene_id: UUID
    index_key: str
    aoi_id: UUID
    status: str = "cropped"
    raster: IndexAoiCropRasterInfo
    stats: IndexStats
    output: IndexAoiCropOutputInfo
    radiometry: Optional[RadiometryInfo] = None


class IndexAoiCropMapOverlayResult(BaseModel):
    """MapLibre overlay metadata for an AOI-cropped derived index (Fase 9F)."""

    scene_id: UUID
    index_key: str
    aoi_id: UUID
    image_url: str = Field(
        description="API path of the cropped preview PNG (relative to API host)"
    )
    width: int
    height: int
    crs_original: str
    bounds_original: IndexMapOverlayBounds
    coordinates_wgs84: list[list[float]] = Field(
        description=(
            "Four [lng, lat] corners in MapLibre image-source order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
        min_length=4,
        max_length=4,
    )


# Backward-compatible alias for Fase 7B callers / OpenAPI.
NdviComputeResult = IndexComputeResult

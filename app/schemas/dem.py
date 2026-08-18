"""Pydantic schemas for experimental DEM / hillshade (v0.1-P5)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DemBounds(BaseModel):
    """Raster bounds in the original CRS (rasterio order)."""

    left: float
    bottom: float
    right: float
    top: float


class DemAssetRead(BaseModel):
    """Catalog entry for an uploaded DEM GeoTIFF."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    asset_path: str = Field(description="Path relative to DATA_ROOT")
    preview_path: Optional[str] = Field(
        default=None,
        description="Hillshade PNG path relative to DATA_ROOT, if generated",
    )
    crs: str
    width: int
    height: int
    bounds: DemBounds
    min_elevation: Optional[float] = None
    max_elevation: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class DemHillshadeResult(BaseModel):
    """Result of generating a hillshade PNG from a DEM."""

    dem_id: UUID
    status: str = "hillshade_created"
    preview_path: str
    width: int
    height: int
    azimuth: float
    altitude: float
    nodata_transparent: bool = True


class DemMapOverlayResult(BaseModel):
    """Metadata to paint a hillshade PNG as a MapLibre image overlay."""

    dem_id: UUID
    image_url: str = Field(
        description="API path of the hillshade PNG (relative to API host)"
    )
    width: int
    height: int
    crs_original: str
    bounds_original: DemBounds
    coordinates_wgs84: list[list[float]] = Field(
        description=(
            "Four [lng, lat] corners in MapLibre image-source order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
        min_length=4,
        max_length=4,
    )


__all__ = [
    "DemAssetRead",
    "DemBounds",
    "DemHillshadeResult",
    "DemMapOverlayResult",
]

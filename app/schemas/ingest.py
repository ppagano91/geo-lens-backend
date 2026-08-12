"""Schemas for local scene ingest (Fase 9A)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.radiometry import RadiometryInfo


class LocalSceneIngestRequest(BaseModel):
    """Register a local GeoTIFF scene folder under DATA_ROOT."""

    scene_path: str = Field(
        ...,
        min_length=1,
        description="Relative path under DATA_ROOT to the scene folder",
        examples=["sample/scenes/landsat8_lc08_225084"],
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Sensor/source hint (landsat-8 or sentinel-2)",
        examples=["landsat-8", "sentinel-2"],
    )
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Optional scene display name",
    )
    overwrite: bool = Field(
        default=False,
        description="If true, replace an existing scene previously ingested from the same path",
    )


class IngestedBandInfo(BaseModel):
    band_key: str
    band_name: str
    asset_path: str
    width: int
    height: int
    crs: Optional[str] = None
    dtype: Optional[str] = None
    nodata: Optional[str] = None
    # Optional band metadata (e.g. Fase 9L alignment / resampling flags).
    metadata: Optional[dict[str, Any]] = None


class AvailableIndexInfo(BaseModel):
    index_key: str
    display_name: str
    compatible: bool
    missing_roles: list[str] = Field(default_factory=list)


class IngestionWarning(BaseModel):
    """Structured ingest warning for API clients."""

    code: str
    title: str
    description: Optional[str] = None
    items: list[str] = Field(default_factory=list)
    severity: Literal["info", "warning", "error"] = "warning"


class LocalSceneIngestResult(BaseModel):
    scene_id: UUID
    name: str
    source: str
    sensor: str
    acquisition_date: date
    scene_path: str
    bands: list[IngestedBandInfo]
    warnings: list[IngestionWarning] = Field(default_factory=list)
    available_indices: list[AvailableIndexInfo] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    radiometry: Optional[RadiometryInfo] = None
    overwritten: bool = False


__all__ = [
    "LocalSceneIngestRequest",
    "IngestedBandInfo",
    "AvailableIndexInfo",
    "IngestionWarning",
    "LocalSceneIngestResult",
]

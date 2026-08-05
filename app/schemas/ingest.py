"""Schemas for local scene ingest (Fase 9A)."""

from __future__ import annotations

from datetime import date
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


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
        description="Sensor/source hint (e.g. landsat-8)",
        examples=["landsat-8"],
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


class AvailableIndexInfo(BaseModel):
    index_key: str
    display_name: str
    compatible: bool
    missing_roles: list[str] = Field(default_factory=list)


class LocalSceneIngestResult(BaseModel):
    scene_id: UUID
    name: str
    source: str
    sensor: str
    acquisition_date: date
    scene_path: str
    bands: list[IngestedBandInfo]
    warnings: list[str] = Field(default_factory=list)
    available_indices: list[AvailableIndexInfo] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    overwritten: bool = False


__all__ = [
    "LocalSceneIngestRequest",
    "IngestedBandInfo",
    "AvailableIndexInfo",
    "LocalSceneIngestResult",
]

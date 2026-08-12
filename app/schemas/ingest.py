"""Schemas for local scene ingest (Fase 9A / 9M.1)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.radiometry import IngestProductLevel, RadiometryInfo


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
    product_level: Optional[IngestProductLevel] = Field(
        default=None,
        description=(
            "Optional radiometry product-level override "
            "(sentinel_l1c / sentinel_l2a / landsat_l2 / unknown)"
        ),
    )
    source_product_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description=(
            "Optional original product id, e.g. "
            "S2B_MSIL1C_20181226T141039_N0207_R110_T20JLL_20181226T172720"
        ),
    )

    @field_validator("source_product_id")
    @classmethod
    def _strip_product_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @model_validator(mode="after")
    def _validate_product_level_for_source(self) -> LocalSceneIngestRequest:
        if self.product_level is None:
            return self
        source = (self.source or "").strip().lower().replace("_", "-")
        if source in {"sentinel-2", "sentinel2", "s2"}:
            if self.product_level not in {"sentinel_l1c", "sentinel_l2a", "unknown"}:
                raise ValueError(
                    "For source=sentinel-2, product_level must be "
                    "sentinel_l1c, sentinel_l2a, or unknown"
                )
        if source in {"landsat-8", "landsat8", "l8"} and self.product_level not in {
            "landsat_l2",
            "unknown",
        }:
            raise ValueError(
                "For source=landsat-8, product_level must be landsat_l2 or unknown"
            )
        return self


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
    metadata_files_detected: list[str] = Field(default_factory=list)
    overwritten: bool = False


__all__ = [
    "LocalSceneIngestRequest",
    "IngestedBandInfo",
    "AvailableIndexInfo",
    "IngestionWarning",
    "LocalSceneIngestResult",
]

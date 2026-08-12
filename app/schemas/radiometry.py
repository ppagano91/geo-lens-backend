"""Radiometry metadata schemas (Fase 9M / 9M.1)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ProductLevel = Literal[
    "landsat_l1",
    "landsat_l2",
    "sentinel_l1c",
    "sentinel_l2a",
    "synthetic",
    "unknown",
]

# Manual ingest override / request values (subset of ProductLevel).
IngestProductLevel = Literal[
    "sentinel_l1c",
    "sentinel_l2a",
    "landsat_l2",
    "unknown",
]

RadiometryType = Literal[
    "dn",
    "toa_reflectance",
    "surface_reflectance",
    "synthetic",
    "unknown",
]

RadiometrySource = Literal[
    "landsat_mtl",
    "sentinel_product_name",
    "sentinel_metadata",
    "sentinel_path",
    "manual",
    "manual_override",
    "manual_product_id",
    "synthetic",
    "unknown",
]


class RadiometryInfo(BaseModel):
    """Compact radiometry block returned by ingest / index / RGB APIs."""

    product_level: ProductLevel = "unknown"
    radiometry_type: RadiometryType = "unknown"
    scale_factor: Optional[float] = None
    offset: Optional[float] = None
    scale_applied: bool = False
    source_product_id: Optional[str] = None
    radiometry_source: RadiometrySource | str = "unknown"
    warning: Optional[str] = Field(
        default=None,
        description="Human-readable radiometry warning (null when known)",
    )


__all__ = [
    "ProductLevel",
    "IngestProductLevel",
    "RadiometryType",
    "RadiometrySource",
    "RadiometryInfo",
]

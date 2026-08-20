"""Pydantic schemas for RGB composite preview / map-overlay (Fase 9H)."""

from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.radiometry import RadiometryInfo

RgbPresetKey = Literal[
    "true_color",
    "false_color_vegetation",
    "swir_urban",
    "moisture_vegetation",
    "agriculture",
    "geology",
    "burn_scar",
    "water_land",
    "atmospheric_penetration",
]

RgbStretchMode = Literal["percentile"]

SPECTRAL_ROLES = frozenset({"blue", "green", "red", "nir", "swir1", "swir2"})


class RgbCompositePreviewRequest(BaseModel):
    """Body for generating an RGB composite PNG from scene bands."""

    preset: RgbPresetKey = Field(
        description="Named RGB combination (true_color, false_color_vegetation, …)"
    )
    red_role: Optional[str] = Field(
        default=None,
        description="Optional override for the red display channel (spectral role)",
    )
    green_role: Optional[str] = Field(
        default=None,
        description="Optional override for the green display channel (spectral role)",
    )
    blue_role: Optional[str] = Field(
        default=None,
        description="Optional override for the blue display channel (spectral role)",
    )
    stretch: RgbStretchMode = "percentile"
    p_min: float = Field(default=2.0, ge=0.0, le=100.0)
    p_max: float = Field(default=98.0, ge=0.0, le=100.0)
    overwrite: bool = True

    @field_validator("red_role", "green_role", "blue_role")
    @classmethod
    def _normalize_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        role = value.strip().lower()
        if not role:
            return None
        if role not in SPECTRAL_ROLES:
            raise ValueError(
                f"Unknown spectral role '{value}'. "
                f"Expected one of: {', '.join(sorted(SPECTRAL_ROLES))}"
            )
        return role

    @model_validator(mode="after")
    def _validate_percentile_range(self) -> RgbCompositePreviewRequest:
        if self.p_max <= self.p_min:
            raise ValueError(
                f"p_max ({self.p_max}) must be greater than p_min ({self.p_min})"
            )
        return self


class RgbCompositeOutputInfo(BaseModel):
    """PNG written under DATA_ROOT."""

    asset_path: str = Field(description="Path relative to DATA_ROOT")


class RgbCompositePreviewResult(BaseModel):
    """RGB composite PNG generation summary."""

    scene_id: UUID
    preset: str
    status: str = "generated"
    sensor: str
    bands_used: dict[str, str] = Field(
        description="Display channel → physical band_key (red/green/blue)"
    )
    width: int
    height: int
    crs: Optional[str] = None
    output: RgbCompositeOutputInfo
    radiometry: Optional[RadiometryInfo] = None


class RgbCompositeAoiPreviewRequest(RgbCompositePreviewRequest):
    """Body for generating an RGB composite PNG cropped by a saved AOI."""

    aoi_id: UUID = Field(description="Saved AOI used to crop source bands before RGB")


class RgbCompositeAoiPreviewResult(BaseModel):
    """AOI-cropped RGB composite PNG generation summary (Fase 9H.1)."""

    scene_id: UUID
    aoi_id: UUID
    preset: str
    status: str = "generated"
    sensor: str
    bands_used: dict[str, str] = Field(
        description="Display channel → physical band_key (red/green/blue)"
    )
    width: int
    height: int
    crs: Optional[str] = None
    output: RgbCompositeOutputInfo
    radiometry: Optional[RadiometryInfo] = None


class RgbCompositeMapOverlayBounds(BaseModel):
    """Raster bounds in the original CRS (rasterio order)."""

    left: float
    bottom: float
    right: float
    top: float


class RgbCompositeMapOverlayResult(BaseModel):
    """Metadata to paint an RGB composite PNG as a MapLibre image overlay."""

    scene_id: UUID
    preset: str
    image_url: str = Field(
        description="API path of the existing RGB preview PNG (relative to API host)"
    )
    width: int
    height: int
    crs_original: str
    bounds_original: RgbCompositeMapOverlayBounds
    coordinates_wgs84: list[list[float]] = Field(
        description=(
            "Four [lng, lat] corners in MapLibre image-source order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
        min_length=4,
        max_length=4,
    )


class RgbCompositeAoiMapOverlayResult(BaseModel):
    """MapLibre overlay metadata for an AOI-cropped RGB composite PNG."""

    scene_id: UUID
    aoi_id: UUID
    preset: str
    image_url: str = Field(
        description="API path of the existing AOI RGB preview PNG"
    )
    width: int
    height: int
    crs_original: str
    bounds_original: RgbCompositeMapOverlayBounds
    coordinates_wgs84: list[list[float]] = Field(
        description=(
            "Four [lng, lat] corners in MapLibre image-source order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
        min_length=4,
        max_length=4,
    )


__all__ = [
    "RgbCompositePreviewRequest",
    "RgbCompositeAoiPreviewRequest",
    "RgbCompositeOutputInfo",
    "RgbCompositePreviewResult",
    "RgbCompositeAoiPreviewResult",
    "RgbCompositeMapOverlayBounds",
    "RgbCompositeMapOverlayResult",
    "RgbCompositeAoiMapOverlayResult",
    "RgbPresetKey",
    "RgbStretchMode",
    "SPECTRAL_ROLES",
]

"""RGB band composites from local scene GeoTIFF bands (Fase 9H).

First approximation of an SCP / Band Set style module: named presets resolve
spectral roles → sensor band keys, stretch to uint8 RGBA PNG, and expose
MapLibre overlay metadata. Does not write multiband GeoTIFF stacks, edit
stretch interactively, crop by AOI, or touch spectral index compute.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.band import RasterBand
from app.raster.preview import PreviewWriteError, write_preview_png
from app.raster.readers import (
    RasterArray,
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
    read_raster_metadata,
)
from app.raster.rgb_stretch import render_rgb_rgba
from app.raster.sensors import detect_sensor, resolve_band_key
from app.repositories.scene_repository import SceneRepository
from app.schemas.rgb_composite import (
    RgbCompositeMapOverlayBounds,
    RgbCompositeMapOverlayResult,
    RgbCompositeOutputInfo,
    RgbCompositePreviewRequest,
    RgbCompositePreviewResult,
)
from app.services.asset_storage_service import AssetStorageService
from app.services.index_map_overlay_service import (
    IndexMapOverlayError,
    corners_to_wgs84,
)
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    MissingRequiredBandError,
)
from app.services.scene_service import SceneNotFoundError


@dataclass(frozen=True)
class RgbCompositeSpec:
    """Named RGB preset: display channels → spectral roles."""

    key: str
    display_name: str
    # Display channel → spectral role (red/green/blue keys).
    roles: Mapping[str, str]


RGB_COMPOSITE_REGISTRY: dict[str, RgbCompositeSpec] = {
    "true_color": RgbCompositeSpec(
        key="true_color",
        display_name="Color verdadero",
        roles={"red": "red", "green": "green", "blue": "blue"},
    ),
    "false_color_vegetation": RgbCompositeSpec(
        key="false_color_vegetation",
        display_name="Falso color vegetación",
        roles={"red": "nir", "green": "red", "blue": "green"},
    ),
    "swir_urban": RgbCompositeSpec(
        key="swir_urban",
        display_name="SWIR urbano",
        roles={"red": "swir2", "green": "swir1", "blue": "red"},
    ),
    "moisture_vegetation": RgbCompositeSpec(
        key="moisture_vegetation",
        display_name="Humedad / vegetación",
        roles={"red": "swir1", "green": "nir", "blue": "red"},
    ),
}


class UnsupportedRgbPresetError(Exception):
    """Requested RGB preset is not in the local registry."""

    def __init__(self, preset: str) -> None:
        self.preset = preset
        super().__init__(f"RGB composite preset '{preset}' is not supported")


class RgbCompositeExistsError(Exception):
    """PNG already exists and overwrite was not requested."""

    def __init__(self, scene_id: UUID, preset: str, asset_path: str) -> None:
        self.scene_id = scene_id
        self.preset = preset
        self.asset_path = asset_path
        super().__init__(
            f"RGB composite PNG already exists for scene {scene_id} preset "
            f"'{preset}' at '{asset_path}'. Pass overwrite=true to replace it."
        )


class RgbCompositePngNotFoundError(Exception):
    """RGB preview PNG is missing; POST .../rgb-composites/preview must run first."""

    def __init__(self, scene_id: UUID, preset: str, asset_path: str) -> None:
        self.scene_id = scene_id
        self.preset = preset
        self.asset_path = asset_path
        super().__init__(
            f"RGB composite PNG not found for scene {scene_id} preset '{preset}' "
            f"at '{asset_path}'. Generate it first with "
            f"POST /api/v1/scenes/{scene_id}/rgb-composites/preview"
        )


class RgbCompositeService:
    """Orchestrate band resolution, stretch, PNG write, and map-overlay metadata."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        data_root: Path | str | None = None,
    ) -> None:
        self.repository = SceneRepository(db) if db is not None else None
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    @data_root.setter
    def data_root(self, value: Path | str) -> None:
        self._storage = AssetStorageService(value)

    def create_preview(
        self,
        scene_id: UUID,
        request: RgbCompositePreviewRequest,
    ) -> RgbCompositePreviewResult:
        """Read three bands, stretch to RGBA PNG, write under derived/…/rgb/."""
        if self.repository is None:
            raise RuntimeError("RgbCompositeService requires a DB session for preview")

        spec = self._resolve_spec(request.preset)
        display_roles = self._effective_roles(spec, request)

        scene = self.repository.get_by_id(scene_id)
        if scene is None or not scene.is_active:
            raise SceneNotFoundError(str(scene_id))

        sensor = self._detect_scene_sensor(scene)
        bands_by_key = {band.band_key: band for band in scene.bands}

        role_bands: dict[str, RasterBand] = {}
        # Unique spectral roles needed (a role may map to more than one channel).
        spectral_roles = list(dict.fromkeys(display_roles.values()))
        for spectral_role in spectral_roles:
            band_key = resolve_band_key(sensor, spectral_role)
            role_bands[spectral_role] = self._require_band(
                scene_id, bands_by_key, band_key
            )

        role_arrays: dict[str, RasterArray] = {
            role: read_raster_array(band.asset_path, self.data_root)
            for role, band in role_bands.items()
        }
        self._validate_aligned(role_arrays)

        asset_path = self._storage.build_derived_rgb_asset_path(
            scene_id, spec.key, "png"
        )
        if self._storage.exists(asset_path) and not request.overwrite:
            raise RgbCompositeExistsError(scene_id, spec.key, asset_path)

        red_arr = role_arrays[display_roles["red"]]
        green_arr = role_arrays[display_roles["green"]]
        blue_arr = role_arrays[display_roles["blue"]]

        rgba = render_rgb_rgba(
            red_arr.data,
            green_arr.data,
            blue_arr.data,
            red_nodata=red_arr.nodata,
            green_nodata=green_arr.nodata,
            blue_nodata=blue_arr.nodata,
            p_min=request.p_min,
            p_max=request.p_max,
        )
        write_preview_png(asset_path, self.data_root, rgba)

        bands_used = {
            channel: role_bands[spectral_role].band_key
            for channel, spectral_role in display_roles.items()
        }
        reference = red_arr

        return RgbCompositePreviewResult(
            scene_id=scene_id,
            preset=spec.key,
            status="generated",
            sensor=sensor,
            bands_used=bands_used,
            width=reference.width,
            height=reference.height,
            crs=reference.crs,
            output=RgbCompositeOutputInfo(asset_path=asset_path),
        )

    def resolve_preview_png(self, scene_id: UUID, preset: str) -> Path:
        """Return absolute path of an existing RGB PNG (does not regenerate)."""
        spec = self._resolve_spec(preset)
        asset_path = self._storage.build_derived_rgb_asset_path(
            scene_id, spec.key, "png"
        )
        if not self._storage.exists(asset_path):
            raise RgbCompositePngNotFoundError(scene_id, spec.key, asset_path)
        return self._storage.resolve_read_path(asset_path)

    def get_map_overlay(
        self, scene_id: UUID, preset: str
    ) -> RgbCompositeMapOverlayResult:
        """Return MapLibre overlay metadata; georef from a source band of the preset."""
        if self.repository is None:
            raise RuntimeError(
                "RgbCompositeService requires a DB session for map-overlay"
            )

        spec = self._resolve_spec(preset)
        png_asset = self._storage.build_derived_rgb_asset_path(
            scene_id, spec.key, "png"
        )
        if not self._storage.exists(png_asset):
            raise RgbCompositePngNotFoundError(scene_id, spec.key, png_asset)

        scene = self.repository.get_by_id(scene_id)
        if scene is None or not scene.is_active:
            raise SceneNotFoundError(str(scene_id))

        sensor = self._detect_scene_sensor(scene)
        bands_by_key = {band.band_key: band for band in scene.bands}
        # Use the red display channel's source band for georeferencing.
        red_role = spec.roles["red"]
        band_key = resolve_band_key(sensor, red_role)
        ref_band = self._require_band(scene_id, bands_by_key, band_key)

        meta = read_raster_metadata(ref_band.asset_path, self.data_root)
        if meta.width is None or meta.height is None:
            raise IndexMapOverlayError(
                f"Reference band '{band_key}' for scene {scene_id} RGB preset "
                f"'{spec.key}' has no width/height"
            )
        if not meta.bounds:
            raise IndexMapOverlayError(
                f"Reference band '{band_key}' for scene {scene_id} RGB preset "
                f"'{spec.key}' has no bounds"
            )
        if not meta.crs:
            raise IndexMapOverlayError(
                f"Reference band '{band_key}' for scene {scene_id} RGB preset "
                f"'{spec.key}' has no CRS; cannot georeference overlay"
            )

        left = float(meta.bounds["left"])
        bottom = float(meta.bounds["bottom"])
        right = float(meta.bounds["right"])
        top = float(meta.bounds["top"])

        coordinates = corners_to_wgs84(
            meta.crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )

        image_url = (
            f"/api/v1/scenes/{scene_id}/rgb-composites/{spec.key}/preview.png"
        )

        return RgbCompositeMapOverlayResult(
            scene_id=scene_id,
            preset=spec.key,
            image_url=image_url,
            width=int(meta.width),
            height=int(meta.height),
            crs_original=meta.crs,
            bounds_original=RgbCompositeMapOverlayBounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            ),
            coordinates_wgs84=coordinates,
        )

    @staticmethod
    def _resolve_spec(preset: str) -> RgbCompositeSpec:
        key = (preset or "").strip().lower()
        spec = RGB_COMPOSITE_REGISTRY.get(key)
        if spec is None:
            raise UnsupportedRgbPresetError(key)
        return spec

    @staticmethod
    def _effective_roles(
        spec: RgbCompositeSpec,
        request: RgbCompositePreviewRequest,
    ) -> dict[str, str]:
        roles = dict(spec.roles)
        if request.red_role:
            roles["red"] = request.red_role
        if request.green_role:
            roles["green"] = request.green_role
        if request.blue_role:
            roles["blue"] = request.blue_role
        return roles

    @staticmethod
    def _detect_scene_sensor(scene: Any) -> str:
        source = getattr(scene, "source", None)
        metadata = getattr(scene, "metadata_", None)
        if metadata is None:
            metadata = getattr(scene, "metadata", None)
        return detect_sensor(source=source, metadata=metadata)

    @staticmethod
    def _require_band(
        scene_id: UUID,
        bands_by_key: dict[str, RasterBand],
        band_key: str,
    ) -> RasterBand:
        band = bands_by_key.get(band_key)
        if band is None:
            raise MissingRequiredBandError(scene_id, band_key)
        return band

    @staticmethod
    def _validate_aligned(role_arrays: Mapping[str, RasterArray]) -> None:
        items = list(role_arrays.items())
        if len(items) < 1:
            raise IncompatibleRasterBandsError(
                "At least one band is required for RGB composite"
            )

        ref_key, ref = items[0]
        if ref.count != 1:
            raise IncompatibleRasterBandsError(
                f"Band {ref_key} must be a single-band raster"
            )

        for other_key, other in items[1:]:
            label = f"{ref_key}/{other_key}"
            if other.count != 1:
                raise IncompatibleRasterBandsError(
                    f"Bands {label} must be single-band rasters"
                )
            if ref.crs != other.crs:
                raise IncompatibleRasterBandsError(
                    f"Bands {label} have different CRS: {ref.crs!r} vs {other.crs!r}"
                )
            if ref.width != other.width or ref.height != other.height:
                raise IncompatibleRasterBandsError(
                    f"Bands {label} have different dimensions: "
                    f"{ref.width}x{ref.height} vs {other.width}x{other.height}"
                )
            if ref.transform != other.transform:
                raise IncompatibleRasterBandsError(
                    f"Bands {label} have different geotransforms"
                )
            if ref.data.shape != other.data.shape:
                raise IncompatibleRasterBandsError(
                    f"Bands {label} have incompatible array shapes: "
                    f"{ref.data.shape} vs {other.data.shape}"
                )


__all__ = [
    "RgbCompositeService",
    "RgbCompositeSpec",
    "RGB_COMPOSITE_REGISTRY",
    "UnsupportedRgbPresetError",
    "RgbCompositeExistsError",
    "RgbCompositePngNotFoundError",
    "MissingRequiredBandError",
    "IncompatibleRasterBandsError",
    "SceneNotFoundError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "PreviewWriteError",
    "IndexMapOverlayError",
]

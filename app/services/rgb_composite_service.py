"""RGB band composites from local scene GeoTIFF bands (Fase 9H / 9H.1).

Named presets resolve spectral roles → sensor band keys, stretch to uint8 RGBA
PNG, and expose MapLibre overlay metadata.

Fase 9H.1: crop source bands by a saved AOI first (``rasterio.mask``), then
stretch only the cropped window. Georef for AOI overlays is stored in a
``.georef.json`` sidecar (same folder reserved for a future RGB GeoTIFF).

Fase 9I: registers catalog rows in ``raster_derived_assets`` (paths + metadata
only; never PNG bytes). Does not write multiband GeoTIFF stacks or touch
index compute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_geom
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
    resolve_asset_path,
)
from app.raster.rgb_stretch import render_rgb_rgba
from app.raster.sensors import detect_sensor, resolve_band_key
from app.repositories.scene_repository import SceneRepository
from app.schemas.rgb_composite import (
    RgbCompositeAoiMapOverlayResult,
    RgbCompositeAoiPreviewRequest,
    RgbCompositeAoiPreviewResult,
    RgbCompositeMapOverlayBounds,
    RgbCompositeMapOverlayResult,
    RgbCompositeOutputInfo,
    RgbCompositePreviewRequest,
    RgbCompositePreviewResult,
)
from app.services.aoi_service import AoiNotFoundError, AoiService
from app.services.asset_storage_service import AssetStorageService
from app.services.derived_asset_service import DerivedAssetService
from app.services.geometry import GeometryValidationError
from app.services.index_aoi_crop_service import IndexAoiReprojectionError
from app.services.index_map_overlay_service import (
    IndexMapOverlayError,
    corners_to_wgs84,
)
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    MissingRequiredBandError,
)
from app.services.radiometry_service import RadiometryService
from app.services.scene_service import SceneNotFoundError

_WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class RgbCompositeSpec:
    """Named RGB preset: display channels → spectral roles."""

    key: str
    display_name: str
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


@dataclass(frozen=True)
class _BandProfile:
    """Lightweight alignment metadata without loading pixel arrays."""

    band_key: str
    asset_path: str
    width: int
    height: int
    crs: str
    transform: tuple[float, float, float, float, float, float]
    count: int
    nodata: float | None


@dataclass(frozen=True)
class _CroppedChannel:
    data: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str
    width: int
    height: int


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


class RgbAoiCompositePngNotFoundError(Exception):
    """AOI RGB PNG missing; POST .../preview-by-aoi must run first."""

    def __init__(
        self,
        scene_id: UUID,
        aoi_id: UUID,
        preset: str,
        asset_path: str,
    ) -> None:
        self.scene_id = scene_id
        self.aoi_id = aoi_id
        self.preset = preset
        self.asset_path = asset_path
        super().__init__(
            f"AOI RGB composite PNG not found for scene {scene_id} AOI {aoi_id} "
            f"preset '{preset}' at '{asset_path}'. Generate it first with "
            f"POST /api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi"
        )


class RgbAoiNoIntersectionError(Exception):
    """AOI geometry does not intersect the source bands."""

    def __init__(self, scene_id: UUID, aoi_id: UUID, preset: str) -> None:
        self.scene_id = scene_id
        self.aoi_id = aoi_id
        self.preset = preset
        super().__init__(
            f"AOI {aoi_id} does not intersect RGB bands for scene {scene_id} "
            f"preset '{preset}'"
        )


class RgbCompositeService:
    """Orchestrate band resolution, stretch, PNG write, and map-overlay metadata."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        data_root: Path | str | None = None,
    ) -> None:
        self._db = db
        self.repository = SceneRepository(db) if db is not None else None
        self._storage = AssetStorageService(data_root)
        self._radiometry = RadiometryService()

    @property
    def radiometry_service(self) -> RadiometryService:
        service = getattr(self, "_radiometry", None)
        if service is None:
            service = RadiometryService()
            self._radiometry = service
        return service

    def _maybe_db(self):
        """DB session if constructed with one (unit stubs may omit ``_db``)."""
        return getattr(self, "_db", None)

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

        radiometry = self.radiometry_service.detect_scene_radiometry(
            scene,
            bands=list(scene.bands),
        )

        asset_path = self._storage.build_derived_rgb_asset_path(
            scene_id, spec.key, "png"
        )
        if self._storage.exists(asset_path) and not request.overwrite:
            raise RgbCompositeExistsError(scene_id, spec.key, asset_path)

        red_arr = role_arrays[display_roles["red"]]
        green_arr = role_arrays[display_roles["green"]]
        blue_arr = role_arrays[display_roles["blue"]]

        red_data = self.radiometry_service.apply_radiometric_scaling(
            red_arr.data, red_arr.nodata, radiometry
        )
        green_data = self.radiometry_service.apply_radiometric_scaling(
            green_arr.data, green_arr.nodata, radiometry
        )
        blue_data = self.radiometry_service.apply_radiometric_scaling(
            blue_arr.data, blue_arr.nodata, radiometry
        )

        rgba = render_rgb_rgba(
            red_data,
            green_data,
            blue_data,
            red_nodata=None,
            green_nodata=None,
            blue_nodata=None,
            p_min=request.p_min,
            p_max=request.p_max,
        )
        write_preview_png(asset_path, self.data_root, rgba)

        bands_used = {
            channel: role_bands[spectral_role].band_key
            for channel, spectral_role in display_roles.items()
        }
        reference = red_arr
        radiometry_meta = radiometry.as_nested_metadata()

        if self._maybe_db() is not None:
            DerivedAssetService(self._maybe_db()).create_or_update_derived_asset(
                scene_id=scene_id,
                asset_type="rgb_composite",
                product_key=spec.key,
                asset_path=asset_path,
                preview_path=asset_path,
                update_preview_path=True,
                crs=reference.crs,
                width=reference.width,
                height=reference.height,
                dtype="uint8",
                metadata={
                    "preset": spec.key,
                    "sensor": sensor,
                    "bands_used": bands_used,
                    "stretch": request.stretch,
                    "p_min": request.p_min,
                    "p_max": request.p_max,
                    **radiometry_meta,
                },
            )

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
            radiometry=radiometry.to_info(),
        )

    def create_preview_by_aoi(
        self,
        scene_id: UUID,
        request: RgbCompositeAoiPreviewRequest,
    ) -> RgbCompositeAoiPreviewResult:
        """Crop source bands by AOI, then stretch RGB for the cropped window."""
        if self._maybe_db() is None or self.repository is None:
            raise RuntimeError(
                "RgbCompositeService requires a DB session for preview-by-aoi"
            )

        spec = self._resolve_spec(request.preset)
        display_roles = self._effective_roles(spec, request)
        aoi_id = request.aoi_id

        scene = self.repository.get_by_id(scene_id)
        if scene is None or not scene.is_active:
            raise SceneNotFoundError(str(scene_id))

        aoi = AoiService(self._maybe_db()).get(aoi_id)
        geometry = aoi.geometry
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise GeometryValidationError(
                f"AOI {aoi_id} has invalid or empty geometry"
            )

        sensor = self._detect_scene_sensor(scene)
        bands_by_key = {band.band_key: band for band in scene.bands}

        role_bands: dict[str, RasterBand] = {}
        spectral_roles = list(dict.fromkeys(display_roles.values()))
        for spectral_role in spectral_roles:
            band_key = resolve_band_key(sensor, spectral_role)
            role_bands[spectral_role] = self._require_band(
                scene_id, bands_by_key, band_key
            )

        radiometry = self.radiometry_service.detect_scene_radiometry(
            scene,
            bands=list(scene.bands),
        )

        profiles = {
            role: self._read_band_profile(band)
            for role, band in role_bands.items()
        }
        self._validate_profiles_aligned(profiles)

        asset_path = self._storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )
        if self._storage.exists(asset_path) and not request.overwrite:
            raise RgbCompositeExistsError(scene_id, spec.key, asset_path)

        ref_profile = profiles[display_roles["red"]]
        geom_in_raster_crs = self._reproject_aoi(
            geometry, ref_profile.crs, aoi_id=aoi_id
        )

        cropped: dict[str, _CroppedChannel] = {}
        for spectral_role, band in role_bands.items():
            cropped[spectral_role] = self._crop_band_by_aoi(
                band.asset_path,
                geom_in_raster_crs,
                scene_id=scene_id,
                aoi_id=aoi_id,
                preset=spec.key,
            )

        ref_crop = cropped[display_roles["red"]]
        for spectral_role, channel in cropped.items():
            if (
                channel.width != ref_crop.width
                or channel.height != ref_crop.height
                or channel.transform != ref_crop.transform
                or channel.crs != ref_crop.crs
            ):
                raise IncompatibleRasterBandsError(
                    f"AOI-cropped bands are not aligned after mask "
                    f"({display_roles['red']} vs {spectral_role})"
                )

        red = cropped[display_roles["red"]]
        green = cropped[display_roles["green"]]
        blue = cropped[display_roles["blue"]]

        red_data = self.radiometry_service.apply_radiometric_scaling(
            red.data, None, radiometry
        )
        green_data = self.radiometry_service.apply_radiometric_scaling(
            green.data, None, radiometry
        )
        blue_data = self.radiometry_service.apply_radiometric_scaling(
            blue.data, None, radiometry
        )

        rgba = render_rgb_rgba(
            red_data,
            green_data,
            blue_data,
            red_nodata=None,
            green_nodata=None,
            blue_nodata=None,
            p_min=request.p_min,
            p_max=request.p_max,
        )
        write_preview_png(asset_path, self.data_root, rgba)

        left, bottom, right, top = array_bounds(
            ref_crop.height,
            ref_crop.width,
            Affine(*ref_crop.transform),
        )
        georef_path = self._write_georef_sidecar(
            scene_id,
            aoi_id,
            spec.key,
            crs=ref_crop.crs,
            width=ref_crop.width,
            height=ref_crop.height,
            transform=ref_crop.transform,
            left=float(left),
            bottom=float(bottom),
            right=float(right),
            top=float(top),
        )

        bands_used = {
            channel: role_bands[spectral_role].band_key
            for channel, spectral_role in display_roles.items()
        }
        radiometry_meta = radiometry.as_nested_metadata()

        if self._maybe_db() is not None:
            DerivedAssetService(self._maybe_db()).create_or_update_derived_asset(
                scene_id=scene_id,
                aoi_id=aoi_id,
                asset_type="rgb_composite_aoi",
                product_key=spec.key,
                asset_path=asset_path,
                preview_path=asset_path,
                georef_path=georef_path,
                update_preview_path=True,
                update_georef_path=True,
                crs=ref_crop.crs,
                width=ref_crop.width,
                height=ref_crop.height,
                dtype="uint8",
                metadata={
                    "preset": spec.key,
                    "sensor": sensor,
                    "bands_used": bands_used,
                    "stretch": request.stretch,
                    "p_min": request.p_min,
                    "p_max": request.p_max,
                    **radiometry_meta,
                },
            )

        return RgbCompositeAoiPreviewResult(
            scene_id=scene_id,
            aoi_id=aoi_id,
            preset=spec.key,
            status="generated",
            sensor=sensor,
            bands_used=bands_used,
            width=ref_crop.width,
            height=ref_crop.height,
            crs=ref_crop.crs,
            output=RgbCompositeOutputInfo(asset_path=asset_path),
            radiometry=radiometry.to_info(),
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

    def resolve_aoi_preview_png(
        self, scene_id: UUID, aoi_id: UUID, preset: str
    ) -> Path:
        """Return absolute path of an existing AOI RGB PNG."""
        spec = self._resolve_spec(preset)
        asset_path = self._storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )
        if not self._storage.exists(asset_path):
            raise RgbAoiCompositePngNotFoundError(
                scene_id, aoi_id, spec.key, asset_path
            )
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

    def get_aoi_map_overlay(
        self, scene_id: UUID, aoi_id: UUID, preset: str
    ) -> RgbCompositeAoiMapOverlayResult:
        """Return MapLibre overlay metadata for an AOI-cropped RGB PNG."""
        spec = self._resolve_spec(preset)
        png_asset = self._storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )
        if not self._storage.exists(png_asset):
            raise RgbAoiCompositePngNotFoundError(
                scene_id, aoi_id, spec.key, png_asset
            )

        georef = self._read_georef_sidecar(scene_id, aoi_id, spec.key)
        crs = str(georef["crs"])
        left = float(georef["bounds"]["left"])
        bottom = float(georef["bounds"]["bottom"])
        right = float(georef["bounds"]["right"])
        top = float(georef["bounds"]["top"])
        width = int(georef["width"])
        height = int(georef["height"])

        coordinates = corners_to_wgs84(
            crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )

        image_url = (
            f"/api/v1/scenes/{scene_id}/rgb-composites/aois/{aoi_id}/"
            f"{spec.key}/preview.png"
        )

        return RgbCompositeAoiMapOverlayResult(
            scene_id=scene_id,
            aoi_id=aoi_id,
            preset=spec.key,
            image_url=image_url,
            width=width,
            height=height,
            crs_original=crs,
            bounds_original=RgbCompositeMapOverlayBounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            ),
            coordinates_wgs84=coordinates,
        )

    def _write_georef_sidecar(
        self,
        scene_id: UUID,
        aoi_id: UUID,
        preset: str,
        *,
        crs: str,
        width: int,
        height: int,
        transform: tuple[float, float, float, float, float, float],
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> str:
        """Persist cropped window georef next to the PNG (future GeoTIFF-ready)."""
        asset_path = self._storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, preset, "georef.json"
        )
        path = self._storage.resolve_write_path(asset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "crs": crs,
            "width": width,
            "height": height,
            "transform": list(transform),
            "bounds": {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return asset_path

    def _read_georef_sidecar(
        self, scene_id: UUID, aoi_id: UUID, preset: str
    ) -> dict[str, Any]:
        asset_path = self._storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, preset, "georef.json"
        )
        if not self._storage.exists(asset_path):
            raise IndexMapOverlayError(
                f"Georef sidecar missing for AOI RGB composite at '{asset_path}'. "
                "Regenerate with POST .../rgb-composites/preview-by-aoi"
            )
        path = self._storage.resolve_read_path(asset_path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexMapOverlayError(
                f"Cannot read georef sidecar '{asset_path}': {exc}"
            ) from exc

    def _read_band_profile(self, band: RasterBand) -> _BandProfile:
        path = resolve_asset_path(band.asset_path, self.data_root)
        path_str = str(path)
        if not path.exists() or not path.is_file():
            raise RasterFileNotFoundError(f"Raster file not found: {path_str}")

        try:
            with rasterio.open(path) as dataset:
                if dataset.crs is None:
                    raise IncompatibleRasterBandsError(
                        f"Band '{band.band_key}' has no CRS"
                    )
                nodata = dataset.nodata
                nodata_f = float(nodata) if nodata is not None else None
                return _BandProfile(
                    band_key=band.band_key,
                    asset_path=band.asset_path,
                    width=int(dataset.width),
                    height=int(dataset.height),
                    crs=dataset.crs.to_string(),
                    transform=tuple(float(v) for v in list(dataset.transform)[:6]),
                    count=int(dataset.count),
                    nodata=nodata_f,
                )
        except IncompatibleRasterBandsError:
            raise
        except (RasterioIOError, OSError, ValueError) as exc:
            raise RasterReadError(
                f"Cannot read raster metadata: {path_str}"
            ) from exc

    def _crop_band_by_aoi(
        self,
        asset_path: str,
        geom_in_raster_crs: dict,
        *,
        scene_id: UUID,
        aoi_id: UUID,
        preset: str,
    ) -> _CroppedChannel:
        path = resolve_asset_path(asset_path, self.data_root)
        path_str = str(path)
        if not path.exists() or not path.is_file():
            raise RasterFileNotFoundError(f"Raster file not found: {path_str}")

        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise IndexAoiReprojectionError(
                        f"Band '{asset_path}' has no CRS; cannot crop by AOI"
                    )
                nodata = float(src.nodata) if src.nodata is not None else 0.0
                try:
                    out_image, out_transform = rasterio_mask(
                        src,
                        [geom_in_raster_crs],
                        crop=True,
                        nodata=nodata,
                        filled=True,
                    )
                except ValueError as exc:
                    message = str(exc).lower()
                    if (
                        "do not overlap" in message
                        or "shapes do not overlap" in message
                    ):
                        raise RgbAoiNoIntersectionError(
                            scene_id, aoi_id, preset
                        ) from exc
                    raise IndexAoiReprojectionError(
                        f"Mask failed for AOI {aoi_id}: {exc}"
                    ) from exc

                if (
                    out_image.size == 0
                    or out_image.shape[-1] == 0
                    or out_image.shape[-2] == 0
                ):
                    raise RgbAoiNoIntersectionError(scene_id, aoi_id, preset)

                raw = np.asarray(out_image[0], dtype=np.float32)
                data = np.where(
                    raw == np.float32(nodata), np.float32(np.nan), raw
                )
                height, width = int(data.shape[0]), int(data.shape[1])
                return _CroppedChannel(
                    data=data,
                    transform=tuple(float(v) for v in list(out_transform)[:6]),
                    crs=src.crs.to_string(),
                    width=width,
                    height=height,
                )
        except (
            RgbAoiNoIntersectionError,
            IndexAoiReprojectionError,
            RasterFileNotFoundError,
        ):
            raise
        except (RasterioIOError, OSError, ValueError) as exc:
            raise RasterReadError(f"Cannot crop raster: {path_str}") from exc

    @staticmethod
    def _reproject_aoi(
        geometry: dict, raster_crs: str, *, aoi_id: UUID
    ) -> dict:
        try:
            return transform_geom(_WGS84, raster_crs, geometry)
        except Exception as exc:  # noqa: BLE001
            raise IndexAoiReprojectionError(
                f"Cannot reproject AOI {aoi_id} from {_WGS84} "
                f"to {raster_crs}: {exc}"
            ) from exc

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

    @staticmethod
    def _validate_profiles_aligned(profiles: Mapping[str, _BandProfile]) -> None:
        items = list(profiles.items())
        if len(items) < 1:
            raise IncompatibleRasterBandsError(
                "At least one band is required for RGB composite"
            )

        ref_key, ref = items[0]
        if ref.count != 1:
            raise IncompatibleRasterBandsError(
                f"Band {ref.band_key} must be a single-band raster"
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


__all__ = [
    "RgbCompositeService",
    "RgbCompositeSpec",
    "RGB_COMPOSITE_REGISTRY",
    "UnsupportedRgbPresetError",
    "RgbCompositeExistsError",
    "RgbCompositePngNotFoundError",
    "RgbAoiCompositePngNotFoundError",
    "RgbAoiNoIntersectionError",
    "MissingRequiredBandError",
    "IncompatibleRasterBandsError",
    "SceneNotFoundError",
    "AoiNotFoundError",
    "GeometryValidationError",
    "IndexAoiReprojectionError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "PreviewWriteError",
    "IndexMapOverlayError",
]

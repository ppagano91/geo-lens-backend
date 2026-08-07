"""Crop derived spectral-index GeoTIFFs by a saved AOI (Fase 9F).

Reads an existing derived index under ``derived/scenes/{scene_id}/``,
reprojects the AOI polygon (EPSG:4326) to the raster CRS, masks with
``rasterio.mask.mask(crop=True)``, and writes cropped GeoTIFF (+ optional PNG)
under ``derived/scenes/{scene_id}/aois/{aoi_id}/``.

Does not crop original bands, recompute indices, or store raster bytes in DB.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import transform_geom
from sqlalchemy.orm import Session

from app.raster.preview import (
    PreviewWriteError,
    render_index_preview_rgba,
    write_preview_png,
)
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_metadata,
)
from app.raster.writers import DEFAULT_INDEX_NODATA, RasterWriteError, write_float32_geotiff
from app.schemas.index_compute import (
    IndexAoiCropMapOverlayResult,
    IndexAoiCropOutputInfo,
    IndexAoiCropRasterInfo,
    IndexAoiCropResult,
    IndexMapOverlayBounds,
    IndexStats,
)
from app.services.aoi_service import AoiNotFoundError, AoiService
from app.services.asset_storage_service import AssetStorageService
from app.services.geometry import GeometryValidationError
from app.services.index_map_overlay_service import (
    IndexMapOverlayError,
    corners_to_wgs84,
)
from app.services.index_preview_service import (
    CroppedGeotiffNotFoundError,
    CroppedPreviewPngNotFoundError,
    DerivedGeotiffNotFoundError,
)
from app.services.local_index_compute_service import (
    LOCAL_INDEX_REGISTRY,
    UnsupportedIndexError,
)
from app.services.scene_service import SceneNotFoundError, SceneService

_WGS84 = "EPSG:4326"


class IndexAoiCropConflictError(Exception):
    """Cropped output already exists and overwrite is false."""

    def __init__(self, asset_path: str) -> None:
        self.asset_path = asset_path
        super().__init__(
            f"Cropped GeoTIFF already exists at '{asset_path}'. "
            "Pass overwrite=true to replace it."
        )


class IndexAoiNoIntersectionError(Exception):
    """AOI geometry does not intersect the derived raster."""

    def __init__(self, scene_id: UUID, index_key: str, aoi_id: UUID) -> None:
        self.scene_id = scene_id
        self.index_key = index_key
        self.aoi_id = aoi_id
        super().__init__(
            f"AOI {aoi_id} does not intersect derived index '{index_key}' "
            f"for scene {scene_id}"
        )


class IndexAoiReprojectionError(Exception):
    """AOI geometry could not be reprojected to the raster CRS."""


class IndexAoiCropService:
    """Orchestrate AOI crop of an existing derived index GeoTIFF."""

    def __init__(
        self,
        db: Session | None = None,
        data_root: Path | str | None = None,
    ) -> None:
        self._db = db
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    def crop_by_aoi(
        self,
        scene_id: UUID,
        index_key: str,
        aoi_id: UUID,
        *,
        overwrite: bool = False,
        generate_preview: bool = True,
    ) -> IndexAoiCropResult:
        """Crop a derived index GeoTIFF with a saved AOI polygon."""
        if self._db is None:
            raise RuntimeError("Database session required for crop-by-aoi")

        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        SceneService(self._db).get(scene_id)
        aoi = AoiService(self._db).get(aoi_id)
        geometry = aoi.geometry
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise GeometryValidationError(
                f"AOI {aoi_id} has invalid or empty geometry"
            )

        input_asset = self._storage.build_derived_asset_path(
            scene_id, spec.key, "tif"
        )
        if not self._storage.exists(input_asset):
            raise DerivedGeotiffNotFoundError(scene_id, spec.key, input_asset)

        output_tif = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "tif"
        )
        output_png = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )

        if not overwrite and self._storage.exists(output_tif):
            raise IndexAoiCropConflictError(output_tif)

        input_path = self._storage.resolve_read_path(input_asset)

        try:
            with rasterio.open(input_path) as src:
                if src.crs is None:
                    raise IndexAoiReprojectionError(
                        f"Derived GeoTIFF '{input_asset}' has no CRS; "
                        "cannot reproject AOI for cropping"
                    )

                src_crs = src.crs.to_string()
                nodata = (
                    float(src.nodata)
                    if src.nodata is not None
                    else DEFAULT_INDEX_NODATA
                )

                try:
                    geom_in_raster_crs = transform_geom(
                        _WGS84,
                        src_crs,
                        geometry,
                    )
                except Exception as exc:  # noqa: BLE001 — surface as 422
                    raise IndexAoiReprojectionError(
                        f"Cannot reproject AOI {aoi_id} from {_WGS84} "
                        f"to {src_crs}: {exc}"
                    ) from exc

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
                        raise IndexAoiNoIntersectionError(
                            scene_id, spec.key, aoi_id
                        ) from exc
                    raise IndexAoiReprojectionError(
                        f"Mask failed for AOI {aoi_id} on index '{spec.key}': {exc}"
                    ) from exc

                if (
                    out_image.size == 0
                    or out_image.shape[-1] == 0
                    or out_image.shape[-2] == 0
                ):
                    raise IndexAoiNoIntersectionError(scene_id, spec.key, aoi_id)

                band = np.asarray(out_image[0], dtype=np.float32)
                data_for_stats = band.copy()
                data_for_stats[np.isclose(data_for_stats, nodata)] = np.nan
                crs_str = src_crs
                transform_tuple = tuple(out_transform)[:6]
        except (
            IndexAoiNoIntersectionError,
            IndexAoiReprojectionError,
        ):
            raise
        except (RasterioIOError, OSError) as exc:
            raise RasterReadError(
                f"Cannot read derived GeoTIFF: {input_path}"
            ) from exc

        write_float32_geotiff(
            output_tif,
            self.data_root,
            band,
            crs=crs_str,
            transform=transform_tuple,
            nodata=nodata,
        )

        png_path: str | None = None
        if generate_preview:
            rgba = render_index_preview_rgba(
                data_for_stats,
                spec.key,
                nodata=nodata,
            )
            write_preview_png(output_png, self.data_root, rgba)
            png_path = output_png

        height, width = int(band.shape[0]), int(band.shape[1])
        stats = self._compute_stats(data_for_stats)

        return IndexAoiCropResult(
            scene_id=scene_id,
            index_key=spec.key,
            aoi_id=aoi_id,
            status="cropped",
            raster=IndexAoiCropRasterInfo(
                width=width,
                height=height,
                crs=crs_str,
                dtype="float32",
                nodata=float(nodata),
            ),
            stats=stats,
            output=IndexAoiCropOutputInfo(
                geotiff_asset_path=output_tif,
                png_asset_path=png_path,
            ),
        )

    def resolve_cropped_geotiff(
        self, scene_id: UUID, index_key: str, aoi_id: UUID
    ) -> Path:
        """Return absolute path of an existing AOI-cropped GeoTIFF."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        asset_path = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "tif"
        )
        if not self._storage.exists(asset_path):
            raise CroppedGeotiffNotFoundError(
                scene_id, spec.key, aoi_id, asset_path
            )
        return self._storage.resolve_read_path(asset_path)

    def resolve_cropped_png(
        self, scene_id: UUID, index_key: str, aoi_id: UUID
    ) -> Path:
        """Return absolute path of an existing AOI-cropped preview PNG."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        asset_path = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )
        if not self._storage.exists(asset_path):
            raise CroppedPreviewPngNotFoundError(
                scene_id, spec.key, aoi_id, asset_path
            )
        return self._storage.resolve_read_path(asset_path)

    def get_map_overlay(
        self, scene_id: UUID, index_key: str, aoi_id: UUID
    ) -> IndexAoiCropMapOverlayResult:
        """Return MapLibre overlay metadata for an AOI-cropped index."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        tif_asset = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "tif"
        )
        png_asset = self._storage.build_derived_aoi_asset_path(
            scene_id, aoi_id, spec.key, "png"
        )

        if not self._storage.exists(tif_asset):
            raise CroppedGeotiffNotFoundError(
                scene_id, spec.key, aoi_id, tif_asset
            )
        if not self._storage.exists(png_asset):
            raise CroppedPreviewPngNotFoundError(
                scene_id, spec.key, aoi_id, png_asset
            )

        meta = read_raster_metadata(tif_asset, self.data_root)
        if meta.width is None or meta.height is None:
            raise IndexMapOverlayError(
                f"Cropped GeoTIFF for scene {scene_id} index '{spec.key}' "
                f"AOI {aoi_id} has no width/height"
            )
        if not meta.bounds:
            raise IndexMapOverlayError(
                f"Cropped GeoTIFF for scene {scene_id} index '{spec.key}' "
                f"AOI {aoi_id} has no bounds"
            )
        if not meta.crs:
            raise IndexMapOverlayError(
                f"Cropped GeoTIFF for scene {scene_id} index '{spec.key}' "
                f"AOI {aoi_id} has no CRS; cannot georeference overlay"
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
            f"/api/v1/scenes/{scene_id}/indices/{spec.key}/aois/{aoi_id}/download.png"
        )

        return IndexAoiCropMapOverlayResult(
            scene_id=scene_id,
            index_key=spec.key,
            aoi_id=aoi_id,
            image_url=image_url,
            width=int(meta.width),
            height=int(meta.height),
            crs_original=meta.crs,
            bounds_original=IndexMapOverlayBounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            ),
            coordinates_wgs84=coordinates,
        )

    @staticmethod
    def _compute_stats(array: np.ndarray) -> IndexStats:
        valid_mask = ~np.isnan(array)
        valid_pixels = int(valid_mask.sum())
        nodata_pixels = int(array.size - valid_pixels)

        if valid_pixels == 0:
            return IndexStats(
                min=None,
                max=None,
                mean=None,
                valid_pixels=0,
                nodata_pixels=nodata_pixels,
            )

        valid = array[valid_mask]
        return IndexStats(
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            mean=float(np.mean(valid)),
            valid_pixels=valid_pixels,
            nodata_pixels=nodata_pixels,
        )


__all__ = [
    "IndexAoiCropService",
    "IndexAoiCropConflictError",
    "IndexAoiNoIntersectionError",
    "IndexAoiReprojectionError",
    "AoiNotFoundError",
    "SceneNotFoundError",
    "GeometryValidationError",
    "UnsupportedIndexError",
    "DerivedGeotiffNotFoundError",
    "CroppedGeotiffNotFoundError",
    "CroppedPreviewPngNotFoundError",
    "IndexMapOverlayError",
    "RasterWriteError",
    "RasterReadError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "PreviewWriteError",
]

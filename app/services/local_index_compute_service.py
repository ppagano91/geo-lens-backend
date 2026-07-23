"""Compute spectral indices from local GeoTIFF bands registered on a scene.

Fase 7B: in-memory NDVI only (no write-back, previews, or async jobs).
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.band import RasterBand
from app.raster.formulas import calculate_ndvi
from app.raster.readers import (
    RasterArray,
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
)
from app.repositories.scene_repository import SceneRepository
from app.schemas.index_compute import (
    IndexBandsUsed,
    IndexBandUsed,
    IndexRasterInfo,
    IndexStats,
    NdviComputeResult,
)
from app.services.scene_service import SceneNotFoundError

# Sentinel-2-like keys for NDVI (NIR / Red).
NDVI_RED_BAND_KEY = "B04"
NDVI_NIR_BAND_KEY = "B08"


class MissingRequiredBandError(Exception):
    """Scene is missing a band required by the requested index."""

    def __init__(self, scene_id: UUID, band_key: str) -> None:
        self.scene_id = scene_id
        self.band_key = band_key
        super().__init__(
            f"Scene {scene_id} is missing required band '{band_key}'"
        )


class IncompatibleRasterBandsError(Exception):
    """Required bands exist but are not spatially aligned for index math."""


class LocalIndexComputeService:
    """Orchestrate local band lookup, alignment checks, and index formulas."""

    def __init__(self, db: Session) -> None:
        self.repository = SceneRepository(db)
        self.data_root = settings.data_root_path

    def compute_ndvi(self, scene_id: UUID) -> NdviComputeResult:
        scene = self.repository.get_by_id(scene_id)
        if scene is None:
            raise SceneNotFoundError(str(scene_id))

        bands_by_key = {band.band_key: band for band in scene.bands}
        red_band = self._require_band(scene_id, bands_by_key, NDVI_RED_BAND_KEY)
        nir_band = self._require_band(scene_id, bands_by_key, NDVI_NIR_BAND_KEY)

        red = read_raster_array(red_band.asset_path, self.data_root)
        nir = read_raster_array(nir_band.asset_path, self.data_root)
        self._validate_aligned(red, nir, red_key=NDVI_RED_BAND_KEY, nir_key=NDVI_NIR_BAND_KEY)

        ndvi = calculate_ndvi(nir.data, red.data)
        stats = self._compute_stats(ndvi)

        return NdviComputeResult(
            scene_id=scene_id,
            index="NDVI",
            status="computed",
            bands_used=IndexBandsUsed(
                red=IndexBandUsed(band_key=red_band.band_key, band_id=red_band.id),
                nir=IndexBandUsed(band_key=nir_band.band_key, band_id=nir_band.id),
            ),
            raster=IndexRasterInfo(
                width=red.width,
                height=red.height,
                crs=red.crs,
                dtype="float32",
            ),
            stats=stats,
        )

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
    def _validate_aligned(
        red: RasterArray,
        nir: RasterArray,
        *,
        red_key: str,
        nir_key: str,
    ) -> None:
        if red.count != 1 or nir.count != 1:
            raise IncompatibleRasterBandsError(
                f"Bands {red_key}/{nir_key} must be single-band rasters"
            )
        if red.crs != nir.crs:
            raise IncompatibleRasterBandsError(
                f"Bands {red_key}/{nir_key} have different CRS: {red.crs!r} vs {nir.crs!r}"
            )
        if red.width != nir.width or red.height != nir.height:
            raise IncompatibleRasterBandsError(
                f"Bands {red_key}/{nir_key} have different dimensions: "
                f"{red.width}x{red.height} vs {nir.width}x{nir.height}"
            )
        if red.transform != nir.transform:
            raise IncompatibleRasterBandsError(
                f"Bands {red_key}/{nir_key} have different geotransforms"
            )
        if red.data.shape != nir.data.shape:
            raise IncompatibleRasterBandsError(
                f"Bands {red_key}/{nir_key} have incompatible array shapes: "
                f"{red.data.shape} vs {nir.data.shape}"
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
    "LocalIndexComputeService",
    "MissingRequiredBandError",
    "IncompatibleRasterBandsError",
    "SceneNotFoundError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "NDVI_RED_BAND_KEY",
    "NDVI_NIR_BAND_KEY",
]

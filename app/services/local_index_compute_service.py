"""Compute spectral indices from local GeoTIFF bands registered on a scene.

Fase 7C: generic in-memory compute via an index registry (no write-back,
previews, AOI crop, or async jobs). Fase 7B NDVI remains available as a
thin wrapper over the same path.

Fase 7D: optional persistence of the computed float32 array as a derived
GeoTIFF under DATA_ROOT (CRS/transform preserved; nodata = -9999).

PNG previews of derived GeoTIFFs live in IndexPreviewService (Fase 7E).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.band import RasterBand
from app.raster.formulas import (
    calculate_nbr,
    calculate_ndmi,
    calculate_ndvi,
    calculate_ndwi,
)
from app.raster.readers import (
    RasterArray,
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
)
from app.raster.writers import (
    DEFAULT_INDEX_NODATA,
    RasterWriteError,
    write_float32_geotiff,
)
from app.repositories.scene_repository import SceneRepository
from app.schemas.index_compute import (
    IndexBandUsed,
    IndexComputeResult,
    IndexComputeSaveResult,
    IndexOutputInfo,
    IndexRasterInfo,
    IndexStats,
    NdviComputeResult,
)
from app.services.scene_service import SceneNotFoundError

# Sentinel-2-like keys (kept for Fase 7B imports / tests).
NDVI_RED_BAND_KEY = "B04"
NDVI_NIR_BAND_KEY = "B08"

FormulaFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class LocalIndexSpec:
    """Internal mapping: index_key → required bands + NumPy formula."""

    key: str
    display_name: str
    # Role → Sentinel-like band_key (aligned with spectral_index_definitions).
    required_bands: Mapping[str, str]
    # Argument order expected by ``formula`` (roles from required_bands).
    formula_roles: Sequence[str]
    formula: FormulaFn


LOCAL_INDEX_REGISTRY: dict[str, LocalIndexSpec] = {
    "ndvi": LocalIndexSpec(
        key="ndvi",
        display_name="NDVI",
        required_bands={"red": NDVI_RED_BAND_KEY, "nir": NDVI_NIR_BAND_KEY},
        formula_roles=("nir", "red"),
        formula=calculate_ndvi,
    ),
    "ndwi": LocalIndexSpec(
        key="ndwi",
        display_name="NDWI",
        required_bands={"green": "B03", "nir": "B08"},
        formula_roles=("green", "nir"),
        formula=calculate_ndwi,
    ),
    "nbr": LocalIndexSpec(
        key="nbr",
        display_name="NBR",
        required_bands={"nir": "B08", "swir2": "B12"},
        formula_roles=("nir", "swir2"),
        formula=calculate_nbr,
    ),
    "ndmi": LocalIndexSpec(
        key="ndmi",
        display_name="NDMI",
        required_bands={"nir": "B08", "swir1": "B11"},
        formula_roles=("nir", "swir1"),
        formula=calculate_ndmi,
    ),
}


@dataclass(frozen=True)
class _PreparedIndex:
    """Shared compute payload for in-memory and save paths."""

    spec: LocalIndexSpec
    role_bands: Mapping[str, RasterBand]
    index_array: np.ndarray
    stats: IndexStats
    reference: RasterArray


class UnsupportedIndexError(Exception):
    """Requested index_key is not available for local compute."""

    def __init__(self, index_key: str) -> None:
        self.index_key = index_key
        super().__init__(f"Spectral index '{index_key}' is not supported for local compute")


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
        """Fase 7B compatibility wrapper around :meth:`compute_index`."""
        return self.compute_index(scene_id, "ndvi")

    def compute_index(self, scene_id: UUID, index_key: str) -> IndexComputeResult:
        prepared = self._prepare_index(scene_id, index_key)
        return self._to_compute_result(scene_id, prepared)

    def compute_and_save_index(
        self,
        scene_id: UUID,
        index_key: str,
    ) -> IndexComputeSaveResult:
        """Compute an index and persist it as a float32 GeoTIFF under DATA_ROOT."""
        prepared = self._prepare_index(scene_id, index_key)
        asset_path = self._derived_asset_path(scene_id, prepared.spec.key)
        resolved = write_float32_geotiff(
            asset_path,
            self.data_root,
            prepared.index_array,
            crs=prepared.reference.crs,
            transform=prepared.reference.transform,
            nodata=DEFAULT_INDEX_NODATA,
        )
        base = self._to_compute_result(scene_id, prepared)
        return IndexComputeSaveResult(
            scene_id=base.scene_id,
            index=base.index,
            status="saved",
            bands_used=base.bands_used,
            raster=base.raster,
            stats=base.stats,
            output=IndexOutputInfo(
                asset_path=asset_path,
                resolved_path=str(resolved),
                nodata=DEFAULT_INDEX_NODATA,
            ),
        )

    def _prepare_index(self, scene_id: UUID, index_key: str) -> _PreparedIndex:
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        scene = self.repository.get_by_id(scene_id)
        if scene is None:
            raise SceneNotFoundError(str(scene_id))

        bands_by_key = {band.band_key: band for band in scene.bands}
        role_bands: dict[str, RasterBand] = {}
        for role, band_key in spec.required_bands.items():
            role_bands[role] = self._require_band(scene_id, bands_by_key, band_key)

        role_arrays: dict[str, RasterArray] = {
            role: read_raster_array(band.asset_path, self.data_root)
            for role, band in role_bands.items()
        }
        self._validate_aligned(role_arrays)

        formula_args = [role_arrays[role].data for role in spec.formula_roles]
        index_array = spec.formula(*formula_args)
        stats = self._compute_stats(index_array)
        reference = next(iter(role_arrays.values()))

        return _PreparedIndex(
            spec=spec,
            role_bands=role_bands,
            index_array=index_array,
            stats=stats,
            reference=reference,
        )

    @staticmethod
    def _derived_asset_path(scene_id: UUID, index_key: str) -> str:
        return f"derived/scenes/{scene_id}/{index_key}.tif"

    @staticmethod
    def _to_compute_result(
        scene_id: UUID,
        prepared: _PreparedIndex,
    ) -> IndexComputeResult:
        reference = prepared.reference
        return IndexComputeResult(
            scene_id=scene_id,
            index=prepared.spec.display_name,
            status="computed",
            bands_used={
                role: IndexBandUsed(band_key=band.band_key, band_id=band.id)
                for role, band in prepared.role_bands.items()
            },
            raster=IndexRasterInfo(
                width=reference.width,
                height=reference.height,
                crs=reference.crs,
                dtype="float32",
            ),
            stats=prepared.stats,
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
    def _validate_aligned(role_arrays: Mapping[str, RasterArray]) -> None:
        items = list(role_arrays.items())
        if len(items) < 2:
            raise IncompatibleRasterBandsError(
                "At least two bands are required for spectral index compute"
            )

        ref_key, ref = items[0]
        for other_key, other in items[1:]:
            LocalIndexComputeService._validate_pair(
                ref,
                other,
                left_key=ref_key,
                right_key=other_key,
            )

    @staticmethod
    def _validate_pair(
        left: RasterArray,
        right: RasterArray,
        *,
        left_key: str,
        right_key: str,
    ) -> None:
        label = f"{left_key}/{right_key}"
        if left.count != 1 or right.count != 1:
            raise IncompatibleRasterBandsError(
                f"Bands {label} must be single-band rasters"
            )
        if left.crs != right.crs:
            raise IncompatibleRasterBandsError(
                f"Bands {label} have different CRS: {left.crs!r} vs {right.crs!r}"
            )
        if left.width != right.width or left.height != right.height:
            raise IncompatibleRasterBandsError(
                f"Bands {label} have different dimensions: "
                f"{left.width}x{left.height} vs {right.width}x{right.height}"
            )
        if left.transform != right.transform:
            raise IncompatibleRasterBandsError(
                f"Bands {label} have different geotransforms"
            )
        if left.data.shape != right.data.shape:
            raise IncompatibleRasterBandsError(
                f"Bands {label} have incompatible array shapes: "
                f"{left.data.shape} vs {right.data.shape}"
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
    "LocalIndexSpec",
    "LOCAL_INDEX_REGISTRY",
    "MissingRequiredBandError",
    "IncompatibleRasterBandsError",
    "UnsupportedIndexError",
    "SceneNotFoundError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "RasterWriteError",
    "DEFAULT_INDEX_NODATA",
    "NDVI_RED_BAND_KEY",
    "NDVI_NIR_BAND_KEY",
]

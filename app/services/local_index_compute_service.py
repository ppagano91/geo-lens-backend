"""Compute spectral indices from local GeoTIFF bands registered on a scene.

Fase 7C: generic in-memory compute via an index registry (no write-back,
previews, AOI crop, or async jobs). Fase 7B NDVI remains available as a
thin wrapper over the same path.

Fase 7D: optional persistence of the computed float32 array as a derived
GeoTIFF under DATA_ROOT (CRS/transform preserved; nodata = -9999).

PNG previews of derived GeoTIFFs live in IndexPreviewService (Fase 7E).

Fase 8B: band keys are resolved from the scene sensor (source / metadata.platform)
via role → band maps (Sentinel-2, Landsat 8, synthetic-sentinel-2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

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
from app.raster.sensors import (
    detect_sensor,
    resolve_band_key,
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
from app.services.asset_storage_service import AssetStorageService
from app.services.derived_asset_service import DerivedAssetService
from app.services.radiometry_service import RadiometryMetadata, RadiometryService
from app.services.scene_service import SceneNotFoundError

# Sentinel-2 keys (default sensor; kept for Fase 7B imports / tests).
NDVI_RED_BAND_KEY = "B04"
NDVI_NIR_BAND_KEY = "B08"

FormulaFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class LocalIndexSpec:
    """Internal mapping: index_key → spectral roles + NumPy formula.

    Physical ``band_key`` values are resolved at compute time from the scene
    sensor map. ``required_bands`` remains the Sentinel-2 defaults for catalog
    alignment and backward-compatible imports.
    """

    key: str
    display_name: str
    # Role → Sentinel-2 band_key (default / catalog-aligned).
    required_bands: Mapping[str, str]
    # Argument order expected by ``formula`` (spectral roles).
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
    radiometry: RadiometryMetadata


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

    def __init__(
        self,
        db: Session,
        *,
        data_root: Path | str | None = None,
    ) -> None:
        self.repository = SceneRepository(db)
        self._storage = AssetStorageService(data_root)
        self._radiometry = RadiometryService()

    @property
    def radiometry_service(self) -> RadiometryService:
        """Lazy accessor so unit-test stubs that use ``__new__`` still work."""
        service = getattr(self, "_radiometry", None)
        if service is None:
            service = RadiometryService()
            self._radiometry = service
        return service

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    @data_root.setter
    def data_root(self, value: Path | str) -> None:
        self._storage = AssetStorageService(value)

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
        asset_path = self._storage.build_derived_asset_path(
            scene_id, prepared.spec.key, "tif"
        )
        resolved = write_float32_geotiff(
            asset_path,
            self.data_root,
            prepared.index_array,
            crs=prepared.reference.crs,
            transform=prepared.reference.transform,
            nodata=DEFAULT_INDEX_NODATA,
        )
        base = self._to_compute_result(scene_id, prepared)
        radiometry_meta = prepared.radiometry.as_nested_metadata()
        DerivedAssetService(self.repository.db).create_or_update_derived_asset(
            scene_id=scene_id,
            asset_type="index",
            product_key=prepared.spec.key,
            asset_path=asset_path,
            crs=prepared.reference.crs,
            width=prepared.reference.width,
            height=prepared.reference.height,
            nodata=str(DEFAULT_INDEX_NODATA),
            dtype="float32",
            stats=base.stats.model_dump(),
            metadata={
                "index_display_name": prepared.spec.display_name,
                "bands_used": {
                    role: {"band_key": band.band_key, "band_id": str(band.id)}
                    for role, band in prepared.role_bands.items()
                },
                **radiometry_meta,
            },
        )
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
            radiometry=base.radiometry,
        )

    def _prepare_index(self, scene_id: UUID, index_key: str) -> _PreparedIndex:
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        scene = self.repository.get_by_id(scene_id)
        if scene is None or not scene.is_active:
            raise SceneNotFoundError(str(scene_id))

        sensor = self._detect_scene_sensor(scene)
        bands_by_key = {band.band_key: band for band in scene.bands}
        role_bands: dict[str, RasterBand] = {}
        for role in spec.formula_roles:
            band_key = resolve_band_key(sensor, role)
            role_bands[role] = self._require_band(scene_id, bands_by_key, band_key)

        role_arrays: dict[str, RasterArray] = {
            role: read_raster_array(band.asset_path, self.data_root)
            for role, band in role_bands.items()
        }
        self._validate_aligned(role_arrays)

        radiometry = self.radiometry_service.detect_scene_radiometry(
            scene,
            bands=list(scene.bands),
        )
        scaled_args = [
            self.radiometry_service.apply_radiometric_scaling(
                role_arrays[role].data,
                role_arrays[role].nodata,
                radiometry,
            )
            for role in spec.formula_roles
        ]
        index_array = spec.formula(*scaled_args)
        stats = self._compute_stats(index_array)
        reference = next(iter(role_arrays.values()))

        return _PreparedIndex(
            spec=spec,
            role_bands=role_bands,
            index_array=index_array,
            stats=stats,
            reference=reference,
            radiometry=radiometry,
        )

    @staticmethod
    def _detect_scene_sensor(scene: Any) -> str:
        """Resolve sensor from scene ``source`` / ``metadata`` (Fase 8B)."""
        source = getattr(scene, "source", None)
        metadata = getattr(scene, "metadata_", None)
        if metadata is None:
            metadata = getattr(scene, "metadata", None)
        return detect_sensor(source=source, metadata=metadata)

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
            radiometry=prepared.radiometry.to_info(),
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

"""Resample / align raster bands onto a reference grid (Fase 9L).

Used for Sentinel-2 native 20 m SWIR (B11/B12) → 10 m reference grid
(B08 or B04). Continuous reflectance uses bilinear resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import reproject

from app.services.asset_storage_service import AssetStorageError, AssetStorageService

DEFAULT_RESAMPLING = Resampling.bilinear
RESAMPLING_METHOD_NAME = "bilinear"


class BandAlignmentError(Exception):
    """Band could not be aligned / resampled onto the reference grid."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of writing an aligned GeoTIFF under DATA_ROOT."""

    relative_asset_path: str
    absolute_path: Path
    width: int
    height: int
    crs: str | None
    transform: list[float]
    dtype: str
    nodata: float | None
    source_resolution: tuple[float, float] | None
    target_resolution: tuple[float, float] | None
    resampling_method: str
    reference_band: str
    original_band_key: str
    aligned_band_key: str

    def as_metadata(self) -> dict[str, Any]:
        """Metadata suitable for ``raster_bands.metadata`` JSONB."""
        return {
            "aligned": True,
            "resampled": True,
            "original_band_key": self.original_band_key,
            "aligned_band_key": self.aligned_band_key,
            "source_resolution": (
                list(self.source_resolution) if self.source_resolution else None
            ),
            "target_resolution": (
                list(self.target_resolution) if self.target_resolution else None
            ),
            "resampling_method": self.resampling_method,
            "reference_band": self.reference_band,
            "original_asset_path": None,  # filled by caller when known
        }


class BandAlignmentService:
    """Align a single-band GeoTIFF onto a reference CRS / transform / size."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    def align_to_reference(
        self,
        *,
        source_asset_path: str,
        destination_asset_path: str,
        reference_crs: str | None,
        reference_transform: list[float] | tuple[float, ...],
        reference_width: int,
        reference_height: int,
        original_band_key: str,
        aligned_band_key: str,
        reference_band: str,
        resampling: Resampling = DEFAULT_RESAMPLING,
        resampling_method: str = RESAMPLING_METHOD_NAME,
    ) -> AlignmentResult:
        """Reproject ``source`` onto the reference grid and write destination.

        Preserves source nodata and uses a dtype compatible with the source
        (falls back to float32 when the source dtype is unsupported for write).
        """
        try:
            src_path = self._storage.resolve_read_path(source_asset_path)
            dst_rel = self._storage.validate_relative_asset_path(destination_asset_path)
            dst_path = self._storage.resolve_write_path(dst_rel)
        except AssetStorageError as exc:
            raise BandAlignmentError(str(exc)) from exc

        if not src_path.is_file():
            raise BandAlignmentError(
                f"Source raster not found for alignment: {source_asset_path}"
            )

        if reference_width <= 0 or reference_height <= 0:
            raise BandAlignmentError(
                f"Invalid reference grid size {reference_width}x{reference_height}"
            )

        if len(reference_transform) < 6:
            raise BandAlignmentError(
                "Reference transform must have at least 6 affine coefficients"
            )

        dst_transform = Affine(*[float(v) for v in reference_transform[:6]])

        try:
            with rasterio.open(src_path) as src:
                if src.count != 1:
                    raise BandAlignmentError(
                        f"Alignment source must be single-band; got count={src.count} "
                        f"({source_asset_path})"
                    )

                src_nodata = src.nodata
                src_dtype = str(src.dtypes[0])
                out_dtype = self._output_dtype(src_dtype)
                source_resolution = (
                    (float(src.res[0]), float(src.res[1])) if src.res else None
                )

                destination = np.empty(
                    (reference_height, reference_width),
                    dtype=np.dtype(out_dtype),
                )
                fill = self._fill_value(src_nodata, out_dtype)
                destination[:] = fill

                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=reference_crs,
                    dst_nodata=src_nodata,
                    resampling=resampling,
                )

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                profile = src.profile.copy()
                profile.update(
                    {
                        "driver": "GTiff",
                        "height": reference_height,
                        "width": reference_width,
                        "count": 1,
                        "dtype": out_dtype,
                        "crs": reference_crs,
                        "transform": dst_transform,
                        "nodata": src_nodata,
                    }
                )
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(destination, 1)

                target_resolution = (
                    (float(dst_transform.a), abs(float(dst_transform.e)))
                    if dst_transform
                    else None
                )
        except BandAlignmentError:
            raise
        except (RasterioIOError, OSError, ValueError, TypeError) as exc:
            raise BandAlignmentError(
                f"Failed to resample '{original_band_key}' onto reference "
                f"'{reference_band}': {exc}"
            ) from exc

        return AlignmentResult(
            relative_asset_path=dst_rel,
            absolute_path=dst_path,
            width=reference_width,
            height=reference_height,
            crs=reference_crs,
            transform=[float(v) for v in dst_transform][:6],
            dtype=out_dtype,
            nodata=float(src_nodata) if src_nodata is not None else None,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            resampling_method=resampling_method,
            reference_band=reference_band,
            original_band_key=original_band_key,
            aligned_band_key=aligned_band_key,
        )

    @staticmethod
    def _output_dtype(src_dtype: str) -> str:
        """Keep a reasonable dtype for reflective continuous data."""
        name = (src_dtype or "").lower()
        if name in {
            "uint8",
            "uint16",
            "uint32",
            "int8",
            "int16",
            "int32",
            "float32",
            "float64",
        }:
            # Prefer float32 for float64 sources to keep aligned assets compact.
            if name == "float64":
                return "float32"
            return name
        return "float32"

    @staticmethod
    def _fill_value(nodata: Optional[float], dtype: str) -> Any:
        if nodata is not None:
            return np.dtype(dtype).type(nodata)
        if dtype.startswith("float"):
            return np.dtype(dtype).type(0)
        return np.dtype(dtype).type(0)


__all__ = [
    "AlignmentResult",
    "BandAlignmentError",
    "BandAlignmentService",
    "DEFAULT_RESAMPLING",
    "RESAMPLING_METHOD_NAME",
]

"""Local GeoTIFF writing helpers for derived float32 rasters (index outputs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine

from app.raster.readers import resolve_asset_path

# Sentinel used for derived spectral-index GeoTIFFs (NaN → this value on write).
DEFAULT_INDEX_NODATA = -9999.0


class RasterWriteError(Exception):
    """Raster could not be written to disk."""


def write_float32_geotiff(
    asset_path: str,
    data_root: Path | str,
    data: np.ndarray,
    *,
    crs: str | None,
    transform: tuple[float, float, float, float, float, float],
    nodata: float = DEFAULT_INDEX_NODATA,
) -> Path:
    """Write a single-band float32 GeoTIFF under DATA_ROOT.

    - Relative ``asset_path`` values are resolved against ``data_root``.
    - Parent directories are created when missing.
    - Existing files are overwritten (rasterio ``"w"``).
    - NaN pixels are replaced with ``nodata`` before writing.
    """
    path = resolve_asset_path(asset_path, data_root)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise RasterWriteError(
            f"Expected 2D array for GeoTIFF write; got shape {array.shape}"
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    out = np.where(np.isnan(array), np.float32(nodata), array).astype(np.float32)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=Affine(*transform),
            nodata=float(nodata),
        ) as dataset:
            dataset.write(out, 1)
    except (RasterioIOError, OSError, ValueError, TypeError) as exc:
        raise RasterWriteError(f"Cannot write GeoTIFF: {path}") from exc

    return path


__all__ = [
    "DEFAULT_INDEX_NODATA",
    "RasterWriteError",
    "write_float32_geotiff",
]

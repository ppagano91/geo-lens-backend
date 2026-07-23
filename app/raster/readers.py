"""Local GeoTIFF reading helpers (metadata, samples, and full arrays). No index math."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError


def _ensure_rasterio_proj_data() -> None:
    """Prefer rasterio's bundled PROJ data over a conflicting system PROJ_LIB.

    On Windows, PostgreSQL/PostGIS often sets PROJ_LIB to an older proj.db that
    breaks rasterio CRS lookups (EPSG). Point PROJ_* at rasterio's copy when
    available.
    """
    bundled = Path(rasterio.__file__).resolve().parent / "proj_data"
    if not (bundled / "proj.db").is_file():
        return
    bundled_str = str(bundled)
    current = os.environ.get("PROJ_LIB") or os.environ.get("PROJ_DATA")
    if current:
        current_path = Path(current)
        # Keep an already-compatible rasterio/pyproj data dir.
        if current_path.resolve() == bundled.resolve():
            return
        if (current_path / "proj.db").is_file() and "rasterio" in current_path.as_posix():
            return
    os.environ["PROJ_LIB"] = bundled_str
    os.environ["PROJ_DATA"] = bundled_str


_ensure_rasterio_proj_data()


class RasterPathError(Exception):
    """Invalid or unsafe asset_path resolution."""


class RasterFileNotFoundError(Exception):
    """Resolved raster file does not exist on disk."""


class RasterReadError(Exception):
    """Raster file exists but could not be opened or read."""


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    exists: bool
    driver: str | None
    width: int | None
    height: int | None
    count: int | None
    dtype: str | None
    dtypes: list[str] | None
    crs: str | None
    bounds: dict[str, float] | None
    transform: list[float] | None
    nodata: float | None
    resolution: tuple[float, float] | None
    indexes: list[int] | None
    is_readable: bool


@dataclass(frozen=True)
class RasterSampleStats:
    path: str
    sample_shape: tuple[int, int]
    sample_min: float | None
    sample_max: float | None
    sample_mean: float | None
    sample_has_nan: bool
    valid_count: int


@dataclass(frozen=True)
class RasterArray:
    """Full single-band array plus georeferencing needed for index alignment checks."""

    path: str
    data: np.ndarray
    width: int
    height: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    nodata: float | None
    count: int


def resolve_asset_path(asset_path: str, data_root: Path | str) -> Path:
    """Resolve asset_path against DATA_ROOT with basic traversal protection.

    - Absolute paths are used as-is (resolved).
    - Relative paths are joined to DATA_ROOT and must stay under DATA_ROOT.
    """
    raw = (asset_path or "").strip()
    if not raw:
        raise RasterPathError("asset_path is empty")

    root = Path(data_root).expanduser().resolve()
    candidate = Path(raw).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RasterPathError(
            f"asset_path escapes DATA_ROOT ({root}): {asset_path}"
        ) from exc
    return resolved


def read_raster_metadata(asset_path: str, data_root: Path | str) -> RasterMetadata:
    """Open a local GeoTIFF and return basic metadata (no full array load)."""
    path = resolve_asset_path(asset_path, data_root)
    path_str = str(path)

    if not path.exists() or not path.is_file():
        raise RasterFileNotFoundError(f"Raster file not found: {path_str}")

    try:
        with rasterio.open(path) as dataset:
            crs = dataset.crs.to_string() if dataset.crs else None
            bounds = {
                "left": float(dataset.bounds.left),
                "bottom": float(dataset.bounds.bottom),
                "right": float(dataset.bounds.right),
                "top": float(dataset.bounds.top),
            }
            transform = list(dataset.transform)[:6]
            res = dataset.res
            resolution = (float(res[0]), float(res[1])) if res else None
            dtypes = [str(dt) for dt in dataset.dtypes]
            nodata = _normalize_nodata(dataset.nodata)

            return RasterMetadata(
                path=path_str,
                exists=True,
                driver=dataset.driver,
                width=int(dataset.width),
                height=int(dataset.height),
                count=int(dataset.count),
                dtype=dtypes[0] if dtypes else None,
                dtypes=dtypes,
                crs=crs,
                bounds=bounds,
                transform=transform,
                nodata=nodata,
                resolution=resolution,
                indexes=list(dataset.indexes),
                is_readable=True,
            )
    except RasterFileNotFoundError:
        raise
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterReadError(f"Cannot read raster metadata: {path_str}") from exc


def read_raster_sample(
    asset_path: str,
    data_root: Path | str,
    max_size: int = 256,
) -> RasterSampleStats:
    """Read a downsampled sample of band 1 and return summary stats only."""
    if max_size < 1:
        raise RasterPathError("max_size must be >= 1")

    path = resolve_asset_path(asset_path, data_root)
    path_str = str(path)

    if not path.exists() or not path.is_file():
        raise RasterFileNotFoundError(f"Raster file not found: {path_str}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.count < 1:
                raise RasterReadError(f"Raster has no bands: {path_str}")

            out_height = min(int(dataset.height), max_size)
            out_width = min(int(dataset.width), max_size)
            data = dataset.read(
                1,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
            )
    except RasterFileNotFoundError:
        raise
    except RasterReadError:
        raise
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterReadError(f"Cannot read raster sample: {path_str}") from exc

    array = np.asarray(data, dtype=np.float64)
    has_nan = bool(np.isnan(array).any())
    valid = array[~np.isnan(array)] if has_nan else array
    valid_count = int(valid.size)

    if valid_count == 0:
        return RasterSampleStats(
            path=path_str,
            sample_shape=(int(array.shape[0]), int(array.shape[1])),
            sample_min=None,
            sample_max=None,
            sample_mean=None,
            sample_has_nan=has_nan,
            valid_count=0,
        )

    return RasterSampleStats(
        path=path_str,
        sample_shape=(int(array.shape[0]), int(array.shape[1])),
        sample_min=float(np.min(valid)),
        sample_max=float(np.max(valid)),
        sample_mean=float(np.mean(valid)),
        sample_has_nan=has_nan,
        valid_count=valid_count,
    )


def read_raster_array(asset_path: str, data_root: Path | str) -> RasterArray:
    """Read band 1 of a local GeoTIFF as float32, masking nodata to NaN.

    Requires a single-band raster. Multi-band files raise ``RasterReadError``.
    """
    path = resolve_asset_path(asset_path, data_root)
    path_str = str(path)

    if not path.exists() or not path.is_file():
        raise RasterFileNotFoundError(f"Raster file not found: {path_str}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise RasterReadError(
                    f"Raster must have exactly 1 band; got {dataset.count}: {path_str}"
                )

            raw = dataset.read(1)
            crs = dataset.crs.to_string() if dataset.crs else None
            transform = tuple(float(v) for v in list(dataset.transform)[:6])
            nodata = _normalize_nodata(dataset.nodata)
            data = _to_float32_with_nodata(raw, nodata)

            return RasterArray(
                path=path_str,
                data=data,
                width=int(dataset.width),
                height=int(dataset.height),
                crs=crs,
                transform=transform,
                nodata=nodata,
                count=int(dataset.count),
            )
    except RasterFileNotFoundError:
        raise
    except RasterReadError:
        raise
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterReadError(f"Cannot read raster array: {path_str}") from exc


def _to_float32_with_nodata(raw: np.ndarray, nodata: float | None) -> np.ndarray:
    """Cast to float32 and replace nodata (and existing NaNs) with NaN."""
    data = np.asarray(raw, dtype=np.float32)
    if nodata is not None:
        data = np.where(data == np.float32(nodata), np.float32(np.nan), data)
    return data


def _normalize_nodata(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


__all__ = [
    "RasterPathError",
    "RasterFileNotFoundError",
    "RasterReadError",
    "RasterMetadata",
    "RasterSampleStats",
    "RasterArray",
    "resolve_asset_path",
    "read_raster_metadata",
    "read_raster_sample",
    "read_raster_array",
]

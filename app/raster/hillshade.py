"""Simple 2D hillshade from a single-band elevation array (v0.1-P5).

Horn's algorithm as used by GDAL ``gdaldem hillshade``. Experimental
visualization only — not a terrain / 3D module.
"""

from __future__ import annotations

import numpy as np

DEFAULT_HILLSHADE_AZIMUTH = 315.0
DEFAULT_HILLSHADE_ALTITUDE = 45.0


def compute_hillshade(
    elevation: np.ndarray,
    *,
    x_cellsize: float,
    y_cellsize: float,
    azimuth_deg: float = DEFAULT_HILLSHADE_AZIMUTH,
    altitude_deg: float = DEFAULT_HILLSHADE_ALTITUDE,
) -> np.ndarray:
    """Return hillshade as float32 in [0, 255]; source nodata stays NaN.

    Uses a 3×3 Horn window. NaN/nodata cells are filled only for the slope
    calculation so neighbors still shade; original nodata is restored after.
    """
    array = np.asarray(elevation, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D elevation array; got shape {array.shape}")

    dx = abs(float(x_cellsize))
    dy = abs(float(y_cellsize))
    if dx == 0.0 or dy == 0.0:
        raise ValueError("Hillshade requires non-zero cell size")

    finite = np.isfinite(array)
    fill_value = float(np.nanmean(array)) if np.any(finite) else 0.0
    filled = np.where(finite, array, fill_value)
    padded = np.pad(filled, 1, mode="edge")
    a = padded[0:-2, 0:-2]
    b = padded[0:-2, 1:-1]
    c = padded[0:-2, 2:]
    d = padded[1:-1, 0:-2]
    f = padded[1:-1, 2:]
    g = padded[2:, 0:-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * dy)

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)

    zenith = np.radians(90.0 - float(altitude_deg))
    azimuth = np.radians(float(azimuth_deg))

    shaded = 255.0 * (
        (np.cos(zenith) * np.cos(slope))
        + (np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect))
    )
    shaded = np.clip(shaded, 0.0, 255.0).astype(np.float32)
    shaded[~finite] = np.nan
    return shaded


def hillshade_to_rgba(hillshade: np.ndarray) -> np.ndarray:
    """Map hillshade [0, 255] to grayscale RGBA; NaN pixels are transparent."""
    array = np.asarray(hillshade, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D hillshade array; got shape {array.shape}")

    height, width = array.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    valid = np.isfinite(array)
    if not np.any(valid):
        return rgba

    gray = np.clip(np.rint(array[valid]), 0, 255).astype(np.uint8)
    rgba[valid, 0] = gray
    rgba[valid, 1] = gray
    rgba[valid, 2] = gray
    rgba[valid, 3] = 255
    return rgba


__all__ = [
    "DEFAULT_HILLSHADE_ALTITUDE",
    "DEFAULT_HILLSHADE_AZIMUTH",
    "compute_hillshade",
    "hillshade_to_rgba",
]

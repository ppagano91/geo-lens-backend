"""Pure NumPy formulas for spectral indices.

This module operates only on in-memory arrays. It does not read GeoTIFF files,
use rasterio/rioxarray, or attach geospatial metadata (CRS, transform, bounds).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_divide(
    numerator: Any,
    denominator: Any,
    nodata_value: float = np.nan,
) -> np.ndarray:
    """Divide arrays safely, returning ``nodata_value`` where denominator is 0.

    Inputs are cast to float32. NaN inputs propagate as NaN in the result.
    Division-by-zero warnings are suppressed.
    """
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)

    result = np.full(np.broadcast_shapes(num.shape, den.shape), nodata_value, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=result, where=den != 0)

    return result.astype(np.float32, copy=False)


def normalized_difference(
    band_a: Any,
    band_b: Any,
    nodata_value: float = np.nan,
) -> np.ndarray:
    """Compute ``(band_a - band_b) / (band_a + band_b)`` via :func:`safe_divide`.

    Raises:
        ValueError: If ``band_a`` and ``band_b`` have different shapes.
    """
    a = np.asarray(band_a, dtype=np.float32)
    b = np.asarray(band_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(
            f"Band arrays must have the same shape; got {a.shape} and {b.shape}."
        )

    return safe_divide(a - b, a + b, nodata_value=nodata_value)


def calculate_ndvi(nir: Any, red: Any) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return normalized_difference(nir, red)


def calculate_ndwi(green: Any, nir: Any) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR)."""
    return normalized_difference(green, nir)


def calculate_nbr(nir: Any, swir2: Any) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2)."""
    return normalized_difference(nir, swir2)


def calculate_ndmi(nir: Any, swir1: Any) -> np.ndarray:
    """NDMI = (NIR - SWIR1) / (NIR + SWIR1)."""
    return normalized_difference(nir, swir1)

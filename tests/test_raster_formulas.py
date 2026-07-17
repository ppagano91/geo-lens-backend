"""Unit tests for pure NumPy spectral index formulas (no raster I/O)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.raster.formulas import (
    calculate_nbr,
    calculate_ndmi,
    calculate_ndvi,
    calculate_ndwi,
    normalized_difference,
    safe_divide,
)


def test_safe_divide_normal_denominator() -> None:
    numerator = np.array([6.0, 9.0, 4.0], dtype=np.float32)
    denominator = np.array([2.0, 3.0, 4.0], dtype=np.float32)

    result = safe_divide(numerator, denominator)

    np.testing.assert_allclose(result, [3.0, 3.0, 1.0], rtol=1e-6)
    assert result.dtype == np.float32


def test_safe_divide_zero_denominator() -> None:
    numerator = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    denominator = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    result = safe_divide(numerator, denominator)

    assert result[0] == pytest.approx(1.0)
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(1.5)
    assert result.dtype == np.float32


def test_normalized_difference_simple_arrays() -> None:
    band_a = np.array([0.8, 0.6, 0.2], dtype=np.float32)
    band_b = np.array([0.2, 0.3, 0.2], dtype=np.float32)

    result = normalized_difference(band_a, band_b)
    expected = np.array(
        [
            (0.8 - 0.2) / (0.8 + 0.2),
            (0.6 - 0.3) / (0.6 + 0.3),
            (0.2 - 0.2) / (0.2 + 0.2),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert result.dtype == np.float32


def test_normalized_difference_incompatible_shapes() -> None:
    band_a = np.array([0.8, 0.6], dtype=np.float32)
    band_b = np.array([0.2, 0.3, 0.1], dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        normalized_difference(band_a, band_b)


def test_calculate_ndvi_known_values() -> None:
    nir = np.array([0.8, 0.6, 0.2], dtype=np.float32)
    red = np.array([0.2, 0.3, 0.2], dtype=np.float32)

    result = calculate_ndvi(nir, red)
    expected = np.array(
        [
            (0.8 - 0.2) / (0.8 + 0.2),
            (0.6 - 0.3) / (0.6 + 0.3),
            (0.2 - 0.2) / (0.2 + 0.2),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_calculate_ndwi_known_values() -> None:
    green = np.array([0.4, 0.5, 0.1], dtype=np.float32)
    nir = np.array([0.2, 0.1, 0.3], dtype=np.float32)

    result = calculate_ndwi(green, nir)
    expected = np.array(
        [
            (0.4 - 0.2) / (0.4 + 0.2),
            (0.5 - 0.1) / (0.5 + 0.1),
            (0.1 - 0.3) / (0.1 + 0.3),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_calculate_nbr_known_values() -> None:
    nir = np.array([0.7, 0.5, 0.3], dtype=np.float32)
    swir2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = calculate_nbr(nir, swir2)
    expected = np.array(
        [
            (0.7 - 0.1) / (0.7 + 0.1),
            (0.5 - 0.2) / (0.5 + 0.2),
            (0.3 - 0.3) / (0.3 + 0.3),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_calculate_ndmi_known_values() -> None:
    nir = np.array([0.6, 0.4, 0.2], dtype=np.float32)
    swir1 = np.array([0.2, 0.4, 0.1], dtype=np.float32)

    result = calculate_ndmi(nir, swir1)
    expected = np.array(
        [
            (0.6 - 0.2) / (0.6 + 0.2),
            (0.4 - 0.4) / (0.4 + 0.4),
            (0.2 - 0.1) / (0.2 + 0.1),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_nan_propagation() -> None:
    nir = np.array([0.8, np.nan, 0.2], dtype=np.float32)
    red = np.array([0.2, 0.3, np.nan], dtype=np.float32)

    result = calculate_ndvi(nir, red)

    assert result[0] == pytest.approx(0.6)
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_result_is_numpy_array() -> None:
    nir = [0.8, 0.6, 0.2]
    red = [0.2, 0.3, 0.2]

    result = calculate_ndvi(nir, red)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_no_rasterio_or_file_io() -> None:
    """Formulas module must not import rasterio or open files."""
    import app.raster.formulas as formulas

    # Intentionally do not assert on sys.modules: other modules (readers) may
    # import rasterio in the same pytest session without polluting formulas.
    assert "rasterio" not in formulas.__dict__
    assert "rioxarray" not in formulas.__dict__

    source = inspect.getsource(formulas)
    assert "import rasterio" not in source
    assert "from rasterio" not in source
    assert "import rioxarray" not in source
    assert "from rioxarray" not in source
    assert "open(" not in source
    assert "Path(" not in source

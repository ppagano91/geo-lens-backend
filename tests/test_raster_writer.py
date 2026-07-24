"""Unit tests for float32 GeoTIFF writer (Fase 7D)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.raster.readers import RasterPathError, read_raster_array
from app.raster.writers import (
    DEFAULT_INDEX_NODATA,
    RasterWriteError,
    write_float32_geotiff,
)


def test_write_float32_geotiff_roundtrip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    transform = tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6]
    data = np.array([[0.5, np.nan], [0.25, -0.1]], dtype=np.float32)

    resolved = write_float32_geotiff(
        "derived/scenes/demo/ndvi.tif",
        data_root,
        data,
        crs="EPSG:4326",
        transform=transform,
    )

    assert resolved.is_file()
    assert resolved == (data_root / "derived" / "scenes" / "demo" / "ndvi.tif").resolve()

    with rasterio.open(resolved) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes[0] == "float32"
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.nodata == DEFAULT_INDEX_NODATA
        assert dataset.width == 2
        assert dataset.height == 2
        assert tuple(float(v) for v in list(dataset.transform)[:6]) == transform
        written = dataset.read(1)
        assert written[0, 0] == pytest.approx(0.5)
        assert written[0, 1] == pytest.approx(DEFAULT_INDEX_NODATA)
        assert written[1, 0] == pytest.approx(0.25)
        assert written[1, 1] == pytest.approx(-0.1)

    # Reader masks nodata back to NaN.
    array = read_raster_array("derived/scenes/demo/ndvi.tif", data_root)
    assert np.isnan(array.data[0, 1])
    assert array.data[0, 0] == pytest.approx(0.5)


def test_write_float32_geotiff_overwrites(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    transform = tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6]
    asset = "derived/scenes/x/ndwi.tif"

    write_float32_geotiff(
        asset,
        data_root,
        np.full((2, 2), 1.0, dtype=np.float32),
        crs="EPSG:4326",
        transform=transform,
    )
    write_float32_geotiff(
        asset,
        data_root,
        np.full((2, 2), 0.25, dtype=np.float32),
        crs="EPSG:4326",
        transform=transform,
    )

    with rasterio.open(data_root / asset) as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(0.25)


def test_write_float32_geotiff_rejects_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    transform = tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6]

    with pytest.raises(RasterPathError, match="escapes DATA_ROOT"):
        write_float32_geotiff(
            "../../outside.tif",
            data_root,
            np.zeros((2, 2), dtype=np.float32),
            crs="EPSG:4326",
            transform=transform,
        )


def test_write_float32_geotiff_rejects_non_2d(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    transform = tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6]

    with pytest.raises(RasterWriteError, match="2D"):
        write_float32_geotiff(
            "derived/bad.tif",
            data_root,
            np.zeros((2, 2, 1), dtype=np.float32),
            crs="EPSG:4326",
            transform=transform,
        )

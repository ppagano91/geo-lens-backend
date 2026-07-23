"""Unit tests for local GeoTIFF reading (self-contained temp rasters)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
    read_raster_metadata,
    read_raster_sample,
    resolve_asset_path,
)


def _write_tiny_geotiff(path: Path, *, nodata: float = 0.0) -> None:
    """Create a 5x5 uint16 GeoTIFF (EPSG:4326) for tests."""
    data = np.arange(25, dtype=np.uint16).reshape(5, 5)
    data[0, 0] = 0  # nodata cell
    transform = from_origin(-58.4, -34.6, 0.01, 0.01)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def test_resolve_relative_asset_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    relative = "sample/scenes/test/B04.tif"

    resolved = resolve_asset_path(relative, data_root)

    assert resolved == (data_root / relative).resolve()
    assert resolved.is_relative_to(data_root.resolve())


def test_resolve_absolute_asset_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    absolute = (tmp_path / "outside" / "B04.tif").resolve()

    resolved = resolve_asset_path(str(absolute), data_root)

    assert resolved == absolute


def test_resolve_rejects_path_traversal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    with pytest.raises(RasterPathError, match="escapes DATA_ROOT"):
        resolve_asset_path("../../etc/passwd", data_root)


def test_missing_raster_file_raises_clear_error(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    missing = "sample/missing.tif"

    with pytest.raises(RasterFileNotFoundError, match="not found"):
        read_raster_metadata(missing, data_root)


def test_read_raster_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    scene_dir = data_root / "sample" / "scenes" / "test_scene"
    scene_dir.mkdir(parents=True)
    raster_path = scene_dir / "B04.tif"
    _write_tiny_geotiff(raster_path)

    meta = read_raster_metadata("sample/scenes/test_scene/B04.tif", data_root)

    assert meta.exists is True
    assert meta.is_readable is True
    assert meta.driver == "GTiff"
    assert meta.width == 5
    assert meta.height == 5
    assert meta.count == 1
    assert meta.dtype == "uint16"
    assert meta.dtypes == ["uint16"]
    assert meta.crs == "EPSG:4326"
    assert meta.nodata == 0.0
    assert meta.indexes == [1]
    assert meta.bounds is not None
    assert meta.transform is not None
    assert len(meta.transform) == 6
    assert meta.resolution is not None
    assert Path(meta.path) == raster_path.resolve()


def test_read_raster_sample_stats(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    scene_dir = data_root / "sample" / "scenes" / "test_scene"
    scene_dir.mkdir(parents=True)
    raster_path = scene_dir / "B04.tif"
    _write_tiny_geotiff(raster_path)

    sample = read_raster_sample(
        "sample/scenes/test_scene/B04.tif",
        data_root,
        max_size=256,
    )

    assert sample.sample_shape == (5, 5)
    assert sample.sample_min == 0.0
    assert sample.sample_max == 24.0
    assert sample.sample_mean == pytest.approx(12.0)
    assert sample.valid_count == 25
    assert sample.sample_has_nan is False
    assert Path(sample.path) == raster_path.resolve()


def test_read_raster_array_float32_and_nodata_mask(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    scene_dir = data_root / "sample" / "scenes" / "test_scene"
    scene_dir.mkdir(parents=True)
    raster_path = scene_dir / "B04.tif"
    _write_tiny_geotiff(raster_path, nodata=0.0)

    array = read_raster_array("sample/scenes/test_scene/B04.tif", data_root)

    assert array.width == 5
    assert array.height == 5
    assert array.count == 1
    assert array.crs == "EPSG:4326"
    assert array.nodata == 0.0
    assert array.data.dtype == np.float32
    assert array.data.shape == (5, 5)
    assert np.isnan(array.data[0, 0])
    assert array.data[0, 1] == pytest.approx(1.0)
    assert len(array.transform) == 6
    assert Path(array.path) == raster_path.resolve()


def test_read_raster_array_rejects_multiband(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    path = data_root / "multi.tif"
    data = np.arange(50, dtype=np.uint16).reshape(2, 5, 5)
    transform = from_origin(-58.4, -34.6, 0.01, 0.01)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=2,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)

    with pytest.raises(RasterReadError, match="exactly 1 band"):
        read_raster_array(str(path.resolve()), data_root)

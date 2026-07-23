"""Unit tests for LocalIndexComputeService (no database required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    LocalIndexComputeService,
    MissingRequiredBandError,
    SceneNotFoundError,
)


def _write_band(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float = 0.0,
    crs: str = "EPSG:4326",
    origin: tuple[float, float] = (-58.4, -34.6),
    pixel_size: float = 0.01,
) -> None:
    height, width = data.shape
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype.name,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def _band(band_key: str, asset_path: Path):
    return SimpleNamespace(
        id=uuid4(),
        band_key=band_key,
        asset_path=str(asset_path.resolve()),
    )


def _scene(*, bands: list):
    return SimpleNamespace(id=uuid4(), bands=bands)


class _FakeRepository:
    def __init__(self, scene) -> None:
        self._scene = scene

    def get_by_id(self, scene_id):
        if self._scene is None or self._scene.id != scene_id:
            return None
        return self._scene


def _service_with_scene(scene) -> LocalIndexComputeService:
    service = LocalIndexComputeService.__new__(LocalIndexComputeService)
    service.repository = _FakeRepository(scene)
    service.data_root = Path(".")
    return service


def test_compute_ndvi_success(tmp_path: Path) -> None:
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    red = np.full((5, 5), 100, dtype=np.uint16)
    nir = np.full((5, 5), 300, dtype=np.uint16)
    red[0, 0] = 0
    nir[0, 0] = 0
    _write_band(red_path, red)
    _write_band(nir_path, nir)

    red_band = _band("B04", red_path)
    nir_band = _band("B08", nir_path)
    scene = _scene(bands=[red_band, nir_band])
    service = _service_with_scene(scene)

    result = service.compute_ndvi(scene.id)

    assert result.index == "NDVI"
    assert result.status == "computed"
    assert result.scene_id == scene.id
    assert result.bands_used.red.band_key == "B04"
    assert result.bands_used.red.band_id == red_band.id
    assert result.bands_used.nir.band_key == "B08"
    assert result.bands_used.nir.band_id == nir_band.id
    assert result.raster.width == 5
    assert result.raster.height == 5
    assert result.raster.crs == "EPSG:4326"
    assert result.raster.dtype == "float32"
    assert result.stats.valid_pixels == 24
    assert result.stats.nodata_pixels == 1
    assert result.stats.min == pytest.approx(0.5)
    assert result.stats.max == pytest.approx(0.5)
    assert result.stats.mean == pytest.approx(0.5)


def test_compute_ndvi_missing_b04(tmp_path: Path) -> None:
    nir_path = tmp_path / "B08.tif"
    _write_band(nir_path, np.full((3, 3), 300, dtype=np.uint16))
    scene = _scene(bands=[_band("B08", nir_path)])
    service = _service_with_scene(scene)

    with pytest.raises(MissingRequiredBandError, match="B04"):
        service.compute_ndvi(scene.id)


def test_compute_ndvi_missing_b08(tmp_path: Path) -> None:
    red_path = tmp_path / "B04.tif"
    _write_band(red_path, np.full((3, 3), 100, dtype=np.uint16))
    scene = _scene(bands=[_band("B04", red_path)])
    service = _service_with_scene(scene)

    with pytest.raises(MissingRequiredBandError, match="B08"):
        service.compute_ndvi(scene.id)


def test_compute_ndvi_incompatible_shapes(tmp_path: Path) -> None:
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    _write_band(red_path, np.full((5, 5), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((10, 10), 300, dtype=np.uint16))
    scene = _scene(bands=[_band("B04", red_path), _band("B08", nir_path)])
    service = _service_with_scene(scene)

    with pytest.raises(IncompatibleRasterBandsError, match="dimensions"):
        service.compute_ndvi(scene.id)


def test_compute_ndvi_respects_nodata(tmp_path: Path) -> None:
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    red = np.array([[0, 100], [100, 100]], dtype=np.uint16)
    nir = np.array([[300, 0], [300, 300]], dtype=np.uint16)
    _write_band(red_path, red)
    _write_band(nir_path, nir)
    scene = _scene(bands=[_band("B04", red_path), _band("B08", nir_path)])
    service = _service_with_scene(scene)

    result = service.compute_ndvi(scene.id)

    assert result.stats.nodata_pixels == 2
    assert result.stats.valid_pixels == 2
    assert result.stats.mean == pytest.approx(0.5)


def test_compute_ndvi_scene_not_found() -> None:
    service = _service_with_scene(None)
    missing_id = uuid4()

    with pytest.raises(SceneNotFoundError):
        service.compute_ndvi(missing_id)


def test_compute_ndvi_incompatible_crs(tmp_path: Path) -> None:
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    data = np.full((4, 4), 100, dtype=np.uint16)
    _write_band(red_path, data, crs="EPSG:4326")
    _write_band(nir_path, data, crs="EPSG:3857")
    scene = _scene(bands=[_band("B04", red_path), _band("B08", nir_path)])
    service = _service_with_scene(scene)

    with pytest.raises(IncompatibleRasterBandsError, match="CRS"):
        service.compute_ndvi(scene.id)

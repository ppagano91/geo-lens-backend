"""Unit tests for LocalIndexComputeService (no database required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.raster.writers import DEFAULT_INDEX_NODATA
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    LocalIndexComputeService,
    MissingRequiredBandError,
    SceneNotFoundError,
    UnsupportedIndexError,
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


def _scene(*, bands: list, source: str = "local", metadata=None):
    return SimpleNamespace(
        id=uuid4(),
        bands=bands,
        source=source,
        metadata_=metadata,
    )


class _FakeRepository:
    def __init__(self, scene) -> None:
        self._scene = scene

    def get_by_id(self, scene_id):
        if self._scene is None or self._scene.id != scene_id:
            return None
        return self._scene


def _service_with_scene(scene, *, data_root: Path | None = None) -> LocalIndexComputeService:
    service = LocalIndexComputeService.__new__(LocalIndexComputeService)
    service.repository = _FakeRepository(scene)
    service.data_root = data_root if data_root is not None else Path(".")
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
    assert result.bands_used["red"].band_key == "B04"
    assert result.bands_used["red"].band_id == red_band.id
    assert result.bands_used["nir"].band_key == "B08"
    assert result.bands_used["nir"].band_id == nir_band.id
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


def test_compute_index_ndwi_success(tmp_path: Path) -> None:
    green_path = tmp_path / "B03.tif"
    nir_path = tmp_path / "B08.tif"
    # NDWI = (green - nir) / (green + nir) → (300-100)/(300+100) = 0.5
    _write_band(green_path, np.full((4, 4), 300, dtype=np.uint16))
    _write_band(nir_path, np.full((4, 4), 100, dtype=np.uint16))
    scene = _scene(bands=[_band("B03", green_path), _band("B08", nir_path)])
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, "ndwi")

    assert result.index == "NDWI"
    assert result.bands_used["green"].band_key == "B03"
    assert result.bands_used["nir"].band_key == "B08"
    assert result.stats.mean == pytest.approx(0.5)
    assert result.stats.valid_pixels == 16


def test_compute_index_nbr_success(tmp_path: Path) -> None:
    nir_path = tmp_path / "B08.tif"
    swir2_path = tmp_path / "B12.tif"
    # NBR = (nir - swir2) / (nir + swir2) → (300-100)/(300+100) = 0.5
    _write_band(nir_path, np.full((3, 3), 300, dtype=np.uint16))
    _write_band(swir2_path, np.full((3, 3), 100, dtype=np.uint16))
    scene = _scene(bands=[_band("B08", nir_path), _band("B12", swir2_path)])
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, "NBR")

    assert result.index == "NBR"
    assert result.bands_used["nir"].band_key == "B08"
    assert result.bands_used["swir2"].band_key == "B12"
    assert result.stats.mean == pytest.approx(0.5)


def test_compute_index_ndmi_success(tmp_path: Path) -> None:
    nir_path = tmp_path / "B08.tif"
    swir1_path = tmp_path / "B11.tif"
    # NDMI = (nir - swir1) / (nir + swir1) → (300-100)/(300+100) = 0.5
    _write_band(nir_path, np.full((3, 3), 300, dtype=np.uint16))
    _write_band(swir1_path, np.full((3, 3), 100, dtype=np.uint16))
    scene = _scene(bands=[_band("B08", nir_path), _band("B11", swir1_path)])
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, "ndmi")

    assert result.index == "NDMI"
    assert result.bands_used["swir1"].band_key == "B11"
    assert result.stats.mean == pytest.approx(0.5)


def test_compute_index_unsupported() -> None:
    service = _service_with_scene(_scene(bands=[]))

    with pytest.raises(UnsupportedIndexError, match="evi"):
        service.compute_index(uuid4(), "evi")


def test_compute_index_missing_required_band_ndwi(tmp_path: Path) -> None:
    nir_path = tmp_path / "B08.tif"
    _write_band(nir_path, np.full((3, 3), 100, dtype=np.uint16))
    scene = _scene(bands=[_band("B08", nir_path)])
    service = _service_with_scene(scene)

    with pytest.raises(MissingRequiredBandError, match="B03"):
        service.compute_index(scene.id, "ndwi")


def test_compute_and_save_ndvi_success(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    red = np.full((5, 5), 100, dtype=np.uint16)
    nir = np.full((5, 5), 300, dtype=np.uint16)
    red[0, 0] = 0
    nir[0, 0] = 0
    _write_band(red_path, red)
    _write_band(nir_path, nir)

    scene = _scene(bands=[_band("B04", red_path), _band("B08", nir_path)])
    service = _service_with_scene(scene, data_root=data_root)

    result = service.compute_and_save_index(scene.id, "ndvi")

    expected_asset = f"derived/scenes/{scene.id}/ndvi.tif"
    expected_path = (data_root / expected_asset).resolve()

    assert result.index == "NDVI"
    assert result.status == "saved"
    assert result.scene_id == scene.id
    assert result.bands_used["red"].band_key == "B04"
    assert result.bands_used["nir"].band_key == "B08"
    assert result.raster.width == 5
    assert result.raster.height == 5
    assert result.raster.crs == "EPSG:4326"
    assert result.raster.dtype == "float32"
    assert result.stats.valid_pixels == 24
    assert result.stats.nodata_pixels == 1
    assert result.stats.mean == pytest.approx(0.5)
    assert result.output.asset_path == expected_asset
    assert result.output.resolved_path == str(expected_path)
    assert result.output.nodata == DEFAULT_INDEX_NODATA
    assert expected_path.is_file()

    with rasterio.open(expected_path) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes[0] == "float32"
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.width == 5
        assert dataset.height == 5
        assert dataset.nodata == DEFAULT_INDEX_NODATA
        band = dataset.read(1)
        assert band[0, 0] == pytest.approx(DEFAULT_INDEX_NODATA)
        assert band[1, 1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("index_key", "display", "band_keys", "roles"),
    [
        ("ndwi", "NDWI", ("B03", "B08"), ("green", "nir")),
        ("nbr", "NBR", ("B08", "B12"), ("nir", "swir2")),
        ("ndmi", "NDMI", ("B08", "B11"), ("nir", "swir1")),
    ],
)
def test_compute_and_save_other_indices(
    tmp_path: Path,
    index_key: str,
    display: str,
    band_keys: tuple[str, str],
    roles: tuple[str, str],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    left_path = tmp_path / f"{band_keys[0]}.tif"
    right_path = tmp_path / f"{band_keys[1]}.tif"
    # First formula arg high / second low → normalized difference 0.5
    # NDWI uses (green, nir) with green high; NBR/NDMI use (nir, swir*) with nir high.
    if index_key == "ndwi":
        _write_band(left_path, np.full((3, 3), 300, dtype=np.uint16))
        _write_band(right_path, np.full((3, 3), 100, dtype=np.uint16))
    else:
        _write_band(left_path, np.full((3, 3), 300, dtype=np.uint16))
        _write_band(right_path, np.full((3, 3), 100, dtype=np.uint16))

    scene = _scene(
        bands=[_band(band_keys[0], left_path), _band(band_keys[1], right_path)]
    )
    service = _service_with_scene(scene, data_root=data_root)

    result = service.compute_and_save_index(scene.id, index_key)

    out_path = Path(result.output.resolved_path)
    assert result.index == display
    assert result.status == "saved"
    assert result.bands_used[roles[0]].band_key == band_keys[0]
    assert result.bands_used[roles[1]].band_key == band_keys[1]
    assert result.stats.mean == pytest.approx(0.5)
    assert result.output.asset_path == f"derived/scenes/{scene.id}/{index_key}.tif"
    assert result.output.nodata == DEFAULT_INDEX_NODATA
    assert out_path.is_file()

    with rasterio.open(out_path) as dataset:
        assert dataset.nodata == DEFAULT_INDEX_NODATA
        assert dataset.dtypes[0] == "float32"
        assert dataset.read(1)[0, 0] == pytest.approx(0.5)


def test_compute_and_save_overwrites_existing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    _write_band(red_path, np.full((2, 2), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))
    scene = _scene(bands=[_band("B04", red_path), _band("B08", nir_path)])
    service = _service_with_scene(scene, data_root=data_root)

    first = service.compute_and_save_index(scene.id, "ndvi")
    second = service.compute_and_save_index(scene.id, "ndvi")

    assert Path(first.output.resolved_path) == Path(second.output.resolved_path)
    assert Path(second.output.resolved_path).is_file()
    assert second.status == "saved"


def test_compute_and_save_unsupported_index(tmp_path: Path) -> None:
    service = _service_with_scene(_scene(bands=[]), data_root=tmp_path)

    with pytest.raises(UnsupportedIndexError, match="evi"):
        service.compute_and_save_index(uuid4(), "evi")


def test_compute_and_save_scene_not_found(tmp_path: Path) -> None:
    service = _service_with_scene(None, data_root=tmp_path)

    with pytest.raises(SceneNotFoundError):
        service.compute_and_save_index(uuid4(), "ndvi")

"""Tests for Fase 7D: compute spectral indices and persist as GeoTIFF."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.config import settings
from app.raster.writers import DEFAULT_INDEX_NODATA
from tests.conftest import requires_database

SCENE_FOOTPRINT = {
    "type": "Polygon",
    "coordinates": [
        [
            [-58.50, -34.50],
            [-58.20, -34.50],
            [-58.20, -34.80],
            [-58.50, -34.80],
            [-58.50, -34.50],
        ]
    ],
}


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


def _scene_payload(*, bands: list[dict], name: str = "index save test") -> dict:
    return {
        "name": name,
        "source": "local",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"purpose": "fase_7d"},
        "bands": bands,
    }


def _band_entry(band_key: str, path: Path, *, band_name: str | None = None) -> dict:
    return {
        "band_key": band_key,
        "band_name": band_name or band_key,
        "asset_path": str(path.resolve()),
        "nodata": "0",
        "dtype": "uint16",
    }


def _create_scene(client, payload: dict) -> str:
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201, create.text
    return create.json()["id"]


@requires_database
def test_compute_and_save_ndvi_success(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_dir = tmp_path / "bands"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"
    red = np.full((5, 5), 100, dtype=np.uint16)
    nir = np.full((5, 5), 300, dtype=np.uint16)
    red[0, 0] = 0
    nir[0, 0] = 0
    _write_band(red_path, red)
    _write_band(nir_path, nir)

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B04", red_path, band_name="Red"),
                _band_entry("B08", nir_path, band_name="NIR"),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save")

    assert response.status_code == 200
    data = response.json()
    expected_asset = f"derived/scenes/{scene_id}/ndvi.tif"
    expected_path = (data_root / expected_asset).resolve()

    assert data["index"] == "NDVI"
    assert data["status"] == "saved"
    assert data["bands_used"]["red"]["band_key"] == "B04"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["raster"]["width"] == 5
    assert data["raster"]["height"] == 5
    assert data["raster"]["crs"] == "EPSG:4326"
    assert data["raster"]["dtype"] == "float32"
    assert data["stats"]["valid_pixels"] == 24
    assert data["stats"]["nodata_pixels"] == 1
    assert data["stats"]["mean"] == pytest.approx(0.5)
    assert data["output"]["asset_path"] == expected_asset
    assert data["output"]["resolved_path"] == str(expected_path)
    assert data["output"]["nodata"] == DEFAULT_INDEX_NODATA
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


@requires_database
@pytest.mark.parametrize(
    ("index_key", "display", "bands"),
    [
        ("ndwi", "NDWI", (("B03", 300), ("B08", 100))),
        ("nbr", "NBR", (("B08", 300), ("B12", 100))),
        ("ndmi", "NDMI", (("B08", 300), ("B11", 100))),
    ],
)
def test_compute_and_save_other_indices(
    client,
    tmp_path: Path,
    monkeypatch,
    index_key: str,
    display: str,
    bands: tuple[tuple[str, int], tuple[str, int]],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_dir = tmp_path / index_key
    scene_dir.mkdir()
    band_entries = []
    for band_key, value in bands:
        path = scene_dir / f"{band_key}.tif"
        _write_band(path, np.full((3, 3), value, dtype=np.uint16))
        band_entries.append(_band_entry(band_key, path))

    scene_id = _create_scene(client, _scene_payload(bands=band_entries, name=index_key))

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/{index_key}/compute-and-save"
    )

    assert response.status_code == 200
    data = response.json()
    out_path = Path(data["output"]["resolved_path"])

    assert data["index"] == display
    assert data["status"] == "saved"
    assert data["stats"]["mean"] == pytest.approx(0.5)
    assert data["output"]["asset_path"] == f"derived/scenes/{scene_id}/{index_key}.tif"
    assert data["output"]["nodata"] == DEFAULT_INDEX_NODATA
    assert out_path.is_file()

    with rasterio.open(out_path) as dataset:
        assert dataset.nodata == DEFAULT_INDEX_NODATA
        assert dataset.dtypes[0] == "float32"
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.read(1)[0, 0] == pytest.approx(0.5)


@requires_database
def test_compute_and_save_unsupported_index(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_dir = tmp_path / "unsupported"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"
    _write_band(red_path, np.full((2, 2), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B04", red_path),
                _band_entry("B08", nir_path),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/evi/compute-and-save")

    assert response.status_code == 404
    assert "evi" in response.json()["detail"].lower()


@requires_database
def test_compute_and_save_scene_not_found(client, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    missing_id = uuid4()

    response = client.post(
        f"/api/v1/scenes/{missing_id}/indices/ndvi/compute-and-save"
    )

    assert response.status_code == 404
    assert str(missing_id) in response.json()["detail"]


@requires_database
def test_in_memory_compute_unchanged(client, tmp_path: Path, monkeypatch) -> None:
    """Fase 7C compute endpoint must not write GeoTIFF files."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_dir = tmp_path / "memory"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"
    _write_band(red_path, np.full((2, 2), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B04", red_path),
                _band_entry("B08", nir_path),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 200
    assert response.json()["status"] == "computed"
    assert "output" not in response.json()
    assert not (data_root / "derived" / "scenes" / scene_id / "ndvi.tif").exists()

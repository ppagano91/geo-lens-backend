"""Tests for Fase 7C: generic local spectral index compute."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

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


def _scene_payload(*, bands: list[dict], name: str = "index compute test") -> dict:
    return {
        "name": name,
        "source": "local",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"purpose": "fase_7c"},
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
def test_generic_compute_ndvi_success(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "ndvi"
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

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "NDVI"
    assert data["status"] == "computed"
    assert data["bands_used"]["red"]["band_key"] == "B04"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["stats"]["valid_pixels"] == 24
    assert data["stats"]["mean"] == pytest.approx(0.5)


@requires_database
def test_generic_compute_ndwi_success(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "ndwi"
    scene_dir.mkdir()
    green_path = scene_dir / "B03.tif"
    nir_path = scene_dir / "B08.tif"
    # (300 - 100) / (300 + 100) = 0.5
    _write_band(green_path, np.full((4, 4), 300, dtype=np.uint16))
    _write_band(nir_path, np.full((4, 4), 100, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B03", green_path, band_name="Green"),
                _band_entry("B08", nir_path, band_name="NIR"),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndwi/compute")

    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "NDWI"
    assert data["bands_used"]["green"]["band_key"] == "B03"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["raster"]["width"] == 4
    assert data["stats"]["mean"] == pytest.approx(0.5)
    assert data["stats"]["valid_pixels"] == 16


@requires_database
def test_generic_compute_nbr_success(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "nbr"
    scene_dir.mkdir()
    nir_path = scene_dir / "B08.tif"
    swir2_path = scene_dir / "B12.tif"
    _write_band(nir_path, np.full((3, 3), 300, dtype=np.uint16))
    _write_band(swir2_path, np.full((3, 3), 100, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B08", nir_path),
                _band_entry("B12", swir2_path, band_name="SWIR2"),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/nbr/compute")

    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "NBR"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["bands_used"]["swir2"]["band_key"] == "B12"
    assert data["stats"]["mean"] == pytest.approx(0.5)


@requires_database
def test_generic_compute_ndmi_success(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "ndmi"
    scene_dir.mkdir()
    nir_path = scene_dir / "B08.tif"
    swir1_path = scene_dir / "B11.tif"
    _write_band(nir_path, np.full((3, 3), 300, dtype=np.uint16))
    _write_band(swir1_path, np.full((3, 3), 100, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[
                _band_entry("B08", nir_path),
                _band_entry("B11", swir1_path, band_name="SWIR1"),
            ]
        ),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndmi/compute")

    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "NDMI"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["bands_used"]["swir1"]["band_key"] == "B11"
    assert data["stats"]["mean"] == pytest.approx(0.5)


@requires_database
def test_generic_compute_unsupported_index(client, tmp_path: Path) -> None:
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

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/evi/compute")

    assert response.status_code == 404
    assert "evi" in response.json()["detail"].lower()


@requires_database
def test_generic_compute_missing_required_band(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "missing"
    scene_dir.mkdir()
    nir_path = scene_dir / "B08.tif"
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))

    scene_id = _create_scene(
        client,
        _scene_payload(bands=[_band_entry("B08", nir_path)]),
    )

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndwi/compute")

    assert response.status_code == 422
    assert "B03" in response.json()["detail"]


@requires_database
def test_legacy_ndvi_endpoint_compatibility(client, tmp_path: Path) -> None:
    """Fase 7B path remains a working alias of the generic compute."""
    scene_dir = tmp_path / "legacy"
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
            ],
            name="legacy NDVI alias",
        ),
    )

    legacy = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")
    generic = client.post(f"/api/v1/scenes/{scene_id}/indices/NDVI/compute")

    assert legacy.status_code == 200
    assert generic.status_code == 200
    assert legacy.json()["index"] == "NDVI"
    assert generic.json()["index"] == "NDVI"
    assert legacy.json()["stats"] == generic.json()["stats"]
    assert legacy.json()["bands_used"]["red"]["band_key"] == "B04"
    assert legacy.json()["bands_used"]["nir"]["band_key"] == "B08"


@requires_database
def test_generic_compute_scene_not_found(client) -> None:
    missing_id = uuid4()

    response = client.post(f"/api/v1/scenes/{missing_id}/indices/ndvi/compute")

    assert response.status_code == 404
    assert str(missing_id) in response.json()["detail"]

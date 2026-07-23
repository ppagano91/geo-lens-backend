"""Tests for Fase 7B: local NDVI compute from registered scene bands."""

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


def _ndvi_scene_payload(
    *,
    red_path: str,
    nir_path: str,
    include_red: bool = True,
    include_nir: bool = True,
) -> dict:
    bands: list[dict] = []
    if include_red:
        bands.append(
            {
                "band_key": "B04",
                "band_name": "Red",
                "asset_path": red_path,
                "nodata": "0",
                "dtype": "uint16",
            }
        )
    if include_nir:
        bands.append(
            {
                "band_key": "B08",
                "band_name": "NIR",
                "asset_path": nir_path,
                "nodata": "0",
                "dtype": "uint16",
            }
        )
    return {
        "name": "NDVI compute test scene",
        "source": "local",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"purpose": "fase_7b"},
        "bands": bands,
    }


def _aligned_red_nir(tmp_path: Path) -> tuple[Path, Path]:
    """Create 5x5 B04/B08 with known NDVI and one shared nodata cell."""
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"

    # nodata=0 at [0,0]; elsewhere red=100, nir=300 → NDVI = 0.5
    red = np.full((5, 5), 100, dtype=np.uint16)
    nir = np.full((5, 5), 300, dtype=np.uint16)
    red[0, 0] = 0
    nir[0, 0] = 0
    # Extra nodata only on red → still nodata in NDVI
    red[1, 1] = 0
    # Extra nodata only on nir
    nir[2, 2] = 0

    _write_band(red_path, red, nodata=0.0)
    _write_band(nir_path, nir, nodata=0.0)
    return red_path, nir_path


@requires_database
def test_compute_ndvi_success(client, tmp_path: Path) -> None:
    red_path, nir_path = _aligned_red_nir(tmp_path)
    payload = _ndvi_scene_payload(
        red_path=str(red_path.resolve()),
        nir_path=str(nir_path.resolve()),
    )
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201
    scene_id = create.json()["id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == scene_id
    assert data["index"] == "NDVI"
    assert data["status"] == "computed"
    assert data["bands_used"]["red"]["band_key"] == "B04"
    assert data["bands_used"]["nir"]["band_key"] == "B08"
    assert data["raster"]["width"] == 5
    assert data["raster"]["height"] == 5
    assert data["raster"]["crs"] == "EPSG:4326"
    assert data["raster"]["dtype"] == "float32"
    # 25 - 3 nodata cells (shared + red-only + nir-only)
    assert data["stats"]["valid_pixels"] == 22
    assert data["stats"]["nodata_pixels"] == 3
    assert data["stats"]["min"] == pytest.approx(0.5)
    assert data["stats"]["max"] == pytest.approx(0.5)
    assert data["stats"]["mean"] == pytest.approx(0.5)


@requires_database
def test_compute_ndvi_missing_b04(client, tmp_path: Path) -> None:
    _, nir_path = _aligned_red_nir(tmp_path)
    payload = _ndvi_scene_payload(
        red_path="unused",
        nir_path=str(nir_path.resolve()),
        include_red=False,
    )
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201
    scene_id = create.json()["id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 422
    assert "B04" in response.json()["detail"]


@requires_database
def test_compute_ndvi_missing_b08(client, tmp_path: Path) -> None:
    red_path, _ = _aligned_red_nir(tmp_path)
    payload = _ndvi_scene_payload(
        red_path=str(red_path.resolve()),
        nir_path="unused",
        include_nir=False,
    )
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201
    scene_id = create.json()["id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 422
    assert "B08" in response.json()["detail"]


@requires_database
def test_compute_ndvi_incompatible_shapes(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "mismatch"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"
    _write_band(red_path, np.full((5, 5), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((10, 10), 300, dtype=np.uint16))

    payload = _ndvi_scene_payload(
        red_path=str(red_path.resolve()),
        nir_path=str(nir_path.resolve()),
    )
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201
    scene_id = create.json()["id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "dimension" in detail or "shape" in detail or "different" in detail


@requires_database
def test_compute_ndvi_respects_nodata(client, tmp_path: Path) -> None:
    scene_dir = tmp_path / "nodata"
    scene_dir.mkdir()
    red_path = scene_dir / "B04.tif"
    nir_path = scene_dir / "B08.tif"

    red = np.array([[0, 100], [100, 100]], dtype=np.uint16)
    nir = np.array([[300, 0], [300, 300]], dtype=np.uint16)
    _write_band(red_path, red, nodata=0.0)
    _write_band(nir_path, nir, nodata=0.0)

    payload = _ndvi_scene_payload(
        red_path=str(red_path.resolve()),
        nir_path=str(nir_path.resolve()),
    )
    create = client.post("/api/v1/scenes", json=payload)
    scene_id = create.json()["id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")

    assert response.status_code == 200
    stats = response.json()["stats"]
    # [0,0] red nodata, [0,1] nir nodata → 2 nodata; 2 valid at 0.5
    assert stats["nodata_pixels"] == 2
    assert stats["valid_pixels"] == 2
    assert stats["mean"] == pytest.approx(0.5)


@requires_database
def test_compute_ndvi_scene_not_found(client) -> None:
    missing_id = uuid4()

    response = client.post(f"/api/v1/scenes/{missing_id}/indices/ndvi/compute")

    assert response.status_code == 404
    assert str(missing_id) in response.json()["detail"]

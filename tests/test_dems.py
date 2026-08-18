"""Tests for experimental DEM upload / hillshade / map overlay (v0.1-P5)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.core.config import settings
from app.raster.hillshade import compute_hillshade, hillshade_to_rgba
from tests.conftest import requires_database

DEMS_URL = "/api/v1/dems"


def _dem_bytes(
    data: np.ndarray,
    *,
    nodata: float | None = -9999.0,
    crs: str | None = "EPSG:4326",
    origin: tuple[float, float] = (-58.45, -34.55),
    pixel_size: float = 0.001,
    count: int = 1,
) -> bytes:
    height, width = data.shape
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=data.dtype.name,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dataset:
            for band in range(1, count + 1):
                dataset.write(data, band)
        return memfile.read()


def _upload_files(
    data: np.ndarray | None = None,
    *,
    filename: str = "relief.tif",
    name: str | None = "Cerro test",
    **kwargs,
) -> dict:
    array = (
        data
        if data is not None
        else np.array(
            [
                [10.0, 12.0, 14.0, 16.0],
                [11.0, 13.0, 15.0, 18.0],
                [12.0, 14.0, 20.0, 22.0],
                [13.0, 16.0, 24.0, 30.0],
            ],
            dtype=np.float32,
        )
    )
    files = {"file": (filename, _dem_bytes(array, **kwargs), "image/tiff")}
    form = {}
    if name is not None:
        form["name"] = name
    return {"files": files, "data": form}


def _assert_valid_coordinates(coords: list) -> None:
    assert len(coords) == 4
    for corner in coords:
        assert len(corner) == 2
        lng, lat = corner
        assert -180.0 <= float(lng) <= 180.0
        assert -90.0 <= float(lat) <= 90.0
    tl, tr, br, bl = coords
    assert tl[1] >= bl[1]
    assert tr[1] >= br[1]
    assert tl[0] <= tr[0]
    assert bl[0] <= br[0]


def test_compute_hillshade_masks_nodata() -> None:
    elevation = np.array(
        [
            [10.0, 12.0, 14.0],
            [11.0, np.nan, 15.0],
            [12.0, 14.0, 20.0],
        ],
        dtype=np.float32,
    )
    shaded = compute_hillshade(elevation, x_cellsize=30.0, y_cellsize=30.0)
    assert np.isnan(shaded[1, 1])
    assert np.isfinite(shaded[0, 0])
    rgba = hillshade_to_rgba(shaded)
    assert rgba[1, 1, 3] == 0
    assert rgba[0, 0, 3] == 255


@requires_database
def test_upload_valid_dem(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    payload = _upload_files()
    response = client.post(f"{DEMS_URL}/upload", **payload)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "Cerro test"
    assert body["crs"] == "EPSG:4326"
    assert body["width"] == 4
    assert body["height"] == 4
    assert body["min_elevation"] == pytest.approx(10.0)
    assert body["max_elevation"] == pytest.approx(30.0)
    assert body["preview_path"] is None
    assert body["metadata"]["count"] == 1
    assert body["metadata"]["nodata"] == pytest.approx(-9999.0)
    assert (data_root / body["asset_path"]).is_file()

    listed = client.get(DEMS_URL)
    assert listed.status_code == 200
    assert any(item["id"] == body["id"] for item in listed.json())

    detail = client.get(f"{DEMS_URL}/{body['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == body["id"]


@requires_database
def test_reject_multiband_raster(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    data = np.full((4, 4), 12.0, dtype=np.float32)
    payload = _upload_files(data, count=3)
    response = client.post(f"{DEMS_URL}/upload", **payload)

    assert response.status_code == 422
    assert "single-band" in response.json()["detail"].lower()
    assert not any(data_root.rglob("*.tif"))


@requires_database
def test_reject_raster_without_crs(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    data = np.full((4, 4), 12.0, dtype=np.float32)
    payload = _upload_files(data, crs=None)
    response = client.post(f"{DEMS_URL}/upload", **payload)

    assert response.status_code == 422
    assert "crs" in response.json()["detail"].lower()


@requires_database
def test_generate_hillshade_and_overlay(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    created = client.post(f"{DEMS_URL}/upload", **_upload_files()).json()
    dem_id = created["id"]

    hillshade = client.post(f"{DEMS_URL}/{dem_id}/hillshade")
    assert hillshade.status_code == 200, hillshade.text
    body = hillshade.json()
    assert body["dem_id"] == dem_id
    assert body["status"] == "hillshade_created"
    assert body["nodata_transparent"] is True
    assert (data_root / body["preview_path"]).is_file()

    png = client.get(f"{DEMS_URL}/{dem_id}/hillshade.png")
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")

    overlay = client.get(f"{DEMS_URL}/{dem_id}/map-overlay")
    assert overlay.status_code == 200, overlay.text
    overlay_body = overlay.json()
    assert overlay_body["dem_id"] == dem_id
    assert overlay_body["image_url"] == f"/api/v1/dems/{dem_id}/hillshade.png"
    assert overlay_body["crs_original"] == "EPSG:4326"
    assert overlay_body["width"] == 4
    assert overlay_body["height"] == 4
    _assert_valid_coordinates(overlay_body["coordinates_wgs84"])
    tl, tr, br, bl = overlay_body["coordinates_wgs84"]
    assert tl == pytest.approx([-58.45, -34.55])
    assert tr == pytest.approx([-58.45 + 0.004, -34.55])
    assert br == pytest.approx([-58.45 + 0.004, -34.55 - 0.004])
    assert bl == pytest.approx([-58.45, -34.55 - 0.004])


@requires_database
def test_hillshade_nodata_is_transparent(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    data = np.linspace(10.0, 40.0, 64, dtype=np.float32).reshape(8, 8)
    data[3, 3] = -9999.0
    created = client.post(
        f"{DEMS_URL}/upload",
        **_upload_files(data, nodata=-9999.0),
    ).json()
    dem_id = created["id"]

    response = client.post(f"{DEMS_URL}/{dem_id}/hillshade")
    assert response.status_code == 200, response.text

    png = client.get(f"{DEMS_URL}/{dem_id}/hillshade.png")
    assert png.status_code == 200
    image = Image.open(BytesIO(png.content)).convert("RGBA")
    pixels = np.array(image)
    assert pixels.shape == (8, 8, 4)
    assert pixels[3, 3, 3] == 0
    assert int(pixels[:, :, 3].max()) == 255


@requires_database
def test_map_overlay_requires_hillshade(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    created = client.post(f"{DEMS_URL}/upload", **_upload_files()).json()
    dem_id = created["id"]

    overlay = client.get(f"{DEMS_URL}/{dem_id}/map-overlay")
    assert overlay.status_code == 404

    png = client.get(f"{DEMS_URL}/{dem_id}/hillshade.png")
    assert png.status_code == 404


@requires_database
def test_get_unknown_dem_returns_404(client) -> None:
    response = client.get(f"{DEMS_URL}/{uuid4()}")
    assert response.status_code == 404

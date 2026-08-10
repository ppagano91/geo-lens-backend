"""Tests for Fase 9H.1: RGB composites cropped by AOI."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.config import settings
from tests.conftest import VALID_POLYGON, requires_database

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

# Outside the sample raster footprint around -58.4 / -34.6
NON_INTERSECTING_AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [-50.0, -20.0],
            [-49.9, -20.0],
            [-49.9, -20.1],
            [-50.0, -20.1],
            [-50.0, -20.0],
        ]
    ],
}

L8_BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")


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


def _write_l8_stack(root: Path, *, shape: tuple[int, int] = (10, 10)) -> dict[str, Path]:
    defaults = {
        "SR_B2": 80,
        "SR_B3": 100,
        "SR_B4": 120,
        "SR_B5": 200,
        "SR_B6": 150,
        "SR_B7": 90,
    }
    paths: dict[str, Path] = {}
    for key in L8_BANDS:
        path = root / f"{key}.tif"
        _write_band(path, np.full(shape, defaults[key], dtype=np.uint16))
        paths[key] = path
    return paths


def _band_entry(band_key: str, path: Path) -> dict:
    return {
        "band_key": band_key,
        "band_name": band_key,
        "asset_path": str(path.resolve()),
        "nodata": "0",
        "dtype": "uint16",
    }


def _scene_payload(*, bands: list[dict]) -> dict:
    return {
        "name": "RGB AOI composite test",
        "source": "landsat-8",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"platform": "Landsat-8", "purpose": "fase_9h1"},
        "bands": bands,
    }


def _create_scene(client, bands: list[dict]) -> str:
    resp = client.post("/api/v1/scenes", json=_scene_payload(bands=bands))
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _create_aoi(client, geometry: dict | None = None) -> str:
    resp = client.post(
        "/api/v1/aois",
        json={"name": "RGB AOI test", "geometry": geometry or VALID_POLYGON},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


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


@requires_database
def test_true_color_by_aoi_landsat_8(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={
            "aoi_id": aoi_id,
            "preset": "true_color",
            "stretch": "percentile",
            "p_min": 2,
            "p_max": 98,
            "overwrite": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scene_id"] == scene_id
    assert body["aoi_id"] == aoi_id
    assert body["preset"] == "true_color"
    assert body["status"] == "generated"
    assert body["sensor"] == "landsat-8"
    assert body["bands_used"] == {
        "red": "SR_B4",
        "green": "SR_B3",
        "blue": "SR_B2",
    }
    assert body["width"] > 0
    assert body["height"] > 0
    expected = f"derived/scenes/{scene_id}/aois/{aoi_id}/rgb/true_color.png"
    assert body["output"]["asset_path"] == expected
    assert (data_root / expected).is_file()
    assert (data_root / expected.replace(".png", ".georef.json")).is_file()


@requires_database
def test_false_color_vegetation_by_aoi(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "false_color_vegetation", "overwrite": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bands_used"] == {
        "red": "SR_B5",
        "green": "SR_B4",
        "blue": "SR_B3",
    }
    assert (
        body["output"]["asset_path"]
        == f"derived/scenes/{scene_id}/aois/{aoi_id}/rgb/false_color_vegetation.png"
    )


@requires_database
def test_aoi_no_intersection_returns_422(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client, NON_INTERSECTING_AOI)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": True},
    )
    assert response.status_code == 422
    assert "does not intersect" in response.json()["detail"].lower()


@requires_database
def test_missing_aoi_returns_404(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    missing_aoi = str(uuid4())

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": missing_aoi, "preset": "true_color", "overwrite": True},
    )
    assert response.status_code == 404


@requires_database
def test_missing_band_returns_422(client, tmp_path: Path, monkeypatch) -> None:
    """Missing required band → 422 (same contract as índices / RGB escena completa)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    bands = [_band_entry(k, p) for k, p in paths.items() if k != "SR_B2"]
    scene_id = _create_scene(client, bands)
    aoi_id = _create_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": True},
    )
    assert response.status_code == 422
    assert "SR_B2" in response.json()["detail"]


@requires_database
def test_overwrite_false_conflict_409(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client)

    first = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": True},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": False},
    )
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"].lower()


@requires_database
def test_aoi_map_overlay_coordinates(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client)

    gen = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": True},
    )
    assert gen.status_code == 200, gen.text

    response = client.get(
        f"/api/v1/scenes/{scene_id}/rgb-composites/aois/{aoi_id}/true_color/map-overlay"
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scene_id"] == scene_id
    assert body["aoi_id"] == aoi_id
    assert body["preset"] == "true_color"
    assert body["image_url"] == (
        f"/api/v1/scenes/{scene_id}/rgb-composites/aois/{aoi_id}/true_color/preview.png"
    )
    assert body["width"] > 0
    assert body["height"] > 0
    _assert_valid_coordinates(body["coordinates_wgs84"])


@requires_database
def test_aoi_download_png_200(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(client, [_band_entry(k, p) for k, p in paths.items()])
    aoi_id = _create_aoi(client)

    gen = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": "true_color", "overwrite": True},
    )
    assert gen.status_code == 200, gen.text

    response = client.get(
        f"/api/v1/scenes/{scene_id}/rgb-composites/aois/{aoi_id}/true_color/download.png"
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert "attachment" in response.headers.get("content-disposition", "").lower()

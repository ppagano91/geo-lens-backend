"""Tests for Fase 9F: crop derived index GeoTIFF by saved AOI."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from app.core.config import settings
from app.raster.writers import DEFAULT_INDEX_NODATA, write_float32_geotiff
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


def _sample_scene_payload(name: str = "Crop test scene") -> dict:
    return {
        "name": name,
        "source": "local",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"platform": "Sentinel-2"},
        "bands": [
            {
                "band_key": "B04",
                "band_name": "Red",
                "asset_path": "sample/scenes/demo/B04.tif",
            }
        ],
    }


def _write_derived_index(
    data_root: Path,
    scene_id,
    index_key: str = "ndvi",
    *,
    crs: str = "EPSG:4326",
    transform: tuple[float, float, float, float, float, float] | None = None,
    data: np.ndarray | None = None,
) -> Path:
    asset = f"derived/scenes/{scene_id}/{index_key}.tif"
    array = (
        data if data is not None else np.full((10, 10), 0.4, dtype=np.float32)
    )
    affine = transform or tuple(from_origin(-58.40, -34.60, 0.01, 0.01))[:6]
    return write_float32_geotiff(
        asset,
        data_root,
        array,
        crs=crs,
        transform=affine,
        nodata=DEFAULT_INDEX_NODATA,
    )


def _create_scene_and_aoi(client, aoi_geometry: dict | None = None) -> tuple[str, str]:
    scene_resp = client.post("/api/v1/scenes", json=_sample_scene_payload())
    assert scene_resp.status_code == 201
    scene_id = scene_resp.json()["id"]

    aoi_resp = client.post(
        "/api/v1/aois",
        json={
            "name": "AOI crop test",
            "geometry": aoi_geometry or VALID_POLYGON,
        },
    )
    assert aoi_resp.status_code == 201
    aoi_id = aoi_resp.json()["id"]
    return scene_id, aoi_id


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
def test_crop_valid_aoi_intersecting_raster(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "ndvi")

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": False, "generate_preview": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == scene_id
    assert body["index_key"] == "ndvi"
    assert body["aoi_id"] == aoi_id
    assert body["status"] == "cropped"
    assert body["raster"]["dtype"] == "float32"
    assert body["raster"]["nodata"] == pytest.approx(DEFAULT_INDEX_NODATA)
    assert body["raster"]["width"] > 0
    assert body["raster"]["height"] > 0
    assert body["raster"]["crs"] == "EPSG:4326"
    assert body["stats"]["valid_pixels"] > 0
    assert "nodata_pixels" in body["stats"]

    expected_tif = f"derived/scenes/{scene_id}/aois/{aoi_id}/ndvi.tif"
    expected_png = f"derived/scenes/{scene_id}/aois/{aoi_id}/ndvi.png"
    assert body["output"]["geotiff_asset_path"] == expected_tif
    assert body["output"]["png_asset_path"] == expected_png
    assert (data_root / expected_tif).is_file()
    assert (data_root / expected_png).is_file()

    with rasterio.open(data_root / expected_tif) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.nodata == pytest.approx(DEFAULT_INDEX_NODATA)
        assert src.crs.to_string() == "EPSG:4326"


@requires_database
def test_crop_aoi_epsg4326_raster_epsg32621(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    # Raster covering roughly the VALID_POLYGON area in UTM 21S (EPSG:32721)
    # Use 32621 as specified (northern) — pick a known WGS84 box and project.
    # VALID_POLYGON is around lng -58.4, lat -34.6 → UTM zone 21S = EPSG:32721.
    # Spec asks for EPSG:32621; we still test AOI 4326 → raster projected CRS.
    west_lng, north_lat = -58.40, -34.60
    xs, ys = warp_transform(
        "EPSG:4326",
        "EPSG:32621",
        [west_lng, west_lng + 0.1],
        [north_lat, north_lat - 0.1],
    )
    west_m, east_m = xs[0], xs[1]
    north_m, south_m = ys[0], ys[1]
    # Guard against south hemisphere northing sign under 32621
    pixel = abs(east_m - west_m) / 20.0
    height_m = abs(north_m - south_m)
    width_px, height_px = 20, max(2, int(height_m / pixel))
    origin_north = max(north_m, south_m)
    origin_west = min(west_m, east_m)
    transform = tuple(from_origin(origin_west, origin_north, pixel, pixel))[:6]

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(
        data_root,
        scene_id,
        "ndwi",
        crs="EPSG:32621",
        transform=transform,
        data=np.full((height_px, width_px), 0.25, dtype=np.float32),
    )

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndwi/crop-by-aoi",
        json={"aoi_id": aoi_id, "generate_preview": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["raster"]["crs"] == "EPSG:32621"
    assert body["output"]["geotiff_asset_path"].endswith(
        f"/aois/{aoi_id}/ndwi.tif"
    )
    assert (data_root / body["output"]["geotiff_asset_path"]).is_file()


@requires_database
def test_crop_aoi_outside_raster_returns_422(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    outside = {
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
    scene_id, aoi_id = _create_scene_and_aoi(client, aoi_geometry=outside)
    _write_derived_index(data_root, scene_id, "ndvi")

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id},
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "does not intersect" in detail or "overlap" in detail


@requires_database
def test_crop_missing_derived_geotiff_returns_404(
    client, tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id},
    )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "geotiff not found" in detail
    assert "compute-and-save" in detail


@requires_database
def test_crop_existing_output_overwrite_false_returns_409(
    client, tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "nbr")

    first = client.post(
        f"/api/v1/scenes/{scene_id}/indices/nbr/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": False},
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/v1/scenes/{scene_id}/indices/nbr/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": False},
    )
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"].lower()


@requires_database
def test_crop_overwrite_true_replaces_output(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "ndmi")

    first = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndmi/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": False, "generate_preview": True},
    )
    assert first.status_code == 200
    tif_path = data_root / first.json()["output"]["geotiff_asset_path"]
    first_mtime = tif_path.stat().st_mtime_ns

    second = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndmi/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": True, "generate_preview": True},
    )
    assert second.status_code == 200
    assert second.json()["status"] == "cropped"
    assert tif_path.is_file()
    assert tif_path.stat().st_mtime_ns >= first_mtime


@requires_database
def test_download_cropped_tif_returns_200(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "ndvi")
    crop = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id, "generate_preview": True},
    )
    assert crop.status_code == 200
    expected = (data_root / crop.json()["output"]["geotiff_asset_path"]).read_bytes()

    response = client.get(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/aois/{aoi_id}/download.tif"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")
    assert "attachment" in response.headers.get("content-disposition", "").lower()
    assert response.content == expected


@requires_database
def test_download_cropped_png_returns_200(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "ndvi")
    crop = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id, "generate_preview": True},
    )
    assert crop.status_code == 200
    expected = (data_root / crop.json()["output"]["png_asset_path"]).read_bytes()

    response = client.get(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/aois/{aoi_id}/download.png"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == expected
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


@requires_database
def test_map_overlay_cropped_returns_valid_coordinates_wgs84(
    client, tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_id, aoi_id = _create_scene_and_aoi(client)
    _write_derived_index(data_root, scene_id, "ndvi")
    crop = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id, "generate_preview": True},
    )
    assert crop.status_code == 200

    response = client.get(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/aois/{aoi_id}/map-overlay"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == scene_id
    assert body["index_key"] == "ndvi"
    assert body["aoi_id"] == aoi_id
    assert body["image_url"] == (
        f"/api/v1/scenes/{scene_id}/indices/ndvi/aois/{aoi_id}/download.png"
    )
    assert body["width"] > 0
    assert body["height"] > 0
    assert body["crs_original"] == "EPSG:4326"
    assert "left" in body["bounds_original"]
    _assert_valid_coordinates(body["coordinates_wgs84"])


@requires_database
def test_crop_missing_aoi_returns_404(client, tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_resp = client.post("/api/v1/scenes", json=_sample_scene_payload())
    scene_id = scene_resp.json()["id"]
    _write_derived_index(data_root, scene_id, "ndvi")
    missing_aoi = str(uuid4())

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": missing_aoi},
    )

    assert response.status_code == 404
    assert missing_aoi in response.json()["detail"]

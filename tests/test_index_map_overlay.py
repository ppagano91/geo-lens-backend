"""Tests for Fase 9E: map-overlay metadata for derived index GeoTIFF + PNG."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from app.core.config import settings
from app.main import app
from app.raster.writers import DEFAULT_INDEX_NODATA, write_float32_geotiff
from app.services.index_map_overlay_service import IndexMapOverlayService
from app.services.index_preview_service import IndexPreviewService


def _write_derived_index(
    data_root: Path,
    scene_id,
    index_key: str,
    *,
    crs: str = "EPSG:4326",
    transform: tuple[float, float, float, float, float, float] | None = None,
    data: np.ndarray | None = None,
) -> Path:
    asset = f"derived/scenes/{scene_id}/{index_key}.tif"
    array = (
        data
        if data is not None
        else np.full((4, 5), 0.4, dtype=np.float32)
    )
    affine = transform or tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6]
    return write_float32_geotiff(
        asset,
        data_root,
        array,
        crs=crs,
        transform=affine,
        nodata=DEFAULT_INDEX_NODATA,
    )


def _create_png(data_root: Path, scene_id, index_key: str = "ndvi") -> Path:
    return Path(
        IndexPreviewService(data_root)
        .create_preview(scene_id, index_key)
        .output.resolved_path
    )


def _assert_valid_coordinates(coords: list) -> None:
    assert len(coords) == 4
    for corner in coords:
        assert len(corner) == 2
        lng, lat = corner
        assert isinstance(lng, (int, float))
        assert isinstance(lat, (int, float))
        assert -180.0 <= float(lng) <= 180.0
        assert -90.0 <= float(lat) <= 90.0

    # MapLibre order: TL, TR, BR, BL — top latitudes >= bottom; west <= east
    tl, tr, br, bl = coords
    assert tl[1] >= bl[1]
    assert tr[1] >= br[1]
    assert tl[0] <= tr[0]
    assert bl[0] <= br[0]


def test_map_overlay_epsg4326_returns_valid_coordinates(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    _write_derived_index(data_root, scene_id, "ndvi")
    _create_png(data_root, scene_id, "ndvi")

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/map-overlay"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == str(scene_id)
    assert body["index_key"] == "ndvi"
    assert body["image_url"] == (
        f"/api/v1/scenes/{scene_id}/indices/ndvi/preview.png"
    )
    assert body["width"] == 5
    assert body["height"] == 4
    assert body["crs_original"] == "EPSG:4326"
    assert body["bounds_original"] == {
        "left": pytest.approx(-58.4),
        "bottom": pytest.approx(-34.6 - 4 * 0.01),
        "right": pytest.approx(-58.4 + 5 * 0.01),
        "top": pytest.approx(-34.6),
    }
    _assert_valid_coordinates(body["coordinates_wgs84"])

    tl, tr, br, bl = body["coordinates_wgs84"]
    assert tl == pytest.approx([-58.4, -34.6])
    assert tr == pytest.approx([-58.4 + 0.05, -34.6])
    assert br == pytest.approx([-58.4 + 0.05, -34.6 - 0.04])
    assert bl == pytest.approx([-58.4, -34.6 - 0.04])


def test_map_overlay_epsg32621_transforms_to_wgs84(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()

    # UTM zone 21N (EPSG:32621) — false easting ~500 km, northing north of equator
    west_m, north_m = 500_000.0, 100_000.0
    pixel = 30.0
    width, height = 5, 4
    transform = tuple(from_origin(west_m, north_m, pixel, pixel))[:6]
    _write_derived_index(
        data_root,
        scene_id,
        "ndwi",
        crs="EPSG:32621",
        transform=transform,
        data=np.full((height, width), 0.2, dtype=np.float32),
    )
    _create_png(data_root, scene_id, "ndwi")

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndwi/map-overlay"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["crs_original"] == "EPSG:32621"
    assert body["bounds_original"]["left"] == pytest.approx(west_m)
    assert body["bounds_original"]["top"] == pytest.approx(north_m)
    _assert_valid_coordinates(body["coordinates_wgs84"])

    expected_xs = [
        west_m,
        west_m + width * pixel,
        west_m + width * pixel,
        west_m,
    ]
    expected_ys = [
        north_m,
        north_m,
        north_m - height * pixel,
        north_m - height * pixel,
    ]
    exp_lngs, exp_lats = warp_transform(
        "EPSG:32621", "EPSG:4326", expected_xs, expected_ys
    )
    for i, corner in enumerate(body["coordinates_wgs84"]):
        assert corner[0] == pytest.approx(exp_lngs[i], abs=1e-6)
        assert corner[1] == pytest.approx(exp_lats[i], abs=1e-6)

    # Projected meters must not be returned as lng/lat
    for lng, _lat in body["coordinates_wgs84"]:
        assert abs(lng) < 180.0
        assert abs(lng - west_m) > 1.0


def test_map_overlay_missing_tif_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    # PNG alone is not enough
    png_dir = data_root / "derived" / "scenes" / str(scene_id)
    png_dir.mkdir(parents=True)
    (png_dir / "ndvi.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/map-overlay"
        )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "geotiff not found" in detail
    assert "compute-and-save" in detail


def test_map_overlay_missing_png_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    _write_derived_index(data_root, scene_id, "nbr")

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/nbr/map-overlay"
        )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "preview png not found" in detail
    assert "/preview" in detail


def test_map_overlay_unsupported_index_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/evi/map-overlay"
        )

    assert response.status_code == 404
    assert "evi" in response.json()["detail"].lower()


def test_map_overlay_service_uses_asset_storage(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    _write_derived_index(data_root, scene_id, "ndmi")
    _create_png(data_root, scene_id, "ndmi")

    result = IndexMapOverlayService(data_root).get_map_overlay(scene_id, "ndmi")
    assert result.index_key == "ndmi"
    assert result.image_url.endswith("/ndmi/preview.png")
    _assert_valid_coordinates(result.coordinates_wgs84)

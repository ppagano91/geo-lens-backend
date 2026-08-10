"""Tests for Fase 9H: RGB composites preview + map-overlay."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from app.core.config import settings
from app.main import app
from app.schemas.rgb_composite import RgbCompositePreviewRequest
from app.services.rgb_composite_service import (
    RgbCompositeService,
    UnsupportedRgbPresetError,
)
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    MissingRequiredBandError,
)
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


def _band(band_key: str, asset_path: Path):
    return SimpleNamespace(
        id=uuid4(),
        band_key=band_key,
        asset_path=str(asset_path.resolve()),
    )


def _scene(*, bands: list, source: str = "landsat-8", metadata=None):
    return SimpleNamespace(
        id=uuid4(),
        bands=bands,
        source=source,
        metadata_=metadata if metadata is not None else {"platform": "Landsat-8"},
        is_active=True,
    )


class _FakeRepository:
    def __init__(self, scene) -> None:
        self._scene = scene

    def get_by_id(self, scene_id):
        if self._scene is None or self._scene.id != scene_id:
            return None
        return self._scene


def _service_with_scene(scene, *, data_root: Path) -> RgbCompositeService:
    service = RgbCompositeService.__new__(RgbCompositeService)
    service.repository = _FakeRepository(scene)
    service.data_root = data_root
    return service


def _write_l8_stack(
    root: Path,
    *,
    values: dict[str, int] | None = None,
    shape: tuple[int, int] = (4, 5),
    crs: str = "EPSG:4326",
    origin: tuple[float, float] = (-58.4, -34.6),
    pixel_size: float = 0.01,
) -> dict[str, Path]:
    defaults = {
        "SR_B2": 80,
        "SR_B3": 100,
        "SR_B4": 120,
        "SR_B5": 200,
        "SR_B6": 150,
        "SR_B7": 90,
    }
    if values:
        defaults.update(values)
    paths: dict[str, Path] = {}
    for key in L8_BANDS:
        path = root / f"{key}.tif"
        _write_band(
            path,
            np.full(shape, defaults[key], dtype=np.uint16),
            crs=crs,
            origin=origin,
            pixel_size=pixel_size,
        )
        paths[key] = path
    return paths


def _assert_valid_coordinates(coords: list) -> None:
    assert len(coords) == 4
    for corner in coords:
        assert len(corner) == 2
        lng, lat = corner
        assert isinstance(lng, (int, float))
        assert isinstance(lat, (int, float))
        assert -180.0 <= float(lng) <= 180.0
        assert -90.0 <= float(lat) <= 90.0

    tl, tr, br, bl = coords
    assert tl[1] >= bl[1]
    assert tr[1] >= br[1]
    assert tl[0] <= tr[0]
    assert bl[0] <= br[0]


def _scene_payload(*, bands: list[dict], name: str = "rgb composite test") -> dict:
    return {
        "name": name,
        "source": "landsat-8",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"platform": "Landsat-8", "purpose": "fase_9h"},
        "bands": bands,
    }


def _band_entry(band_key: str, path: Path) -> dict:
    return {
        "band_key": band_key,
        "band_name": band_key,
        "asset_path": str(path.resolve()),
        "nodata": "0",
        "dtype": "uint16",
    }


def _create_scene(client, payload: dict) -> str:
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201, create.text
    return create.json()["id"]


# --- Unit / service tests (no DB) ---


def test_true_color_landsat_8(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene = _scene(bands=[_band(k, p) for k, p in paths.items()])
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset="true_color", overwrite=True),
    )

    assert result.status == "generated"
    assert result.preset == "true_color"
    assert result.sensor == "landsat-8"
    assert result.bands_used == {
        "red": "SR_B4",
        "green": "SR_B3",
        "blue": "SR_B2",
    }
    assert result.width == 5
    assert result.height == 4
    assert result.crs == "EPSG:4326"
    assert result.output.asset_path == (
        f"derived/scenes/{scene.id}/rgb/true_color.png"
    )
    assert (data_root / result.output.asset_path).is_file()


def test_false_color_vegetation_landsat_8(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene = _scene(bands=[_band(k, p) for k, p in paths.items()])
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(
            preset="false_color_vegetation", overwrite=True
        ),
    )

    assert result.preset == "false_color_vegetation"
    assert result.bands_used == {
        "red": "SR_B5",
        "green": "SR_B4",
        "blue": "SR_B3",
    }
    assert (data_root / result.output.asset_path).is_file()


def test_missing_required_band_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    # Omit blue (SR_B2) required by true_color
    bands = [_band(k, p) for k, p in paths.items() if k != "SR_B2"]
    scene = _scene(bands=bands)
    service = _service_with_scene(scene, data_root=data_root)

    with pytest.raises(MissingRequiredBandError, match="SR_B2"):
        service.create_preview(
            scene.id,
            RgbCompositePreviewRequest(preset="true_color", overwrite=True),
        )


def test_misaligned_bands_raise(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    # Overwrite green with different dimensions
    _write_band(
        paths["SR_B3"],
        np.full((3, 3), 100, dtype=np.uint16),
    )
    scene = _scene(bands=[_band(k, p) for k, p in paths.items()])
    service = _service_with_scene(scene, data_root=data_root)

    with pytest.raises(IncompatibleRasterBandsError, match="dimensions"):
        service.create_preview(
            scene.id,
            RgbCompositePreviewRequest(preset="true_color", overwrite=True),
        )


def test_unsupported_preset_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    service = RgbCompositeService(data_root=data_root)
    with pytest.raises(UnsupportedRgbPresetError):
        service.resolve_preview_png(uuid4(), "not_a_preset")


def test_map_overlay_coordinates_from_source_band(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()

    west_m, north_m = 500_000.0, 100_000.0
    pixel = 30.0
    width, height = 5, 4
    paths = _write_l8_stack(
        band_dir,
        shape=(height, width),
        crs="EPSG:32621",
        origin=(west_m, north_m),
        pixel_size=pixel,
    )
    scene = _scene(bands=[_band(k, p) for k, p in paths.items()])
    service = _service_with_scene(scene, data_root=data_root)

    service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset="true_color", overwrite=True),
    )
    overlay = service.get_map_overlay(scene.id, "true_color")

    assert overlay.preset == "true_color"
    assert overlay.crs_original == "EPSG:32621"
    assert overlay.width == width
    assert overlay.height == height
    assert overlay.image_url == (
        f"/api/v1/scenes/{scene.id}/rgb-composites/true_color/preview.png"
    )
    _assert_valid_coordinates(overlay.coordinates_wgs84)

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
    for i, corner in enumerate(overlay.coordinates_wgs84):
        assert corner[0] == pytest.approx(exp_lngs[i], abs=1e-6)
        assert corner[1] == pytest.approx(exp_lats[i], abs=1e-6)


# --- HTTP tests (require DB) ---


@requires_database
def test_http_true_color_landsat_8(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(
        client,
        _scene_payload(bands=[_band_entry(k, p) for k, p in paths.items()]),
    )

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={
            "preset": "true_color",
            "red_role": "red",
            "green_role": "green",
            "blue_role": "blue",
            "stretch": "percentile",
            "p_min": 2,
            "p_max": 98,
            "overwrite": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preset"] == "true_color"
    assert body["status"] == "generated"
    assert body["sensor"] == "landsat-8"
    assert body["bands_used"] == {
        "red": "SR_B4",
        "green": "SR_B3",
        "blue": "SR_B2",
    }
    assert body["output"]["asset_path"] == (
        f"derived/scenes/{scene_id}/rgb/true_color.png"
    )

    png = client.get(
        f"/api/v1/scenes/{scene_id}/rgb-composites/true_color/preview.png"
    )
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")


@requires_database
def test_http_false_color_vegetation_landsat_8(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(
        client,
        _scene_payload(bands=[_band_entry(k, p) for k, p in paths.items()]),
    )

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "false_color_vegetation", "overwrite": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bands_used"] == {
        "red": "SR_B5",
        "green": "SR_B4",
        "blue": "SR_B3",
    }


@requires_database
def test_http_missing_band_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    bands = [_band_entry(k, p) for k, p in paths.items() if k != "SR_B2"]
    scene_id = _create_scene(client, _scene_payload(bands=bands))

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True},
    )
    assert response.status_code == 422
    assert "SR_B2" in response.json()["detail"]


@requires_database
def test_http_misaligned_bands_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    _write_band(paths["SR_B3"], np.full((2, 2), 50, dtype=np.uint16))
    scene_id = _create_scene(
        client,
        _scene_payload(bands=[_band_entry(k, p) for k, p in paths.items()]),
    )

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True},
    )
    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "dimensions" in detail or "geotransform" in detail or "crs" in detail


@requires_database
def test_http_map_overlay_coordinates_wgs84(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_l8_stack(band_dir)
    scene_id = _create_scene(
        client,
        _scene_payload(bands=[_band_entry(k, p) for k, p in paths.items()]),
    )

    gen = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True},
    )
    assert gen.status_code == 200, gen.text

    response = client.get(
        f"/api/v1/scenes/{scene_id}/rgb-composites/true_color/map-overlay"
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preset"] == "true_color"
    assert body["image_url"] == (
        f"/api/v1/scenes/{scene_id}/rgb-composites/true_color/preview.png"
    )
    _assert_valid_coordinates(body["coordinates_wgs84"])
    tl, tr, br, bl = body["coordinates_wgs84"]
    assert tl == pytest.approx([-58.4, -34.6])
    assert tr == pytest.approx([-58.4 + 0.05, -34.6])
    assert br == pytest.approx([-58.4 + 0.05, -34.6 - 0.04])
    assert bl == pytest.approx([-58.4, -34.6 - 0.04])


def test_preview_png_missing_returns_404(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/rgb-composites/true_color/preview.png"
        )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "rgb composite png not found" in detail
    assert "preview" in detail

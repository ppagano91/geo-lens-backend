"""Unit tests for Fase 7E index PNG preview (no database required)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from app.core.config import settings
from app.main import app
from app.raster.preview import (
    render_index_preview_rgba,
    valid_mask,
)
from app.raster.writers import DEFAULT_INDEX_NODATA, write_float32_geotiff
from app.services.index_preview_service import IndexPreviewService
from app.services.local_index_compute_service import UnsupportedIndexError
from app.raster.readers import RasterFileNotFoundError


def _write_derived_index(
    data_root: Path,
    scene_id,
    index_key: str,
    data: np.ndarray,
) -> Path:
    asset = f"derived/scenes/{scene_id}/{index_key}.tif"
    return write_float32_geotiff(
        asset,
        data_root,
        data,
        crs="EPSG:4326",
        transform=tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6],
        nodata=DEFAULT_INDEX_NODATA,
    )


def test_preview_ndvi_success(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()

    array = np.full((4, 5), 0.5, dtype=np.float32)
    array[0, 0] = np.nan  # will become nodata on write
    _write_derived_index(data_root, scene_id, "ndvi", array)

    service = IndexPreviewService(data_root)
    result = service.create_preview(scene_id, "ndvi")

    expected_png = data_root / "derived" / "scenes" / str(scene_id) / "ndvi.png"
    assert result.index == "NDVI"
    assert result.status == "preview_created"
    assert result.scene_id == scene_id
    assert result.input.asset_path == f"derived/scenes/{scene_id}/ndvi.tif"
    assert result.output.asset_path == f"derived/scenes/{scene_id}/ndvi.png"
    assert result.output.resolved_path == str(expected_png.resolve())
    assert result.width == 5
    assert result.height == 4
    assert expected_png.is_file()

    with Image.open(expected_png) as img:
        assert img.format == "PNG"
        assert img.mode == "RGBA"
        assert img.size == (5, 4)
        rgba = np.array(img)
        assert rgba[0, 0, 3] == 0  # nodata transparent
        assert rgba[1, 1, 3] == 255
        assert tuple(rgba[1, 1, :3].tolist()) != (0, 0, 0)


@pytest.mark.parametrize("index_key,display", [
    ("ndwi", "NDWI"),
    ("nbr", "NBR"),
    ("ndmi", "NDMI"),
])
def test_preview_other_indices_success(
    tmp_path: Path,
    index_key: str,
    display: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    _write_derived_index(
        data_root,
        scene_id,
        index_key,
        np.full((3, 3), 0.4, dtype=np.float32),
    )

    service = IndexPreviewService(data_root)
    result = service.create_preview(scene_id, index_key)

    png_path = Path(result.output.resolved_path)
    assert result.index == display
    assert result.status == "preview_created"
    assert result.output.asset_path == f"derived/scenes/{scene_id}/{index_key}.png"
    assert png_path.is_file()
    with Image.open(png_path) as img:
        assert img.size == (3, 3)
        assert img.mode == "RGBA"


def test_preview_missing_geotiff(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    service = IndexPreviewService(data_root)

    with pytest.raises(RasterFileNotFoundError):
        service.create_preview(scene_id, "ndvi")


def test_preview_unsupported_index(tmp_path: Path) -> None:
    service = IndexPreviewService(tmp_path)
    with pytest.raises(UnsupportedIndexError):
        service.create_preview(uuid4(), "evi")


def test_preview_png_created_on_disk(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    _write_derived_index(
        data_root,
        scene_id,
        "ndvi",
        np.linspace(-0.5, 0.8, 20, dtype=np.float32).reshape(4, 5),
    )

    service = IndexPreviewService(data_root)
    result = service.create_preview(scene_id, "NDVI")  # case-insensitive

    png = Path(result.output.resolved_path)
    assert png.is_file()
    assert png.stat().st_size > 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_nodata_transparent_in_rgba() -> None:
    data = np.array(
        [[DEFAULT_INDEX_NODATA, 0.5], [-1.0, 1.0]],
        dtype=np.float32,
    )
    # Simulate read_raster_array: nodata → NaN
    data = np.where(data == np.float32(DEFAULT_INDEX_NODATA), np.nan, data)

    rgba = render_index_preview_rgba(data, "ndvi", nodata=DEFAULT_INDEX_NODATA)

    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 255
    assert rgba[1, 0, 3] == 255
    assert rgba[1, 1, 3] == 255
    # Low NDVI browner than high NDVI green
    assert int(rgba[1, 0, 1]) < int(rgba[1, 1, 1]) or int(rgba[1, 0, 0]) > int(
        rgba[1, 1, 0]
    )


def test_valid_mask_ignores_nodata_sentinel() -> None:
    data = np.array([[DEFAULT_INDEX_NODATA, 0.2], [np.nan, 0.8]], dtype=np.float32)
    mask = valid_mask(data, nodata=DEFAULT_INDEX_NODATA)
    assert mask.tolist() == [[False, True], [False, True]]


def test_http_preview_ndvi(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    _write_derived_index(
        data_root,
        scene_id,
        "ndvi",
        np.full((2, 2), 0.3, dtype=np.float32),
    )

    with TestClient(app) as client:
        response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "preview_created"
    assert body["index"] == "NDVI"
    assert body["width"] == 2
    assert body["height"] == 2
    assert body["input"]["asset_path"].endswith("ndvi.tif")
    assert body["output"]["asset_path"].endswith("ndvi.png")
    assert Path(body["output"]["resolved_path"]).is_file()


def test_http_preview_missing_geotiff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/preview")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.parametrize("index_key", ["ndwi", "nbr", "ndmi"])
def test_http_preview_other_indices(tmp_path: Path, monkeypatch, index_key: str) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    _write_derived_index(
        data_root,
        scene_id,
        index_key,
        np.full((2, 3), -0.2, dtype=np.float32),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/scenes/{scene_id}/indices/{index_key}/preview"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "preview_created"
    assert Path(body["output"]["resolved_path"]).is_file()

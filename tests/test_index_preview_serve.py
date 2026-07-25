"""Tests for Fase 7E.1: serve existing index preview PNGs over HTTP."""

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
from app.raster.readers import RasterPathError, resolve_asset_path
from app.raster.writers import DEFAULT_INDEX_NODATA, write_float32_geotiff
from app.services.index_preview_service import (
    IndexPreviewService,
    PreviewPngNotFoundError,
)
from app.services.local_index_compute_service import UnsupportedIndexError


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


def _create_png(data_root: Path, scene_id, index_key: str = "ndvi") -> Path:
    _write_derived_index(
        data_root,
        scene_id,
        index_key,
        np.full((3, 3), 0.4, dtype=np.float32),
    )
    return Path(
        IndexPreviewService(data_root).create_preview(scene_id, index_key).output.resolved_path
    )


def test_get_preview_png_returns_image(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    png_path = _create_png(data_root, scene_id, "ndvi")
    expected_bytes = png_path.read_bytes()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/scenes/{scene_id}/indices/ndvi/preview.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    disposition = response.headers.get("content-disposition", "")
    assert "inline" in disposition.lower()
    assert response.content == expected_bytes
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_preview_png_missing_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/scenes/{scene_id}/indices/ndvi/preview.png")

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "preview png not found" in detail
    assert "post" in detail
    assert "/preview" in detail


def test_get_preview_png_unsupported_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/scenes/{scene_id}/indices/evi/preview.png")

    assert response.status_code == 404
    assert "evi" in response.json()["detail"].lower()


def test_resolve_preview_png_path_stays_under_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    png_path = _create_png(data_root, scene_id, "ndwi")

    service = IndexPreviewService(data_root)
    resolved = service.resolve_preview_png(scene_id, "ndwi")

    assert resolved == png_path.resolve()
    assert resolved.is_relative_to(data_root.resolve())
    assert resolved.name == "ndwi.png"
    assert "derived" in resolved.parts
    assert str(scene_id) in resolved.parts


def test_resolve_preview_png_rejects_traversal_asset(tmp_path: Path) -> None:
    """resolve_asset_path must reject paths that escape DATA_ROOT."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(RasterPathError):
        resolve_asset_path("../outside.png", data_root)


def test_resolve_preview_png_unsupported_and_missing(tmp_path: Path) -> None:
    service = IndexPreviewService(tmp_path)
    scene_id = uuid4()

    with pytest.raises(UnsupportedIndexError):
        service.resolve_preview_png(scene_id, "evi")

    with pytest.raises(PreviewPngNotFoundError) as exc_info:
        service.resolve_preview_png(scene_id, "ndvi")

    assert "Generate it first" in str(exc_info.value)
    assert "/indices/ndvi/preview" in str(exc_info.value)


def test_get_preview_png_does_not_regenerate(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    png_path = _create_png(data_root, scene_id, "nbr")
    mtime_before = png_path.stat().st_mtime_ns
    size_before = png_path.stat().st_size

    with TestClient(app) as client:
        response = client.get(f"/api/v1/scenes/{scene_id}/indices/nbr/preview.png")

    assert response.status_code == 200
    assert png_path.stat().st_mtime_ns == mtime_before
    assert png_path.stat().st_size == size_before
    with Image.open(png_path) as img:
        assert img.size == (3, 3)

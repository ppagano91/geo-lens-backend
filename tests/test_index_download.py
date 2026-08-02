"""Tests for Fase 8C: download derived index GeoTIFF and PNG over HTTP."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.core.config import settings
from app.main import app
from app.raster.readers import RasterPathError, resolve_asset_path
from app.raster.writers import DEFAULT_INDEX_NODATA, write_float32_geotiff
from app.services.index_preview_service import (
    DerivedGeotiffNotFoundError,
    IndexPreviewService,
    PreviewPngNotFoundError,
)
from app.services.local_index_compute_service import UnsupportedIndexError


def _write_derived_index(
    data_root: Path,
    scene_id,
    index_key: str,
    data: np.ndarray | None = None,
) -> Path:
    asset = f"derived/scenes/{scene_id}/{index_key}.tif"
    array = (
        data
        if data is not None
        else np.full((3, 3), 0.4, dtype=np.float32)
    )
    return write_float32_geotiff(
        asset,
        data_root,
        array,
        crs="EPSG:4326",
        transform=tuple(from_origin(-58.4, -34.6, 0.01, 0.01))[:6],
        nodata=DEFAULT_INDEX_NODATA,
    )


def _create_png(data_root: Path, scene_id, index_key: str = "ndvi") -> Path:
    _write_derived_index(data_root, scene_id, index_key)
    return Path(
        IndexPreviewService(data_root)
        .create_preview(scene_id, index_key)
        .output.resolved_path
    )


def test_download_tif_existing_returns_200(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    tif_path = _write_derived_index(data_root, scene_id, "ndvi")
    expected_bytes = tif_path.read_bytes()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/download.tif"
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition.lower()
    assert f"{scene_id}_ndvi.tif" in disposition
    assert response.content == expected_bytes


def test_download_png_existing_returns_200(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    png_path = _create_png(data_root, scene_id, "ndwi")
    expected_bytes = png_path.read_bytes()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndwi/download.png"
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition.lower()
    assert f"{scene_id}_ndwi.png" in disposition
    assert response.content == expected_bytes
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_download_tif_missing_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/download.tif"
        )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "geotiff not found" in detail
    assert "compute-and-save" in detail


def test_download_png_missing_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/download.png"
        )

    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "preview png not found" in detail
    assert "/preview" in detail


def test_download_unsupported_index_returns_404(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "data_root", str(tmp_path))
    scene_id = uuid4()

    with TestClient(app) as client:
        tif_response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/evi/download.tif"
        )
        png_response = client.get(
            f"/api/v1/scenes/{scene_id}/indices/evi/download.png"
        )

    assert tif_response.status_code == 404
    assert "evi" in tif_response.json()["detail"].lower()
    assert png_response.status_code == 404
    assert "evi" in png_response.json()["detail"].lower()


def test_download_content_disposition_and_type(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    _write_derived_index(data_root, scene_id, "nbr")
    _create_png(data_root, scene_id, "nbr")

    with TestClient(app) as client:
        tif = client.get(f"/api/v1/scenes/{scene_id}/indices/nbr/download.tif")
        png = client.get(f"/api/v1/scenes/{scene_id}/indices/nbr/download.png")

    assert tif.status_code == 200
    assert tif.headers["content-type"].startswith("image/tiff")
    assert "attachment" in tif.headers.get("content-disposition", "").lower()

    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")
    assert "attachment" in png.headers.get("content-disposition", "").lower()


def test_resolve_derived_geotiff_path_stays_under_data_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scene_id = uuid4()
    tif_path = _write_derived_index(data_root, scene_id, "ndmi")

    service = IndexPreviewService(data_root)
    resolved = service.resolve_derived_geotiff(scene_id, "ndmi")

    assert resolved == tif_path.resolve()
    assert resolved.is_relative_to(data_root.resolve())
    assert resolved.name == "ndmi.tif"
    assert "derived" in resolved.parts
    assert str(scene_id) in resolved.parts


def test_resolve_derived_rejects_traversal_and_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(RasterPathError):
        resolve_asset_path("../outside.tif", data_root)

    service = IndexPreviewService(tmp_path)
    scene_id = uuid4()

    with pytest.raises(UnsupportedIndexError):
        service.resolve_derived_geotiff(scene_id, "evi")

    with pytest.raises(DerivedGeotiffNotFoundError) as tif_exc:
        service.resolve_derived_geotiff(scene_id, "ndvi")
    assert "compute-and-save" in str(tif_exc.value)

    with pytest.raises(PreviewPngNotFoundError) as png_exc:
        service.resolve_preview_png(scene_id, "ndvi")
    assert "Generate it first" in str(png_exc.value)


def test_download_does_not_modify_files(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_id = uuid4()
    png_path = _create_png(data_root, scene_id, "ndvi")
    tif_path = data_root / "derived" / "scenes" / str(scene_id) / "ndvi.tif"
    tif_mtime = tif_path.stat().st_mtime_ns
    tif_size = tif_path.stat().st_size
    png_mtime = png_path.stat().st_mtime_ns
    png_size = png_path.stat().st_size

    with TestClient(app) as client:
        assert (
            client.get(
                f"/api/v1/scenes/{scene_id}/indices/ndvi/download.tif"
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"/api/v1/scenes/{scene_id}/indices/ndvi/download.png"
            ).status_code
            == 200
        )

    assert tif_path.stat().st_mtime_ns == tif_mtime
    assert tif_path.stat().st_size == tif_size
    assert png_path.stat().st_mtime_ns == png_mtime
    assert png_path.stat().st_size == png_size

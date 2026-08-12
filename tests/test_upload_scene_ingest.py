"""Fase 9D: upload Landsat 8 bands from UI and auto-ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.core.config import settings
from tests.conftest import requires_database
from tests.test_local_scene_ingest import LANDSAT_KEYS, MTL_TEXT, _write_band

UPLOAD_URL = "/api/v1/ingest/upload-scene"


def _band_bytes(
    data: np.ndarray,
    *,
    nodata: float = 0.0,
    crs: str = "EPSG:4326",
    origin: tuple[float, float] = (-58.45, -34.55),
    pixel_size: float = 0.001,
) -> bytes:
    height, width = data.shape
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    with MemoryFile() as memfile:
        with memfile.open(
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
        return memfile.read()


def _landsat_upload_files(
    *,
    omit: set[str] | None = None,
    mismatch_size_for: str | None = None,
    with_mtl: bool = True,
    include_invalid_ext: bool = False,
) -> list[tuple[str, tuple[str, bytes, str]]]:
    omit = omit or set()
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for key in LANDSAT_KEYS:
        if key in omit:
            continue
        if mismatch_size_for == key:
            data = np.full((3, 3), 100, dtype=np.uint16)
        else:
            data = np.full((4, 4), 10909 if key != "SR_B5" else 18182, dtype=np.uint16)
        files.append(
            ("files", (f"{key}.tif", _band_bytes(data), "image/tiff"))
        )

    if with_mtl:
        files.append(
            (
                "files",
                (
                    "LC08_L2SP_225084_20260510_20260515_02_T1_MTL.txt",
                    MTL_TEXT.encode("utf-8"),
                    "text/plain",
                ),
            )
        )

    if include_invalid_ext:
        files.append(
            ("files", ("readme.md", b"# not allowed", "text/markdown"))
        )

    return files


@requires_database
def test_upload_landsat8_valid_scene(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8", "name": "Upload test LC08"},
        files=_landsat_upload_files(),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "landsat-8"
    assert body["sensor"] == "landsat-8"
    assert body["name"] == "Upload test LC08"
    assert body["acquisition_date"] == "2026-05-10"
    assert body["scene_path"].startswith("uploaded/scenes/")
    assert body["metadata"]["ingest"]["method"] == "upload-scene"
    assert body["metadata"]["ingest"]["phase"] == "9D"
    assert {b["band_key"] for b in body["bands"]} == set(LANDSAT_KEYS)
    assert all(item["compatible"] for item in body["available_indices"])

    # Files persisted under DATA_ROOT via AssetStorageService layout.
    scene_dir = data_root / body["scene_path"]
    assert scene_dir.is_dir()
    assert (scene_dir / "SR_B5.tif").is_file()


@requires_database
def test_upload_missing_sr_b5_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8"},
        files=_landsat_upload_files(omit={"SR_B5"}),
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "SR_B5" in detail
    assert "Missing required" in detail
    # Failed upload should not leave orphan folders under uploaded/scenes.
    uploaded_root = data_root / "uploaded" / "scenes"
    if uploaded_root.exists():
        assert list(uploaded_root.iterdir()) == []


@requires_database
def test_upload_mismatched_band_size_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8"},
        files=_landsat_upload_files(mismatch_size_for="SR_B6"),
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "mismatch" in detail.lower()
    assert "SR_B6" in detail


@requires_database
def test_upload_invalid_extension_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8"},
        files=_landsat_upload_files(include_invalid_ext=True),
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "extension" in detail or ".md" in detail or "invalid" in detail


@requires_database
def test_upload_unsupported_source_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "modis"},
        files=_landsat_upload_files(),
    )

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "landsat-8" in detail
    assert "sentinel-2" in detail


@requires_database
def test_upload_sentinel2_source_accepted(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Fase 9K: sentinel-2 is a supported upload source (see full suite)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    # Landsat files with sentinel-2 source should fail on missing B0x bands,
    # not on unsupported source.
    response = client.post(
        UPLOAD_URL,
        data={"source": "sentinel-2"},
        files=_landsat_upload_files(with_mtl=False),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Missing required" in detail
    assert "B02" in detail or "B08" in detail
    assert "supports source" not in detail.lower()

@requires_database
def test_upload_file_too_large_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    monkeypatch.setattr(settings, "max_upload_file_bytes", 64)

    tiny = _band_bytes(np.full((2, 2), 1, dtype=np.uint16))
    # Ensure payload exceeds the tiny limit.
    assert len(tiny) > 64

    response = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8"},
        files=[("files", ("SR_B2.tif", tiny, "image/tiff"))],
    )

    assert response.status_code == 422
    assert "max upload size" in response.json()["detail"].lower()


@requires_database
def test_upload_then_compute_ndvi(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    ingest = client.post(
        UPLOAD_URL,
        data={"source": "landsat-8"},
        files=_landsat_upload_files(),
    )
    assert ingest.status_code == 201, ingest.text
    scene_id = ingest.json()["scene_id"]

    compute = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")
    assert compute.status_code == 200, compute.text
    result = compute.json()
    assert result["index"] == "NDVI"
    assert result["status"] == "computed"
    assert result["bands_used"]["red"]["band_key"] == "SR_B4"
    assert result["bands_used"]["nir"]["band_key"] == "SR_B5"
    assert result["stats"]["mean"] == pytest.approx(0.5, abs=1e-4)
    assert result["radiometry"]["product_level"] == "landsat_l2"
    assert result["radiometry"]["scale_applied"] is True


@requires_database
def test_local_scene_still_works_after_upload_endpoint(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Regression: path-based /local-scene must remain intact."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    scene_dir = data_root / "sample" / "scenes" / "local_still_ok"
    scene_dir.mkdir(parents=True)
    for key in LANDSAT_KEYS:
        data = np.full((4, 4), 10909 if key != "SR_B5" else 18182, dtype=np.uint16)
        _write_band(scene_dir / f"{key}.tif", data)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": "sample/scenes/local_still_ok",
            "source": "landsat-8",
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["metadata"]["ingest"]["method"] == "local-scene"

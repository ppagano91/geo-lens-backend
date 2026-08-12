"""Fase 9K/9L: local Sentinel-2 ingest + 20 m → 10 m SWIR alignment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.core.config import settings
from app.services.band_alignment_service import BandAlignmentError
from tests.conftest import requires_database
from tests.test_local_scene_ingest import _write_band
from tests.test_upload_scene_ingest import UPLOAD_URL, _band_bytes

SENTINEL_REQUIRED = ("B02", "B03", "B04", "B08")
SENTINEL_OPTIONAL = ("B11", "B12")


def _stage_sentinel_scene(
    data_root: Path,
    *,
    folder: str = "sample/scenes/sentinel2_ingest_test",
    omit: set[str] | None = None,
    include_swir: bool = False,
    swir_mismatch: bool = False,
    long_names: bool = False,
) -> str:
    scene_dir = data_root / folder
    scene_dir.mkdir(parents=True, exist_ok=True)
    omit = omit or set()

    for key in SENTINEL_REQUIRED:
        if key in omit:
            continue
        name = (
            f"T21HUS_20260510T133211_{key}_10m.tif"
            if long_names
            else f"{key}.tif"
        )
        value = 300 if key == "B08" else 100
        _write_band(scene_dir / name, np.full((4, 4), value, dtype=np.uint16))

    if include_swir:
        for key in SENTINEL_OPTIONAL:
            if key in omit:
                continue
            name = (
                f"T21HUS_20260510T133211_{key}_20m.tif"
                if long_names
                else f"{key}.tif"
            )
            if swir_mismatch:
                # Half resolution → different width/height and transform.
                data = np.full((2, 2), 50, dtype=np.uint16)
                path = scene_dir / name
                transform = from_origin(-58.45, -34.55, 0.002, 0.002)
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=2,
                    width=2,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=0,
                ) as dataset:
                    dataset.write(data, 1)
            else:
                _write_band(
                    scene_dir / name, np.full((4, 4), 50, dtype=np.uint16)
                )

    return folder.replace("\\", "/")


def _sentinel_upload_files(
    *,
    omit: set[str] | None = None,
    include_swir_mismatch: bool = False,
) -> list[tuple[str, tuple[str, bytes, str]]]:
    omit = omit or set()
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for key in SENTINEL_REQUIRED:
        if key in omit:
            continue
        value = 300 if key == "B08" else 100
        files.append(
            (
                "files",
                (
                    f"{key}.tif",
                    _band_bytes(np.full((4, 4), value, dtype=np.uint16)),
                    "image/tiff",
                ),
            )
        )
    if include_swir_mismatch:
        # 2x2 at coarser pixel size — not aligned with 4x4 10 m grid.
        height, width = 2, 2
        transform = from_origin(-58.45, -34.55, 0.002, 0.002)
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(np.full((2, 2), 50, dtype=np.uint16), 1)
            content = memfile.read()
        files.append(("files", ("B11.tif", content, "image/tiff")))
        files.append(("files", ("B12.tif", content, "image/tiff")))
    return files


def _band_by_key(body: dict, key: str) -> dict:
    return next(b for b in body["bands"] if b["band_key"] == key)


@requires_database
def test_ingest_sentinel2_valid_10m_scene(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(data_root, long_names=True)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": scene_path,
            "source": "sentinel-2",
            "name": "Ingest test S2",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "sentinel-2"
    assert body["sensor"] == "sentinel-2"
    assert body["name"] == "Ingest test S2"
    assert body["metadata"]["platform"] == "Sentinel-2"
    assert body["metadata"]["sensor"] == "sentinel-2"
    assert body["metadata"]["ingest"]["phase"] == "9L"
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_REQUIRED)

    by_key = {item["index_key"]: item for item in body["available_indices"]}
    assert by_key["ndvi"]["compatible"] is True
    assert by_key["ndwi"]["compatible"] is True
    assert by_key["nbr"]["compatible"] is False
    assert "swir2" in by_key["nbr"]["missing_roles"]
    assert by_key["ndmi"]["compatible"] is False
    assert "swir1" in by_key["ndmi"]["missing_roles"]


@requires_database
def test_ingest_sentinel2_missing_b08_returns_clear_error(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(data_root, omit={"B08"})

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "B08" in detail
    assert "Missing required" in detail


@requires_database
def test_ingest_sentinel2_resamples_misaligned_swir_to_10m(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(
        data_root, include_swir=True, swir_mismatch=True
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert {b["band_key"] for b in body["bands"]} == set(
        SENTINEL_REQUIRED + SENTINEL_OPTIONAL
    )

    codes = {w["code"] for w in body["warnings"]}
    assert "sentinel_swir_20m_detected" in codes
    assert "sentinel_swir_resampled" in codes

    b08 = _band_by_key(body, "B08")
    b11 = _band_by_key(body, "B11")
    b12 = _band_by_key(body, "B12")
    assert b11["width"] == b08["width"]
    assert b11["height"] == b08["height"]
    assert b11["crs"] == b08["crs"]
    assert b12["width"] == b08["width"]
    assert b12["height"] == b08["height"]
    assert "aligned/B11_10m.tif" in b11["asset_path"]
    assert "aligned/B12_10m.tif" in b12["asset_path"]
    assert (data_root / b11["asset_path"]).is_file()
    assert (data_root / b12["asset_path"]).is_file()
    # Originals are preserved.
    assert (data_root / scene_path / "B11.tif").is_file()
    assert (data_root / scene_path / "B12.tif").is_file()

    by_key = {item["index_key"]: item for item in body["available_indices"]}
    assert by_key["nbr"]["compatible"] is True
    assert by_key["ndmi"]["compatible"] is True


@requires_database
def test_ingest_sentinel2_aligned_swir_grid_matches_b08(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(
        data_root, include_swir=True, swir_mismatch=True
    )

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert ingest.status_code == 201, ingest.text
    body = ingest.json()
    scene_id = body["scene_id"]

    with rasterio.open(data_root / _band_by_key(body, "B08")["asset_path"]) as ref:
        with rasterio.open(
            data_root / _band_by_key(body, "B11")["asset_path"]
        ) as b11:
            assert b11.crs == ref.crs
            assert b11.width == ref.width
            assert b11.height == ref.height
            assert b11.transform == ref.transform
        with rasterio.open(
            data_root / _band_by_key(body, "B12")["asset_path"]
        ) as b12:
            assert b12.crs == ref.crs
            assert b12.width == ref.width
            assert b12.height == ref.height
            assert b12.transform == ref.transform

    bands = client.get(f"/api/v1/scenes/{scene_id}/bands")
    assert bands.status_code == 200, bands.text
    meta_by_key = {b["band_key"]: b["metadata"] for b in bands.json()}
    assert meta_by_key["B11"]["aligned"] is True
    assert meta_by_key["B11"]["resampling_method"] == "bilinear"
    assert meta_by_key["B11"]["reference_band"] == "B08"
    assert meta_by_key["B11"]["original_band_key"] == "B11"
    assert meta_by_key["B11"]["aligned_band_key"] == "B11"
    assert meta_by_key["B12"]["aligned"] is True

    # Ingest response also exposes alignment metadata on band entries.
    b11_ingest = _band_by_key(body, "B11")
    assert b11_ingest["metadata"]["resampled"] is True
    assert b11_ingest["metadata"]["resampling_method"] == "bilinear"
    assert b11_ingest["metadata"]["reference_band"] == "B08"


@requires_database
def test_ingest_sentinel2_then_compute_nbr_ndmi_with_aligned_swir(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(
        data_root, include_swir=True, swir_mismatch=True
    )

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert ingest.status_code == 201, ingest.text
    scene_id = ingest.json()["scene_id"]

    nbr = client.post(f"/api/v1/scenes/{scene_id}/indices/nbr/compute")
    assert nbr.status_code == 200, nbr.text
    nbr_body = nbr.json()
    assert nbr_body["index"] == "NBR"
    assert nbr_body["bands_used"]["nir"]["band_key"] == "B08"
    assert nbr_body["bands_used"]["swir2"]["band_key"] == "B12"

    ndmi = client.post(f"/api/v1/scenes/{scene_id}/indices/ndmi/compute")
    assert ndmi.status_code == 200, ndmi.text
    ndmi_body = ndmi.json()
    assert ndmi_body["index"] == "NDMI"
    assert ndmi_body["bands_used"]["nir"]["band_key"] == "B08"
    assert ndmi_body["bands_used"]["swir1"]["band_key"] == "B11"


@requires_database
def test_ingest_sentinel2_swir_composites_with_aligned_bands(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(
        data_root, include_swir=True, swir_mismatch=True
    )

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert ingest.status_code == 201, ingest.text
    scene_id = ingest.json()["scene_id"]

    swir_urban = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "swir_urban", "overwrite": True},
    )
    assert swir_urban.status_code == 200, swir_urban.text
    assert swir_urban.json()["bands_used"] == {
        "red": "B12",
        "green": "B11",
        "blue": "B04",
    }

    moisture = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "moisture_vegetation", "overwrite": True},
    )
    assert moisture.status_code == 200, moisture.text
    assert moisture.json()["bands_used"] == {
        "red": "B11",
        "green": "B08",
        "blue": "B04",
    }


@requires_database
def test_ingest_sentinel2_missing_swir_keeps_indices_incompatible(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(data_root, include_swir=False)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    by_key = {item["index_key"]: item for item in body["available_indices"]}
    assert by_key["nbr"]["compatible"] is False
    assert "swir2" in by_key["nbr"]["missing_roles"]
    assert by_key["ndmi"]["compatible"] is False
    assert "swir1" in by_key["ndmi"]["missing_roles"]


@requires_database
def test_ingest_sentinel2_resampling_failure_returns_clear_error(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(
        data_root, include_swir=True, swir_mismatch=True
    )

    with patch(
        "app.services.local_scene_ingest_service.BandAlignmentService.align_to_reference",
        side_effect=BandAlignmentError("synthetic alignment failure"),
    ):
        response = client.post(
            "/api/v1/ingest/local-scene",
            json={"scene_path": scene_path, "source": "sentinel-2"},
        )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "Failed to resample" in detail
    assert "synthetic alignment failure" in detail


@requires_database
def test_ingest_sentinel2_then_compute_ndvi_ndwi(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(data_root)

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert ingest.status_code == 201, ingest.text
    scene_id = ingest.json()["scene_id"]

    ndvi = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute")
    assert ndvi.status_code == 200, ndvi.text
    ndvi_body = ndvi.json()
    assert ndvi_body["index"] == "NDVI"
    assert ndvi_body["status"] == "computed"
    assert ndvi_body["bands_used"]["red"]["band_key"] == "B04"
    assert ndvi_body["bands_used"]["nir"]["band_key"] == "B08"
    assert ndvi_body["stats"]["mean"] == pytest.approx(0.5)

    ndwi = client.post(f"/api/v1/scenes/{scene_id}/indices/ndwi/compute")
    assert ndwi.status_code == 200, ndwi.text
    ndwi_body = ndwi.json()
    assert ndwi_body["index"] == "NDWI"
    assert ndwi_body["bands_used"]["green"]["band_key"] == "B03"
    assert ndwi_body["bands_used"]["nir"]["band_key"] == "B08"


@requires_database
def test_ingest_sentinel2_true_color_and_false_color(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_sentinel_scene(data_root)

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert ingest.status_code == 201, ingest.text
    scene_id = ingest.json()["scene_id"]

    true_color = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True},
    )
    assert true_color.status_code == 200, true_color.text
    tc = true_color.json()
    assert tc["preset"] == "true_color"
    assert tc["bands_used"] == {"red": "B04", "green": "B03", "blue": "B02"}

    false_color = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "false_color_vegetation", "overwrite": True},
    )
    assert false_color.status_code == 200, false_color.text
    fc = false_color.json()
    assert fc["preset"] == "false_color_vegetation"
    assert fc["bands_used"] == {"red": "B08", "green": "B04", "blue": "B03"}


@requires_database
def test_upload_sentinel2_valid_scene(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "sentinel-2", "name": "Upload test S2"},
        files=_sentinel_upload_files(),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "sentinel-2"
    assert body["sensor"] == "sentinel-2"
    assert body["metadata"]["platform"] == "Sentinel-2"
    assert body["metadata"]["ingest"]["phase"] == "9L"
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_REQUIRED)


@requires_database
def test_upload_sentinel2_swir_mismatch_resamples(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        UPLOAD_URL,
        data={"source": "sentinel-2"},
        files=_sentinel_upload_files(include_swir_mismatch=True),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert {b["band_key"] for b in body["bands"]} == set(
        SENTINEL_REQUIRED + SENTINEL_OPTIONAL
    )
    assert any(w["code"] == "sentinel_swir_resampled" for w in body["warnings"])
    by_key = {item["index_key"]: item for item in body["available_indices"]}
    assert by_key["nbr"]["compatible"] is True
    assert by_key["ndmi"]["compatible"] is True


@requires_database
def test_landsat8_ingest_still_works_alongside_sentinel(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Regression: Landsat 8 path ingest remains intact after 9L."""
    from tests.test_local_scene_ingest import LANDSAT_KEYS, _stage_landsat_scene

    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "landsat-8"
    assert body["metadata"]["platform"] == "Landsat-8"
    assert {b["band_key"] for b in body["bands"]} == set(LANDSAT_KEYS)
    assert all(item["compatible"] for item in body["available_indices"])

"""Fase 9K: local Sentinel-2 scene ingest (10 m bands) + index/RGB smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.core.config import settings
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
    assert body["metadata"]["ingest"]["phase"] == "9K"
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
def test_ingest_sentinel2_skips_misaligned_swir_with_warning(
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
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_REQUIRED)
    codes = {w["code"] for w in body["warnings"]}
    assert "sentinel_swir_not_aligned" in codes or "sentinel_swir_skipped" in codes
    by_key = {item["index_key"]: item for item in body["available_indices"]}
    assert by_key["ndvi"]["compatible"] is True
    assert by_key["nbr"]["compatible"] is False
    assert by_key["ndmi"]["compatible"] is False


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
    assert body["metadata"]["ingest"]["phase"] == "9K"
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_REQUIRED)


@requires_database
def test_upload_sentinel2_swir_mismatch_warns_and_continues(
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
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_REQUIRED)
    assert any(
        w["code"] in ("sentinel_swir_not_aligned", "sentinel_swir_skipped")
        for w in body["warnings"]
    )


@requires_database
def test_landsat8_ingest_still_works_alongside_sentinel(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Regression: Landsat 8 path ingest remains intact after 9K."""
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

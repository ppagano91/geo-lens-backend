"""Fase 9A: local Landsat 8 scene ingest from a GeoTIFF folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.config import settings
from tests.conftest import requires_database

LANDSAT_KEYS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")

MTL_TEXT = """
GROUP = LANDSAT_METADATA_FILE
  GROUP = PRODUCT_CONTENTS
    LANDSAT_PRODUCT_ID = "LC08_L2SP_225084_20260510_20260515_02_T1"
    PROCESSING_LEVEL = "L2SP"
    COLLECTION_NUMBER = 02
  END_GROUP = PRODUCT_CONTENTS
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    SENSOR_ID = "OLI_TIRS"
    WRS_PATH = 225
    WRS_ROW = 84
    DATE_ACQUIRED = 2026-05-10
    CLOUD_COVER = 1.77
  END_GROUP = IMAGE_ATTRIBUTES
END_GROUP = LANDSAT_METADATA_FILE
END
"""


def _write_band(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float = 0.0,
    crs: str = "EPSG:4326",
    origin: tuple[float, float] = (-58.45, -34.55),
    pixel_size: float = 0.001,
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


def _stage_landsat_scene(
    data_root: Path,
    *,
    folder: str = "sample/scenes/landsat8_ingest_test",
    with_mtl: bool = True,
    omit: set[str] | None = None,
    mismatch_size_for: str | None = None,
) -> str:
    scene_dir = data_root / folder
    scene_dir.mkdir(parents=True, exist_ok=True)
    omit = omit or set()

    for key in LANDSAT_KEYS:
        if key in omit:
            continue
        path = scene_dir / f"{key}.tif"
        if mismatch_size_for == key:
            data = np.full((3, 3), 100, dtype=np.uint16)
        else:
            data = np.full((4, 4), 100 if key != "SR_B5" else 300, dtype=np.uint16)
        _write_band(path, data)

    if with_mtl:
        (scene_dir / "LC08_L2SP_225084_20260510_20260515_02_T1_MTL.txt").write_text(
            MTL_TEXT, encoding="utf-8"
        )
    return folder.replace("\\", "/")


@requires_database
def test_ingest_landsat8_valid_scene(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": scene_path,
            "source": "landsat-8",
            "name": "Ingest test LC08",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "landsat-8"
    assert body["sensor"] == "landsat-8"
    assert body["name"] == "Ingest test LC08"
    assert body["acquisition_date"] == "2026-05-10"
    assert body["metadata"]["platform"] == "Landsat-8"
    assert body["metadata"]["sensor"] == "landsat-8"
    assert body["metadata"]["ingest_scene_path"] == scene_path
    assert {b["band_key"] for b in body["bands"]} == set(LANDSAT_KEYS)
    assert all(item["compatible"] for item in body["available_indices"])
    assert body["overwritten"] is False


@requires_database
def test_ingest_missing_sr_b5_returns_clear_error(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root, omit={"SR_B5"})

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )

    assert response.status_code == 422
    assert "SR_B5" in response.json()["detail"]
    assert "Missing required" in response.json()["detail"]


@requires_database
def test_ingest_mismatched_band_size_returns_error(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root, mismatch_size_for="SR_B6")

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "mismatch" in detail.lower()
    assert "SR_B6" in detail


@requires_database
def test_ingest_path_outside_data_root_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": "../outside", "source": "landsat-8"},
    )

    assert response.status_code == 422
    assert "DATA_ROOT" in response.json()["detail"] or "escapes" in response.json()[
        "detail"
    ]


@requires_database
def test_ingest_sets_landsat8_source_and_platform(
    client, tmp_path: Path, monkeypatch
) -> None:
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

    scene = client.get(f"/api/v1/scenes/{body['scene_id']}")
    assert scene.status_code == 200
    scene_body = scene.json()
    assert scene_body["source"] == "landsat-8"
    assert scene_body["metadata"]["platform"] == "Landsat-8"
    assert {b["band_key"] for b in scene_body["bands"]} == set(LANDSAT_KEYS)


@requires_database
def test_ingest_then_compute_ndvi(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root)

    ingest = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
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
    assert result["stats"]["mean"] == pytest.approx(0.5)


@requires_database
def test_ingest_duplicate_without_overwrite_returns_409(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(data_root)

    first = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )
    assert first.status_code == 201, first.text

    second = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )
    assert second.status_code == 409
    assert "already ingested" in second.json()["detail"].lower()


@requires_database
def test_ingest_without_mtl_still_works(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_landsat_scene(
        data_root,
        folder="sample/scenes/LC08_L2SP_225084_20260510_crop",
        with_mtl=False,
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "landsat-8"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "landsat-8"
    assert body["metadata"]["platform"] == "Landsat-8"
    assert any("No MTL.txt" in w for w in body["warnings"])
    assert body["acquisition_date"] == "2026-05-10"  # from folder name token

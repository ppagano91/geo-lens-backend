"""Tests for Fase 9I: DB catalog of derived raster products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sqlalchemy import inspect, text

from app.core.config import settings
from app.models.derived_asset import RasterDerivedAsset
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


def _band_entry(band_key: str, path: Path) -> dict:
    return {
        "band_key": band_key,
        "band_name": band_key,
        "asset_path": str(path.resolve()),
        "nodata": "0",
        "dtype": "uint16",
    }


def _scene_payload(*, bands: list[dict], name: str = "derived catalog test") -> dict:
    return {
        "name": name,
        "source": "landsat-8",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"platform": "Landsat-8", "purpose": "fase_9i"},
        "bands": bands,
    }


def _write_ndvi_bands(root: Path) -> list[dict]:
    red = root / "SR_B4.tif"
    nir = root / "SR_B5.tif"
    _write_band(red, np.full((8, 8), 100, dtype=np.uint16))
    _write_band(nir, np.full((8, 8), 300, dtype=np.uint16))
    return [_band_entry("SR_B4", red), _band_entry("SR_B5", nir)]


def _write_l8_stack(root: Path) -> list[dict]:
    defaults = {
        "SR_B2": 80,
        "SR_B3": 100,
        "SR_B4": 120,
        "SR_B5": 200,
        "SR_B6": 150,
        "SR_B7": 90,
    }
    bands: list[dict] = []
    for key in L8_BANDS:
        path = root / f"{key}.tif"
        _write_band(path, np.full((8, 8), defaults[key], dtype=np.uint16))
        bands.append(_band_entry(key, path))
    return bands


def _create_scene(client, bands: list[dict]) -> str:
    resp = client.post("/api/v1/scenes", json=_scene_payload(bands=bands))
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _create_aoi(client) -> str:
    resp = client.post(
        "/api/v1/aois",
        json={"name": "Derived AOI", "geometry": VALID_POLYGON},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@requires_database
def test_migration_creates_raster_derived_assets_table(db_session) -> None:
    inspector = inspect(db_session.get_bind())
    assert inspector.has_table("raster_derived_assets")
    columns = {col["name"] for col in inspector.get_columns("raster_derived_assets")}
    expected = {
        "id",
        "scene_id",
        "aoi_id",
        "asset_type",
        "product_key",
        "asset_path",
        "preview_path",
        "georef_path",
        "crs",
        "width",
        "height",
        "nodata",
        "dtype",
        "stats",
        "metadata",
        "is_active",
        "deleted_at",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(columns)
    # No binary / bytea column for raster payloads.
    col_types = {
        col["name"]: str(col["type"]).lower()
        for col in inspector.get_columns("raster_derived_assets")
    }
    for name, type_name in col_types.items():
        assert "bytea" not in type_name, f"{name} must not store bytes"
        assert "blob" not in type_name, f"{name} must not store bytes"


@requires_database
def test_compute_and_save_registers_index_asset(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_ndvi_bands(band_dir))

    response = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save"
    )
    assert response.status_code == 200, response.text

    listed = client.get(f"/api/v1/scenes/{scene_id}/derived-assets")
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_type"] == "index"
    assert row["product_key"] == "ndvi"
    assert row["aoi_id"] is None
    assert row["asset_path"] == f"derived/scenes/{scene_id}/ndvi.tif"
    assert row["width"] == 8
    assert row["height"] == 8
    assert isinstance(row["asset_path"], str)
    assert "bytes" not in (row.get("metadata") or {})


@requires_database
def test_crop_by_aoi_registers_index_aoi_crop(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_ndvi_bands(band_dir))
    aoi_id = _create_aoi(client)

    save = client.post(f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save")
    assert save.status_code == 200, save.text

    crop = client.post(
        f"/api/v1/scenes/{scene_id}/indices/ndvi/crop-by-aoi",
        json={"aoi_id": aoi_id, "overwrite": True, "generate_preview": True},
    )
    assert crop.status_code == 200, crop.text

    listed = client.get(
        f"/api/v1/scenes/{scene_id}/derived-assets",
        params={"asset_type": "index_aoi_crop"},
    )
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_type"] == "index_aoi_crop"
    assert row["product_key"] == "ndvi"
    assert row["aoi_id"] == aoi_id
    assert row["asset_path"].endswith(f"/aois/{aoi_id}/ndvi.tif")
    assert row["preview_path"] is not None
    assert row["preview_path"].endswith(".png")


@requires_database
def test_rgb_composite_registers_asset(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_l8_stack(band_dir))

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={
            "preset": "true_color",
            "stretch": "percentile",
            "p_min": 2,
            "p_max": 98,
            "overwrite": True,
        },
    )
    assert response.status_code == 200, response.text

    listed = client.get(
        f"/api/v1/scenes/{scene_id}/derived-assets",
        params={"asset_type": "rgb_composite"},
    )
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_type"] == "rgb_composite"
    assert row["product_key"] == "true_color"
    assert row["asset_path"] == f"derived/scenes/{scene_id}/rgb/true_color.png"
    assert row["aoi_id"] is None


@requires_database
def test_rgb_composite_by_aoi_registers_asset(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_l8_stack(band_dir))
    aoi_id = _create_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={
            "aoi_id": aoi_id,
            "preset": "false_color_vegetation",
            "stretch": "percentile",
            "p_min": 2,
            "p_max": 98,
            "overwrite": True,
        },
    )
    assert response.status_code == 200, response.text

    listed = client.get(
        f"/api/v1/scenes/{scene_id}/derived-assets",
        params={"asset_type": "rgb_composite_aoi"},
    )
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_type"] == "rgb_composite_aoi"
    assert row["product_key"] == "false_color_vegetation"
    assert row["aoi_id"] == aoi_id
    assert row["georef_path"] is not None
    assert row["georef_path"].endswith(".georef.json")

    detail = client.get(f"/api/v1/derived-assets/{row['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == row["id"]


@requires_database
def test_list_derived_assets_by_scene(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_l8_stack(band_dir))

    assert (
        client.post(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save"
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
            json={"preset": "true_color", "overwrite": True},
        ).status_code
        == 200
    )

    listed = client.get(f"/api/v1/scenes/{scene_id}/derived-assets")
    assert listed.status_code == 200
    rows = listed.json()
    types = {row["asset_type"] for row in rows}
    assert "index" in types
    assert "rgb_composite" in types
    assert all(row["scene_id"] == scene_id for row in rows)


@requires_database
def test_overwrite_true_updates_derived_asset_row(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_l8_stack(band_dir))

    first = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True},
    )
    assert first.status_code == 200, first.text
    rows_before = client.get(f"/api/v1/scenes/{scene_id}/derived-assets").json()
    assert len(rows_before) == 1
    asset_id = rows_before[0]["id"]
    created_at = rows_before[0]["created_at"]

    second = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "true_color", "overwrite": True, "p_min": 5, "p_max": 95},
    )
    assert second.status_code == 200, second.text

    rows_after = client.get(f"/api/v1/scenes/{scene_id}/derived-assets").json()
    assert len(rows_after) == 1
    assert rows_after[0]["id"] == asset_id
    assert rows_after[0]["created_at"] == created_at
    assert rows_after[0]["metadata"]["p_min"] == 5
    assert rows_after[0]["metadata"]["p_max"] == 95


@requires_database
def test_derived_assets_store_paths_not_bytes(
    client, db_session, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_ndvi_bands(band_dir))
    assert (
        client.post(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save"
        ).status_code
        == 200
    )

    row = db_session.execute(
        text(
            "SELECT asset_path, preview_path, pg_typeof(asset_path)::text AS path_type "
            "FROM raster_derived_assets WHERE scene_id = :scene_id"
        ),
        {"scene_id": scene_id},
    ).mappings().one()
    assert row["asset_path"].startswith("derived/scenes/")
    assert row["path_type"] == "text"
    assert isinstance(row["asset_path"], str)
    assert len(row["asset_path"]) < 500

    # ORM model must not expose a bytes payload field.
    assert not hasattr(RasterDerivedAsset, "payload")
    assert not hasattr(RasterDerivedAsset, "data")
    assert not hasattr(RasterDerivedAsset, "content")


@requires_database
def test_soft_delete_derived_asset(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    scene_id = _create_scene(client, _write_ndvi_bands(band_dir))
    assert (
        client.post(
            f"/api/v1/scenes/{scene_id}/indices/ndvi/compute-and-save"
        ).status_code
        == 200
    )
    asset_id = client.get(f"/api/v1/scenes/{scene_id}/derived-assets").json()[0]["id"]

    deleted = client.delete(f"/api/v1/derived-assets/{asset_id}")
    assert deleted.status_code == 204

    listed = client.get(f"/api/v1/scenes/{scene_id}/derived-assets")
    assert listed.status_code == 200
    assert listed.json() == []

    detail = client.get(f"/api/v1/derived-assets/{asset_id}")
    assert detail.status_code == 404

    # Physical file remains.
    tif = data_root / f"derived/scenes/{scene_id}/ndvi.tif"
    assert tif.is_file()

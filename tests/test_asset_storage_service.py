"""Unit tests for AssetStorageService (Fase 9C)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.services.asset_storage_service import (
    AssetStorageError,
    AssetStorageService,
)


def test_resolve_read_path_valid_relative(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "sample" / "scenes" / "demo").mkdir(parents=True)
    relative = "sample/scenes/demo/B04.tif"

    storage = AssetStorageService(data_root)
    resolved = storage.resolve_read_path(relative)

    assert resolved == (data_root / relative).resolve()
    assert resolved.is_relative_to(data_root.resolve())


def test_resolve_write_path_matches_read(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    relative = "derived/scenes/x/ndvi.tif"

    storage = AssetStorageService(data_root)
    assert storage.resolve_write_path(relative) == storage.resolve_read_path(relative)


def test_rejects_absolute_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    absolute = str((tmp_path / "outside" / "B04.tif").resolve())

    storage = AssetStorageService(data_root)
    with pytest.raises(AssetStorageError, match="relative"):
        storage.validate_relative_asset_path(absolute)
    with pytest.raises(AssetStorageError, match="relative"):
        storage.resolve_read_path(absolute)
    with pytest.raises(AssetStorageError, match="relative"):
        storage.resolve_write_path(absolute)


def test_rejects_path_traversal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    storage = AssetStorageService(data_root)
    with pytest.raises(AssetStorageError, match="escapes DATA_ROOT"):
        storage.validate_relative_asset_path("../../etc/passwd")
    with pytest.raises(AssetStorageError, match="escapes DATA_ROOT"):
        storage.resolve_read_path("../outside.tif")


def test_exists_checks_file_under_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    scene_dir = data_root / "sample" / "scenes" / "demo"
    scene_dir.mkdir(parents=True)
    raster = scene_dir / "B04.tif"
    raster.write_bytes(b"not-a-real-tif")

    storage = AssetStorageService(data_root)
    assert storage.exists("sample/scenes/demo/B04.tif") is True
    assert storage.exists("sample/scenes/demo/missing.tif") is False


def test_build_derived_asset_path() -> None:
    scene_id = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_derived_asset_path(scene_id, "ndvi", "tif")
        == f"derived/scenes/{scene_id}/ndvi.tif"
    )
    assert (
        storage.build_derived_asset_path(scene_id, "NDWI", ".png")
        == f"derived/scenes/{scene_id}/ndwi.png"
    )


def test_build_derived_aoi_asset_path() -> None:
    scene_id = uuid4()
    aoi_id = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_derived_aoi_asset_path(scene_id, aoi_id, "ndvi", "tif")
        == f"derived/scenes/{scene_id}/aois/{aoi_id}/ndvi.tif"
    )
    assert (
        storage.build_derived_aoi_asset_path(scene_id, aoi_id, "NDMI", ".png")
        == f"derived/scenes/{scene_id}/aois/{aoi_id}/ndmi.png"
    )
    with pytest.raises(AssetStorageError, match="aoi_id"):
        storage.build_derived_aoi_asset_path(scene_id, "  ", "ndvi", "tif")


def test_build_derived_rgb_asset_path() -> None:
    scene_id = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_derived_rgb_asset_path(scene_id, "true_color", "png")
        == f"derived/scenes/{scene_id}/rgb/true_color.png"
    )
    assert (
        storage.build_derived_rgb_asset_path(scene_id, "SWIR_Urban", ".png")
        == f"derived/scenes/{scene_id}/rgb/swir_urban.png"
    )
    with pytest.raises(AssetStorageError, match="preset"):
        storage.build_derived_rgb_asset_path(scene_id, "  ", "png")


def test_build_derived_aoi_rgb_asset_path() -> None:
    scene_id = uuid4()
    aoi_id = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, "true_color", "png"
        )
        == f"derived/scenes/{scene_id}/aois/{aoi_id}/rgb/true_color.png"
    )
    assert (
        storage.build_derived_aoi_rgb_asset_path(
            scene_id, aoi_id, "true_color", "georef.json"
        )
        == f"derived/scenes/{scene_id}/aois/{aoi_id}/rgb/true_color.georef.json"
    )
    with pytest.raises(AssetStorageError, match="aoi_id"):
        storage.build_derived_aoi_rgb_asset_path(scene_id, "  ", "true_color", "png")


def test_build_aligned_band_asset_path() -> None:
    scene_id = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_aligned_band_asset_path(scene_id, "B11_10m.tif")
        == f"derived/scenes/{scene_id}/aligned/B11_10m.tif"
    )
    assert (
        storage.build_aligned_band_asset_path(scene_id, "B12_10m.tif")
        == f"derived/scenes/{scene_id}/aligned/B12_10m.tif"
    )
    with pytest.raises(AssetStorageError, match="filename"):
        storage.build_aligned_band_asset_path(scene_id, "../B11.tif")


def test_build_uploaded_scene_dir() -> None:
    scene_slug = uuid4()
    storage = AssetStorageService(".")

    assert (
        storage.build_uploaded_scene_dir(scene_slug)
        == f"uploaded/scenes/{scene_slug}"
    )
    with pytest.raises(AssetStorageError, match="empty"):
        storage.build_uploaded_scene_dir("  ")


def test_build_uploaded_dem_asset_path() -> None:
    dem_id = uuid4()
    storage = AssetStorageService(".")

    assert storage.build_uploaded_dem_dir(dem_id) == f"uploaded/dems/{dem_id}"
    assert (
        storage.build_uploaded_dem_asset_path(dem_id, "dem.tif")
        == f"uploaded/dems/{dem_id}/dem.tif"
    )
    assert (
        storage.build_uploaded_dem_asset_path(dem_id, "hillshade.png")
        == f"uploaded/dems/{dem_id}/hillshade.png"
    )
    with pytest.raises(AssetStorageError, match="empty"):
        storage.build_uploaded_dem_dir("  ")
    with pytest.raises(AssetStorageError, match="filename"):
        storage.build_uploaded_dem_asset_path(dem_id, "../dem.tif")


def test_validate_normalizes_slashes(tmp_path: Path) -> None:
    storage = AssetStorageService(tmp_path)
    assert (
        storage.validate_relative_asset_path(r"sample\scenes\demo\B04.tif")
        == "sample/scenes/demo/B04.tif"
    )


def test_empty_asset_path_rejected(tmp_path: Path) -> None:
    storage = AssetStorageService(tmp_path)
    with pytest.raises(AssetStorageError, match="empty"):
        storage.validate_relative_asset_path("   ")

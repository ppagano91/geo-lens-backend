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

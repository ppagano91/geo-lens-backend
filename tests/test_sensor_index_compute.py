"""Fase 8B: index compute resolves bands via sensor maps (Sentinel-2 / Landsat 8)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.local_index_compute_service import LocalIndexComputeService


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


def _band(band_key: str, asset_path: Path):
    return SimpleNamespace(
        id=uuid4(),
        band_key=band_key,
        asset_path=str(asset_path.resolve()),
    )


def _scene(*, bands: list, source: str = "local", metadata=None):
    return SimpleNamespace(
        id=uuid4(),
        bands=bands,
        source=source,
        metadata_=metadata,
        is_active=True,
    )


class _FakeRepository:
    def __init__(self, scene) -> None:
        self._scene = scene
        self.db = object()

    def get_by_id(self, scene_id):
        if self._scene is None or self._scene.id != scene_id:
            return None
        return self._scene


def _service_with_scene(scene, *, data_root: Path | None = None) -> LocalIndexComputeService:
    service = LocalIndexComputeService.__new__(LocalIndexComputeService)
    service.repository = _FakeRepository(scene)
    service.data_root = data_root if data_root is not None else Path(".")
    return service


def _write_pair(tmp_path: Path, left_key: str, right_key: str, left_val: int, right_val: int):
    left_path = tmp_path / f"{left_key}.tif"
    right_path = tmp_path / f"{right_key}.tif"
    _write_band(left_path, np.full((4, 4), left_val, dtype=np.uint16))
    _write_band(right_path, np.full((4, 4), right_val, dtype=np.uint16))
    return left_path, right_path


@pytest.mark.parametrize(
    ("index_key", "display", "roles", "s2_keys", "values"),
    [
        # NDVI = (nir - red) / (nir + red) → (300-100)/(300+100) = 0.5
        ("ndvi", "NDVI", ("nir", "red"), ("B08", "B04"), (300, 100)),
        # NDWI = (green - nir) / (green + nir) → (300-100)/(300+100) = 0.5
        ("ndwi", "NDWI", ("green", "nir"), ("B03", "B08"), (300, 100)),
        # NBR = (nir - swir2) / (nir + swir2) → 0.5
        ("nbr", "NBR", ("nir", "swir2"), ("B08", "B12"), (300, 100)),
        # NDMI = (nir - swir1) / (nir + swir1) → 0.5
        ("ndmi", "NDMI", ("nir", "swir1"), ("B08", "B11"), (300, 100)),
    ],
)
def test_compute_indices_sentinel_2_band_keys(
    tmp_path: Path,
    index_key: str,
    display: str,
    roles: tuple[str, str],
    s2_keys: tuple[str, str],
    values: tuple[int, int],
) -> None:
    left_path, right_path = _write_pair(
        tmp_path, s2_keys[0], s2_keys[1], values[0], values[1]
    )
    scene = _scene(
        bands=[_band(s2_keys[0], left_path), _band(s2_keys[1], right_path)],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, index_key)

    assert result.index == display
    assert result.status == "computed"
    assert result.bands_used[roles[0]].band_key == s2_keys[0]
    assert result.bands_used[roles[1]].band_key == s2_keys[1]
    assert result.stats.mean == pytest.approx(0.5)
    assert result.stats.valid_pixels == 16


@pytest.mark.parametrize(
    ("index_key", "display", "roles", "l8_keys", "values"),
    [
        # Reflectance 0.3 / 0.1 after L2 scale+offset → index mean 0.5
        ("ndvi", "NDVI", ("nir", "red"), ("SR_B5", "SR_B4"), (18182, 10909)),
        ("ndwi", "NDWI", ("green", "nir"), ("SR_B3", "SR_B5"), (18182, 10909)),
        ("nbr", "NBR", ("nir", "swir2"), ("SR_B5", "SR_B7"), (18182, 10909)),
        ("ndmi", "NDMI", ("nir", "swir1"), ("SR_B5", "SR_B6"), (18182, 10909)),
    ],
)
def test_compute_indices_landsat_8_band_keys(
    tmp_path: Path,
    index_key: str,
    display: str,
    roles: tuple[str, str],
    l8_keys: tuple[str, str],
    values: tuple[int, int],
) -> None:
    left_path, right_path = _write_pair(
        tmp_path, l8_keys[0], l8_keys[1], values[0], values[1]
    )
    scene = _scene(
        bands=[_band(l8_keys[0], left_path), _band(l8_keys[1], right_path)],
        source="landsat-8",
        metadata={"platform": "Landsat-8"},
    )
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, index_key)

    assert result.index == display
    assert result.status == "computed"
    assert result.bands_used[roles[0]].band_key == l8_keys[0]
    assert result.bands_used[roles[1]].band_key == l8_keys[1]
    assert result.stats.mean == pytest.approx(0.5, abs=1e-4)
    assert result.stats.valid_pixels == 16
    assert result.radiometry is not None
    assert result.radiometry.product_level == "landsat_l2"
    assert result.radiometry.radiometry_type == "surface_reflectance"
    assert result.radiometry.scale_applied is True


def test_compute_and_save_landsat_8_ndvi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_svc = MagicMock()
    monkeypatch.setattr(
        "app.services.local_index_compute_service.DerivedAssetService",
        MagicMock(return_value=mock_svc),
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    red_path = tmp_path / "SR_B4.tif"
    nir_path = tmp_path / "SR_B5.tif"
    _write_band(red_path, np.full((3, 3), 10909, dtype=np.uint16))
    _write_band(nir_path, np.full((3, 3), 18182, dtype=np.uint16))

    scene = _scene(
        bands=[_band("SR_B4", red_path), _band("SR_B5", nir_path)],
        source="landsat-8",
        metadata={"platform": "Landsat-8"},
    )
    service = _service_with_scene(scene, data_root=data_root)
    service.repository.db = object()

    result = service.compute_and_save_index(scene.id, "ndvi")

    assert result.status == "saved"
    assert result.bands_used["red"].band_key == "SR_B4"
    assert result.bands_used["nir"].band_key == "SR_B5"
    assert result.stats.mean == pytest.approx(0.5, abs=1e-4)
    assert Path(result.output.resolved_path).is_file()
    assert result.radiometry is not None
    assert result.radiometry.scale_applied is True
    mock_svc.create_or_update_derived_asset.assert_called_once()
    saved_meta = mock_svc.create_or_update_derived_asset.call_args.kwargs["metadata"]
    assert saved_meta["radiometry"]["product_level"] == "landsat_l2"


def test_synthetic_scene_still_uses_sentinel_like_keys(tmp_path: Path) -> None:
    """Seed-style synthetic scenes keep B0x keys without renaming."""
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    _write_band(red_path, np.full((2, 2), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))
    scene = _scene(
        bands=[_band("B04", red_path), _band("B08", nir_path)],
        source="local",
        metadata={"type": "synthetic", "platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, "ndvi")

    assert result.bands_used["red"].band_key == "B04"
    assert result.bands_used["nir"].band_key == "B08"
    assert result.stats.mean == pytest.approx(0.5)
    assert result.radiometry is not None
    assert result.radiometry.product_level == "synthetic"
    assert result.radiometry.scale_applied is False


def test_landsat_scene_does_not_require_sentinel_keys(tmp_path: Path) -> None:
    """Landsat SR_* bands work without renaming to B0x."""
    from app.services.local_index_compute_service import MissingRequiredBandError

    red_path = tmp_path / "SR_B4.tif"
    nir_path = tmp_path / "SR_B5.tif"
    _write_band(red_path, np.full((2, 2), 10909, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 18182, dtype=np.uint16))
    scene = _scene(
        bands=[_band("SR_B4", red_path), _band("SR_B5", nir_path)],
        source="landsat-8",
        metadata={"platform": "Landsat-8"},
    )
    service = _service_with_scene(scene)

    result = service.compute_index(scene.id, "ndvi")
    assert result.bands_used["red"].band_key == "SR_B4"

    # Same files registered only as Landsat keys fail under Sentinel detection.
    sentinel_scene = _scene(
        bands=[_band("SR_B4", red_path), _band("SR_B5", nir_path)],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    sentinel_service = _service_with_scene(sentinel_scene)
    with pytest.raises(MissingRequiredBandError, match="B0[48]"):
        sentinel_service.compute_index(sentinel_scene.id, "ndvi")


def test_legacy_scene_without_radiometry_metadata_still_computes(
    tmp_path: Path,
) -> None:
    """Old scenes without radiometry keys remain computable (unknown / no scale)."""
    red_path = tmp_path / "B04.tif"
    nir_path = tmp_path / "B08.tif"
    _write_band(red_path, np.full((2, 2), 100, dtype=np.uint16))
    _write_band(nir_path, np.full((2, 2), 300, dtype=np.uint16))
    scene = _scene(
        bands=[_band("B04", red_path), _band("B08", nir_path)],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene)
    result = service.compute_index(scene.id, "ndvi")
    assert result.stats.mean == pytest.approx(0.5)
    assert result.radiometry is not None
    assert result.radiometry.product_level == "unknown"
    assert result.radiometry.scale_applied is False
    assert result.radiometry.warning is not None
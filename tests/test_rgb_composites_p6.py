"""v0.1-P6: additional RGB composite presets (roles, not hardcoded bands)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.config import settings
from app.schemas.rgb_composite import RgbCompositePreviewRequest
from app.services.local_index_compute_service import MissingRequiredBandError
from app.services.rgb_composite_service import (
    RGB_COMPOSITE_REGISTRY,
    RgbCompositeService,
)
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
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
S2_NO_SWIR = ("B02", "B03", "B04", "B08")

EXISTING_PRESETS = (
    "true_color",
    "false_color_vegetation",
    "swir_urban",
    "moisture_vegetation",
)

NEW_PRESETS = (
    "agriculture",
    "geology",
    "burn_scar",
    "water_land",
    "atmospheric_penetration",
)

SWIR_DEPENDENT_PRESETS = (
    "agriculture",
    "geology",
    "burn_scar",
    "atmospheric_penetration",
    "swir_urban",
    "moisture_vegetation",
)

EXPECTED_S2_BANDS = {
    "agriculture": {"red": "B11", "green": "B08", "blue": "B02"},
    "geology": {"red": "B12", "green": "B11", "blue": "B02"},
    "burn_scar": {"red": "B12", "green": "B08", "blue": "B04"},
    "water_land": {"red": "B08", "green": "B03", "blue": "B02"},
    "atmospheric_penetration": {"red": "B12", "green": "B11", "blue": "B08"},
    "true_color": {"red": "B04", "green": "B03", "blue": "B02"},
    "false_color_vegetation": {"red": "B08", "green": "B04", "blue": "B03"},
    "swir_urban": {"red": "B12", "green": "B11", "blue": "B04"},
    "moisture_vegetation": {"red": "B11", "green": "B08", "blue": "B04"},
}

EXPECTED_L8_BANDS = {
    "agriculture": {"red": "SR_B6", "green": "SR_B5", "blue": "SR_B2"},
    "geology": {"red": "SR_B7", "green": "SR_B6", "blue": "SR_B2"},
    "burn_scar": {"red": "SR_B7", "green": "SR_B5", "blue": "SR_B4"},
    "water_land": {"red": "SR_B5", "green": "SR_B3", "blue": "SR_B2"},
    "atmospheric_penetration": {"red": "SR_B7", "green": "SR_B6", "blue": "SR_B5"},
    "true_color": {"red": "SR_B4", "green": "SR_B3", "blue": "SR_B2"},
    "false_color_vegetation": {"red": "SR_B5", "green": "SR_B4", "blue": "SR_B3"},
    "swir_urban": {"red": "SR_B7", "green": "SR_B6", "blue": "SR_B4"},
    "moisture_vegetation": {"red": "SR_B6", "green": "SR_B5", "blue": "SR_B4"},
}


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


def _scene(*, bands: list, source: str, metadata: dict):
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

    def get_by_id(self, scene_id):
        if self._scene is None or self._scene.id != scene_id:
            return None
        return self._scene


def _service_with_scene(scene, *, data_root: Path) -> RgbCompositeService:
    service = RgbCompositeService.__new__(RgbCompositeService)
    service.repository = _FakeRepository(scene)
    service.data_root = data_root
    return service


def _write_stack(
    root: Path,
    band_keys: tuple[str, ...],
    *,
    values: dict[str, int] | None = None,
    shape: tuple[int, int] = (4, 5),
) -> dict[str, Path]:
    defaults = {key: 80 + (i * 20) for i, key in enumerate(band_keys)}
    if values:
        defaults.update(values)
    paths: dict[str, Path] = {}
    for key in band_keys:
        path = root / f"{key}.tif"
        _write_band(path, np.full(shape, defaults[key], dtype=np.uint16))
        paths[key] = path
    return paths


def _band_entry(band_key: str, path: Path) -> dict:
    return {
        "band_key": band_key,
        "band_name": band_key,
        "asset_path": str(path.resolve()),
        "nodata": "0",
        "dtype": "uint16",
    }


def _scene_payload(*, bands: list[dict], source: str, platform: str) -> dict:
    return {
        "name": f"P6 RGB {source}",
        "source": source,
        "acquisition_date": "2025-03-01",
        "cloud_cover": 5.0,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {"platform": platform, "purpose": "v0.1-p6"},
        "bands": bands,
    }


def _create_scene(client, payload: dict) -> str:
    create = client.post("/api/v1/scenes", json=payload)
    assert create.status_code == 201, create.text
    return create.json()["id"]


def _create_aoi(client) -> str:
    resp = client.post(
        "/api/v1/aois",
        json={"name": "P6 RGB AOI", "geometry": VALID_POLYGON},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_registry_keeps_existing_presets_and_adds_new_ones() -> None:
    for key in EXISTING_PRESETS:
        assert key in RGB_COMPOSITE_REGISTRY
    for key in NEW_PRESETS:
        assert key in RGB_COMPOSITE_REGISTRY
    assert RGB_COMPOSITE_REGISTRY["agriculture"].roles == {
        "red": "swir1",
        "green": "nir",
        "blue": "blue",
    }
    assert RGB_COMPOSITE_REGISTRY["geology"].roles == {
        "red": "swir2",
        "green": "swir1",
        "blue": "blue",
    }
    assert RGB_COMPOSITE_REGISTRY["burn_scar"].roles == {
        "red": "swir2",
        "green": "nir",
        "blue": "red",
    }
    assert RGB_COMPOSITE_REGISTRY["water_land"].roles == {
        "red": "nir",
        "green": "green",
        "blue": "blue",
    }
    assert RGB_COMPOSITE_REGISTRY["atmospheric_penetration"].roles == {
        "red": "swir2",
        "green": "swir1",
        "blue": "nir",
    }


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_new_presets_sentinel2(tmp_path: Path, preset: str) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, S2_BANDS)
    scene = _scene(
        bands=[_band(k, p) for k, p in paths.items()],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset=preset, overwrite=True),  # type: ignore[arg-type]
    )

    assert result.status == "generated"
    assert result.preset == preset
    assert result.sensor == "sentinel-2"
    assert result.bands_used == EXPECTED_S2_BANDS[preset]
    assert result.radiometry is not None
    assert (data_root / result.output.asset_path).is_file()
    assert result.output.asset_path == (
        f"derived/scenes/{scene.id}/rgb/{preset}.png"
    )


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_new_presets_landsat8(tmp_path: Path, preset: str) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, L8_BANDS)
    scene = _scene(
        bands=[_band(k, p) for k, p in paths.items()],
        source="landsat-8",
        metadata={"platform": "Landsat-8"},
    )
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset=preset, overwrite=True),  # type: ignore[arg-type]
    )

    assert result.status == "generated"
    assert result.preset == preset
    assert result.sensor == "landsat-8"
    assert result.bands_used == EXPECTED_L8_BANDS[preset]
    assert result.radiometry is not None
    assert (data_root / result.output.asset_path).is_file()


@pytest.mark.parametrize("preset", EXISTING_PRESETS)
def test_existing_presets_still_work_sentinel2(tmp_path: Path, preset: str) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, S2_BANDS)
    scene = _scene(
        bands=[_band(k, p) for k, p in paths.items()],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset=preset, overwrite=True),  # type: ignore[arg-type]
    )

    assert result.preset == preset
    assert result.bands_used == EXPECTED_S2_BANDS[preset]
    assert (data_root / result.output.asset_path).is_file()


@pytest.mark.parametrize("preset", SWIR_DEPENDENT_PRESETS)
def test_missing_sentinel2_swir_raises_for_swir_presets(
    tmp_path: Path, preset: str
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, S2_NO_SWIR)
    scene = _scene(
        bands=[_band(k, p) for k, p in paths.items()],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene, data_root=data_root)

    with pytest.raises(MissingRequiredBandError) as exc_info:
        service.create_preview(
            scene.id,
            RgbCompositePreviewRequest(preset=preset, overwrite=True),  # type: ignore[arg-type]
        )
    assert exc_info.value.band_key in {"B11", "B12"}


def test_water_land_works_without_sentinel2_swir(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, S2_NO_SWIR)
    scene = _scene(
        bands=[_band(k, p) for k, p in paths.items()],
        source="sentinel-2",
        metadata={"platform": "Sentinel-2"},
    )
    service = _service_with_scene(scene, data_root=data_root)

    result = service.create_preview(
        scene.id,
        RgbCompositePreviewRequest(preset="water_land", overwrite=True),
    )
    assert result.bands_used == EXPECTED_S2_BANDS["water_land"]


@requires_database
@pytest.mark.parametrize("preset", ("agriculture", "burn_scar"))
def test_new_presets_by_aoi(client, tmp_path: Path, monkeypatch, preset: str) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, L8_BANDS)
    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[_band_entry(k, p) for k, p in paths.items()],
            source="landsat-8",
            platform="Landsat-8",
        ),
    )
    aoi_id = _create_aoi(client)

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview-by-aoi",
        json={"aoi_id": aoi_id, "preset": preset, "overwrite": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preset"] == preset
    assert body["bands_used"] == EXPECTED_L8_BANDS[preset]
    assert body["radiometry"] is not None
    expected = f"derived/scenes/{scene_id}/aois/{aoi_id}/rgb/{preset}.png"
    assert body["output"]["asset_path"] == expected
    assert (data_root / expected).is_file()


@requires_database
def test_http_missing_sentinel2_swir_returns_422(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, S2_NO_SWIR)
    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[_band_entry(k, p) for k, p in paths.items()],
            source="sentinel-2",
            platform="Sentinel-2",
        ),
    )

    response = client.post(
        f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
        json={"preset": "geology", "overwrite": True},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "B11" in detail or "B12" in detail


@requires_database
def test_new_presets_register_derived_assets_with_radiometry(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    band_dir = tmp_path / "bands"
    band_dir.mkdir()
    paths = _write_stack(band_dir, L8_BANDS)
    scene_id = _create_scene(
        client,
        _scene_payload(
            bands=[_band_entry(k, p) for k, p in paths.items()],
            source="landsat-8",
            platform="Landsat-8",
        ),
    )

    for preset in ("agriculture", "burn_scar"):
        response = client.post(
            f"/api/v1/scenes/{scene_id}/rgb-composites/preview",
            json={"preset": preset, "overwrite": True},
        )
        assert response.status_code == 200, response.text
        assert response.json()["radiometry"] is not None

    listed = client.get(
        f"/api/v1/scenes/{scene_id}/derived-assets",
        params={"asset_type": "rgb_composite"},
    )
    assert listed.status_code == 200
    rows = listed.json()
    by_key = {row["product_key"]: row for row in rows}
    assert set(by_key) == {"agriculture", "burn_scar"}
    for preset, row in by_key.items():
        assert row["asset_path"] == f"derived/scenes/{scene_id}/rgb/{preset}.png"
        assert "radiometry" in row["metadata"]
        assert row["metadata"]["preset"] == preset
        assert row["metadata"]["bands_used"] == EXPECTED_L8_BANDS[preset]

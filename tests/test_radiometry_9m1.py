"""Fase 9M.1: Sentinel-2 radiometry via SAFE metadata / product id / override."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.config import settings
from app.services.radiometry_service import (
    SENTINEL_REFLECTANCE_OFFSET,
    SENTINEL_REFLECTANCE_SCALE,
    RadiometryService,
)
from tests.conftest import requires_database

SENTINEL_KEYS = ("B02", "B03", "B04", "B08")
PRODUCT_ID_L1C = (
    "S2B_MSIL1C_20181226T141039_N0207_R110_T20JLL_20181226T172720"
)
PRODUCT_ID_L2A = (
    "S2A_MSIL2A_20190101T142711_N0211_R053_T20JLL_20190101T163012"
)


def _write_band(path: Path, data: np.ndarray, *, nodata: float = 0.0) -> None:
    height, width = data.shape
    transform = from_origin(-58.4, -34.6, 0.01, 0.01)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def _stage_renamed_sentinel_bands(
    data_root: Path,
    *,
    folder: str = "sample/scenes/t20jll_converted",
    with_safe: str | None = None,
) -> str:
    """Band names like T20JLL_…_B02.tif (no MSIL1C/MSIL2A in filenames)."""
    scene_dir = data_root / folder
    scene_dir.mkdir(parents=True, exist_ok=True)
    for key in SENTINEL_KEYS:
        path = scene_dir / f"T20JLL_20181226T141039_{key}.tif"
        value = 100 if key != "B08" else 300
        _write_band(path, np.full((4, 4), value, dtype=np.uint16))

    if with_safe == "l1c":
        (scene_dir / "MTD_MSIL1C.xml").write_text(
            f"<Level_1C_User_Product><PRODUCT_URI>{PRODUCT_ID_L1C}.SAFE"
            f"</PRODUCT_URI></Level_1C_User_Product>",
            encoding="utf-8",
        )
    elif with_safe == "l2a":
        (scene_dir / "MTD_MSIL2A.xml").write_text(
            f"<Level_2A_User_Product><PRODUCT_URI>{PRODUCT_ID_L2A}.SAFE"
            f"</PRODUCT_URI></Level_2A_User_Product>",
            encoding="utf-8",
        )
    return folder.replace("\\", "/")


def test_manual_product_level_l1c(service: RadiometryService | None = None) -> None:
    svc = service or RadiometryService()
    meta = svc.detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        name="T20JLL_20181226T141039",
        product_level="sentinel_l1c",
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l1c"
    assert meta.radiometry_type == "toa_reflectance"
    assert meta.scale_factor == pytest.approx(SENTINEL_REFLECTANCE_SCALE)
    assert meta.offset == pytest.approx(SENTINEL_REFLECTANCE_OFFSET)
    assert meta.scale_applied is True
    assert meta.radiometry_source == "manual_override"


def test_manual_product_level_l2a() -> None:
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        product_level="sentinel_l2a",
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l2a"
    assert meta.radiometry_type == "surface_reflectance"
    assert meta.scale_applied is True
    assert meta.radiometry_source == "manual_override"


def test_source_product_id_msil1c() -> None:
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        name="T20JLL_converted",
        source_product_id=PRODUCT_ID_L1C,
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l1c"
    assert meta.radiometry_type == "toa_reflectance"
    assert meta.scale_factor == pytest.approx(0.0001)
    assert meta.offset == pytest.approx(0)
    assert meta.scale_applied is True
    assert meta.radiometry_source == "manual_product_id"
    assert meta.source_product_id == PRODUCT_ID_L1C


def test_source_product_id_msil2a() -> None:
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        source_product_id=PRODUCT_ID_L2A,
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l2a"
    assert meta.radiometry_type == "surface_reflectance"
    assert meta.radiometry_source == "manual_product_id"


def test_safe_metadata_file_l1c(tmp_path: Path) -> None:
    xml = tmp_path / "MTD_MSIL1C.xml"
    xml.write_text(
        f"<root><PRODUCT_URI>{PRODUCT_ID_L1C}.SAFE</PRODUCT_URI></root>",
        encoding="utf-8",
    )
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        name="T20JLL_converted",
        metadata_files=[xml],
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l1c"
    assert meta.radiometry_source == "sentinel_metadata"
    assert "MTD_MSIL1C.xml" in meta.metadata_files_detected


def test_safe_metadata_file_l2a(tmp_path: Path) -> None:
    xml = tmp_path / "MTD_MSIL2A.xml"
    xml.write_text("<root>MSIL2A</root>", encoding="utf-8")
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        metadata_files=[xml],
        prefer_stored=False,
    )
    assert meta.product_level == "sentinel_l2a"
    assert meta.radiometry_source == "sentinel_metadata"


def test_renamed_bands_without_hints_are_unknown() -> None:
    meta = RadiometryService().detect_scene_radiometry(
        source="sentinel-2",
        band_keys=list(SENTINEL_KEYS),
        name="T20JLL_20181226T141039",
        scene_path="sample/scenes/t20jll_converted",
        prefer_stored=False,
    )
    assert meta.product_level == "unknown"
    assert meta.scale_applied is False
    assert meta.radiometry_warning is not None


@requires_database
def test_ingest_renamed_bands_with_product_level_l1c(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(data_root)

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": scene_path,
            "source": "sentinel-2",
            "product_level": "sentinel_l1c",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l1c"
    assert body["radiometry"]["radiometry_type"] == "toa_reflectance"
    assert body["radiometry"]["scale_factor"] == pytest.approx(0.0001)
    assert body["radiometry"]["offset"] == pytest.approx(0)
    assert body["radiometry"]["scale_applied"] is True
    assert body["radiometry"]["radiometry_source"] == "manual_override"
    assert any(w["code"] == "radiometry_manual_override" for w in body["warnings"])
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_KEYS)


@requires_database
def test_ingest_renamed_bands_with_product_level_l2a(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(data_root, folder="sample/scenes/t20_l2a")

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": scene_path,
            "source": "sentinel-2",
            "product_level": "sentinel_l2a",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l2a"
    assert body["radiometry"]["radiometry_type"] == "surface_reflectance"
    assert body["radiometry"]["scale_applied"] is True


@requires_database
def test_ingest_with_source_product_id_msil1c(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(
        data_root, folder="sample/scenes/t20_pid"
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={
            "scene_path": scene_path,
            "source": "sentinel-2",
            "source_product_id": PRODUCT_ID_L1C,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l1c"
    assert body["radiometry"]["radiometry_source"] == "manual_product_id"
    assert body["radiometry"]["source_product_id"] == PRODUCT_ID_L1C
    assert any(
        w["code"] == "radiometry_detected_from_product_id" for w in body["warnings"]
    )


@requires_database
def test_ingest_with_safe_mtd_msil1c(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(
        data_root, folder="sample/scenes/t20_safe_l1c", with_safe="l1c"
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l1c"
    assert body["radiometry"]["radiometry_source"] == "sentinel_metadata"
    assert "MTD_MSIL1C.xml" in body["metadata_files_detected"]
    assert any(
        w["code"] == "radiometry_detected_from_safe_metadata" for w in body["warnings"]
    )
    # SAFE file must not appear as a raster band.
    assert all(not b["band_key"].lower().endswith(".xml") for b in body["bands"])


@requires_database
def test_ingest_with_safe_mtd_msil2a(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(
        data_root, folder="sample/scenes/t20_safe_l2a", with_safe="l2a"
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l2a"
    assert body["radiometry"]["radiometry_source"] == "sentinel_metadata"


@requires_database
def test_ingest_renamed_bands_unknown_without_hints(
    client, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))
    scene_path = _stage_renamed_sentinel_bands(
        data_root, folder="sample/scenes/t20_unknown"
    )

    response = client.post(
        "/api/v1/ingest/local-scene",
        json={"scene_path": scene_path, "source": "sentinel-2"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "unknown"
    assert body["radiometry"]["scale_applied"] is False
    assert any(w["code"] == "radiometry_unknown" for w in body["warnings"])


@requires_database
def test_upload_with_mtd_msil1c_xml(client, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    files = []
    for key in SENTINEL_KEYS:
        band_path = tmp_path / f"T20JLL_20181226T141039_{key}.tif"
        value = 100 if key != "B08" else 300
        _write_band(band_path, np.full((4, 4), value, dtype=np.uint16))
        files.append(
            (
                "files",
                (band_path.name, band_path.read_bytes(), "image/tiff"),
            )
        )
    xml_name = "MTD_MSIL1C.xml"
    xml_bytes = (
        f"<root><PRODUCT_URI>{PRODUCT_ID_L1C}.SAFE</PRODUCT_URI></root>"
    ).encode("utf-8")
    files.append(("files", (xml_name, xml_bytes, "application/xml")))

    response = client.post(
        "/api/v1/ingest/upload-scene",
        data={"source": "sentinel-2"},
        files=files,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["radiometry"]["product_level"] == "sentinel_l1c"
    assert body["radiometry"]["radiometry_source"] == "sentinel_metadata"
    assert "MTD_MSIL1C.xml" in body["metadata_files_detected"]
    assert {b["band_key"] for b in body["bands"]} == set(SENTINEL_KEYS)


@requires_database
def test_landsat8_ingest_unchanged(client, tmp_path: Path, monkeypatch) -> None:
    from tests.test_local_scene_ingest import _stage_landsat_scene

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
    assert body["radiometry"]["product_level"] == "landsat_l2"
    assert body["radiometry"]["radiometry_type"] == "surface_reflectance"
    assert body["radiometry"]["scale_applied"] is True

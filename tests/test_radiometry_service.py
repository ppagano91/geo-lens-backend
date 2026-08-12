"""Unit tests for RadiometryService (Fase 9M)."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.services.radiometry_service import (
    LANDSAT_L2_OFFSET,
    LANDSAT_L2_SCALE,
    SENTINEL_REFLECTANCE_OFFSET,
    SENTINEL_REFLECTANCE_SCALE,
    UNKNOWN_RADIOMETRY_WARNING,
    RadiometryMetadata,
    RadiometryService,
)


@pytest.fixture
def service() -> RadiometryService:
    return RadiometryService()


def test_detect_landsat_l2_surface_reflectance(service: RadiometryService) -> None:
    meta = service.detect_scene_radiometry(
        source="landsat-8",
        name="LC08_L2SP_225084_20260510_20260515_02_T1",
        band_keys=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        product_id="LC08_L2SP_225084_20260510_20260515_02_T1",
    )
    assert meta.product_level == "landsat_l2"
    assert meta.radiometry_type == "surface_reflectance"
    assert meta.scale_factor == pytest.approx(LANDSAT_L2_SCALE)
    assert meta.offset == pytest.approx(LANDSAT_L2_OFFSET)
    assert meta.scale_applied is True
    assert meta.radiometry_warning is None


def test_detect_sentinel_l1c_toa(service: RadiometryService) -> None:
    meta = service.detect_scene_radiometry(
        source="sentinel-2",
        name="S2A_MSIL1C_20240101T142711_N0509_R053_T20HNH",
        band_keys=["B02", "B03", "B04", "B08"],
        product_id="S2A_MSIL1C_20240101T142711_N0509_R053_T20HNH",
    )
    assert meta.product_level == "sentinel_l1c"
    assert meta.radiometry_type == "toa_reflectance"
    assert meta.scale_factor == pytest.approx(SENTINEL_REFLECTANCE_SCALE)
    assert meta.offset == pytest.approx(SENTINEL_REFLECTANCE_OFFSET)
    assert meta.scale_applied is True


def test_detect_sentinel_l2a_surface(service: RadiometryService) -> None:
    meta = service.detect_scene_radiometry(
        source="sentinel-2",
        name="S2B_MSIL2A_20240202T142719_N0510_R053_T20HNH",
        band_keys=["B02", "B03", "B04", "B08", "B11", "B12"],
        product_id="S2B_MSIL2A_20240202T142719_N0510_R053_T20HNH",
    )
    assert meta.product_level == "sentinel_l2a"
    assert meta.radiometry_type == "surface_reflectance"
    assert meta.scale_factor == pytest.approx(SENTINEL_REFLECTANCE_SCALE)
    assert meta.offset == pytest.approx(SENTINEL_REFLECTANCE_OFFSET)
    assert meta.scale_applied is True


def test_detect_synthetic_no_scale(service: RadiometryService) -> None:
    meta = service.detect_scene_radiometry(
        source="synthetic-sentinel-2",
        name="synthetic_scene",
        band_keys=["B02", "B04", "B08"],
    )
    assert meta.product_level == "synthetic"
    assert meta.radiometry_type == "synthetic"
    assert meta.scale_factor is None
    assert meta.offset is None
    assert meta.scale_applied is False


def test_detect_unknown_when_sentinel_level_missing(
    service: RadiometryService,
) -> None:
    meta = service.detect_scene_radiometry(
        source="sentinel-2",
        name="local_s2_crop",
        band_keys=["B02", "B03", "B04", "B08"],
    )
    assert meta.product_level == "unknown"
    assert meta.radiometry_type == "unknown"
    assert meta.scale_applied is False
    assert meta.radiometry_warning == UNKNOWN_RADIOMETRY_WARNING


def test_apply_landsat_scaling_preserves_nodata(service: RadiometryService) -> None:
    radiometry = RadiometryMetadata(
        product_level="landsat_l2",
        radiometry_type="surface_reflectance",
        scale_factor=LANDSAT_L2_SCALE,
        offset=LANDSAT_L2_OFFSET,
        scale_applied=True,
    )
    array = np.array([[10909.0, np.nan], [0.0, 18182.0]], dtype=np.float32)
    scaled = service.apply_radiometric_scaling(array, nodata=0.0, radiometry=radiometry)
    assert np.isnan(scaled[0, 1])
    assert np.isnan(scaled[1, 0])
    assert scaled[0, 0] == pytest.approx(0.1, abs=1e-5)
    assert scaled[1, 1] == pytest.approx(0.3, abs=1e-5)


def test_apply_scaling_noop_when_not_applied(service: RadiometryService) -> None:
    radiometry = RadiometryMetadata(
        product_level="unknown",
        radiometry_type="unknown",
        scale_applied=False,
    )
    array = np.array([[100.0, 200.0]], dtype=np.float32)
    scaled = service.apply_radiometric_scaling(array, nodata=None, radiometry=radiometry)
    np.testing.assert_array_equal(scaled, array)


def test_prefers_stored_scene_metadata(service: RadiometryService) -> None:
    scene = SimpleNamespace(
        id=uuid4(),
        source="landsat-8",
        name="anything",
        metadata_={
            "product_level": "landsat_l2",
            "radiometry_type": "surface_reflectance",
            "scale_factor": LANDSAT_L2_SCALE,
            "offset": LANDSAT_L2_OFFSET,
            "scale_applied": True,
            "radiometry_source": "landsat_mtl",
        },
    )
    meta = service.detect_scene_radiometry(scene)
    assert meta.product_level == "landsat_l2"
    assert meta.scale_applied is True


def test_build_response_metadata(service: RadiometryService) -> None:
    radiometry = RadiometryMetadata(
        product_level="sentinel_l1c",
        radiometry_type="toa_reflectance",
        scale_factor=SENTINEL_REFLECTANCE_SCALE,
        offset=0.0,
        scale_applied=True,
        radiometry_warning=None,
    )
    payload = service.build_radiometry_response_metadata(radiometry)
    assert payload["product_level"] == "sentinel_l1c"
    assert payload["warning"] is None
    assert payload["scale_applied"] is True

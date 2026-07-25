"""Unit tests for sensor detection and band-role maps (Fase 8B)."""

from __future__ import annotations

import pytest

from app.raster.sensors import (
    DEFAULT_SENSOR,
    LANDSAT_8_BAND_MAP,
    SENTINEL_2_BAND_MAP,
    SENSOR_LANDSAT_8,
    SENSOR_SENTINEL_2,
    SENSOR_SYNTHETIC_SENTINEL_2,
    detect_sensor,
    normalize_sensor_token,
    resolve_band_key,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sentinel-2", SENSOR_SENTINEL_2),
        ("sentinel2", SENSOR_SENTINEL_2),
        ("S2", SENSOR_SENTINEL_2),
        ("Landsat-8", SENSOR_LANDSAT_8),
        ("landsat8", SENSOR_LANDSAT_8),
        ("LC08", SENSOR_LANDSAT_8),
        ("L8", SENSOR_LANDSAT_8),
        ("synthetic-sentinel-2", SENSOR_SYNTHETIC_SENTINEL_2),
        ("synthetic", SENSOR_SYNTHETIC_SENTINEL_2),
        ("local", None),
        ("", None),
    ],
)
def test_normalize_sensor_token(raw: str, expected: str | None) -> None:
    assert normalize_sensor_token(raw) == expected


def test_detect_sensor_from_platform_sentinel() -> None:
    assert (
        detect_sensor(source="local", metadata={"platform": "Sentinel-2"})
        == SENSOR_SENTINEL_2
    )


def test_detect_sensor_synthetic_sentinel_from_type_and_platform() -> None:
    assert (
        detect_sensor(
            source="local",
            metadata={"type": "synthetic", "platform": "Sentinel-2"},
        )
        == SENSOR_SYNTHETIC_SENTINEL_2
    )


def test_detect_sensor_from_source_landsat() -> None:
    assert detect_sensor(source="landsat-8", metadata=None) == SENSOR_LANDSAT_8


def test_detect_sensor_from_metadata_sensor() -> None:
    assert (
        detect_sensor(
            source="local",
            metadata={"sensor": "landsat-8", "platform": "Sentinel-2"},
        )
        == SENSOR_LANDSAT_8
    )


def test_detect_sensor_defaults_to_sentinel() -> None:
    assert detect_sensor(source="local", metadata={"purpose": "test"}) == DEFAULT_SENSOR


def test_detect_sensor_ignores_unknown_platform() -> None:
    assert detect_sensor(source="local", metadata={"platform": "Planet"}) == DEFAULT_SENSOR


def test_resolve_band_keys_sentinel_and_landsat() -> None:
    assert resolve_band_key(SENSOR_SENTINEL_2, "red") == SENTINEL_2_BAND_MAP["red"]
    assert resolve_band_key(SENSOR_LANDSAT_8, "red") == LANDSAT_8_BAND_MAP["red"]
    assert resolve_band_key(SENSOR_SYNTHETIC_SENTINEL_2, "nir") == "B08"
    assert resolve_band_key(SENSOR_LANDSAT_8, "swir2") == "SR_B7"

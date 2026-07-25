"""Sensor detection and spectral role → band_key maps for local index compute.

Fase 8B: resolve physical band keys from scene ``source`` / ``metadata.platform``
without a DB migration. Index formulas stay role-based (nir, red, …).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

# Canonical sensor ids used by local compute.
SENSOR_SENTINEL_2 = "sentinel-2"
SENSOR_LANDSAT_8 = "landsat-8"
SENSOR_SYNTHETIC_SENTINEL_2 = "synthetic-sentinel-2"

SUPPORTED_SENSORS: tuple[str, ...] = (
    SENSOR_SENTINEL_2,
    SENSOR_LANDSAT_8,
    SENSOR_SYNTHETIC_SENTINEL_2,
)

# Role → native product band_key.
SENTINEL_2_BAND_MAP: dict[str, str] = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
}

LANDSAT_8_BAND_MAP: dict[str, str] = {
    "blue": "SR_B2",
    "green": "SR_B3",
    "red": "SR_B4",
    "nir": "SR_B5",
    "swir1": "SR_B6",
    "swir2": "SR_B7",
}

# Synthetic fixtures reuse Sentinel-2-like keys (seed / create_sample_rasters).
SYNTHETIC_SENTINEL_2_BAND_MAP: dict[str, str] = dict(SENTINEL_2_BAND_MAP)

SENSOR_BAND_MAPS: dict[str, Mapping[str, str]] = {
    SENSOR_SENTINEL_2: SENTINEL_2_BAND_MAP,
    SENSOR_LANDSAT_8: LANDSAT_8_BAND_MAP,
    SENSOR_SYNTHETIC_SENTINEL_2: SYNTHETIC_SENTINEL_2_BAND_MAP,
}

# Normalized token → canonical sensor id.
_SENSOR_ALIASES: dict[str, str] = {
    "sentinel-2": SENSOR_SENTINEL_2,
    "sentinel2": SENSOR_SENTINEL_2,
    "s2": SENSOR_SENTINEL_2,
    "msi": SENSOR_SENTINEL_2,
    "landsat-8": SENSOR_LANDSAT_8,
    "landsat8": SENSOR_LANDSAT_8,
    "l8": SENSOR_LANDSAT_8,
    "lc08": SENSOR_LANDSAT_8,
    "oli": SENSOR_LANDSAT_8,
    "synthetic-sentinel-2": SENSOR_SYNTHETIC_SENTINEL_2,
    "synthetic-sentinel2": SENSOR_SYNTHETIC_SENTINEL_2,
    "synthetic": SENSOR_SYNTHETIC_SENTINEL_2,
}

DEFAULT_SENSOR = SENSOR_SENTINEL_2


class UnknownSensorError(Exception):
    """Explicit sensor token did not match a supported sensor."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"Unsupported sensor '{token}'")


def normalize_sensor_token(value: str) -> Optional[str]:
    """Map a free-form platform/source string to a canonical sensor id, if known."""
    token = value.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in token:
        token = token.replace("--", "-")
    token = token.strip("-")
    if not token:
        return None
    return _SENSOR_ALIASES.get(token)


def _is_synthetic_metadata(metadata: Mapping[str, Any] | None, source: str | None) -> bool:
    if metadata:
        raw_type = metadata.get("type")
        if isinstance(raw_type, str) and raw_type.strip().lower() == "synthetic":
            return True
    if source and "synthetic" in source.strip().lower():
        return True
    return False


def detect_sensor(
    *,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Detect sensor from ``metadata.sensor`` / ``metadata.platform``, then ``source``.

    Priority:
    1. ``metadata["sensor"]``
    2. ``metadata["platform"]``
    3. ``source``
    4. Synthetic type without an explicit non-sentinel sensor → ``synthetic-sentinel-2``
    5. Default ``sentinel-2`` (backward compatible with existing B0x scenes)

    When platform/source resolves to Sentinel-2 and the scene is marked synthetic,
    returns ``synthetic-sentinel-2`` (same band keys as Sentinel-2).
    """
    meta = metadata or {}

    for key in ("sensor", "platform"):
        raw = meta.get(key)
        if isinstance(raw, str) and raw.strip():
            detected = normalize_sensor_token(raw)
            if detected is None:
                # Unrecognized platform/sensor strings are ignored (no migration).
                continue
            if detected == SENSOR_SENTINEL_2 and _is_synthetic_metadata(meta, source):
                return SENSOR_SYNTHETIC_SENTINEL_2
            return detected

    if source and source.strip():
        detected = normalize_sensor_token(source)
        if detected is not None:
            return detected
        # Free-form sources like "local" are not sensors; keep looking.

    if _is_synthetic_metadata(meta, source):
        return SENSOR_SYNTHETIC_SENTINEL_2

    return DEFAULT_SENSOR


def get_band_map(sensor: str) -> Mapping[str, str]:
    """Return role → band_key map for a canonical sensor id."""
    try:
        return SENSOR_BAND_MAPS[sensor]
    except KeyError as exc:
        raise UnknownSensorError(sensor) from exc


def resolve_band_key(sensor: str, role: str) -> str:
    """Resolve a spectral role to the native band_key for ``sensor``."""
    band_map = get_band_map(sensor)
    try:
        return band_map[role]
    except KeyError as exc:
        raise KeyError(f"Role '{role}' is not defined for sensor '{sensor}'") from exc


__all__ = [
    "SENSOR_SENTINEL_2",
    "SENSOR_LANDSAT_8",
    "SENSOR_SYNTHETIC_SENTINEL_2",
    "SUPPORTED_SENSORS",
    "SENTINEL_2_BAND_MAP",
    "LANDSAT_8_BAND_MAP",
    "SYNTHETIC_SENTINEL_2_BAND_MAP",
    "SENSOR_BAND_MAPS",
    "DEFAULT_SENSOR",
    "UnknownSensorError",
    "normalize_sensor_token",
    "detect_sensor",
    "get_band_map",
    "resolve_band_key",
]

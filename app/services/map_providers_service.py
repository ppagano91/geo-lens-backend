"""Build public map-provider config from env (v0.1-P5.2).

No tile proxy, no DB, no caching. The MapTiler TileJSON URL includes the
API key when configured; do not log that key.
"""

from __future__ import annotations

from urllib.parse import urlencode

from app.core.config import settings
from app.schemas.map_providers import (
    MapProviderInfo,
    MapProvidersConfig,
    MaptilerProviderConfig,
)

MAPTILER_TERRAIN_RGB_TILES_JSON = (
    "https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json"
)


def _configured_maptiler_key() -> str | None:
    key = settings.maptiler_api_key
    if key is None:
        return None
    stripped = key.strip()
    return stripped or None


def _maptiler_tiles_json_url(key: str) -> str:
    return f"{MAPTILER_TERRAIN_RGB_TILES_JSON}?{urlencode({'key': key})}"


def get_map_providers_config() -> MapProvidersConfig:
    key = _configured_maptiler_key()
    maptiler_enabled = key is not None
    tiles_url = _maptiler_tiles_json_url(key) if key is not None else None

    return MapProvidersConfig(
        maptiler=MaptilerProviderConfig(
            enabled=maptiler_enabled,
            terrain_rgb_tiles_json_url=tiles_url,
        ),
        providers=[
            MapProviderInfo(
                id="maptiler",
                name="MapTiler Terrain RGB",
                type="terrain",
                requires_key=True,
                available=maptiler_enabled,
            ),
            MapProviderInfo(
                id="aws-terrarium",
                name="AWS / Mapzen Terrarium",
                type="terrain",
                requires_key=False,
                available=True,
            ),
            MapProviderInfo(
                id="maplibre-demo",
                name="MapLibre demo terrain",
                type="terrain",
                requires_key=False,
                available=True,
            ),
        ],
    )


__all__ = [
    "MAPTILER_TERRAIN_RGB_TILES_JSON",
    "get_map_providers_config",
]

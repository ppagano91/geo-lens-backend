"""Public map-provider configuration (v0.1-P5.2)."""

from fastapi import APIRouter

from app.schemas.map_providers import MapProvidersConfig
from app.services.map_providers_service import get_map_providers_config

router = APIRouter()


@router.get(
    "/config",
    response_model=MapProvidersConfig,
    summary="Map provider config for external terrain",
)
def get_map_providers() -> MapProvidersConfig:
    """Return which terrain providers the UI may enable.

    MapTiler is available only when ``MAPTILER_API_KEY`` is set. The TileJSON
    URL includes the key so MapLibre can fetch tiles directly (no proxy).
    """
    return get_map_providers_config()

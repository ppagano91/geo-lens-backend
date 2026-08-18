"""Public map-provider config for the frontend (v0.1-P5.2).

Does not expose full Settings. MapTiler TileJSON URL may include the API
key; that is intentional until a tile proxy exists.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MaptilerProviderConfig(BaseModel):
    enabled: bool
    terrain_rgb_tiles_json_url: Optional[str] = None


class MapProviderInfo(BaseModel):
    id: str
    name: str
    type: Literal["terrain"] = "terrain"
    requires_key: bool
    available: bool


class MapProvidersConfig(BaseModel):
    maptiler: MaptilerProviderConfig
    providers: list[MapProviderInfo] = Field(min_length=1)


__all__ = [
    "MapProviderInfo",
    "MapProvidersConfig",
    "MaptilerProviderConfig",
]

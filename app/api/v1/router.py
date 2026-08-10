from fastapi import APIRouter

from app.api.v1.endpoints import (
    aois,
    derived_assets,
    health,
    indices,
    ingest,
    raster_bands,
    scenes,
    spatial_coverage,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(aois.router, prefix="/aois", tags=["aois"])
api_router.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
api_router.include_router(
    derived_assets.router,
    prefix="/derived-assets",
    tags=["derived-assets"],
)
api_router.include_router(indices.router, prefix="/indices", tags=["indices"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(
    spatial_coverage.router,
    prefix="/spatial-coverage",
    tags=["spatial-coverage"],
)
api_router.include_router(
    raster_bands.router,
    prefix="/raster-bands",
    tags=["raster-bands"],
)

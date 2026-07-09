from fastapi import APIRouter

from app.api.v1.endpoints import aois, health, indices, scenes

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(aois.router, prefix="/aois", tags=["aois"])
api_router.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
api_router.include_router(indices.router, prefix="/indices", tags=["indices"])

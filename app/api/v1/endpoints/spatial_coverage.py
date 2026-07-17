from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.coverage import SpatialCoverageRead
from app.services.aoi_service import AoiNotFoundError
from app.services.coverage_service import SpatialCoverageService
from app.services.scene_service import SceneNotFoundError

router = APIRouter()


@router.get(
    "/aoi/{aoi_id}/scene/{scene_id}",
    response_model=SpatialCoverageRead,
)
def get_spatial_coverage(
    aoi_id: UUID,
    scene_id: UUID,
    db: Session = Depends(get_db),
) -> SpatialCoverageRead:
    """Evaluate spatial coverage of an AOI against a scene footprint (PostGIS)."""
    service = SpatialCoverageService(db)
    try:
        return service.evaluate(aoi_id=aoi_id, scene_id=scene_id)
    except AoiNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI {aoi_id} not found",
        ) from exc
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.aoi import Aoi
from app.repositories.aoi_repository import AoiRepository
from app.schemas.aoi import AoiCreate, AoiRead
from app.services.geometry import (
    GeometryValidationError,
    db_element_to_geojson,
    polygon_to_db_element,
    validate_polygon_geojson,
)


class AoiNotFoundError(Exception):
    pass


class AoiService:
    def __init__(self, db: Session) -> None:
        self.repository = AoiRepository(db)

    def create(self, payload: AoiCreate) -> AoiRead:
        polygon = validate_polygon_geojson(payload.geometry)
        aoi = Aoi(
            name=payload.name,
            description=payload.description,
            geom=polygon_to_db_element(polygon),
            properties=payload.properties,
        )
        created = self.repository.create(aoi)
        return self._to_read(created)

    def list(
        self, limit: int, offset: int, *, include_inactive: bool = False
    ) -> list[AoiRead]:
        aois = self.repository.list(
            limit=limit, offset=offset, include_inactive=include_inactive
        )
        return [self._to_read(aoi) for aoi in aois]

    def get(self, aoi_id: UUID) -> AoiRead:
        aoi = self.repository.get_by_id(aoi_id)
        if aoi is None or not aoi.is_active:
            raise AoiNotFoundError(str(aoi_id))
        return self._to_read(aoi)

    def delete(self, aoi_id: UUID) -> None:
        """Logically deactivate an AOI. Idempotent if already inactive."""
        aoi = self.repository.get_by_id(aoi_id)
        if aoi is None:
            raise AoiNotFoundError(str(aoi_id))
        self.repository.soft_delete(aoi)

    @staticmethod
    def _to_read(aoi: Aoi) -> AoiRead:
        return AoiRead(
            id=aoi.id,
            name=aoi.name,
            description=aoi.description,
            geometry=db_element_to_geojson(aoi.geom),
            properties=aoi.properties,
            is_active=aoi.is_active,
            deleted_at=aoi.deleted_at,
            created_at=aoi.created_at,
            updated_at=aoi.updated_at,
        )


__all__ = ["AoiService", "AoiNotFoundError", "GeometryValidationError"]

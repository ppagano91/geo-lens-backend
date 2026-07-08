from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.band import RasterBand
from app.models.scene import RasterScene
from app.repositories.scene_repository import SceneRepository
from app.schemas.band import BandCreate, BandRead
from app.schemas.scene import SceneCreate, SceneListItem, SceneRead
from app.services.geometry import (
    GeometryValidationError,
    db_element_to_geojson,
    polygon_to_db_element,
    validate_polygon_geojson,
)


class SceneNotFoundError(Exception):
    pass


class BandKeyDuplicateError(Exception):
    pass


class SceneService:
    def __init__(self, db: Session) -> None:
        self.repository = SceneRepository(db)

    def create(self, payload: SceneCreate) -> SceneRead:
        self._validate_unique_band_keys(payload.bands)
        polygon = validate_polygon_geojson(payload.footprint)
        scene = RasterScene(
            name=payload.name,
            source=payload.source,
            acquisition_date=payload.acquisition_date,
            cloud_cover=payload.cloud_cover,
            footprint=polygon_to_db_element(polygon),
            metadata_=payload.metadata,
            bands=[self._band_from_create(band) for band in payload.bands],
        )
        created = self.repository.create(scene)
        return self._to_read(created)

    def list(self, limit: int, offset: int) -> list[SceneListItem]:
        scenes = self.repository.list(limit=limit, offset=offset)
        return [self._to_list_item(scene) for scene in scenes]

    def get(self, scene_id: UUID) -> SceneRead:
        scene = self.repository.get_by_id(scene_id)
        if scene is None:
            raise SceneNotFoundError(str(scene_id))
        return self._to_read(scene)

    def list_bands(self, scene_id: UUID) -> list[BandRead]:
        scene = self.repository.get_by_id(scene_id)
        if scene is None:
            raise SceneNotFoundError(str(scene_id))
        bands = self.repository.list_bands(scene_id)
        return [self._band_to_read(band) for band in bands]

    def delete(self, scene_id: UUID) -> None:
        scene = self.repository.get_by_id(scene_id)
        if scene is None:
            raise SceneNotFoundError(str(scene_id))
        self.repository.delete(scene)

    @staticmethod
    def _validate_unique_band_keys(bands: list[BandCreate]) -> None:
        keys = [band.band_key for band in bands]
        if len(keys) != len(set(keys)):
            raise BandKeyDuplicateError("duplicate band_key within the same scene")

    @staticmethod
    def _band_from_create(payload: BandCreate) -> RasterBand:
        return RasterBand(
            band_key=payload.band_key,
            band_name=payload.band_name,
            description=payload.description,
            resolution=payload.resolution,
            asset_path=payload.asset_path,
            nodata=payload.nodata,
            dtype=payload.dtype,
            metadata_=payload.metadata,
        )

    @staticmethod
    def _band_to_read(band: RasterBand) -> BandRead:
        return BandRead(
            id=band.id,
            scene_id=band.scene_id,
            band_key=band.band_key,
            band_name=band.band_name,
            description=band.description,
            resolution=band.resolution,
            asset_path=band.asset_path,
            nodata=band.nodata,
            dtype=band.dtype,
            metadata=band.metadata_,
            created_at=band.created_at,
        )

    def _to_list_item(self, scene: RasterScene) -> SceneListItem:
        return SceneListItem(
            id=scene.id,
            name=scene.name,
            source=scene.source,
            acquisition_date=scene.acquisition_date,
            cloud_cover=scene.cloud_cover,
            footprint=db_element_to_geojson(scene.footprint),
            metadata=scene.metadata_,
            created_at=scene.created_at,
            updated_at=scene.updated_at,
        )

    def _to_read(self, scene: RasterScene) -> SceneRead:
        return SceneRead(
            **self._to_list_item(scene).model_dump(),
            bands=[self._band_to_read(band) for band in scene.bands],
        )


__all__ = [
    "SceneService",
    "SceneNotFoundError",
    "BandKeyDuplicateError",
    "GeometryValidationError",
]

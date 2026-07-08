from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models.band import RasterBand
from app.models.scene import RasterScene


class SceneRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, scene: RasterScene) -> RasterScene:
        self.db.add(scene)
        self.db.commit()
        self.db.refresh(scene)
        return scene

    def get_by_id(self, scene_id: UUID) -> RasterScene | None:
        stmt = (
            select(RasterScene)
            .where(RasterScene.id == scene_id)
            .options(joinedload(RasterScene.bands))
        )
        return self.db.scalars(stmt).unique().one_or_none()

    def list(self, limit: int, offset: int) -> list[RasterScene]:
        stmt = (
            select(RasterScene)
            .order_by(RasterScene.acquisition_date.desc(), RasterScene.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(stmt).all())

    def list_bands(self, scene_id: UUID) -> list[RasterBand]:
        stmt = (
            select(RasterBand)
            .where(RasterBand.scene_id == scene_id)
            .order_by(RasterBand.band_key)
        )
        return list(self.db.scalars(stmt).all())

    def delete(self, scene: RasterScene) -> None:
        self.db.delete(scene)
        self.db.commit()

from __future__ import annotations

from datetime import datetime, timezone
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

    def find_by_ingest_scene_path(self, scene_path: str) -> RasterScene | None:
        """Return an active scene previously ingested from the same relative path."""
        normalized = scene_path.strip().replace("\\", "/")
        stmt = (
            select(RasterScene)
            .where(RasterScene.metadata_["ingest_scene_path"].as_string() == normalized)
            .where(RasterScene.is_active.is_(True))
            .options(joinedload(RasterScene.bands))
            .limit(1)
        )
        return self.db.scalars(stmt).unique().one_or_none()

    def list(
        self, limit: int, offset: int, *, include_inactive: bool = False
    ) -> list[RasterScene]:
        stmt = select(RasterScene)
        if not include_inactive:
            stmt = stmt.where(RasterScene.is_active.is_(True))
        stmt = stmt.order_by(
            RasterScene.acquisition_date.desc(),
            RasterScene.created_at.desc(),
        ).limit(limit).offset(offset)
        return list(self.db.scalars(stmt).all())

    def list_bands(self, scene_id: UUID) -> list[RasterBand]:
        stmt = (
            select(RasterBand)
            .where(RasterBand.scene_id == scene_id)
            .order_by(RasterBand.band_key)
        )
        return list(self.db.scalars(stmt).all())

    def soft_delete(self, scene: RasterScene) -> RasterScene:
        """Deactivate a scene without removing bands or files."""
        if scene.is_active:
            scene.is_active = False
            scene.deleted_at = datetime.now(timezone.utc)
            self.db.add(scene)
            self.db.commit()
            self.db.refresh(scene)
        return scene

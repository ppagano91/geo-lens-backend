from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.derived_asset import RasterDerivedAsset


class DerivedAssetRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, asset: RasterDerivedAsset) -> RasterDerivedAsset:
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def save(self, asset: RasterDerivedAsset) -> RasterDerivedAsset:
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def get_by_id(self, asset_id: UUID) -> RasterDerivedAsset | None:
        stmt = select(RasterDerivedAsset).where(RasterDerivedAsset.id == asset_id)
        return self.db.scalars(stmt).one_or_none()

    def find_by_scene_aoi_product(
        self,
        scene_id: UUID,
        asset_type: str,
        product_key: str,
        *,
        aoi_id: UUID | None = None,
        include_inactive: bool = False,
    ) -> RasterDerivedAsset | None:
        """Lookup by natural key (scene + type + product + optional AOI)."""
        stmt = (
            select(RasterDerivedAsset)
            .where(RasterDerivedAsset.scene_id == scene_id)
            .where(RasterDerivedAsset.asset_type == asset_type)
            .where(RasterDerivedAsset.product_key == product_key)
        )
        if aoi_id is None:
            stmt = stmt.where(RasterDerivedAsset.aoi_id.is_(None))
        else:
            stmt = stmt.where(RasterDerivedAsset.aoi_id == aoi_id)
        if not include_inactive:
            stmt = stmt.where(RasterDerivedAsset.is_active.is_(True))
        stmt = stmt.limit(1)
        return self.db.scalars(stmt).one_or_none()

    def find_by_scene(
        self,
        scene_id: UUID,
        *,
        include_inactive: bool = False,
    ) -> list[RasterDerivedAsset]:
        stmt = select(RasterDerivedAsset).where(
            RasterDerivedAsset.scene_id == scene_id
        )
        if not include_inactive:
            stmt = stmt.where(RasterDerivedAsset.is_active.is_(True))
        stmt = stmt.order_by(
            RasterDerivedAsset.created_at.desc(),
            RasterDerivedAsset.product_key.asc(),
        )
        return list(self.db.scalars(stmt).all())

    def find_by_scene_and_type(
        self,
        scene_id: UUID,
        asset_type: str,
        *,
        include_inactive: bool = False,
    ) -> list[RasterDerivedAsset]:
        stmt = (
            select(RasterDerivedAsset)
            .where(RasterDerivedAsset.scene_id == scene_id)
            .where(RasterDerivedAsset.asset_type == asset_type)
        )
        if not include_inactive:
            stmt = stmt.where(RasterDerivedAsset.is_active.is_(True))
        stmt = stmt.order_by(
            RasterDerivedAsset.created_at.desc(),
            RasterDerivedAsset.product_key.asc(),
        )
        return list(self.db.scalars(stmt).all())

    def list(
        self,
        *,
        scene_id: UUID | None = None,
        asset_type: str | None = None,
        product_key: str | None = None,
        aoi_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> list[RasterDerivedAsset]:
        stmt = select(RasterDerivedAsset)
        if scene_id is not None:
            stmt = stmt.where(RasterDerivedAsset.scene_id == scene_id)
        if asset_type is not None:
            stmt = stmt.where(RasterDerivedAsset.asset_type == asset_type)
        if product_key is not None:
            stmt = stmt.where(RasterDerivedAsset.product_key == product_key)
        if aoi_id is not None:
            stmt = stmt.where(RasterDerivedAsset.aoi_id == aoi_id)
        if not include_inactive:
            stmt = stmt.where(RasterDerivedAsset.is_active.is_(True))
        stmt = (
            stmt.order_by(
                RasterDerivedAsset.created_at.desc(),
                RasterDerivedAsset.product_key.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(stmt).all())

    def soft_delete(self, asset: RasterDerivedAsset) -> RasterDerivedAsset:
        """Deactivate a catalog row without removing files from storage."""
        if asset.is_active:
            asset.is_active = False
            asset.deleted_at = datetime.now(timezone.utc)
            self.db.add(asset)
            self.db.commit()
            self.db.refresh(asset)
        return asset

    def restore(self, asset: RasterDerivedAsset) -> RasterDerivedAsset:
        """Reactivate a soft-deleted catalog row. Does not touch files."""
        if not asset.is_active:
            asset.is_active = True
            asset.deleted_at = None
            self.db.add(asset)
            self.db.commit()
            self.db.refresh(asset)
        return asset


__all__ = ["DerivedAssetRepository"]

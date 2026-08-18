from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dem_asset import DemAsset


class DemAssetRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, asset: DemAsset) -> DemAsset:
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def save(self, asset: DemAsset) -> DemAsset:
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def get_by_id(self, dem_id: UUID) -> DemAsset | None:
        return self.db.get(DemAsset, dem_id)

    def list(self, *, limit: int = 50, offset: int = 0) -> list[DemAsset]:
        stmt = (
            select(DemAsset)
            .order_by(DemAsset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(stmt).all())


__all__ = ["DemAssetRepository"]

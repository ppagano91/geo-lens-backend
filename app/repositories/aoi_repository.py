from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.aoi import Aoi


class AoiRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, aoi: Aoi) -> Aoi:
        self.db.add(aoi)
        self.db.commit()
        self.db.refresh(aoi)
        return aoi

    def get_by_id(self, aoi_id: UUID) -> Optional[Aoi]:
        return self.db.get(Aoi, aoi_id)

    def list(
        self, limit: int, offset: int, *, include_inactive: bool = False
    ) -> list[Aoi]:
        stmt = select(Aoi)
        if not include_inactive:
            stmt = stmt.where(Aoi.is_active.is_(True))
        stmt = stmt.order_by(Aoi.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt).all())

    def soft_delete(self, aoi: Aoi) -> Aoi:
        if aoi.is_active:
            aoi.is_active = False
            aoi.deleted_at = datetime.now(timezone.utc)
            self.db.add(aoi)
            self.db.commit()
            self.db.refresh(aoi)
        return aoi

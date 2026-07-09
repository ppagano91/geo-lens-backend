from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.spectral_index import SpectralIndexDefinition


class SpectralIndexRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list(
        self,
        category: str | None = None,
        is_active: bool | None = None,
    ) -> list[SpectralIndexDefinition]:
        stmt = select(SpectralIndexDefinition)

        if category is not None:
            stmt = stmt.where(SpectralIndexDefinition.category == category)

        if is_active is not None:
            stmt = stmt.where(SpectralIndexDefinition.is_active == is_active)

        stmt = stmt.order_by(SpectralIndexDefinition.key)
        return list(self.db.scalars(stmt).all())

    def get_by_key(self, index_key: str) -> SpectralIndexDefinition | None:
        normalized_key = index_key.lower()
        stmt = select(SpectralIndexDefinition).where(
            func.lower(SpectralIndexDefinition.key) == normalized_key
        )
        return self.db.scalars(stmt).one_or_none()

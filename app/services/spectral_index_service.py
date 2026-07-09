from __future__ import annotations

from sqlalchemy.orm import Session

from app.repositories.spectral_index_repository import SpectralIndexRepository
from app.schemas.spectral_index import SpectralIndexDefinitionListItem, SpectralIndexDefinitionRead


class SpectralIndexNotFoundError(Exception):
    pass


class SpectralIndexService:
    def __init__(self, db: Session) -> None:
        self.repository = SpectralIndexRepository(db)

    def list_indices(
        self,
        category: str | None = None,
        is_active: bool | None = None,
    ) -> list[SpectralIndexDefinitionListItem]:
        effective_is_active = True if is_active is None else is_active
        indices = self.repository.list(category=category, is_active=effective_is_active)
        return [self._to_read(index) for index in indices]

    def get_index_by_key(self, index_key: str) -> SpectralIndexDefinitionRead:
        index = self.repository.get_by_key(index_key)
        if index is None:
            raise SpectralIndexNotFoundError(index_key.lower())
        return self._to_read(index)

    @staticmethod
    def _to_read(index) -> SpectralIndexDefinitionRead:
        return SpectralIndexDefinitionRead(
            id=index.id,
            key=index.key,
            name=index.name,
            description=index.description,
            formula=index.formula,
            required_bands=index.required_bands,
            category=index.category,
            output_range=index.output_range,
            interpretation=index.interpretation,
            is_active=index.is_active,
            created_at=index.created_at,
            updated_at=index.updated_at,
        )


__all__ = ["SpectralIndexService", "SpectralIndexNotFoundError"]

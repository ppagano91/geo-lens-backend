from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SpectralIndexDefinition(Base):
    """Catalog entry for a spectral index formula and metadata.

    This phase stores definitions only; raster calculation is out of scope.
    """

    __tablename__ = "spectral_index_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    formula: Mapped[str] = mapped_column(String(255), nullable=False)
    required_bands: Mapped[dict] = mapped_column(JSONB, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    output_range: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    interpretation: Mapped[str] = mapped_column(String(2048), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_spectral_index_definitions_category", "category"),
        Index("ix_spectral_index_definitions_is_active", "is_active"),
    )

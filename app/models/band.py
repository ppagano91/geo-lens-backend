from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.scene import RasterScene


class RasterBand(Base):
    """Spectral band metadata associated with a raster scene."""

    __tablename__ = "raster_bands"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("raster_scenes.id", ondelete="CASCADE"),
        nullable=False,
    )
    band_key: Mapped[str] = mapped_column(String(20), nullable=False)
    band_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    resolution: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    # Relative reference under DATA_ROOT (via AssetStorageService); never file bytes.
    asset_path: Mapped[str] = mapped_column(Text, nullable=False)
    nodata: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dtype: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    scene: Mapped[RasterScene] = relationship(back_populates="bands")

    __table_args__ = (
        UniqueConstraint("scene_id", "band_key", name="uq_raster_bands_scene_band_key"),
        Index("ix_raster_bands_scene_id", "scene_id"),
    )

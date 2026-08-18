from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DemAsset(Base):
    """Catalog entry for an experimental DEM / relief raster under DATA_ROOT.

    Independent of satellite scenes. Stores metadata and relative path
    references only — never GeoTIFF/PNG bytes.
    """

    __tablename__ = "dem_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_path: Mapped[str] = mapped_column(Text, nullable=False)
    preview_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    crs: Mapped[str] = mapped_column(String(128), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    bounds: Mapped[dict] = mapped_column(JSONB, nullable=False)
    min_elevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_elevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
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

    __table_args__ = (Index("ix_dem_assets_created_at", "created_at"),)


__all__ = ["DemAsset"]

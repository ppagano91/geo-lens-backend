from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.aoi import Aoi
    from app.models.scene import RasterScene


class RasterDerivedAsset(Base):
    """Catalog entry for a derived raster product under DATA_ROOT.

    Stores metadata and relative path references only — never GeoTIFF/PNG bytes.
    Natural key: (scene_id, aoi_id, asset_type, product_key) with partial unique
    indexes so nullable ``aoi_id`` is unique when NULL.
    """

    __tablename__ = "raster_derived_assets"

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
    aoi_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("aois.id", ondelete="SET NULL"),
        nullable=True,
    )
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)
    product_key: Mapped[str] = mapped_column(String(100), nullable=False)
    # Relative references under DATA_ROOT (via AssetStorageService); never file bytes.
    asset_path: Mapped[str] = mapped_column(Text, nullable=False)
    preview_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    georef_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    crs: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    nodata: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dtype: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    stats: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
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

    scene: Mapped[RasterScene] = relationship()
    aoi: Mapped[Optional[Aoi]] = relationship()

    __table_args__ = (
        Index("ix_raster_derived_assets_scene_id", "scene_id"),
        Index("ix_raster_derived_assets_aoi_id", "aoi_id"),
        Index("ix_raster_derived_assets_asset_type", "asset_type"),
        Index("ix_raster_derived_assets_product_key", "product_key"),
        Index("ix_raster_derived_assets_is_active", "is_active"),
        Index(
            "uq_derived_assets_scene_type_product",
            "scene_id",
            "asset_type",
            "product_key",
            unique=True,
            postgresql_where=text("aoi_id IS NULL AND is_active IS TRUE"),
        ),
        Index(
            "uq_derived_assets_scene_aoi_type_product",
            "scene_id",
            "aoi_id",
            "asset_type",
            "product_key",
            unique=True,
            postgresql_where=text("aoi_id IS NOT NULL AND is_active IS TRUE"),
        ),
    )


__all__ = ["RasterDerivedAsset"]

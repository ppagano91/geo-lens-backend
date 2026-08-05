from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Date, DateTime, Index, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.band import RasterBand


class RasterScene(Base):
    """Satellite scene metadata persisted in PostGIS.

    Geometries are stored as MultiPolygon with SRID 4326. The API accepts
    Polygon GeoJSON on input and returns Polygon GeoJSON when the stored
    geometry is a single polygon.
    """

    __tablename__ = "raster_scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    acquisition_date: Mapped[date] = mapped_column(Date, nullable=False)
    cloud_cover: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    footprint: Mapped[object] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False),
        nullable=False,
    )
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

    bands: Mapped[list[RasterBand]] = relationship(
        back_populates="scene",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_raster_scenes_footprint", "footprint", postgresql_using="gist"),
        Index("ix_raster_scenes_acquisition_date", "acquisition_date"),
        Index("ix_raster_scenes_source", "source"),
        Index("ix_raster_scenes_is_active", "is_active"),
    )

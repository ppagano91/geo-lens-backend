"""raster scenes and bands tables

Revision ID: 002_raster_scenes
Revises: 001_initial_aois
Create Date: 2026-07-07

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_raster_scenes"
down_revision: Union[str, None] = "001_initial_aois"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "raster_scenes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("acquisition_date", sa.Date(), nullable=False),
        sa.Column("cloud_cover", sa.Numeric(), nullable=True),
        sa.Column(
            "footprint",
            Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_raster_scenes_footprint",
        "raster_scenes",
        ["footprint"],
        unique=False,
        postgresql_using="gist",
    )
    op.create_index(
        "ix_raster_scenes_acquisition_date",
        "raster_scenes",
        ["acquisition_date"],
        unique=False,
    )
    op.create_index(
        "ix_raster_scenes_source",
        "raster_scenes",
        ["source"],
        unique=False,
    )

    op.create_table(
        "raster_bands",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("band_key", sa.String(length=20), nullable=False),
        sa.Column("band_name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("resolution", sa.Numeric(), nullable=True),
        sa.Column("asset_path", sa.Text(), nullable=False),
        sa.Column("nodata", sa.String(length=50), nullable=True),
        sa.Column("dtype", sa.String(length=50), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["scene_id"], ["raster_scenes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scene_id", "band_key", name="uq_raster_bands_scene_band_key"),
    )
    op.create_index(
        "ix_raster_bands_scene_id",
        "raster_bands",
        ["scene_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_raster_bands_scene_id", table_name="raster_bands")
    op.drop_table("raster_bands")
    op.drop_index(
        "ix_raster_scenes_source",
        table_name="raster_scenes",
    )
    op.drop_index(
        "ix_raster_scenes_acquisition_date",
        table_name="raster_scenes",
    )
    op.drop_index(
        "ix_raster_scenes_footprint",
        table_name="raster_scenes",
        postgresql_using="gist",
    )
    op.drop_table("raster_scenes")

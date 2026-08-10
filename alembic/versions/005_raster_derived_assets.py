"""raster_derived_assets catalog table

Revision ID: 005_raster_derived_assets
Revises: 004_soft_delete_aois_scenes
Create Date: 2026-08-10

Catalog of derived products (indexes, AOI crops, RGB composites). Stores
metadata and relative path references under DATA_ROOT — never raster bytes.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "005_raster_derived_assets"
down_revision: Union[str, None] = "004_soft_delete_aois_scenes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "raster_derived_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("aoi_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("asset_type", sa.String(length=50), nullable=False),
        sa.Column("product_key", sa.String(length=100), nullable=False),
        sa.Column("asset_path", sa.Text(), nullable=False),
        sa.Column("preview_path", sa.Text(), nullable=True),
        sa.Column("georef_path", sa.Text(), nullable=True),
        sa.Column("crs", sa.String(length=128), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("nodata", sa.String(length=50), nullable=True),
        sa.Column("dtype", sa.String(length=50), nullable=True),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["scene_id"],
            ["raster_scenes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["aoi_id"],
            ["aois.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_raster_derived_assets_scene_id",
        "raster_derived_assets",
        ["scene_id"],
    )
    op.create_index(
        "ix_raster_derived_assets_aoi_id",
        "raster_derived_assets",
        ["aoi_id"],
    )
    op.create_index(
        "ix_raster_derived_assets_asset_type",
        "raster_derived_assets",
        ["asset_type"],
    )
    op.create_index(
        "ix_raster_derived_assets_product_key",
        "raster_derived_assets",
        ["product_key"],
    )
    op.create_index(
        "ix_raster_derived_assets_is_active",
        "raster_derived_assets",
        ["is_active"],
    )
    op.create_index(
        "uq_derived_assets_scene_type_product",
        "raster_derived_assets",
        ["scene_id", "asset_type", "product_key"],
        unique=True,
        postgresql_where=sa.text("aoi_id IS NULL AND is_active IS TRUE"),
    )
    op.create_index(
        "uq_derived_assets_scene_aoi_type_product",
        "raster_derived_assets",
        ["scene_id", "aoi_id", "asset_type", "product_key"],
        unique=True,
        postgresql_where=sa.text("aoi_id IS NOT NULL AND is_active IS TRUE"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_derived_assets_scene_aoi_type_product",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "uq_derived_assets_scene_type_product",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "ix_raster_derived_assets_is_active",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "ix_raster_derived_assets_product_key",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "ix_raster_derived_assets_asset_type",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "ix_raster_derived_assets_aoi_id",
        table_name="raster_derived_assets",
    )
    op.drop_index(
        "ix_raster_derived_assets_scene_id",
        table_name="raster_derived_assets",
    )
    op.drop_table("raster_derived_assets")

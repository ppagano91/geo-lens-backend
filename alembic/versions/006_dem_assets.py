"""dem_assets catalog table

Revision ID: 006_dem_assets
Revises: 005_raster_derived_assets
Create Date: 2026-08-14

Experimental DEM / relief catalog. Stores metadata and relative path
references under DATA_ROOT — never raster bytes. Independent of
raster_scenes (v0.1-P5).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "006_dem_assets"
down_revision: Union[str, None] = "005_raster_derived_assets"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dem_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("asset_path", sa.Text(), nullable=False),
        sa.Column("preview_path", sa.Text(), nullable=True),
        sa.Column("crs", sa.String(length=128), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("bounds", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("min_elevation", sa.Float(), nullable=True),
        sa.Column("max_elevation", sa.Float(), nullable=True),
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
    op.create_index("ix_dem_assets_created_at", "dem_assets", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_dem_assets_created_at", table_name="dem_assets")
    op.drop_table("dem_assets")

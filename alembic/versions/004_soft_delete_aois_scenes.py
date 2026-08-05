"""soft delete columns for aois and raster_scenes

Revision ID: 004_soft_delete_aois_scenes
Revises: 003_spectral_index_definitions
Create Date: 2026-08-05

Adds is_active and deleted_at for logical deactivation (no physical deletes).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004_soft_delete_aois_scenes"
down_revision: Union[str, None] = "003_spectral_index_definitions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "aois",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "aois",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_aois_is_active", "aois", ["is_active"])

    op.add_column(
        "raster_scenes",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "raster_scenes",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_raster_scenes_is_active", "raster_scenes", ["is_active"])


def downgrade() -> None:
    op.drop_index("ix_raster_scenes_is_active", table_name="raster_scenes")
    op.drop_column("raster_scenes", "deleted_at")
    op.drop_column("raster_scenes", "is_active")

    op.drop_index("ix_aois_is_active", table_name="aois")
    op.drop_column("aois", "deleted_at")
    op.drop_column("aois", "is_active")

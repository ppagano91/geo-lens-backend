"""spectral index definitions catalog

Revision ID: 003_spectral_index_definitions
Revises: 002_raster_scenes
Create Date: 2026-07-08

Initial NDVI, NDWI, NBR and NDMI definitions are seeded in this migration.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_spectral_index_definitions"
down_revision: Union[str, None] = "002_raster_scenes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INITIAL_INDICES = [
    {
        "id": "a1000001-0000-4000-8000-000000000001",
        "key": "ndvi",
        "name": "Normalized Difference Vegetation Index",
        "description": (
            "Índice normalizado usado para estimar vigor o actividad fotosintética de la vegetación."
        ),
        "formula": "(NIR - RED) / (NIR + RED)",
        "required_bands": {"nir": "B08", "red": "B04"},
        "category": "vegetation",
        "output_range": {"min": -1, "max": 1},
        "interpretation": (
            "Valores altos suelen asociarse a vegetación más vigorosa; valores bajos o negativos "
            "suelen asociarse a agua, suelo desnudo, áreas urbanas o ausencia de vegetación."
        ),
    },
    {
        "id": "a1000001-0000-4000-8000-000000000002",
        "key": "ndwi",
        "name": "Normalized Difference Water Index",
        "description": (
            "Índice normalizado usado para resaltar agua superficial o humedad relativa "
            "usando verde e infrarrojo cercano."
        ),
        "formula": "(GREEN - NIR) / (GREEN + NIR)",
        "required_bands": {"green": "B03", "nir": "B08"},
        "category": "water",
        "output_range": {"min": -1, "max": 1},
        "interpretation": (
            "Valores positivos pueden asociarse a presencia de agua superficial; valores bajos "
            "suelen corresponder a vegetación, suelo o áreas urbanas."
        ),
    },
    {
        "id": "a1000001-0000-4000-8000-000000000003",
        "key": "nbr",
        "name": "Normalized Burn Ratio",
        "description": (
            "Índice normalizado usado para identificar áreas quemadas o analizar severidad "
            "relativa de incendios."
        ),
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "required_bands": {"nir": "B08", "swir2": "B12"},
        "category": "burn",
        "output_range": {"min": -1, "max": 1},
        "interpretation": (
            "Cambios entre NBR antes y después de un evento pueden ayudar a identificar áreas "
            "quemadas o con pérdida de cobertura vegetal."
        ),
    },
    {
        "id": "a1000001-0000-4000-8000-000000000004",
        "key": "ndmi",
        "name": "Normalized Difference Moisture Index",
        "description": "Índice normalizado usado para estimar humedad de la vegetación.",
        "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "required_bands": {"nir": "B08", "swir1": "B11"},
        "category": "moisture",
        "output_range": {"min": -1, "max": 1},
        "interpretation": (
            "Valores más altos suelen asociarse a mayor contenido de humedad en vegetación; "
            "valores bajos pueden indicar estrés hídrico o menor humedad."
        ),
    },
]


def upgrade() -> None:
    op.create_table(
        "spectral_index_definitions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=False),
        sa.Column("formula", sa.String(length=255), nullable=False),
        sa.Column(
            "required_bands",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column(
            "output_range",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("interpretation", sa.String(length=2048), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.UniqueConstraint("key"),
    )
    op.create_index(
        "ix_spectral_index_definitions_category",
        "spectral_index_definitions",
        ["category"],
        unique=False,
    )
    op.create_index(
        "ix_spectral_index_definitions_is_active",
        "spectral_index_definitions",
        ["is_active"],
        unique=False,
    )

    spectral_index_table = sa.table(
        "spectral_index_definitions",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("key", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.String()),
        sa.column("formula", sa.String()),
        sa.column("required_bands", postgresql.JSONB()),
        sa.column("category", sa.String()),
        sa.column("output_range", postgresql.JSONB()),
        sa.column("interpretation", sa.String()),
        sa.column("is_active", sa.Boolean()),
    )

    op.bulk_insert(
        spectral_index_table,
        [
            {
                "id": index["id"],
                "key": index["key"],
                "name": index["name"],
                "description": index["description"],
                "formula": index["formula"],
                "required_bands": index["required_bands"],
                "category": index["category"],
                "output_range": index["output_range"],
                "interpretation": index["interpretation"],
                "is_active": True,
            }
            for index in INITIAL_INDICES
        ],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_spectral_index_definitions_is_active",
        table_name="spectral_index_definitions",
    )
    op.drop_index(
        "ix_spectral_index_definitions_category",
        table_name="spectral_index_definitions",
    )
    op.drop_table("spectral_index_definitions")

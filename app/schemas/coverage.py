from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class CoverageStatus(str, Enum):
    full = "full"
    partial = "partial"
    none = "none"


class SpatialCoverageRead(BaseModel):
    aoi_id: UUID
    scene_id: UUID
    coverage_status: CoverageStatus
    intersects: bool
    covered: bool
    coverage_percent: float = Field(
        ...,
        description=(
            "Porcentaje del área del AOI cubierta por el footprint. "
            "Estimación vía ST_Transform a EPSG:3857."
        ),
    )
    message: str

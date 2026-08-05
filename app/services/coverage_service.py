from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.aoi import Aoi
from app.models.scene import RasterScene
from app.schemas.coverage import CoverageStatus, SpatialCoverageRead
from app.services.aoi_service import AoiNotFoundError
from app.services.scene_service import SceneNotFoundError

# Web Mercator approximation for area ratios in EPSG:4326 geometries.
# Documented as an estimate (not equal-area); sufficient for coverage %.
_AREA_SRID = 3857

_MESSAGES = {
    CoverageStatus.full: (
        "El AOI está completamente cubierto por el footprint de la escena."
    ),
    CoverageStatus.partial: (
        "El AOI intersecta parcialmente el footprint de la escena."
    ),
    CoverageStatus.none: "El AOI está fuera del footprint de la escena.",
}


class SpatialCoverageService:
    """Evaluate AOI vs scene footprint coverage using PostGIS (no raster I/O)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def evaluate(self, aoi_id: UUID, scene_id: UUID) -> SpatialCoverageRead:
        aoi = self.db.get(Aoi, aoi_id)
        if aoi is None or not aoi.is_active:
            raise AoiNotFoundError(str(aoi_id))
        scene = self.db.get(RasterScene, scene_id)
        if scene is None or not scene.is_active:
            raise SceneNotFoundError(str(scene_id))

        row = self.db.execute(
            text(
                """
                SELECT
                    ST_Intersects(a.geom, s.footprint) AS intersects,
                    ST_Covers(s.footprint, a.geom) AS covered,
                    CASE
                        WHEN ST_Area(ST_Transform(a.geom, :srid)) <= 0 THEN 0.0
                        ELSE (
                            ST_Area(
                                ST_Transform(
                                    ST_Intersection(a.geom, s.footprint),
                                    :srid
                                )
                            )
                            / ST_Area(ST_Transform(a.geom, :srid))
                        ) * 100.0
                    END AS coverage_percent
                FROM aois AS a
                CROSS JOIN raster_scenes AS s
                WHERE a.id = :aoi_id
                  AND s.id = :scene_id
                  AND a.is_active IS TRUE
                  AND s.is_active IS TRUE
                """
            ),
            {"aoi_id": aoi_id, "scene_id": scene_id, "srid": _AREA_SRID},
        ).one()

        intersects = bool(row.intersects)
        covered = bool(row.covered)
        coverage_percent = round(float(row.coverage_percent or 0.0), 1)

        if covered:
            status = CoverageStatus.full
            # Covers implies full coverage; clamp floating-point noise.
            coverage_percent = 100.0
        elif intersects:
            status = CoverageStatus.partial
        else:
            status = CoverageStatus.none
            coverage_percent = 0.0

        return SpatialCoverageRead(
            aoi_id=aoi_id,
            scene_id=scene_id,
            coverage_status=status,
            intersects=intersects,
            covered=covered,
            coverage_percent=coverage_percent,
            message=_MESSAGES[status],
        )


__all__ = [
    "SpatialCoverageService",
    "AoiNotFoundError",
    "SceneNotFoundError",
]

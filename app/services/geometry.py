from copy import deepcopy
from typing import Any

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.validation import explain_validity

EXPECTED_SRID = 4326


class GeometryValidationError(ValueError):
    pass


def validate_polygon_geojson(geojson: dict[str, Any]) -> Polygon:
    """Validate Polygon GeoJSON and return a Shapely Polygon.

    If the exterior ring is not closed, the ring is closed automatically
    by repeating the first coordinate at the end.
    """
    if not geojson:
        raise GeometryValidationError("geometry is required")

    if geojson.get("type") != "Polygon":
        raise GeometryValidationError("geometry.type must be Polygon")

    coordinates = geojson.get("coordinates")
    if not coordinates or not isinstance(coordinates, list):
        raise GeometryValidationError("geometry.coordinates are required")

    if not coordinates[0] or not isinstance(coordinates[0], list):
        raise GeometryValidationError("geometry.coordinates must include an exterior ring")

    normalized = deepcopy(geojson)
    exterior_ring = list(normalized["coordinates"][0])

    if len(exterior_ring) < 3:
        raise GeometryValidationError(
            "polygon exterior ring must have at least 3 distinct points"
        )

    if exterior_ring[0] != exterior_ring[-1]:
        exterior_ring.append(exterior_ring[0])
        normalized["coordinates"][0] = exterior_ring

    if len(exterior_ring) < 4:
        raise GeometryValidationError(
            "polygon exterior ring must have at least 4 coordinates when closed"
        )

    polygon = shape(normalized)

    if polygon.is_empty:
        raise GeometryValidationError("geometry must not be empty")

    if not isinstance(polygon, Polygon):
        raise GeometryValidationError("geometry must be a Polygon")

    if not polygon.is_valid:
        raise GeometryValidationError(f"invalid geometry: {explain_validity(polygon)}")

    return polygon


def polygon_to_db_element(polygon: Polygon):
    """Convert a Shapely Polygon to a GeoAlchemy2 element stored as MultiPolygon."""
    multi = MultiPolygon([polygon])
    return from_shape(multi, srid=EXPECTED_SRID)


def db_element_to_geojson(db_geometry) -> dict[str, Any]:
    """Convert a PostGIS geometry to GeoJSON for API responses."""
    shapely_geom = to_shape(db_geometry)

    if isinstance(shapely_geom, MultiPolygon) and len(shapely_geom.geoms) == 1:
        return mapping(shapely_geom.geoms[0])

    return mapping(shapely_geom)

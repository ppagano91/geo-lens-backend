import pytest

from app.services.geometry import GeometryValidationError, validate_polygon_geojson

VALID_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-58.40, -34.60],
            [-58.38, -34.60],
            [-58.38, -34.62],
            [-58.40, -34.62],
            [-58.40, -34.60],
        ]
    ],
}


def test_validate_polygon_geojson_accepts_valid_polygon() -> None:
    polygon = validate_polygon_geojson(VALID_POLYGON)

    assert polygon.geom_type == "Polygon"
    assert polygon.is_valid


def test_validate_polygon_geojson_closes_open_ring() -> None:
    open_ring = {
        "type": "Polygon",
        "coordinates": [
            [
                [-58.40, -34.60],
                [-58.38, -34.60],
                [-58.38, -34.62],
                [-58.40, -34.62],
            ]
        ],
    }

    polygon = validate_polygon_geojson(open_ring)

    assert polygon.is_valid
    exterior = list(polygon.exterior.coords)
    assert exterior[0] == exterior[-1]


def test_validate_polygon_geojson_rejects_too_few_points() -> None:
    invalid = {
        "type": "Polygon",
        "coordinates": [[[-58.40, -34.60], [-58.38, -34.60]]],
    }

    with pytest.raises(GeometryValidationError):
        validate_polygon_geojson(invalid)


def test_validate_polygon_geojson_rejects_non_polygon() -> None:
    invalid = {
        "type": "Point",
        "coordinates": [-58.40, -34.60],
    }

    with pytest.raises(GeometryValidationError):
        validate_polygon_geojson(invalid)

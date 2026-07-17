from uuid import UUID, uuid4

from tests.conftest import VALID_POLYGON, requires_database
from tests.test_scenes import SCENE_FOOTPRINT

# Overlaps SCENE_FOOTPRINT on the west edge only (partial coverage).
PARTIAL_AOI_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-58.55, -34.60],
            [-58.45, -34.60],
            [-58.45, -34.65],
            [-58.55, -34.65],
            [-58.55, -34.60],
        ]
    ],
}

# Completely outside SCENE_FOOTPRINT.
DISJOINT_AOI_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-57.50, -33.50],
            [-57.40, -33.50],
            [-57.40, -33.60],
            [-57.50, -33.60],
            [-57.50, -33.50],
        ]
    ],
}


def _create_aoi(client, geometry: dict, name: str = "AOI coverage") -> str:
    response = client.post(
        "/api/v1/aois",
        json={"name": name, "geometry": geometry},
    )
    assert response.status_code == 201
    return response.json()["id"]


def _create_scene(client) -> str:
    response = client.post(
        "/api/v1/scenes",
        json={
            "name": "Scene coverage",
            "source": "local",
            "acquisition_date": "2025-03-01",
            "cloud_cover": 5.0,
            "footprint": SCENE_FOOTPRINT,
            "bands": [
                {
                    "band_key": "B04",
                    "band_name": "Red",
                    "asset_path": "data/sample/scenes/scene_before/B04.tif",
                },
                {
                    "band_key": "B08",
                    "band_name": "NIR",
                    "asset_path": "data/sample/scenes/scene_before/B08.tif",
                },
            ],
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def _coverage_url(aoi_id: str, scene_id: str) -> str:
    return f"/api/v1/spatial-coverage/aoi/{aoi_id}/scene/{scene_id}"


@requires_database
def test_spatial_coverage_full(client) -> None:
    aoi_id = _create_aoi(client, VALID_POLYGON, name="AOI full")
    scene_id = _create_scene(client)

    response = client.get(_coverage_url(aoi_id, scene_id))

    assert response.status_code == 200
    data = response.json()
    assert data["aoi_id"] == aoi_id
    assert data["scene_id"] == scene_id
    assert data["coverage_status"] == "full"
    assert data["intersects"] is True
    assert data["covered"] is True
    assert data["coverage_percent"] == 100.0
    assert "completamente cubierto" in data["message"].lower()


@requires_database
def test_spatial_coverage_partial(client) -> None:
    aoi_id = _create_aoi(client, PARTIAL_AOI_POLYGON, name="AOI partial")
    scene_id = _create_scene(client)

    response = client.get(_coverage_url(aoi_id, scene_id))

    assert response.status_code == 200
    data = response.json()
    assert data["coverage_status"] == "partial"
    assert data["intersects"] is True
    assert data["covered"] is False
    assert 0.0 < data["coverage_percent"] < 100.0
    assert "parcialmente" in data["message"].lower()


@requires_database
def test_spatial_coverage_none(client) -> None:
    aoi_id = _create_aoi(client, DISJOINT_AOI_POLYGON, name="AOI none")
    scene_id = _create_scene(client)

    response = client.get(_coverage_url(aoi_id, scene_id))

    assert response.status_code == 200
    data = response.json()
    assert data["coverage_status"] == "none"
    assert data["intersects"] is False
    assert data["covered"] is False
    assert data["coverage_percent"] == 0.0
    assert "fuera" in data["message"].lower()


@requires_database
def test_spatial_coverage_aoi_not_found(client) -> None:
    scene_id = _create_scene(client)
    missing_aoi = str(uuid4())

    response = client.get(_coverage_url(missing_aoi, scene_id))

    assert response.status_code == 404
    assert "AOI" in response.json()["detail"]


@requires_database
def test_spatial_coverage_scene_not_found(client) -> None:
    aoi_id = _create_aoi(client, VALID_POLYGON)
    missing_scene = str(uuid4())

    response = client.get(_coverage_url(aoi_id, missing_scene))

    assert response.status_code == 404
    assert "Scene" in response.json()["detail"]


@requires_database
def test_health_still_works(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@requires_database
def test_aois_scenes_indices_still_work(client) -> None:
    aoi_response = client.post(
        "/api/v1/aois",
        json={"name": "AOI smoke", "geometry": VALID_POLYGON},
    )
    assert aoi_response.status_code == 201
    aoi_id = aoi_response.json()["id"]
    UUID(aoi_id)

    scene_id = _create_scene(client)
    scene_response = client.get(f"/api/v1/scenes/{scene_id}")
    assert scene_response.status_code == 200

    indices_response = client.get("/api/v1/indices")
    assert indices_response.status_code == 200
    keys = {item["key"] for item in indices_response.json()}
    assert "ndvi" in keys

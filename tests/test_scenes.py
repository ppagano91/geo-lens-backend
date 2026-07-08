from uuid import UUID

from tests.conftest import VALID_POLYGON, requires_database

SCENE_FOOTPRINT = {
    "type": "Polygon",
    "coordinates": [
        [
            [-58.50, -34.50],
            [-58.20, -34.50],
            [-58.20, -34.80],
            [-58.50, -34.80],
            [-58.50, -34.50],
        ]
    ],
}


def _sample_bands() -> list[dict]:
    return [
        {
            "band_key": "B02",
            "band_name": "Blue",
            "description": "Blue band",
            "resolution": 10,
            "asset_path": "data/sample/scenes/scene_before/B02.tif",
            "nodata": "0",
            "dtype": "uint16",
        },
        {
            "band_key": "B03",
            "band_name": "Green",
            "asset_path": "data/sample/scenes/scene_before/B03.tif",
        },
    ]


def _sample_scene_payload() -> dict:
    return {
        "name": "Sentinel-2 sample before",
        "source": "local",
        "acquisition_date": "2025-03-01",
        "cloud_cover": 12.5,
        "footprint": SCENE_FOOTPRINT,
        "metadata": {
            "platform": "Sentinel-2",
            "processing_level": "L2A",
            "note": "Local metadata only.",
        },
        "bands": _sample_bands(),
    }


@requires_database
def test_create_scene_with_bands(client) -> None:
    payload = _sample_scene_payload()

    response = client.post("/api/v1/scenes", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["source"] == payload["source"]
    assert data["acquisition_date"] == payload["acquisition_date"]
    assert data["footprint"]["type"] == "Polygon"
    assert len(data["bands"]) == 2
    assert data["bands"][0]["band_key"] == "B02"
    UUID(data["id"])


@requires_database
def test_list_scenes(client) -> None:
    create_response = client.post("/api/v1/scenes", json=_sample_scene_payload())
    assert create_response.status_code == 201

    response = client.get("/api/v1/scenes")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(item["name"] == "Sentinel-2 sample before" for item in data)
    assert "bands" not in data[0]


@requires_database
def test_get_scene_by_id(client) -> None:
    create_response = client.post("/api/v1/scenes", json=_sample_scene_payload())
    scene_id = create_response.json()["id"]

    response = client.get(f"/api/v1/scenes/{scene_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == scene_id
    assert len(data["bands"]) == 2


@requires_database
def test_list_scene_bands(client) -> None:
    create_response = client.post("/api/v1/scenes", json=_sample_scene_payload())
    scene_id = create_response.json()["id"]

    response = client.get(f"/api/v1/scenes/{scene_id}/bands")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert {band["band_key"] for band in data} == {"B02", "B03"}


@requires_database
def test_delete_scene(client) -> None:
    create_response = client.post("/api/v1/scenes", json=_sample_scene_payload())
    scene_id = create_response.json()["id"]

    delete_response = client.delete(f"/api/v1/scenes/{scene_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/api/v1/scenes/{scene_id}")
    assert get_response.status_code == 404


@requires_database
def test_create_scene_invalid_footprint_returns_422(client) -> None:
    payload = _sample_scene_payload()
    payload["footprint"] = {
        "type": "Polygon",
        "coordinates": [[[-58.40, -34.60], [-58.38, -34.60]]],
    }

    response = client.post("/api/v1/scenes", json=payload)

    assert response.status_code == 422


@requires_database
def test_create_scene_duplicate_band_key_returns_422(client) -> None:
    payload = _sample_scene_payload()
    payload["bands"] = [
        {
            "band_key": "B02",
            "band_name": "Blue",
            "asset_path": "data/sample/B02.tif",
        },
        {
            "band_key": "B02",
            "band_name": "Blue duplicate",
            "asset_path": "data/sample/B02_dup.tif",
        },
    ]

    response = client.post("/api/v1/scenes", json=payload)

    assert response.status_code == 422

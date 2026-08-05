from uuid import UUID

from tests.conftest import VALID_POLYGON, requires_database


@requires_database
def test_create_aoi(client) -> None:
    payload = {
        "name": "AOI de prueba",
        "description": "Polígono de prueba en Buenos Aires",
        "geometry": VALID_POLYGON,
        "properties": {"source": "manual"},
    }

    response = client.post("/api/v1/aois", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["description"] == payload["description"]
    assert data["geometry"]["type"] == "Polygon"
    assert data["properties"] == payload["properties"]
    UUID(data["id"])


@requires_database
def test_create_aoi_invalid_geometry_returns_422(client) -> None:
    payload = {
        "name": "AOI inválido",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-58.40, -34.60], [-58.38, -34.60]]],
        },
    }

    response = client.post("/api/v1/aois", json=payload)

    assert response.status_code == 422


@requires_database
def test_list_aois(client) -> None:
    payload = {
        "name": "AOI listado",
        "geometry": VALID_POLYGON,
    }
    create_response = client.post("/api/v1/aois", json=payload)
    assert create_response.status_code == 201

    response = client.get("/api/v1/aois")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(item["name"] == "AOI listado" for item in data)


@requires_database
def test_get_aoi_by_id(client) -> None:
    create_response = client.post(
        "/api/v1/aois",
        json={"name": "AOI detalle", "geometry": VALID_POLYGON},
    )
    aoi_id = create_response.json()["id"]

    response = client.get(f"/api/v1/aois/{aoi_id}")

    assert response.status_code == 200
    assert response.json()["id"] == aoi_id


@requires_database
def test_get_aoi_not_found(client) -> None:
    response = client.get("/api/v1/aois/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


@requires_database
def test_delete_aoi(client) -> None:
    create_response = client.post(
        "/api/v1/aois",
        json={"name": "AOI borrar", "geometry": VALID_POLYGON},
    )
    aoi_id = create_response.json()["id"]

    delete_response = client.delete(f"/api/v1/aois/{aoi_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/api/v1/aois/{aoi_id}")
    assert get_response.status_code == 404

    list_response = client.get("/api/v1/aois")
    assert list_response.status_code == 200
    assert all(item["id"] != aoi_id for item in list_response.json())

    inactive_list = client.get("/api/v1/aois?include_inactive=true")
    assert inactive_list.status_code == 200
    deactivated = next(item for item in inactive_list.json() if item["id"] == aoi_id)
    assert deactivated["is_active"] is False
    assert deactivated["deleted_at"] is not None

    # Idempotent soft-delete
    second_delete = client.delete(f"/api/v1/aois/{aoi_id}")
    assert second_delete.status_code == 204

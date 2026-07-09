from tests.conftest import requires_database


@requires_database
def test_list_indices_includes_seed_definitions(client) -> None:
    response = client.get("/api/v1/indices")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    keys = {item["key"] for item in data}
    assert {"ndvi", "ndwi", "nbr", "ndmi"}.issubset(keys)


@requires_database
def test_get_ndvi_by_key(client) -> None:
    response = client.get("/api/v1/indices/ndvi")

    assert response.status_code == 200
    data = response.json()
    assert data["key"] == "ndvi"
    assert data["name"] == "Normalized Difference Vegetation Index"
    assert data["formula"] == "(NIR - RED) / (NIR + RED)"
    assert data["category"] == "vegetation"


@requires_database
def test_get_ndvi_case_insensitive(client) -> None:
    response = client.get("/api/v1/indices/NDVI")

    assert response.status_code == 200
    assert response.json()["key"] == "ndvi"


@requires_database
def test_get_unknown_index_returns_404(client) -> None:
    response = client.get("/api/v1/indices/unknown")

    assert response.status_code == 404


@requires_database
def test_filter_indices_by_category_vegetation(client) -> None:
    response = client.get("/api/v1/indices", params={"category": "vegetation"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert all(item["category"] == "vegetation" for item in data)
    keys = {item["key"] for item in data}
    assert "ndvi" in keys


@requires_database
def test_ndvi_required_bands(client) -> None:
    response = client.get("/api/v1/indices/ndvi")

    assert response.status_code == 200
    required_bands = response.json()["required_bands"]
    assert required_bands["nir"] == "B08"
    assert required_bands["red"] == "B04"


@requires_database
def test_ndwi_required_bands(client) -> None:
    response = client.get("/api/v1/indices/ndwi")

    assert response.status_code == 200
    required_bands = response.json()["required_bands"]
    assert required_bands["green"] == "B03"
    assert required_bands["nir"] == "B08"


@requires_database
def test_nbr_required_bands(client) -> None:
    response = client.get("/api/v1/indices/nbr")

    assert response.status_code == 200
    required_bands = response.json()["required_bands"]
    assert required_bands["nir"] == "B08"
    assert required_bands["swir2"] == "B12"


@requires_database
def test_ndmi_required_bands(client) -> None:
    response = client.get("/api/v1/indices/ndmi")

    assert response.status_code == 200
    required_bands = response.json()["required_bands"]
    assert required_bands["nir"] == "B08"
    assert required_bands["swir1"] == "B11"


@requires_database
def test_calculate_endpoint_not_available(client) -> None:
    response = client.post("/api/v1/indices/calculate")

    assert response.status_code in {404, 405}

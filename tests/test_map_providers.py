"""Tests for GET /api/v1/map-providers/config (v0.1-P5.2)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

CONFIG_URL = "/api/v1/map-providers/config"
MAPTILER_TILEJSON = (
    "https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json"
)


def _provider_by_id(body: dict, provider_id: str) -> dict:
    match = next(
        (item for item in body["providers"] if item["id"] == provider_id),
        None,
    )
    assert match is not None, f"missing provider {provider_id}"
    return match


def test_config_without_maptiler_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "maptiler_api_key", None)

    with TestClient(app) as client:
        response = client.get(CONFIG_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["maptiler"] == {
        "enabled": False,
        "terrain_rgb_tiles_json_url": None,
    }
    assert _provider_by_id(body, "maptiler")["available"] is False
    assert _provider_by_id(body, "maptiler")["requires_key"] is True
    assert _provider_by_id(body, "aws-terrarium")["available"] is True
    assert _provider_by_id(body, "aws-terrarium")["requires_key"] is False
    assert _provider_by_id(body, "maplibre-demo")["available"] is True
    assert "maptiler_api_key" not in str(body).lower()


def test_config_with_blank_maptiler_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "maptiler_api_key", "   ")

    with TestClient(app) as client:
        response = client.get(CONFIG_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["maptiler"]["enabled"] is False
    assert body["maptiler"]["terrain_rgb_tiles_json_url"] is None
    assert _provider_by_id(body, "maptiler")["available"] is False
    assert _provider_by_id(body, "aws-terrarium")["available"] is True


def test_config_with_maptiler_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "maptiler_api_key", "unit-test-key")

    with TestClient(app) as client:
        response = client.get(CONFIG_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["maptiler"]["enabled"] is True
    assert body["maptiler"]["terrain_rgb_tiles_json_url"] == (
        f"{MAPTILER_TILEJSON}?key=unit-test-key"
    )
    assert _provider_by_id(body, "maptiler")["available"] is True
    assert _provider_by_id(body, "aws-terrarium")["available"] is True
    assert _provider_by_id(body, "maplibre-demo")["available"] is True
    assert "maptiler_api_key" not in body

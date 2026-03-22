"""Tests for the API server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nirasa.serving import api_server
from nirasa.serving.api_server import app


@pytest.fixture()
def client():
    """Test client with model NOT loaded (default state)."""
    # Ensure model is not loaded
    api_server._model = None
    api_server._tokenizer = None
    api_server._model_name = ""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_503_when_model_not_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 503

    def test_health_returns_200_when_model_loaded(self, client):
        # Simulate a loaded model with mocks
        api_server._model = "fake-model"
        api_server._tokenizer = "fake-tokenizer"
        api_server._model_name = "test-model"

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "test-model"


class TestModelsEndpoint:
    def test_list_models(self, client):
        api_server._model_name = "nirasa-7b-th"
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "nirasa-7b-th"


class TestChatCompletionsEndpoint:
    def test_returns_503_when_model_not_loaded(self, client):
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "สวัสดี"}],
        })
        assert response.status_code == 503

    def test_rejects_invalid_temperature(self, client):
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "สวัสดี"}],
            "temperature": 5.0,
        })
        assert response.status_code == 422

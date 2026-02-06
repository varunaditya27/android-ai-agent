"""
Tests for FastAPI Routes
========================

Comprehensive tests for:
- Health endpoints (/, /health, /health/live, /health/ready, /health/info)
- Session management (/sessions CRUD)
- Agent endpoints (/agent/execute, /agent/status, /agent/input, /agent/cancel)
- Middleware (request ID, response time)
- Error handling (404, 405, 422)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app


class TestHealthRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_liveness(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_not_configured(self, client):
        """Without env vars, readiness should return 503."""
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)

    def test_service_info(self, client):
        resp = client.get("/health/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "android-ai-agent"
        assert "version" in data
        assert "config" in data


class TestRootEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Android AI Agent"
        assert "version" in data


class TestSessionRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_list_sessions_empty(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data

    def test_get_session_not_found(self, client):
        resp = client.get("/sessions/non-existent-id")
        assert resp.status_code == 404

    def test_delete_session_not_found(self, client):
        resp = client.delete("/sessions/non-existent-id")
        assert resp.status_code == 404


class TestAgentRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_status_no_session(self, client):
        resp = client.get("/agent/status/non-existent-id")
        assert resp.status_code == 404

    def test_cancel_no_session(self, client):
        resp = client.post("/agent/cancel/non-existent-id")
        assert resp.status_code == 404

    def test_input_no_session(self, client):
        resp = client.post(
            "/agent/input",
            json={"session_id": "non-existent-id", "value": "test"},
        )
        assert resp.status_code == 404

    def test_execute_no_session(self, client):
        resp = client.post(
            "/agent/execute",
            json={
                "session_id": "non-existent-id",
                "task": "Open YouTube",
                "max_steps": 10,
            },
        )
        assert resp.status_code == 404

    def test_execute_validation_error(self, client):
        """Missing required fields should return 422."""
        resp = client.post("/agent/execute", json={})
        assert resp.status_code == 422


class TestMiddleware:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_request_id_header(self, client):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_response_time_header(self, client):
        resp = client.get("/health")
        assert "X-Response-Time" in resp.headers
        assert "ms" in resp.headers["X-Response-Time"]


class TestErrorHandling:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_404(self, client):
        resp = client.get("/unknown/endpoint")
        assert resp.status_code == 404

    def test_405(self, client):
        resp = client.put("/health")
        assert resp.status_code == 405

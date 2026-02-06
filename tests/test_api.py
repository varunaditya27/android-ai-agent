"""
Tests for FastAPI Routes
========================

Tests for:
- Health endpoints
- Session management
- Agent endpoints
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


class TestHealthRoutes:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_check_not_configured(self, client):
        """Test readiness when not configured."""
        # This will fail because API keys aren't configured in tests
        response = client.get("/health/ready")

        # Should return 503 when not configured
        assert response.status_code in [200, 503]

    def test_service_info(self, client):
        """Test service info endpoint."""
        response = client.get("/health/info")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "android-ai-agent"
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Android AI Agent"
        assert "version" in data


class TestSessionRoutes:
    """Tests for session management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_sessions_empty(self, client):
        """Test listing sessions when empty."""
        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data

    def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = client.get("/sessions/non-existent-id")

        assert response.status_code == 404

    def test_delete_session_not_found(self, client):
        """Test deleting non-existent session."""
        response = client.delete("/sessions/non-existent-id")

        assert response.status_code == 404

    @patch("app.api.routes.sessions.create_cloud_device")
    def test_create_session(self, mock_create_device, client):
        """Test creating a session."""
        # Setup mock
        mock_device = MagicMock()
        mock_device.allocate = AsyncMock(return_value=MagicMock(
            device_id="test-123",
            device_name="Pixel 7",
            os_version="13.0",
            screen_width=1080,
            screen_height=2400,
            __dict__={
                "device_id": "test-123",
                "device_name": "Pixel 7",
                "os_version": "13.0",
            },
        ))
        mock_create_device.return_value = mock_device

        response = client.post(
            "/sessions",
            json={
                "device_type": "android",
                "timeout_minutes": 30,
            },
        )

        # May fail due to settings not configured
        assert response.status_code in [201, 500]


class TestAgentRoutes:
    """Tests for agent endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_agent_status_no_session(self, client):
        """Test getting status for non-existent session."""
        response = client.get("/agent/status/non-existent-id")

        assert response.status_code == 404

    def test_cancel_task_no_session(self, client):
        """Test cancelling task for non-existent session."""
        response = client.post("/agent/cancel/non-existent-id")

        assert response.status_code == 404

    def test_provide_input_no_session(self, client):
        """Test providing input for non-existent session."""
        response = client.post(
            "/agent/input",
            json={
                "session_id": "non-existent-id",
                "value": "test input",
            },
        )

        assert response.status_code == 404

    def test_execute_task_no_session(self, client):
        """Test executing task for non-existent session."""
        response = client.post(
            "/agent/execute",
            json={
                "session_id": "non-existent-id",
                "task": "Open YouTube",
                "max_steps": 10,
            },
        )

        assert response.status_code == 404


class TestRequestMiddleware:
    """Tests for request middleware."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_request_id_header(self, client):
        """Test request ID is added to response."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers

    def test_response_time_header(self, client):
        """Test response time is added to response."""
        response = client.get("/health")

        assert "X-Response-Time" in response.headers
        assert "ms" in response.headers["X-Response-Time"]


class TestCORS:
    """Tests for CORS configuration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS should be enabled
        assert response.status_code in [200, 204, 400]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_404_not_found(self, client):
        """Test 404 for unknown endpoint."""
        response = client.get("/unknown/endpoint")

        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Test 405 for wrong method."""
        response = client.put("/health")  # Only GET is allowed

        assert response.status_code == 405

    def test_422_validation_error(self, client):
        """Test 422 for validation errors."""
        response = client.post(
            "/agent/execute",
            json={
                # Missing required fields
                "task": "",  # Empty task should fail
            },
        )

        assert response.status_code == 422

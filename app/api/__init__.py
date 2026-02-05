"""
API Module
==========

FastAPI routes and WebSocket handlers for the Android AI Agent.

This package contains:
    - routes/: REST API endpoints
    - websocket: WebSocket handler for real-time streaming
"""

from app.api.routes import health, sessions, agent
from app.api.websocket import WebSocketManager

__all__ = [
    "health",
    "sessions",
    "agent",
    "WebSocketManager",
]

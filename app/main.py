"""
Android AI Agent - Main Application
====================================

FastAPI application entry point for the Android AI Agent service.

This module sets up:
- FastAPI application with CORS
- Route registration
- WebSocket endpoints
- Middleware (logging, error handling)
- Lifespan management (startup/shutdown)

Usage:
    # Development
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import health_router, sessions_router, agent_router
from app.api.websocket import websocket_endpoint
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

# Setup logging
settings = get_settings()
setup_logging(
    level=settings.server.log_level,
    json_logs=not settings.server.debug,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting Android AI Agent",
        version=__version__,
        environment=settings.server.environment,
    )

    # Initialize any resources here
    # e.g., database connections, caches, etc.

    yield

    # Shutdown
    logger.info("Shutting down Android AI Agent")

    # Cleanup any resources here
    # e.g., close database connections, flush caches, etc.


# Create FastAPI application
app = FastAPI(
    title="Android AI Agent",
    description=(
        "AI-powered mobile automation agent for blind users. "
        "Enables natural language control of Android devices through "
        "cloud-based device farms."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.server.debug else None,
    redoc_url="/redoc" if settings.server.debug else None,
    openapi_url="/openapi.json" if settings.server.debug else None,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()

    # Generate request ID
    request_id = request.headers.get("X-Request-ID", str(time.time()))

    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    # Log response
    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.server.debug else "An unexpected error occurred",
        },
    )


# Register routers
app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(agent_router)


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for real-time agent streaming.

    Connect to receive live updates during task execution.

    Args:
        websocket: WebSocket connection.
        session_id: Session identifier.
    """
    await websocket_endpoint(websocket, session_id)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "Android AI Agent",
        "version": __version__,
        "docs": "/docs" if settings.server.debug else None,
        "health": "/health",
    }


# Run directly (for development)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.server.log_level.lower(),
    )

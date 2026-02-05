"""
Health Check Routes
===================

Endpoints for health monitoring and service status.

Includes:
- Basic health check
- Readiness probe
- Detailed status information
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings, Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    summary="Basic health check",
    response_description="Service health status",
)
async def health_check() -> dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        Simple status message indicating service is running.
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/ready",
    summary="Readiness probe",
    response_description="Service readiness status",
)
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Readiness probe for Kubernetes/container orchestration.

    Checks if all required dependencies are available:
    - LLM API key configured
    - Cloud device credentials configured

    Returns:
        Readiness status and component health.

    Raises:
        HTTPException: If service is not ready.
    """
    checks: dict[str, bool] = {}
    all_ready = True

    # Check LLM configuration
    llm_ready = bool(settings.llm.api_key)
    checks["llm_configured"] = llm_ready
    if not llm_ready:
        all_ready = False
        logger.warning("LLM API key not configured")

    # Check cloud device configuration
    cloud_ready = bool(settings.cloud_device.api_key)
    checks["cloud_device_configured"] = cloud_ready
    if not cloud_ready:
        all_ready = False
        logger.warning("Cloud device API key not configured")

    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "checks": checks,
            },
        )

    return {
        "status": "ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/live",
    summary="Liveness probe",
    response_description="Service liveness status",
)
async def liveness_check() -> dict[str, str]:
    """
    Liveness probe for container orchestration.

    This endpoint should return quickly and indicate
    that the service process is alive.

    Returns:
        Simple alive status.
    """
    return {"status": "alive"}


@router.get(
    "/info",
    summary="Service information",
    response_description="Detailed service information",
)
async def service_info(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Get detailed service information.

    Returns:
        Service version, configuration, and environment info.
    """
    from app import __version__

    return {
        "service": "android-ai-agent",
        "version": __version__,
        "environment": settings.server.environment,
        "config": {
            "llm_model": settings.llm.model_name,
            "cloud_provider": settings.cloud_device.provider,
            "max_steps": settings.agent.max_steps,
            "debug_mode": settings.server.debug,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

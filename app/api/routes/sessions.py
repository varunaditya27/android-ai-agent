"""
Session Management Routes
=========================

Endpoints for managing device sessions.

Sessions represent active connections to cloud Android devices.
Each session includes:
- Device allocation and configuration
- Session lifecycle management
- Device status monitoring
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings, Settings
from app.device.cloud_provider import create_cloud_device, CloudDevice, DeviceInfo
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# In-memory session storage (use Redis for production)
_sessions: dict[str, dict[str, Any]] = {}


# Request/Response Models
class CreateSessionRequest(BaseModel):
    """Request to create a new device session."""

    device_type: str = Field(
        default="android",
        description="Type of device to allocate",
    )
    device_name: Optional[str] = Field(
        default=None,
        description="Specific device name (e.g., 'Google Pixel 7')",
    )
    os_version: Optional[str] = Field(
        default=None,
        description="Specific OS version (e.g., '13.0')",
    )
    timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Session timeout in minutes",
    )


class SessionResponse(BaseModel):
    """Response containing session information."""

    session_id: str
    status: str
    device_info: Optional[dict[str, Any]] = None
    created_at: str
    expires_at: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""

    sessions: list[SessionResponse]
    total: int


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new device session",
)
async def create_session(
    request: CreateSessionRequest,
    settings: Settings = Depends(get_settings),
) -> SessionResponse:
    """
    Create a new device session.

    Allocates a cloud Android device and initializes a session.
    The session ID is used for all subsequent agent operations.

    Args:
        request: Session creation parameters.

    Returns:
        Created session information.

    Raises:
        HTTPException: If device allocation fails.
    """
    session_id = str(uuid.uuid4())
    logger.info(
        "Creating session",
        session_id=session_id,
        device_type=request.device_type,
    )

    try:
        # Create cloud device client
        device = create_cloud_device(
            provider=settings.cloud_device.provider,
            api_key=settings.cloud_device.api_key,
            base_url=settings.cloud_device.base_url or "",
        )

        # Allocate device
        device_info = await device.allocate(
            device_name=request.device_name,
            os_version=request.os_version,
        )

        # Calculate expiry
        from datetime import timedelta

        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(minutes=request.timeout_minutes)

        # Store session
        _sessions[session_id] = {
            "device": device,
            "device_info": device_info,
            "status": "active",
            "created_at": created_at,
            "expires_at": expires_at,
        }

        logger.info(
            "Session created",
            session_id=session_id,
            device_name=device_info.device_name if device_info else "unknown",
        )

        return SessionResponse(
            session_id=session_id,
            status="active",
            device_info=device_info.__dict__ if device_info else None,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
        )

    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to allocate device: {str(e)}",
        )


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List all active sessions",
)
async def list_sessions() -> SessionListResponse:
    """
    List all active sessions.

    Returns:
        List of active session information.
    """
    sessions = []
    for session_id, session_data in _sessions.items():
        sessions.append(
            SessionResponse(
                session_id=session_id,
                status=session_data["status"],
                device_info=session_data["device_info"].__dict__
                if session_data.get("device_info")
                else None,
                created_at=session_data["created_at"].isoformat(),
                expires_at=session_data["expires_at"].isoformat()
                if session_data.get("expires_at")
                else None,
            )
        )

    return SessionListResponse(
        sessions=sessions,
        total=len(sessions),
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get details of a specific session.

    Args:
        session_id: The session identifier.

    Returns:
        Session information.

    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session_data = _sessions[session_id]

    return SessionResponse(
        session_id=session_id,
        status=session_data["status"],
        device_info=session_data["device_info"].__dict__
        if session_data.get("device_info")
        else None,
        created_at=session_data["created_at"].isoformat(),
        expires_at=session_data["expires_at"].isoformat()
        if session_data.get("expires_at")
        else None,
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(session_id: str) -> None:
    """
    Delete a session and release the device.

    Args:
        session_id: The session identifier.

    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session_data = _sessions[session_id]
    device: CloudDevice = session_data.get("device")

    try:
        # Release device
        if device:
            await device.release()

        # Remove session
        del _sessions[session_id]
        logger.info("Session deleted", session_id=session_id)

    except Exception as e:
        logger.error("Error deleting session", session_id=session_id, error=str(e))
        # Still remove from memory
        _sessions.pop(session_id, None)


@router.get(
    "/{session_id}/screenshot",
    summary="Capture device screenshot",
)
async def capture_screenshot(session_id: str) -> dict[str, str]:
    """
    Capture a screenshot from the session's device.

    Args:
        session_id: The session identifier.

    Returns:
        Base64-encoded screenshot image.

    Raises:
        HTTPException: If session not found or screenshot fails.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session_data = _sessions[session_id]
    device: CloudDevice = session_data.get("device")

    if not device:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Device not available",
        )

    try:
        screenshot_b64 = await device.capture_screenshot()
        return {
            "screenshot": screenshot_b64,
            "format": "png",
            "encoding": "base64",
        }
    except Exception as e:
        logger.error("Screenshot failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to capture screenshot: {str(e)}",
        )


# Helper function to get session device
def get_session_device(session_id: str) -> CloudDevice:
    """
    Get the device for a session.

    Args:
        session_id: The session identifier.

    Returns:
        The CloudDevice instance.

    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session_data = _sessions[session_id]
    device = session_data.get("device")

    if not device:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Device not available",
        )

    return device


# Expose sessions for other modules
def get_sessions_store() -> dict[str, dict[str, Any]]:
    """Get the sessions store for internal use."""
    return _sessions

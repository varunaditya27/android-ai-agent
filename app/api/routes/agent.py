"""
Agent Routes
============

Endpoints for interacting with the AI agent.

Provides:
- Task execution (single request)
- Task streaming (WebSocket upgrade)
- Agent state and control
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.agent import ReActAgent, AgentConfig, TaskResult
from app.api.routes.sessions import get_session_device, get_sessions_store
from app.config import get_settings, Settings
from app.llm.client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# Store active agents
_active_agents: dict[str, ReActAgent] = {}


# Request/Response Models
class ExecuteTaskRequest(BaseModel):
    """Request to execute a task."""

    session_id: str = Field(
        description="Session ID for the device to use",
    )
    task: str = Field(
        description="Natural language task description",
        min_length=1,
        max_length=1000,
    )
    max_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum steps to attempt",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=600,
        description="Task timeout in seconds",
    )


class TaskStatusResponse(BaseModel):
    """Response containing task status."""

    session_id: str
    status: str
    current_step: int
    thinking: Optional[str] = None
    last_action: Optional[str] = None
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Response containing task result."""

    success: bool
    result: str
    steps_taken: int
    duration_seconds: float
    error: Optional[str] = None
    history: list[dict[str, Any]] = []


class ProvideInputRequest(BaseModel):
    """Request to provide user input."""

    session_id: str
    value: str = Field(
        description="The input value to provide",
    )


@router.post(
    "/execute",
    response_model=TaskResultResponse,
    summary="Execute a task",
)
async def execute_task(
    request: ExecuteTaskRequest,
    settings: Settings = Depends(get_settings),
) -> TaskResultResponse:
    """
    Execute a task using the AI agent.

    This is a blocking endpoint that runs the full task
    and returns the result. For streaming updates, use
    the WebSocket endpoint instead.

    Args:
        request: Task execution parameters.

    Returns:
        Task result with full history.

    Raises:
        HTTPException: If execution fails.
    """
    logger.info(
        "Executing task",
        session_id=request.session_id,
        task=request.task[:50] + "..." if len(request.task) > 50 else request.task,
    )

    try:
        # Get device from session
        device = get_session_device(request.session_id)

        # Create LLM client
        llm_client = LLMClient(
            api_key=settings.llm.api_key,
            model=settings.llm.model_name,
            base_url=settings.llm.api_base,
        )

        # Create agent
        agent_config = AgentConfig(
            max_steps=request.max_steps,
            step_timeout=float(request.timeout_seconds) / request.max_steps,
        )

        agent = ReActAgent(
            llm_client=llm_client,
            device=device,
            config=agent_config,
        )

        # Store agent reference
        _active_agents[request.session_id] = agent

        try:
            # Run task
            import asyncio

            result = await asyncio.wait_for(
                agent.run(request.task),
                timeout=float(request.timeout_seconds),
            )

            return TaskResultResponse(
                success=result.success,
                result=result.result,
                steps_taken=result.steps_taken,
                duration_seconds=result.duration_seconds,
                error=result.error,
                history=result.history,
            )

        finally:
            # Cleanup
            _active_agents.pop(request.session_id, None)

    except asyncio.TimeoutError:
        logger.error("Task timed out", session_id=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Task execution timed out",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Task execution failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task execution failed: {str(e)}",
        )


@router.get(
    "/status/{session_id}",
    response_model=TaskStatusResponse,
    summary="Get agent status",
)
async def get_agent_status(session_id: str) -> TaskStatusResponse:
    """
    Get the current status of an active agent.

    Args:
        session_id: The session identifier.

    Returns:
        Current agent status.

    Raises:
        HTTPException: If no active agent for session.
    """
    agent = _active_agents.get(session_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active agent for session: {session_id}",
        )

    state = agent.get_state()
    last_step = state.get("history", [])[-1] if state.get("history") else None

    return TaskStatusResponse(
        session_id=session_id,
        status=state.get("status", "unknown"),
        current_step=state.get("current_step", 0),
        thinking=last_step.get("thinking") if last_step else None,
        last_action=last_step.get("action_type") if last_step else None,
        error=state.get("result") if state.get("status") == "FAILED" else None,
    )


@router.post(
    "/input",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Provide user input",
)
async def provide_input(request: ProvideInputRequest) -> dict[str, str]:
    """
    Provide requested user input to an active agent.

    Use this when the agent is waiting for input
    (e.g., credentials, OTP codes).

    Args:
        request: Input value to provide.

    Returns:
        Acknowledgment message.

    Raises:
        HTTPException: If no active agent or not waiting for input.
    """
    agent = _active_agents.get(request.session_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active agent for session: {request.session_id}",
        )

    state = agent.get_state()
    if state.get("status") != "WAITING_INPUT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not waiting for input",
        )

    await agent.provide_input(request.value)
    logger.info("Input provided", session_id=request.session_id)

    return {"message": "Input received"}


@router.post(
    "/cancel/{session_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel task execution",
)
async def cancel_task(session_id: str) -> dict[str, str]:
    """
    Cancel an active task execution.

    Args:
        session_id: The session identifier.

    Returns:
        Acknowledgment message.

    Raises:
        HTTPException: If no active agent for session.
    """
    agent = _active_agents.get(session_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active agent for session: {session_id}",
        )

    agent.cancel()
    logger.info("Task cancelled", session_id=session_id)

    return {"message": "Task cancellation requested"}


@router.post(
    "/quick-action",
    summary="Execute a quick action",
)
async def quick_action(
    session_id: str,
    action: str,
    params: Optional[dict[str, Any]] = None,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Execute a quick action without full agent loop.

    Useful for simple actions that don't need AI reasoning:
    - tap: Tap at coordinates or element
    - swipe: Swipe in a direction
    - type: Type text
    - back: Press back
    - home: Go to home screen

    Args:
        session_id: The session identifier.
        action: Action type to perform.
        params: Action parameters.

    Returns:
        Action result.

    Raises:
        HTTPException: If action fails.
    """
    device = get_session_device(session_id)
    params = params or {}

    try:
        result = None

        if action == "tap":
            x = params.get("x", 0)
            y = params.get("y", 0)
            result = await device.tap(x, y)
        elif action == "swipe":
            direction = params.get("direction", "up")
            screen_info = await device.get_screen_info()
            width = screen_info.get("width", 1080)
            height = screen_info.get("height", 2400)
            cx, cy = width // 2, height // 2

            swipe_map = {
                "up": (cx, cy + 300, cx, cy - 300),
                "down": (cx, cy - 300, cx, cy + 300),
                "left": (cx + 300, cy, cx - 300, cy),
                "right": (cx - 300, cy, cx + 300, cy),
            }
            coords = swipe_map.get(direction, swipe_map["up"])
            result = await device.swipe(*coords)
        elif action == "type":
            text = params.get("text", "")
            result = await device.type_text(text)
        elif action == "back":
            result = await device.press_back()
        elif action == "home":
            result = await device.press_home()
        elif action == "launch":
            package = params.get("package", "")
            result = await device.launch_app(package)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {action}",
            )

        return {
            "success": result.success if result else False,
            "message": result.message if result else "No result",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Quick action failed", action=action, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Action failed: {str(e)}",
        )

"""
WebSocket Handler
=================

WebSocket endpoint for real-time agent streaming.

Provides:
- Real-time task progress updates
- Screenshot streaming
- Bidirectional communication for user input
"""

import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

from app.agent import ReActAgent, AgentConfig, StepResult
from app.api.routes.sessions import get_sessions_store
from app.config import get_settings
from app.device.screenshot import resize_for_llm
from app.llm.client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types."""

    # Client -> Server
    START_TASK = "start_task"
    PROVIDE_INPUT = "provide_input"
    CANCEL_TASK = "cancel_task"
    PING = "ping"

    # Server -> Client
    TASK_STARTED = "task_started"
    STEP_UPDATE = "step_update"
    INPUT_REQUIRED = "input_required"
    SCREENSHOT = "screenshot"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    ERROR = "error"
    PONG = "pong"


@dataclass
class WSMessage:
    """WebSocket message structure."""

    type: str
    data: dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({"type": self.type, "data": self.data})

    @classmethod
    def from_json(cls, text: str) -> "WSMessage":
        """Parse from JSON string."""
        parsed = json.loads(text)
        return cls(type=parsed.get("type", ""), data=parsed.get("data", {}))


class WebSocketManager:
    """
    Manages WebSocket connections for agent streaming.

    Handles:
    - Connection lifecycle
    - Message routing
    - Agent task coordination
    """

    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        self._connections: dict[str, WebSocket] = {}
        self._agents: dict[str, ReActAgent] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._input_events: dict[str, asyncio.Event] = {}
        self._input_values: dict[str, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            session_id: Session identifier.

        Returns:
            True if connection accepted.
        """
        # Verify session exists
        sessions = get_sessions_store()
        if session_id not in sessions:
            await websocket.close(code=4004, reason="Session not found")
            return False

        await websocket.accept()
        self._connections[session_id] = websocket
        logger.info("WebSocket connected", session_id=session_id)
        return True

    async def disconnect(self, session_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            session_id: Session identifier.
        """
        # Cancel running task
        if session_id in self._tasks:
            self._tasks[session_id].cancel()
            del self._tasks[session_id]

        # Cleanup
        self._connections.pop(session_id, None)
        self._agents.pop(session_id, None)
        self._input_events.pop(session_id, None)
        self._input_values.pop(session_id, None)

        logger.info("WebSocket disconnected", session_id=session_id)

    async def send_message(self, session_id: str, message: WSMessage) -> bool:
        """
        Send a message to a connected client.

        Args:
            session_id: Session identifier.
            message: Message to send.

        Returns:
            True if sent successfully.
        """
        websocket = self._connections.get(session_id)
        if not websocket:
            return False

        try:
            await websocket.send_text(message.to_json())
            return True
        except Exception as e:
            logger.error("Failed to send message", session_id=session_id, error=str(e))
            return False

    async def handle_message(self, session_id: str, raw_message: str) -> None:
        """
        Handle an incoming WebSocket message.

        Args:
            session_id: Session identifier.
            raw_message: Raw message text.
        """
        try:
            message = WSMessage.from_json(raw_message)

            if message.type == MessageType.PING.value:
                await self.send_message(
                    session_id,
                    WSMessage(type=MessageType.PONG.value, data={}),
                )

            elif message.type == MessageType.START_TASK.value:
                task = message.data.get("task", "")
                max_steps = message.data.get("max_steps", 30)
                await self._start_task(session_id, task, max_steps)

            elif message.type == MessageType.PROVIDE_INPUT.value:
                value = message.data.get("value", "")
                await self._provide_input(session_id, value)

            elif message.type == MessageType.CANCEL_TASK.value:
                await self._cancel_task(session_id)

            else:
                logger.warning("Unknown message type", type=message.type)

        except json.JSONDecodeError:
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.ERROR.value,
                    data={"message": "Invalid JSON"},
                ),
            )
        except Exception as e:
            logger.error("Error handling message", error=str(e))
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.ERROR.value,
                    data={"message": str(e)},
                ),
            )

    async def _start_task(self, session_id: str, task: str, max_steps: int) -> None:
        """Start a new task for the session."""
        # Check if task already running
        if session_id in self._tasks:
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.ERROR.value,
                    data={"message": "Task already running"},
                ),
            )
            return

        # Get session device
        sessions = get_sessions_store()
        session_data = sessions.get(session_id)
        if not session_data:
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.ERROR.value,
                    data={"message": "Session not found"},
                ),
            )
            return

        device = session_data.get("device")
        if not device:
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.ERROR.value,
                    data={"message": "Device not available"},
                ),
            )
            return

        # Get settings
        settings = get_settings()

        # Create LLM client
        from app.llm.models import LLMConfig

        llm_config = LLMConfig(
            api_key=settings.llm.gemini_api_key,
            model=settings.llm.llm_model,
            max_output_tokens=settings.llm.llm_max_output_tokens,
            temperature=settings.llm.llm_temperature,
            top_p=settings.llm.llm_top_p,
            top_k=settings.llm.llm_top_k,
        )
        llm_client = LLMClient(llm_config)

        # Create input event for this session
        self._input_events[session_id] = asyncio.Event()

        # Create agent with callbacks
        agent = ReActAgent(
            llm_client=llm_client,
            device=device,
            config=AgentConfig(max_steps=max_steps),
            on_step=lambda step: asyncio.create_task(
                self._on_step(session_id, step)
            ),
            on_input_required=lambda prompt: self._sync_input_handler(session_id, prompt),
        )

        self._agents[session_id] = agent

        # Notify task started
        await self.send_message(
            session_id,
            WSMessage(
                type=MessageType.TASK_STARTED.value,
                data={"task": task, "max_steps": max_steps},
            ),
        )

        # Start task in background
        self._tasks[session_id] = asyncio.create_task(
            self._run_task(session_id, agent, task)
        )

    async def _run_task(self, session_id: str, agent: ReActAgent, task: str) -> None:
        """Run the agent task."""
        try:
            result = await agent.run(task)

            if result.success:
                await self.send_message(
                    session_id,
                    WSMessage(
                        type=MessageType.TASK_COMPLETED.value,
                        data={
                            "result": result.result,
                            "steps_taken": result.steps_taken,
                            "duration_seconds": result.duration_seconds,
                        },
                    ),
                )
            else:
                await self.send_message(
                    session_id,
                    WSMessage(
                        type=MessageType.TASK_FAILED.value,
                        data={
                            "error": result.error or result.result,
                            "steps_taken": result.steps_taken,
                        },
                    ),
                )

        except asyncio.CancelledError:
            logger.info("Task cancelled", session_id=session_id)
        except Exception as e:
            logger.exception("Task failed", error=str(e))
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.TASK_FAILED.value,
                    data={"error": str(e)},
                ),
            )
        finally:
            self._tasks.pop(session_id, None)
            self._agents.pop(session_id, None)
            self._input_events.pop(session_id, None)

    async def _on_step(self, session_id: str, step: StepResult) -> None:
        """Handle step completion callback."""
        # Send step update
        await self.send_message(
            session_id,
            WSMessage(
                type=MessageType.STEP_UPDATE.value,
                data={
                    "thinking": step.thinking,
                    "action_type": step.action_type,
                    "action_message": step.action_message,
                    "success": step.success,
                    "error": step.error,
                },
            ),
        )

        # Handle input required
        if step.requires_input:
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.INPUT_REQUIRED.value,
                    data={"prompt": step.input_prompt},
                ),
            )

        # Send screenshot
        agent = self._agents.get(session_id)
        if agent and agent.state.last_screenshot:
            # Resize for sending
            resized = resize_for_llm(agent.state.last_screenshot, max_width=800, max_height=800)
            await self.send_message(
                session_id,
                WSMessage(
                    type=MessageType.SCREENSHOT.value,
                    data={"image": resized},
                ),
            )

    def _sync_input_handler(self, session_id: str, prompt: str) -> str:
        """
        Synchronous input handler for agent callback.

        This is a workaround since the agent callback is sync.
        We use an event to wait for input from WebSocket.
        """
        # This will be called from sync context
        # We need to wait for input via the event
        event = self._input_events.get(session_id)
        if event:
            # Clear event and wait
            event.clear()

            # Use a timeout to avoid indefinite blocking
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(asyncio.wait_for(event.wait(), timeout=300))
            except asyncio.TimeoutError:
                return ""

        return self._input_values.get(session_id, "")

    async def _provide_input(self, session_id: str, value: str) -> None:
        """Provide user input to waiting agent."""
        self._input_values[session_id] = value

        event = self._input_events.get(session_id)
        if event:
            event.set()

        logger.info("Input provided via WebSocket", session_id=session_id)

    async def _cancel_task(self, session_id: str) -> None:
        """Cancel the running task."""
        task = self._tasks.get(session_id)
        if task:
            task.cancel()
            logger.info("Task cancellation requested", session_id=session_id)


# Global WebSocket manager instance
ws_manager = WebSocketManager()


async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint handler.

    Args:
        websocket: The WebSocket connection.
        session_id: Session identifier from path.
    """
    if not await ws_manager.connect(websocket, session_id):
        return

    try:
        while True:
            message = await websocket.receive_text()
            await ws_manager.handle_message(session_id, message)

    except WebSocketDisconnect:
        await ws_manager.disconnect(session_id)
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        await ws_manager.disconnect(session_id)
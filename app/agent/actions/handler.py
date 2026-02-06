"""
Action Handler
==============

Central dispatcher for executing agent actions on the device.

Handles action validation, execution, and result reporting.
Provides a unified interface for all action types.

Usage:
    from app.agent.actions import ActionHandler

    handler = ActionHandler(device)
    result = await handler.execute(action_result, elements)
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.llm.response_parser import ActionResult, ActionType
from app.perception.ui_parser import UIElement
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ActionExecutionResult:
    """
    Result of action execution.

    Attributes:
        success: Whether action executed successfully.
        message: Human-readable result message.
        error: Error message if failed.
        requires_input: Whether user input is needed.
        input_prompt: Prompt for user input.
        data: Additional data from action.
    """

    success: bool
    message: str = ""
    error: Optional[str] = None
    requires_input: bool = False
    input_prompt: Optional[str] = None
    data: Optional[dict] = None


class ActionHandler:
    """
    Central handler for executing agent actions.

    Routes actions to appropriate handlers and manages
    execution flow, retries, and error handling.
    """

    def __init__(
        self,
        device: CloudDevice,
        retry_count: int = 2,
        action_delay: float = 0.5,
    ) -> None:
        """
        Initialize action handler.

        Args:
            device: Cloud device to execute actions on.
            retry_count: Number of retries on failure.
            action_delay: Delay after action in seconds.
        """
        self.device = device
        self.retry_count = retry_count
        self.action_delay = action_delay

    async def execute(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """
        Execute an action on the device.

        Args:
            action: Parsed action to execute.
            elements: Current UI elements for reference.

        Returns:
            ActionExecutionResult with status and details.
        """
        logger.info(
            "Executing action",
            action_type=action.action_type.name,
            params=action.params,
        )

        # Route to appropriate handler
        handlers = {
            ActionType.TAP: self._handle_tap,
            ActionType.LONG_PRESS: self._handle_long_press,
            ActionType.SWIPE: self._handle_swipe,
            ActionType.SCROLL: self._handle_swipe,  # Scroll is a swipe
            ActionType.TYPE: self._handle_type,
            ActionType.LAUNCH: self._handle_launch,
            ActionType.BACK: self._handle_back,
            ActionType.HOME: self._handle_home,
            ActionType.WAIT: self._handle_wait,
            ActionType.FINISH: self._handle_finish,
            ActionType.REQUEST_INPUT: self._handle_request_input,
        }

        handler = handlers.get(action.action_type)
        if not handler:
            return ActionExecutionResult(
                success=False,
                error=f"Unknown action type: {action.action_type}",
            )

        # Execute with retry
        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                result = await handler(action, elements)

                if result.success:
                    # Add delay after successful action
                    if action.action_type not in (ActionType.FINISH, ActionType.REQUEST_INPUT, ActionType.WAIT):
                        await asyncio.sleep(self.action_delay)

                    logger.info(
                        "Action succeeded",
                        action_type=action.action_type.name,
                        message=result.message,
                    )
                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Action attempt failed",
                    action_type=action.action_type.name,
                    attempt=attempt + 1,
                    error=str(e),
                )

            if attempt < self.retry_count:
                await asyncio.sleep(0.5)  # Brief delay before retry

        logger.error(
            "Action failed after retries",
            action_type=action.action_type.name,
            error=last_error,
        )

        return ActionExecutionResult(
            success=False,
            error=last_error or "Action failed",
        )

    async def _handle_tap(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle tap action."""
        params = action.params

        # Tap by element index
        if "element_id" in params:
            element_id = int(params["element_id"])
            element = self._find_element(elements, element_id)

            if not element:
                return ActionExecutionResult(
                    success=False,
                    error=f"Element {element_id} not found",
                )

            x, y = element.center_x, element.center_y
            result = await self.device.tap(x, y)

            return ActionExecutionResult(
                success=result.success,
                message=f"Tapped on '{element.display_text}' at ({x}, {y})",
                error=result.error,
            )

        # Tap by coordinates
        elif "x" in params and "y" in params:
            x = int(params["x"])
            y = int(params["y"])
            result = await self.device.tap(x, y)

            return ActionExecutionResult(
                success=result.success,
                message=f"Tapped at ({x}, {y})",
                error=result.error,
            )

        return ActionExecutionResult(
            success=False,
            error="Tap requires element_id or x,y coordinates",
        )

    async def _handle_long_press(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle long press action."""
        params = action.params
        duration = int(params.get("duration_ms", 1000))

        if "element_id" in params:
            element_id = int(params["element_id"])
            element = self._find_element(elements, element_id)

            if not element:
                return ActionExecutionResult(
                    success=False,
                    error=f"Element {element_id} not found",
                )

            x, y = element.center_x, element.center_y
            result = await self.device.long_press(x, y, duration)

            return ActionExecutionResult(
                success=result.success,
                message=f"Long pressed on '{element.display_text}'",
                error=result.error,
            )

        elif "x" in params and "y" in params:
            x = int(params["x"])
            y = int(params["y"])
            result = await self.device.long_press(x, y, duration)

            return ActionExecutionResult(
                success=result.success,
                message=f"Long pressed at ({x}, {y})",
                error=result.error,
            )

        return ActionExecutionResult(
            success=False,
            error="Long press requires element_id or x,y coordinates",
        )

    async def _handle_swipe(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle swipe/scroll action."""
        params = action.params
        direction = params.get("direction", "").lower()

        if direction:
            result = await self.device.swipe_direction(direction)
            return ActionExecutionResult(
                success=result.success,
                message=f"Swiped {direction}",
                error=result.error,
            )

        # Manual swipe with coordinates
        if all(k in params for k in ["start_x", "start_y", "end_x", "end_y"]):
            result = await self.device.swipe(
                start_x=int(params["start_x"]),
                start_y=int(params["start_y"]),
                end_x=int(params["end_x"]),
                end_y=int(params["end_y"]),
                duration_ms=int(params.get("duration_ms", 300)),
            )

            return ActionExecutionResult(
                success=result.success,
                message="Swipe completed",
                error=result.error,
            )

        return ActionExecutionResult(
            success=False,
            error="Swipe requires direction or start/end coordinates",
        )

    async def _handle_type(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle text input action."""
        text = action.params.get("text", "")

        if not text:
            return ActionExecutionResult(
                success=False,
                error="Type action requires text parameter",
            )

        result = await self.device.type_text(text)

        # Mask text in message for security
        display_text = text[:3] + "***" if len(text) > 3 else "***"

        return ActionExecutionResult(
            success=result.success,
            message=f"Typed '{display_text}'",
            error=result.error,
        )

    async def _handle_launch(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle app launch action."""
        app_name = action.params.get("app", "")

        if not app_name:
            return ActionExecutionResult(
                success=False,
                error="Launch action requires app name",
            )

        # Use the comprehensive package resolver with 80+ apps and fuzzy matching
        from app.agent.actions.launch_app import resolve_package_name

        package_name = resolve_package_name(app_name)

        result = await self.device.launch_app(package_name)

        return ActionExecutionResult(
            success=result.success,
            message=f"Launched {app_name}",
            error=result.error,
        )

    async def _handle_back(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle back button press."""
        result = await self.device.press_key("KEYCODE_BACK")

        return ActionExecutionResult(
            success=result.success,
            message="Pressed back button",
            error=result.error,
        )

    async def _handle_home(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle home button press."""
        result = await self.device.press_key("KEYCODE_HOME")

        return ActionExecutionResult(
            success=result.success,
            message="Pressed home button",
            error=result.error,
        )

    async def _handle_wait(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle wait action."""
        seconds = float(action.params.get("seconds", 2.0))
        seconds = min(seconds, 10.0)  # Cap at 10 seconds

        await asyncio.sleep(seconds)

        return ActionExecutionResult(
            success=True,
            message=f"Waited {seconds} seconds",
        )

    async def _handle_finish(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle task completion."""
        message = action.params.get("message", "Task completed")

        return ActionExecutionResult(
            success=True,
            message=message,
            data={"finished": True, "result": message},
        )

    async def _handle_request_input(
        self,
        action: ActionResult,
        elements: list[UIElement],
    ) -> ActionExecutionResult:
        """Handle input request (credentials, OTP, etc.)."""
        prompt = action.params.get("prompt", "Please provide input")

        return ActionExecutionResult(
            success=True,
            message="Input required",
            requires_input=True,
            input_prompt=prompt,
        )

    def _find_element(
        self,
        elements: list[UIElement],
        index: int,
    ) -> Optional[UIElement]:
        """Find element by index."""
        for elem in elements:
            if elem.index == index:
                return elem
        return None

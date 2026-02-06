"""
Tests for Action Handlers
=========================

Comprehensive tests for:
- ActionHandler routing and execution
- All action types (Tap, LongPress, Swipe, Type, Launch, Back, Home, Wait, Finish, RequestInput)
- Action result handling, retries, and error cases
- Launch app package resolution via resolve_package_name
- Swipe coordinate calculation
- System key codes
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.actions.handler import ActionHandler, ActionExecutionResult
from app.agent.actions.launch_app import resolve_package_name, APP_PACKAGES
from app.agent.actions.swipe import SwipeDirection, calculate_swipe_coords
from app.agent.actions.system import KeyCode
from app.device.cloud_provider import ActionResult
from app.llm.response_parser import ActionType, ParsedAction
from app.perception.ui_parser import UIElement
from tests.conftest import _make_bounds


# ---------------------------------------------------------------------------
# ActionHandler tests
# ---------------------------------------------------------------------------


class TestActionHandler:
    """Tests for the ActionHandler dispatcher."""

    @pytest.fixture
    def handler(self, mock_device):
        """Create an ActionHandler with zero action delay for fast tests."""
        return ActionHandler(mock_device, action_delay=0)

    # --- Tap ---

    @pytest.mark.asyncio
    async def test_tap_by_element_id(self, handler, mock_device, sample_elements):
        action = ParsedAction(action_type=ActionType.TAP, params={"element_id": 2})
        result = await handler.execute(action, sample_elements)

        assert result.success is True
        mock_device.tap.assert_called_once_with(540, 540)

    @pytest.mark.asyncio
    async def test_tap_by_coordinates(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.TAP, params={"x": 500, "y": 300})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.tap.assert_called_with(500, 300)

    @pytest.mark.asyncio
    async def test_tap_invalid_element_id(self, handler, mock_device, sample_elements):
        action = ParsedAction(action_type=ActionType.TAP, params={"element_id": 999})
        result = await handler.execute(action, sample_elements)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tap_missing_params(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.TAP, params={})
        result = await handler.execute(action, [])

        assert result.success is False

    # --- Long Press ---

    @pytest.mark.asyncio
    async def test_long_press_by_element(self, handler, mock_device, sample_elements):
        action = ParsedAction(action_type=ActionType.LONG_PRESS, params={"element_id": 6})
        result = await handler.execute(action, sample_elements)

        assert result.success is True
        mock_device.long_press.assert_called_once_with(540, 940, 1000)

    @pytest.mark.asyncio
    async def test_long_press_with_duration(self, handler, mock_device, sample_elements):
        action = ParsedAction(
            action_type=ActionType.LONG_PRESS,
            params={"element_id": 6, "duration_ms": 2000},
        )
        result = await handler.execute(action, sample_elements)

        assert result.success is True
        mock_device.long_press.assert_called_once_with(540, 940, 2000)

    @pytest.mark.asyncio
    async def test_long_press_by_coords(self, handler, mock_device):
        action = ParsedAction(
            action_type=ActionType.LONG_PRESS,
            params={"x": 100, "y": 200},
        )
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.long_press.assert_called_once_with(100, 200, 1000)

    # --- Swipe / Scroll ---

    @pytest.mark.asyncio
    async def test_swipe_direction(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.SWIPE, params={"direction": "up"})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.swipe_direction.assert_called_once_with("up")

    @pytest.mark.asyncio
    async def test_scroll_routes_to_swipe(self, handler, mock_device):
        """SCROLL action type should route to the same handler as SWIPE."""
        action = ParsedAction(action_type=ActionType.SCROLL, params={"direction": "down"})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.swipe_direction.assert_called_once_with("down")

    @pytest.mark.asyncio
    async def test_swipe_manual_coords(self, handler, mock_device):
        action = ParsedAction(
            action_type=ActionType.SWIPE,
            params={"start_x": 100, "start_y": 500, "end_x": 100, "end_y": 200},
        )
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.swipe.assert_called_once()

    @pytest.mark.asyncio
    async def test_swipe_missing_direction(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.SWIPE, params={})
        result = await handler.execute(action, [])

        assert result.success is False

    # --- Type ---

    @pytest.mark.asyncio
    async def test_type_text(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.TYPE, params={"text": "hello world"})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.type_text.assert_called_with("hello world")

    @pytest.mark.asyncio
    async def test_type_empty_text(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.TYPE, params={"text": ""})
        result = await handler.execute(action, [])

        assert result.success is False

    @pytest.mark.asyncio
    async def test_type_masks_sensitive_text_in_message(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.TYPE, params={"text": "secretPassword123"})
        result = await handler.execute(action, [])

        assert result.success is True
        assert "secretPassword123" not in result.message

    # --- Launch ---

    @pytest.mark.asyncio
    async def test_launch_app_known(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.LAUNCH, params={"app": "YouTube"})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.launch_app.assert_called_with("com.google.android.youtube")

    @pytest.mark.asyncio
    async def test_launch_app_case_insensitive(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.LAUNCH, params={"app": "gmail"})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.launch_app.assert_called_with("com.google.android.gm")

    @pytest.mark.asyncio
    async def test_launch_app_package_name(self, handler, mock_device):
        """If caller passes a package name directly, it should be used as-is."""
        action = ParsedAction(
            action_type=ActionType.LAUNCH,
            params={"app": "com.custom.myapp"},
        )
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.launch_app.assert_called_with("com.custom.myapp")

    @pytest.mark.asyncio
    async def test_launch_missing_app(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.LAUNCH, params={})
        result = await handler.execute(action, [])

        assert result.success is False

    # --- Back / Home ---

    @pytest.mark.asyncio
    async def test_back_button(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.BACK, params={})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.press_key.assert_called_with("KEYCODE_BACK")

    @pytest.mark.asyncio
    async def test_home_button(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.HOME, params={})
        result = await handler.execute(action, [])

        assert result.success is True
        mock_device.press_key.assert_called_with("KEYCODE_HOME")

    # --- Wait ---

    @pytest.mark.asyncio
    async def test_wait_action(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.WAIT, params={"seconds": 1})
        result = await handler.execute(action, [])

        assert result.success is True

    @pytest.mark.asyncio
    async def test_wait_capped_at_10s(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.WAIT, params={"seconds": 60})
        result = await handler.execute(action, [])

        assert result.success is True
        assert "10" in result.message  # Should say waited 10 seconds, not 60

    # --- Finish ---

    @pytest.mark.asyncio
    async def test_finish(self, handler, mock_device):
        action = ParsedAction(
            action_type=ActionType.FINISH,
            params={"message": "Task completed successfully"},
        )
        result = await handler.execute(action, [])

        assert result.success is True
        assert result.message == "Task completed successfully"

    # --- RequestInput ---

    @pytest.mark.asyncio
    async def test_request_input(self, handler, mock_device):
        action = ParsedAction(
            action_type=ActionType.REQUEST_INPUT,
            params={"prompt": "Enter your password"},
        )
        result = await handler.execute(action, [])

        assert result.success is True
        assert result.requires_input is True
        assert result.input_prompt == "Enter your password"

    # --- Unknown action ---

    @pytest.mark.asyncio
    async def test_unknown_action_type(self, handler, mock_device):
        action = ParsedAction(action_type=ActionType.UNKNOWN, params={})
        result = await handler.execute(action, [])

        assert result.success is False

    # --- Retry logic ---

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, handler, mock_device, sample_elements):
        """Handler should retry up to retry_count times on transient failure."""
        mock_device.tap = AsyncMock(
            side_effect=[
                ActionResult(success=False, error="transient"),
                ActionResult(success=True),
            ]
        )
        action = ParsedAction(action_type=ActionType.TAP, params={"element_id": 2})
        result = await handler.execute(action, sample_elements)

        assert result.success is True
        assert mock_device.tap.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, handler, mock_device, sample_elements):
        """After exhausting retries, handler returns failure."""
        mock_device.tap = AsyncMock(
            return_value=ActionResult(success=False, error="persistent error")
        )
        action = ParsedAction(action_type=ActionType.TAP, params={"element_id": 2})
        result = await handler.execute(action, sample_elements)

        assert result.success is False
        assert mock_device.tap.call_count == 3  # 1 original + 2 retries


# ---------------------------------------------------------------------------
# ActionExecutionResult tests
# ---------------------------------------------------------------------------


class TestActionExecutionResult:
    def test_success(self):
        r = ActionExecutionResult(success=True, message="OK")
        assert r.success is True
        assert r.error is None
        assert r.requires_input is False

    def test_failure(self):
        r = ActionExecutionResult(success=False, error="boom")
        assert r.success is False
        assert r.error == "boom"

    def test_input_required(self):
        r = ActionExecutionResult(
            success=True,
            message="need input",
            requires_input=True,
            input_prompt="Enter password",
        )
        assert r.requires_input is True
        assert r.input_prompt == "Enter password"


# ---------------------------------------------------------------------------
# launch_app.py resolve_package_name tests
# ---------------------------------------------------------------------------


class TestResolvePackageName:
    def test_exact_match(self):
        assert resolve_package_name("YouTube") == "com.google.android.youtube"

    def test_case_insensitive(self):
        assert resolve_package_name("youtube") == "com.google.android.youtube"
        assert resolve_package_name("YOUTUBE") == "com.google.android.youtube"

    def test_chrome(self):
        assert resolve_package_name("Chrome") == "com.android.chrome"

    def test_gmail(self):
        assert resolve_package_name("Gmail") == "com.google.android.gm"

    def test_whatsapp(self):
        assert resolve_package_name("WhatsApp") == "com.whatsapp"

    def test_instagram(self):
        assert resolve_package_name("Instagram") == "com.instagram.android"

    def test_unknown_returns_input(self):
        assert resolve_package_name("com.unknown.app") == "com.unknown.app"

    def test_fuzzy_match(self):
        """Fuzzy matching should resolve partial/variant names."""
        pkg = resolve_package_name("google maps")
        assert "maps" in pkg.lower()

    def test_common_apps_coverage(self):
        """All common apps should resolve to a package (not themselves)."""
        common = [
            "YouTube", "Chrome", "Gmail", "Maps", "Calendar",
            "Settings", "Camera", "WhatsApp", "Instagram",
        ]
        for app in common:
            result = resolve_package_name(app)
            assert result != app.lower(), f"Missing mapping for {app}"
            assert "." in result, f"Expected package name for {app}, got: {result}"


# ---------------------------------------------------------------------------
# Swipe coordinate calculation tests
# ---------------------------------------------------------------------------


class TestSwipeCoordinates:
    def test_swipe_up(self):
        params = calculate_swipe_coords(
            SwipeDirection.UP, screen_width=1080, screen_height=2400
        )
        assert params.start_x == params.end_x  # same horizontal center
        assert params.start_y > params.end_y    # start below, end above

    def test_swipe_down(self):
        params = calculate_swipe_coords(
            SwipeDirection.DOWN, screen_width=1080, screen_height=2400
        )
        assert params.start_x == params.end_x
        assert params.start_y < params.end_y  # start above, end below

    def test_swipe_left(self):
        params = calculate_swipe_coords(
            SwipeDirection.LEFT, screen_width=1080, screen_height=2400
        )
        assert params.start_y == params.end_y  # same vertical center
        assert params.start_x > params.end_x    # start right, end left

    def test_swipe_right(self):
        params = calculate_swipe_coords(
            SwipeDirection.RIGHT, screen_width=1080, screen_height=2400
        )
        assert params.start_y == params.end_y
        assert params.start_x < params.end_x  # start left, end right


# ---------------------------------------------------------------------------
# System key codes
# ---------------------------------------------------------------------------


class TestKeyCode:
    def test_back(self):
        assert KeyCode.BACK == "KEYCODE_BACK"

    def test_home(self):
        assert KeyCode.HOME == "KEYCODE_HOME"

    def test_enter(self):
        assert KeyCode.ENTER == "KEYCODE_ENTER"

    def test_volume_up(self):
        assert KeyCode.VOLUME_UP == "KEYCODE_VOLUME_UP"

    def test_volume_down(self):
        assert KeyCode.VOLUME_DOWN == "KEYCODE_VOLUME_DOWN"

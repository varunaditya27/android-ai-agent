"""
Tests for Action Handlers
=========================

Tests for:
- ActionHandler routing
- Individual action implementations
- Action result handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.actions.handler import ActionHandler, ActionExecutionResult
from app.agent.actions.tap import TapTarget, tap_element, find_tap_target
from app.agent.actions.swipe import SwipeDirection, calculate_swipe_coords
from app.agent.actions.type_text import type_text, find_input_element
from app.agent.actions.launch_app import resolve_package_name, APP_PACKAGES
from app.agent.actions.system import KeyCode, SystemActions
from app.llm.response_parser import ActionType, ParsedAction
from app.perception.ui_parser import UIElement


class TestActionHandler:
    """Tests for ActionHandler class."""

    @pytest.fixture
    def action_handler(self, mock_device):
        """Create an ActionHandler with mock device."""
        return ActionHandler(mock_device, action_delay=0)

    @pytest.mark.asyncio
    async def test_handle_tap_element(self, action_handler, mock_device):
        """Test handling tap action with element_id."""
        elements = [
            UIElement(
                index=0,
                element_type="Button",
                display_text="Submit",
                clickable=True,
                bounds=(100, 100, 200, 150),
                center=(150, 125),
            ),
        ]

        action = ParsedAction(
            action_type=ActionType.TAP,
            params={"element_id": 0},
        )

        result = await action_handler.execute(action, elements)

        assert result.success is True
        mock_device.tap.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tap_coordinates(self, action_handler, mock_device):
        """Test handling tap action with coordinates."""
        action = ParsedAction(
            action_type=ActionType.TAP,
            params={"x": 500, "y": 300},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.tap.assert_called_with(500, 300)

    @pytest.mark.asyncio
    async def test_handle_swipe(self, action_handler, mock_device):
        """Test handling swipe action."""
        action = ParsedAction(
            action_type=ActionType.SWIPE,
            params={"direction": "up"},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.swipe.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_type(self, action_handler, mock_device):
        """Test handling type action."""
        action = ParsedAction(
            action_type=ActionType.TYPE,
            params={"text": "hello world"},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.type_text.assert_called_with("hello world")

    @pytest.mark.asyncio
    async def test_handle_launch(self, action_handler, mock_device):
        """Test handling launch action."""
        action = ParsedAction(
            action_type=ActionType.LAUNCH,
            params={"app": "YouTube"},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.launch_app.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_back(self, action_handler, mock_device):
        """Test handling back action."""
        action = ParsedAction(
            action_type=ActionType.BACK,
            params={},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.press_back.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_home(self, action_handler, mock_device):
        """Test handling home action."""
        action = ParsedAction(
            action_type=ActionType.HOME,
            params={},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        mock_device.press_home.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_finish(self, action_handler, mock_device):
        """Test handling finish action."""
        action = ParsedAction(
            action_type=ActionType.FINISH,
            params={"message": "Task completed"},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        assert result.message == "Task completed"

    @pytest.mark.asyncio
    async def test_handle_request_input(self, action_handler, mock_device):
        """Test handling request input action."""
        action = ParsedAction(
            action_type=ActionType.REQUEST_INPUT,
            params={"prompt": "Enter password"},
        )

        result = await action_handler.execute(action, [])

        assert result.success is True
        assert result.requires_input is True
        assert result.input_prompt == "Enter password"

    @pytest.mark.asyncio
    async def test_handle_invalid_element_id(self, action_handler, mock_device):
        """Test handling tap with invalid element_id."""
        elements = [
            UIElement(
                index=0,
                element_type="Button",
                display_text="Submit",
                bounds=(100, 100, 200, 150),
            ),
        ]

        action = ParsedAction(
            action_type=ActionType.TAP,
            params={"element_id": 99},  # Invalid
        )

        result = await action_handler.execute(action, elements)

        assert result.success is False
        assert "not found" in result.error.lower()


class TestTapActions:
    """Tests for tap action functions."""

    def test_tap_target_from_element(self):
        """Test TapTarget creation from element."""
        element = UIElement(
            index=0,
            element_type="Button",
            display_text="Submit",
            bounds=(100, 100, 200, 150),
            center=(150, 125),
        )

        target = TapTarget.from_element(element)

        assert target.x == 150
        assert target.y == 125
        assert target.element_index == 0

    def test_find_tap_target_by_id(self):
        """Test finding tap target by element ID."""
        elements = [
            UIElement(index=0, element_type="Button", display_text="A", bounds=(0, 0, 100, 50), center=(50, 25)),
            UIElement(index=1, element_type="Button", display_text="B", bounds=(100, 0, 200, 50), center=(150, 25)),
        ]

        target = find_tap_target(elements, element_id=1)

        assert target is not None
        assert target.x == 150
        assert target.y == 25

    def test_find_tap_target_by_coords(self):
        """Test finding tap target by coordinates."""
        target = find_tap_target([], x=300, y=400)

        assert target is not None
        assert target.x == 300
        assert target.y == 400

    def test_find_tap_target_none(self):
        """Test find returns None when no target found."""
        target = find_tap_target([], element_id=5)  # No elements
        assert target is None


class TestSwipeActions:
    """Tests for swipe action functions."""

    def test_calculate_swipe_coords_up(self):
        """Test calculating swipe coordinates for up direction."""
        x1, y1, x2, y2 = calculate_swipe_coords(
            SwipeDirection.UP,
            screen_width=1080,
            screen_height=2400,
        )

        # Should swipe from bottom to top
        assert x1 == x2  # Same x (center)
        assert y1 > y2  # Start below end

    def test_calculate_swipe_coords_down(self):
        """Test calculating swipe coordinates for down direction."""
        x1, y1, x2, y2 = calculate_swipe_coords(
            SwipeDirection.DOWN,
            screen_width=1080,
            screen_height=2400,
        )

        assert x1 == x2
        assert y1 < y2  # Start above end

    def test_calculate_swipe_coords_left(self):
        """Test calculating swipe coordinates for left direction."""
        x1, y1, x2, y2 = calculate_swipe_coords(
            SwipeDirection.LEFT,
            screen_width=1080,
            screen_height=2400,
        )

        assert y1 == y2
        assert x1 > x2  # Start right of end

    def test_calculate_swipe_coords_right(self):
        """Test calculating swipe coordinates for right direction."""
        x1, y1, x2, y2 = calculate_swipe_coords(
            SwipeDirection.RIGHT,
            screen_width=1080,
            screen_height=2400,
        )

        assert y1 == y2
        assert x1 < x2  # Start left of end

    def test_swipe_direction_from_string(self):
        """Test SwipeDirection parsing from string."""
        assert SwipeDirection.from_string("up") == SwipeDirection.UP
        assert SwipeDirection.from_string("DOWN") == SwipeDirection.DOWN
        assert SwipeDirection.from_string("Left") == SwipeDirection.LEFT
        assert SwipeDirection.from_string("invalid") == SwipeDirection.UP  # Default


class TestTypeActions:
    """Tests for type action functions."""

    def test_find_input_element_editable(self):
        """Test finding editable input element."""
        elements = [
            UIElement(index=0, element_type="TextView", display_text="Label", editable=False, bounds=(0, 0, 100, 50)),
            UIElement(index=1, element_type="EditText", display_text="", editable=True, focused=True, bounds=(0, 50, 100, 100)),
        ]

        input_elem = find_input_element(elements)

        assert input_elem is not None
        assert input_elem.index == 1

    def test_find_input_element_focused(self):
        """Test finding focused input element."""
        elements = [
            UIElement(index=0, element_type="EditText", display_text="", editable=True, focused=False, bounds=(0, 0, 100, 50)),
            UIElement(index=1, element_type="EditText", display_text="", editable=True, focused=True, bounds=(0, 50, 100, 100)),
        ]

        input_elem = find_input_element(elements, prefer_focused=True)

        assert input_elem is not None
        assert input_elem.focused is True

    def test_find_input_element_none(self):
        """Test returns None when no input found."""
        elements = [
            UIElement(index=0, element_type="Button", display_text="Submit", editable=False, bounds=(0, 0, 100, 50)),
        ]

        input_elem = find_input_element(elements)
        assert input_elem is None


class TestLaunchAppActions:
    """Tests for app launching functions."""

    def test_resolve_package_youtube(self):
        """Test resolving YouTube package name."""
        package = resolve_package_name("YouTube")
        assert package == "com.google.android.youtube"

    def test_resolve_package_chrome(self):
        """Test resolving Chrome package name."""
        package = resolve_package_name("Chrome")
        assert package == "com.android.chrome"

    def test_resolve_package_case_insensitive(self):
        """Test package resolution is case-insensitive."""
        package1 = resolve_package_name("youtube")
        package2 = resolve_package_name("YOUTUBE")
        package3 = resolve_package_name("YouTube")

        assert package1 == package2 == package3

    def test_resolve_package_unknown(self):
        """Test resolving unknown app returns input."""
        package = resolve_package_name("com.unknown.app")
        assert package == "com.unknown.app"

    def test_app_packages_coverage(self):
        """Test common apps are in package mapping."""
        common_apps = [
            "YouTube", "Chrome", "Gmail", "Maps", "Calendar",
            "Settings", "Camera", "WhatsApp", "Instagram", "Twitter",
        ]

        for app in common_apps:
            assert resolve_package_name(app) != app.lower(), f"Missing package for {app}"


class TestSystemActions:
    """Tests for system action functions."""

    def test_key_codes_exist(self):
        """Test key codes are defined."""
        assert KeyCode.BACK == 4
        assert KeyCode.HOME == 3
        assert KeyCode.ENTER == 66
        assert KeyCode.VOLUME_UP == 24

    @pytest.mark.asyncio
    async def test_system_actions_back(self, mock_device):
        """Test SystemActions back button."""
        system = SystemActions(mock_device)
        await system.back()
        mock_device.press_back.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_actions_home(self, mock_device):
        """Test SystemActions home button."""
        system = SystemActions(mock_device)
        await system.home()
        mock_device.press_home.assert_called_once()


class TestActionExecutionResult:
    """Tests for ActionExecutionResult class."""

    def test_success_result(self):
        """Test successful result creation."""
        result = ActionExecutionResult(
            success=True,
            message="Action completed",
        )

        assert result.success is True
        assert result.error is None
        assert result.requires_input is False

    def test_failure_result(self):
        """Test failure result creation."""
        result = ActionExecutionResult(
            success=False,
            message="",
            error="Element not found",
        )

        assert result.success is False
        assert result.error == "Element not found"

    def test_input_required_result(self):
        """Test input required result."""
        result = ActionExecutionResult(
            success=True,
            message="Waiting for input",
            requires_input=True,
            input_prompt="Enter your password",
        )

        assert result.requires_input is True
        assert result.input_prompt == "Enter your password"

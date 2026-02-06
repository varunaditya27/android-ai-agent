"""
Tests for conftest fixtures and shared test utilities.

This module provides pytest fixtures used across all tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from app.device.cloud_provider import CloudDevice, DeviceInfo, ActionResult
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


@pytest.fixture
def mock_device() -> MagicMock:
    """Create a mock device for testing."""
    device = MagicMock(spec=CloudDevice)

    # Setup device info
    device.info = DeviceInfo(
        device_id="test-device-123",
        platform="android",
        os_version="14.0",
        screen_width=1080,
        screen_height=2400,
        model="Test Pixel 8",
        manufacturer="Google",
    )
    device.is_connected = True

    # Setup async mocks
    device.connect = AsyncMock(return_value=True)
    device.disconnect = AsyncMock(return_value=None)

    device.capture_screenshot = AsyncMock(
        return_value="base64_screenshot_data_here"
    )

    device.get_ui_hierarchy = AsyncMock(
        return_value={
            "elements": [
                {"class": "android.widget.TextView", "text": "Home", "clickable": True, "bounds": {"left": 0, "top": 0, "right": 100, "bottom": 50}, "center_x": 50, "center_y": 25},
                {"class": "android.widget.EditText", "text": "Search", "clickable": True, "bounds": {"left": 0, "top": 50, "right": 1080, "bottom": 150}, "center_x": 540, "center_y": 100},
                {"class": "android.widget.Button", "text": "Settings", "clickable": True, "bounds": {"left": 0, "top": 150, "right": 200, "bottom": 200}, "center_x": 100, "center_y": 175},
            ]
        }
    )

    device.get_current_app = AsyncMock(
        return_value="com.google.android.apps.messaging"
    )

    device.tap = AsyncMock(
        return_value=ActionResult(success=True)
    )

    device.long_press = AsyncMock(
        return_value=ActionResult(success=True)
    )

    device.swipe = AsyncMock(
        return_value=ActionResult(success=True)
    )

    device.type_text = AsyncMock(
        return_value=ActionResult(success=True)
    )

    device.press_key = AsyncMock(
        return_value=ActionResult(success=True)
    )

    device.launch_app = AsyncMock(
        return_value=ActionResult(success=True)
    )

    return device


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for testing."""
    client = MagicMock(spec=LLMClient)

    # Default response (Gemini format)
    client.complete_with_vision = AsyncMock(
        return_value=LLMResponse(
            content='<think>I see a Settings button. I will tap it.</think>\n<answer>do(action="Tap", element_id=2)</answer>',
            model="gemini-2.0-flash",
            usage={"prompt_tokens": 100, "candidates_tokens": 50, "total_tokens": 150},
        )
    )

    client.complete = AsyncMock(
        return_value=LLMResponse(
            content="Test response",
            model="gemini-2.0-flash",
            usage={"prompt_tokens": 50, "candidates_tokens": 25, "total_tokens": 75},
        )
    )

    return client


@pytest.fixture
def sample_ui_hierarchy() -> str:
    """Sample UI hierarchy XML for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <hierarchy rotation="0">
        <node index="0" text="Android AI Agent" resource-id="com.app:id/title"
              class="android.widget.TextView" package="com.app"
              content-desc="" checkable="false" checked="false"
              clickable="false" enabled="true" focusable="false"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[100,100][500,150]"/>

        <node index="1" text="" resource-id="com.app:id/search_input"
              class="android.widget.EditText" package="com.app"
              content-desc="Search input" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[50,200][1030,280]"/>

        <node index="2" text="Login" resource-id="com.app:id/login_button"
              class="android.widget.Button" package="com.app"
              content-desc="" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[400,500][680,580]"/>

        <node index="3" text="Email" resource-id="com.app:id/email_input"
              class="android.widget.EditText" package="com.app"
              content-desc="Enter your email" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[100,600][980,680]"/>

        <node index="4" text="Password" resource-id="com.app:id/password_input"
              class="android.widget.EditText" package="com.app"
              content-desc="Enter your password" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="true" selected="false"
              bounds="[100,700][980,780]"/>

        <node index="5" text="Remember me" resource-id="com.app:id/remember_checkbox"
              class="android.widget.CheckBox" package="com.app"
              content-desc="" checkable="true" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[100,800][400,850]"/>

        <node index="6" text="Submit" resource-id="com.app:id/submit_button"
              class="android.widget.Button" package="com.app"
              content-desc="Submit form" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true"
              focused="false" scrollable="false" long-clickable="false"
              password="false" selected="false"
              bounds="[400,900][680,980]"/>
    </hierarchy>
    """


@pytest.fixture
def sample_elements() -> list[dict[str, Any]]:
    """Sample parsed UI elements for testing."""
    return [
        {
            "index": 0,
            "text": "Android AI Agent",
            "element_type": "TextView",
            "clickable": False,
            "editable": False,
            "bounds": (100, 100, 500, 150),
            "center": (300, 125),
        },
        {
            "index": 1,
            "text": "",
            "content_desc": "Search input",
            "element_type": "EditText",
            "clickable": True,
            "editable": True,
            "bounds": (50, 200, 1030, 280),
            "center": (540, 240),
        },
        {
            "index": 2,
            "text": "Login",
            "element_type": "Button",
            "clickable": True,
            "editable": False,
            "bounds": (400, 500, 680, 580),
            "center": (540, 540),
        },
    ]
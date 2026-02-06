"""
Shared Test Fixtures
====================

Pytest fixtures used across all test modules.
Provides correctly-typed mocks matching the actual codebase APIs.
"""

import os

# Set required env vars BEFORE any app.* imports to avoid pydantic validation errors
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("GROQ_API_KEY", "gsk_test-groq-key-for-testing")

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from app.device.cloud_provider import CloudDevice, DeviceInfo, ActionResult
from app.llm.client import LLMClient
from app.llm.models import LLMResponse
from app.perception.ui_parser import UIElement


def _make_bounds(left: int, top: int, right: int, bottom: int) -> dict[str, int]:
    """Helper to create a properly-formatted bounds dict."""
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "center_x": (left + right) // 2,
        "center_y": (top + bottom) // 2,
        "width": right - left,
        "height": bottom - top,
    }


# ---------------------------------------------------------------------------
# Device mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_device() -> MagicMock:
    """Create a mock CloudDevice with all abstract methods mocked."""
    device = MagicMock(spec=CloudDevice)

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
    device.state = "connected"

    # Async lifecycle
    device.connect = AsyncMock(return_value=True)
    device.disconnect = AsyncMock(return_value=None)

    # Async observation
    # A 1x1 black PNG base64
    device.capture_screenshot = AsyncMock(return_value="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
    device.get_ui_hierarchy = AsyncMock(return_value={
        "elements": [
            {
                "class": "android.widget.TextView",
                "text": "Home",
                "content_desc": "",
                "resource_id": "",
                "clickable": True,
                "enabled": True,
                "focusable": False,
                "focused": False,
                "scrollable": False,
                "long_clickable": False,
                "checkable": False,
                "checked": False,
                "bounds": _make_bounds(0, 0, 100, 50),
            },
            {
                "class": "android.widget.EditText",
                "text": "",
                "content_desc": "Search",
                "resource_id": "com.app:id/search",
                "clickable": True,
                "enabled": True,
                "focusable": True,
                "focused": False,
                "scrollable": False,
                "long_clickable": False,
                "checkable": False,
                "checked": False,
                "bounds": _make_bounds(0, 50, 1080, 150),
            },
            {
                "class": "android.widget.Button",
                "text": "Settings",
                "content_desc": "",
                "resource_id": "com.app:id/settings_btn",
                "clickable": True,
                "enabled": True,
                "focusable": True,
                "focused": False,
                "scrollable": False,
                "long_clickable": False,
                "checkable": False,
                "checked": False,
                "bounds": _make_bounds(0, 150, 200, 200),
            },
        ]
    })

    device.get_current_app = AsyncMock(return_value="com.google.android.apps.messaging")

    # Async actions – all return ActionResult
    device.tap = AsyncMock(return_value=ActionResult(success=True))
    device.long_press = AsyncMock(return_value=ActionResult(success=True))
    device.swipe = AsyncMock(return_value=ActionResult(success=True))
    device.swipe_direction = AsyncMock(return_value=ActionResult(success=True))
    device.type_text = AsyncMock(return_value=ActionResult(success=True))
    device.press_key = AsyncMock(return_value=ActionResult(success=True))
    device.launch_app = AsyncMock(return_value=ActionResult(success=True))

    return device


# ---------------------------------------------------------------------------
# LLM client mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)

    client.complete_with_vision = AsyncMock(
        return_value=LLMResponse(
            content='<think>I see a Settings button. I will tap it.</think>\n<answer>do(action="Tap", element_id=2)</answer>',
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "candidates_tokens": 50, "total_tokens": 150},
        )
    )

    client.complete = AsyncMock(
        return_value=LLMResponse(
            content="Test response",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 50, "candidates_tokens": 25, "total_tokens": 75},
        )
    )

    client.get_api_call_count = MagicMock(return_value=0)
    client.reset_api_call_count = MagicMock()
    client.close = AsyncMock()

    return client


# ---------------------------------------------------------------------------
# UI element helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_elements() -> list[UIElement]:
    """Sample parsed UIElement list for testing."""
    return [
        UIElement(
            index=0,
            element_class="android.widget.TextView",
            text="Android AI Agent",
            content_desc="",
            resource_id="com.app:id/title",
            bounds=_make_bounds(100, 100, 500, 150),
            clickable=False,
            enabled=True,
        ),
        UIElement(
            index=1,
            element_class="android.widget.EditText",
            text="",
            content_desc="Search input",
            resource_id="com.app:id/search_input",
            bounds=_make_bounds(50, 200, 1030, 280),
            clickable=True,
            enabled=True,
            focusable=True,
            editable=True,
        ),
        UIElement(
            index=2,
            element_class="android.widget.Button",
            text="Login",
            content_desc="",
            resource_id="com.app:id/login_button",
            bounds=_make_bounds(400, 500, 680, 580),
            clickable=True,
            enabled=True,
            focusable=True,
        ),
        UIElement(
            index=3,
            element_class="android.widget.EditText",
            text="Email",
            content_desc="Enter your email",
            resource_id="com.app:id/email_input",
            bounds=_make_bounds(100, 600, 980, 680),
            clickable=True,
            enabled=True,
            focusable=True,
            editable=True,
        ),
        UIElement(
            index=4,
            element_class="android.widget.EditText",
            text="Password",
            content_desc="Enter your password",
            resource_id="com.app:id/password_input",
            bounds=_make_bounds(100, 700, 980, 780),
            clickable=True,
            enabled=True,
            focusable=True,
            editable=True,
        ),
        UIElement(
            index=5,
            element_class="android.widget.CheckBox",
            text="Remember me",
            content_desc="",
            resource_id="com.app:id/remember_checkbox",
            bounds=_make_bounds(100, 800, 400, 850),
            clickable=True,
            enabled=True,
            focusable=True,
            checkable=True,
            checked=False,
        ),
        UIElement(
            index=6,
            element_class="android.widget.Button",
            text="Submit",
            content_desc="Submit form",
            resource_id="com.app:id/submit_button",
            bounds=_make_bounds(400, 900, 680, 980),
            clickable=True,
            enabled=True,
            focusable=True,
        ),
    ]


@pytest.fixture
def sample_ui_hierarchy_xml() -> str:
    """Sample UI hierarchy XML for testing UIParser XML parsing."""
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

    <node index="5" text="Submit" resource-id="com.app:id/submit_button"
          class="android.widget.Button" package="com.app"
          content-desc="Submit form" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" long-clickable="false"
          password="false" selected="false"
          bounds="[400,900][680,980]"/>
</hierarchy>"""

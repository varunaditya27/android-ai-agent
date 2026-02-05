"""
BrowserStack Cloud Device Client
================================

Implementation of CloudDevice for BrowserStack App Automate.

BrowserStack provides cloud-hosted real Android devices.
This client handles device provisioning, screenshot capture,
UI hierarchy retrieval, and input actions via Appium.

Usage:
    from app.device import BrowserStackDevice

    device = BrowserStackDevice(username="user", access_key="key")
    await device.connect()
    screenshot = await device.capture_screenshot()
"""

import asyncio
import base64
import time
from typing import Any, Optional

import aiohttp

from app.config import get_settings
from app.device.cloud_provider import (
    ActionResult,
    CloudDevice,
    DeviceInfo,
    DeviceState,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BrowserStackDevice(CloudDevice):
    """
    BrowserStack App Automate device implementation.

    Provides device control via BrowserStack's Appium REST API including:
    - Session management
    - Screenshot capture
    - UI hierarchy retrieval (via Appium page source)
    - Touch and gesture input
    - App launching
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        username: Optional[str] = None,
        access_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        """
        Initialize BrowserStack device client.

        Args:
            device_id: Device/OS combination (e.g., "Google Pixel 6-12.0").
            username: BrowserStack username.
            access_key: BrowserStack access key.
            api_url: API base URL.
        """
        super().__init__(device_id)

        settings = get_settings()
        self.username = username or settings.device.browserstack_username
        self.access_key = access_key or settings.device.browserstack_access_key
        self.api_url = (api_url or settings.device.browserstack_api_url).rstrip("/")

        # Appium session URL (set after session creation)
        self._appium_url: Optional[str] = None
        self._session_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.username or not self.access_key:
            logger.warning("BrowserStack credentials not configured")

    def _get_auth(self) -> aiohttp.BasicAuth:
        """Get HTTP basic auth for BrowserStack."""
        return aiohttp.BasicAuth(self.username, self.access_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                auth=self._get_auth(),
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session

    async def connect(self) -> bool:
        """
        Connect to a BrowserStack device via Appium.

        Creates a new Appium session with specified capabilities.

        Returns:
            True if connection successful.
        """
        self.state = DeviceState.CONNECTING
        logger.info("Connecting to BrowserStack device", device_id=self.device_id)

        # Parse device_id into device and OS version
        device_name = "Google Pixel 6"
        os_version = "12.0"

        if self.device_id and "-" in self.device_id:
            parts = self.device_id.rsplit("-", 1)
            device_name = parts[0]
            os_version = parts[1] if len(parts) > 1 else "12.0"

        try:
            session = await self._get_session()

            # Appium desired capabilities for BrowserStack
            capabilities = {
                "bstack:options": {
                    "deviceName": device_name,
                    "osVersion": os_version,
                    "realMobile": True,
                    "local": False,
                    "debug": True,
                    "networkLogs": False,
                    "appiumVersion": "2.0.0",
                },
                "platformName": "Android",
                "appium:automationName": "UiAutomator2",
                "appium:noReset": False,
                "appium:fullReset": False,
                # Start with a launcher/home screen
                "appium:appPackage": "com.google.android.apps.nexuslauncher",
                "appium:appActivity": "com.google.android.apps.nexuslauncher.NexusLauncherActivity",
            }

            # Create Appium session
            appium_hub_url = "https://hub-cloud.browserstack.com/wd/hub"

            async with session.post(
                f"{appium_hub_url}/session",
                json={"capabilities": {"alwaysMatch": capabilities}},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status not in (200, 303):
                    error_text = await response.text()
                    logger.error(
                        "Failed to create BrowserStack session",
                        status=response.status,
                        error=error_text,
                    )
                    self.state = DeviceState.ERROR
                    return False

                data = await response.json()
                self._session_id = data.get("value", {}).get("sessionId")
                caps = data.get("value", {}).get("capabilities", {})

                if not self._session_id:
                    logger.error("No session ID in response", data=data)
                    self.state = DeviceState.ERROR
                    return False

                self._appium_url = f"{appium_hub_url}/session/{self._session_id}"

                # Parse device info from capabilities
                window_size = caps.get("deviceScreenSize", "1080x2340")
                width, height = 1080, 2340
                if "x" in window_size:
                    width, height = map(int, window_size.split("x"))

                self.info = DeviceInfo(
                    device_id=self._session_id,
                    platform="android",
                    os_version=caps.get("platformVersion", os_version),
                    screen_width=width,
                    screen_height=height,
                    model=caps.get("deviceModel", device_name),
                    manufacturer=caps.get("deviceManufacturer", ""),
                )

            self.state = DeviceState.CONNECTED
            logger.info(
                "Connected to BrowserStack device",
                session_id=self._session_id,
                device=device_name,
                os_version=os_version,
            )
            return True

        except aiohttp.ClientError as e:
            logger.error("Connection error", error=str(e))
            self.state = DeviceState.ERROR
            return False
        except Exception as e:
            logger.error("Unexpected error during connect", error=str(e))
            self.state = DeviceState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect and cleanup the Appium session."""
        if self._appium_url and self._session_id:
            try:
                session = await self._get_session()
                async with session.delete(self._appium_url) as response:
                    if response.status == 200:
                        logger.info("BrowserStack session terminated")
            except Exception as e:
                logger.warning("Error terminating session", error=str(e))

        if self._session and not self._session.closed:
            await self._session.close()

        self._appium_url = None
        self._session_id = None
        self.state = DeviceState.DISCONNECTED
        logger.info("Disconnected from BrowserStack")

    async def capture_screenshot(self) -> str:
        """
        Capture current screen as base64-encoded PNG.

        Returns:
            Base64-encoded screenshot string.
        """
        if not self.is_connected or not self._appium_url:
            raise RuntimeError("Device not connected")

        session = await self._get_session()
        start_time = time.time()

        async with session.get(f"{self._appium_url}/screenshot") as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Screenshot failed: {error_text}")

            data = await response.json()
            screenshot_b64 = data.get("value", "")

            duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                "Screenshot captured",
                duration_ms=duration_ms,
            )

            return screenshot_b64

    async def get_ui_hierarchy(self) -> dict[str, Any]:
        """
        Get the UI hierarchy via Appium page source.

        Returns:
            Dictionary with 'xml' and 'elements' keys.
        """
        if not self.is_connected or not self._appium_url:
            raise RuntimeError("Device not connected")

        session = await self._get_session()

        async with session.get(f"{self._appium_url}/source") as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"UI hierarchy failed: {error_text}")

            data = await response.json()
            xml_content = data.get("value", "")

            # Parse XML to extract elements
            elements = self._parse_ui_xml(xml_content)

            logger.debug("UI hierarchy retrieved", element_count=len(elements))

            return {"xml": xml_content, "elements": elements}

    def _parse_ui_xml(self, xml_content: str) -> list[dict[str, Any]]:
        """Parse Appium page source XML into element list."""
        from xml.etree import ElementTree

        elements = []
        try:
            root = ElementTree.fromstring(xml_content)
            self._extract_elements_recursive(root, elements, 0)
        except ElementTree.ParseError as e:
            logger.warning("Failed to parse UI XML", error=str(e))
        return elements

    def _extract_elements_recursive(
        self,
        node: Any,
        elements: list[dict[str, Any]],
        index: int,
    ) -> int:
        """Recursively extract elements from XML tree."""
        # Extract bounds
        bounds_str = node.attrib.get("bounds", "[0,0][0,0]")
        bounds = self._parse_bounds(bounds_str)

        element = {
            "index": index,
            "class": node.attrib.get("class", node.tag),
            "text": node.attrib.get("text", ""),
            "content_desc": node.attrib.get("content-desc", ""),
            "resource_id": node.attrib.get("resource-id", ""),
            "clickable": node.attrib.get("clickable", "false") == "true",
            "enabled": node.attrib.get("enabled", "true") == "true",
            "focusable": node.attrib.get("focusable", "false") == "true",
            "focused": node.attrib.get("focused", "false") == "true",
            "scrollable": node.attrib.get("scrollable", "false") == "true",
            "long_clickable": node.attrib.get("long-clickable", "false") == "true",
            "checkable": node.attrib.get("checkable", "false") == "true",
            "checked": node.attrib.get("checked", "false") == "true",
            "bounds": bounds,
        }

        has_content = element["text"] or element["content_desc"]
        is_interactive = element["clickable"] or element["focusable"] or element["scrollable"]

        if has_content or is_interactive:
            elements.append(element)
            index += 1

        for child in node:
            index = self._extract_elements_recursive(child, elements, index)

        return index

    def _parse_bounds(self, bounds_str: str) -> dict[str, int]:
        """Parse bounds string."""
        import re

        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return {
                "left": x1,
                "top": y1,
                "right": x2,
                "bottom": y2,
                "center_x": (x1 + x2) // 2,
                "center_y": (y1 + y2) // 2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        return {"left": 0, "top": 0, "right": 0, "bottom": 0, "center_x": 0, "center_y": 0, "width": 0, "height": 0}

    async def tap(self, x: int, y: int) -> ActionResult:
        """Perform a tap at coordinates using Appium W3C actions."""
        return await self._perform_touch_action(
            [
                {"type": "pointerMove", "duration": 0, "x": x, "y": y},
                {"type": "pointerDown", "button": 0},
                {"type": "pause", "duration": 100},
                {"type": "pointerUp", "button": 0},
            ]
        )

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> ActionResult:
        """Perform a long press at coordinates."""
        return await self._perform_touch_action(
            [
                {"type": "pointerMove", "duration": 0, "x": x, "y": y},
                {"type": "pointerDown", "button": 0},
                {"type": "pause", "duration": duration_ms},
                {"type": "pointerUp", "button": 0},
            ]
        )

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> ActionResult:
        """Perform a swipe gesture."""
        return await self._perform_touch_action(
            [
                {"type": "pointerMove", "duration": 0, "x": start_x, "y": start_y},
                {"type": "pointerDown", "button": 0},
                {"type": "pointerMove", "duration": duration_ms, "x": end_x, "y": end_y},
                {"type": "pointerUp", "button": 0},
            ]
        )

    async def type_text(self, text: str) -> ActionResult:
        """Type text into focused element."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        try:
            session = await self._get_session()

            # First get the active element
            async with session.get(f"{self._appium_url}/element/active") as response:
                if response.status != 200:
                    # No active element, try typing anyway
                    pass
                data = await response.json()
                element_id = data.get("value", {}).get("ELEMENT")

            if element_id:
                # Send keys to active element
                async with session.post(
                    f"{self._appium_url}/element/{element_id}/value",
                    json={"text": text},
                ) as response:
                    if response.status == 200:
                        return ActionResult(success=True)
                    error_text = await response.text()
                    return ActionResult(success=False, error=error_text)
            else:
                # Fallback: use keyboard
                async with session.post(
                    f"{self._appium_url}/keys",
                    json={"value": list(text)},
                ) as response:
                    if response.status == 200:
                        return ActionResult(success=True)
                    error_text = await response.text()
                    return ActionResult(success=False, error=error_text)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def press_key(self, key_code: str) -> ActionResult:
        """Press a system key (back, home, etc.)."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        # Map key codes to Appium key codes
        key_map = {
            "KEYCODE_BACK": 4,
            "KEYCODE_HOME": 3,
            "KEYCODE_MENU": 82,
            "KEYCODE_ENTER": 66,
            "KEYCODE_SEARCH": 84,
            "back": 4,
            "home": 3,
            "menu": 82,
            "enter": 66,
        }

        android_key = key_map.get(key_code, key_map.get(key_code.upper()))
        if android_key is None:
            return ActionResult(success=False, error=f"Unknown key code: {key_code}")

        try:
            session = await self._get_session()
            async with session.post(
                f"{self._appium_url}/appium/device/press_keycode",
                json={"keycode": android_key},
            ) as response:
                if response.status == 200:
                    return ActionResult(success=True)
                error_text = await response.text()
                return ActionResult(success=False, error=error_text)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def launch_app(self, package_name: str) -> ActionResult:
        """Launch an application by package name."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        try:
            session = await self._get_session()
            async with session.post(
                f"{self._appium_url}/appium/device/activate_app",
                json={"bundleId": package_name},
            ) as response:
                if response.status == 200:
                    return ActionResult(success=True)
                error_text = await response.text()
                return ActionResult(success=False, error=error_text)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def get_current_app(self) -> str:
        """Get the currently focused app's package name."""
        if not self.is_connected or not self._appium_url:
            return ""

        try:
            session = await self._get_session()
            async with session.get(
                f"{self._appium_url}/appium/device/current_package"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value", "")
        except Exception as e:
            logger.warning("Failed to get current app", error=str(e))

        return ""

    async def _perform_touch_action(self, actions: list[dict]) -> ActionResult:
        """Perform W3C touch actions."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            session = await self._get_session()

            payload = {
                "actions": [
                    {
                        "type": "pointer",
                        "id": "finger1",
                        "parameters": {"pointerType": "touch"},
                        "actions": actions,
                    }
                ]
            }

            async with session.post(
                f"{self._appium_url}/actions",
                json=payload,
            ) as response:
                duration_ms = int((time.time() - start_time) * 1000)

                if response.status == 200:
                    return ActionResult(success=True, duration_ms=duration_ms)
                error_text = await response.text()
                return ActionResult(success=False, error=error_text, duration_ms=duration_ms)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

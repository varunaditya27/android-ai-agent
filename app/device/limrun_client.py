"""
Limrun Cloud Device Client
==========================

Implementation of CloudDevice for Limrun cloud Android service.

Limrun provides cloud-hosted Android devices accessible via API.
This client handles device provisioning, screenshot capture,
UI hierarchy retrieval, and input actions.

Usage:
    from app.device import LimrunDevice

    device = LimrunDevice(api_key="your-key")
    await device.connect()
    screenshot = await device.capture_screenshot()
"""

import asyncio
import base64
import time
from typing import Any, Optional
from xml.etree import ElementTree

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


class LimrunDevice(CloudDevice):
    """
    Limrun cloud device implementation.

    Provides full device control via Limrun's REST API including:
    - Device provisioning and management
    - Screenshot capture
    - UI hierarchy (accessibility tree) retrieval
    - Touch and gesture input
    - App launching and key presses
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        """
        Initialize Limrun device client.

        Args:
            device_id: Optional specific device to use.
            api_key: Limrun API key (defaults to config).
            api_url: Limrun API URL (defaults to config).
        """
        super().__init__(device_id)

        settings = get_settings()
        self.api_key = api_key or settings.device.limrun_api_key
        self.api_url = (api_url or settings.device.limrun_api_url).rstrip("/")

        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None

        if not self.api_key:
            logger.warning("Limrun API key not configured")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def connect(self) -> bool:
        """
        Connect to a Limrun cloud device.

        Provisions a new device session or connects to existing device.

        Returns:
            True if connection successful.
        """
        self.state = DeviceState.CONNECTING
        logger.info("Connecting to Limrun device", device_id=self.device_id)

        try:
            session = await self._get_session()

            # Request a device session
            payload = {}
            if self.device_id:
                payload["device_id"] = self.device_id

            async with session.post(
                f"{self.api_url}/devices/session",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        "Failed to create device session",
                        status=response.status,
                        error=error_text,
                    )
                    self.state = DeviceState.ERROR
                    return False

                data = await response.json()
                self._session_id = data.get("session_id")
                device_info = data.get("device", {})

                self.info = DeviceInfo(
                    device_id=device_info.get("id", self.device_id or "unknown"),
                    platform="android",
                    os_version=device_info.get("os_version", ""),
                    screen_width=device_info.get("screen_width", 1080),
                    screen_height=device_info.get("screen_height", 2340),
                    model=device_info.get("model", ""),
                    manufacturer=device_info.get("manufacturer", ""),
                )

            self.state = DeviceState.CONNECTED
            logger.info(
                "Connected to Limrun device",
                session_id=self._session_id,
                device_model=self.info.model,
                screen_size=f"{self.info.screen_width}x{self.info.screen_height}",
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
        """Disconnect from the device and cleanup."""
        if self._session_id:
            try:
                session = await self._get_session()
                async with session.delete(
                    f"{self.api_url}/devices/session/{self._session_id}"
                ) as response:
                    if response.status == 200:
                        logger.info("Device session terminated", session_id=self._session_id)
            except Exception as e:
                logger.warning("Error terminating session", error=str(e))

        if self._session and not self._session.closed:
            await self._session.close()

        self._session_id = None
        self.state = DeviceState.DISCONNECTED
        logger.info("Disconnected from Limrun device")

    async def capture_screenshot(self) -> str:
        """
        Capture current screen as base64-encoded PNG.

        Returns:
            Base64-encoded screenshot string.

        Raises:
            RuntimeError: If device not connected.
        """
        if not self.is_connected or not self._session_id:
            raise RuntimeError("Device not connected")

        session = await self._get_session()
        start_time = time.time()

        async with session.get(
            f"{self.api_url}/devices/session/{self._session_id}/screenshot"
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Screenshot failed: {error_text}")

            # Response should be PNG image bytes
            image_bytes = await response.read()
            screenshot_b64 = base64.b64encode(image_bytes).decode("utf-8")

            duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                "Screenshot captured",
                size_kb=len(image_bytes) // 1024,
                duration_ms=duration_ms,
            )

            return screenshot_b64

    async def get_ui_hierarchy(self) -> dict[str, Any]:
        """
        Get the UI accessibility tree.

        Returns:
            Dictionary with 'xml' key containing raw XML and 'elements' with parsed elements.

        Raises:
            RuntimeError: If device not connected.
        """
        if not self.is_connected or not self._session_id:
            raise RuntimeError("Device not connected")

        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/devices/session/{self._session_id}/ui-hierarchy"
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"UI hierarchy failed: {error_text}")

            data = await response.json()

            # Parse XML if provided
            xml_content = data.get("xml", "")
            elements = data.get("elements", [])

            if xml_content and not elements:
                # Parse XML to extract elements
                elements = self._parse_ui_xml(xml_content)

            logger.debug(
                "UI hierarchy retrieved",
                element_count=len(elements),
            )

            return {"xml": xml_content, "elements": elements}

    def _parse_ui_xml(self, xml_content: str) -> list[dict[str, Any]]:
        """Parse accessibility tree XML into element list."""
        elements = []
        try:
            root = ElementTree.fromstring(xml_content)
            self._extract_elements_recursive(root, elements, 0)
        except ElementTree.ParseError as e:
            logger.warning("Failed to parse UI XML", error=str(e))
        return elements

    def _extract_elements_recursive(
        self,
        node: ElementTree.Element,
        elements: list[dict[str, Any]],
        index: int,
    ) -> int:
        """Recursively extract elements from XML tree."""
        # Extract element properties
        bounds_str = node.attrib.get("bounds", "[0,0][0,0]")
        bounds = self._parse_bounds(bounds_str)

        element = {
            "index": index,
            "class": node.attrib.get("class", ""),
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

        # Only include meaningful elements
        has_content = element["text"] or element["content_desc"]
        is_interactive = (
            element["clickable"]
            or element["focusable"]
            or element["scrollable"]
            or element["checkable"]
        )

        if has_content or is_interactive:
            elements.append(element)
            index += 1

        # Process children
        for child in node:
            index = self._extract_elements_recursive(child, elements, index)

        return index

    def _parse_bounds(self, bounds_str: str) -> dict[str, int]:
        """Parse bounds string like '[0,0][100,200]' into coordinates."""
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
        """Perform a tap at coordinates."""
        return await self._perform_action("tap", {"x": x, "y": y})

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> ActionResult:
        """Perform a long press at coordinates."""
        return await self._perform_action(
            "long_press", {"x": x, "y": y, "duration_ms": duration_ms}
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
        return await self._perform_action(
            "swipe",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "duration_ms": duration_ms,
            },
        )

    async def type_text(self, text: str) -> ActionResult:
        """Type text into focused element."""
        return await self._perform_action("type", {"text": text})

    async def press_key(self, key_code: str) -> ActionResult:
        """Press a system key."""
        return await self._perform_action("key", {"key_code": key_code})

    async def launch_app(self, package_name: str) -> ActionResult:
        """Launch an application by package name."""
        return await self._perform_action("launch", {"package": package_name})

    async def get_current_app(self) -> str:
        """Get the currently focused app's package name."""
        if not self.is_connected or not self._session_id:
            return ""

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.api_url}/devices/session/{self._session_id}/current-app"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("package", "")
        except Exception as e:
            logger.warning("Failed to get current app", error=str(e))

        return ""

    async def _perform_action(
        self, action_type: str, params: dict[str, Any]
    ) -> ActionResult:
        """Perform a device action via API."""
        if not self.is_connected or not self._session_id:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.api_url}/devices/session/{self._session_id}/action",
                json={"type": action_type, **params},
            ) as response:
                duration_ms = int((time.time() - start_time) * 1000)

                if response.status == 200:
                    logger.debug(
                        "Action performed",
                        action=action_type,
                        duration_ms=duration_ms,
                    )
                    return ActionResult(success=True, duration_ms=duration_ms)
                else:
                    error_text = await response.text()
                    logger.warning(
                        "Action failed",
                        action=action_type,
                        status=response.status,
                        error=error_text,
                    )
                    return ActionResult(
                        success=False,
                        error=error_text,
                        duration_ms=duration_ms,
                    )

        except Exception as e:
            logger.error("Action error", action=action_type, error=str(e))
            return ActionResult(success=False, error=str(e))

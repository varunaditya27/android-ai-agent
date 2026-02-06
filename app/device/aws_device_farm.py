"""
AWS Device Farm Cloud Device Client
====================================

Implementation of CloudDevice for AWS Device Farm remote access sessions.

AWS Device Farm provides cloud-hosted real Android devices in us-west-2.
This client handles:
- Remote access session lifecycle (create → poll → connect → stop)
- Device listing and selection
- Screenshot capture via Appium WebDriver
- UI hierarchy retrieval via Appium page source
- Touch, gesture, and text input via W3C Actions

Prerequisites:
    - AWS account with Device Farm access
    - IAM credentials with devicefarm:* permissions
    - A Device Farm project created in us-west-2
    - boto3 installed

Usage:
    from app.device.aws_device_farm import AWSDeviceFarmDevice

    device = AWSDeviceFarmDevice(
        project_arn="arn:aws:devicefarm:us-west-2:ACCOUNT:project:PROJECT_ID",
    )
    await device.connect()
    screenshot = await device.capture_screenshot()
    await device.tap(500, 300)
    await device.disconnect()
"""

import asyncio
import base64
import re
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

# AWS Device Farm is only available in us-west-2
AWS_DEVICE_FARM_REGION = "us-west-2"

# Session polling
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 60  # 5 min max wait for session to start


class AWSDeviceFarmDevice(CloudDevice):
    """
    AWS Device Farm remote access session implementation.

    Creates a remote access session on a real Android device in AWS,
    then controls it via the Appium/WebDriver endpoint provided by
    the session.

    Lifecycle:
        1. create_remote_access_session() → session ARN
        2. Poll get_remote_access_session() until status == RUNNING
        3. Extract endpoints.remoteDriverEndpoint (Appium URL)
        4. Use Appium REST API for device control
        5. stop_remote_access_session() on disconnect
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        project_arn: Optional[str] = None,
        device_arn: Optional[str] = None,
    ) -> None:
        """
        Initialize AWS Device Farm device client.

        Args:
            device_id: Not used directly; kept for CloudDevice interface.
            project_arn: ARN of the Device Farm project.
            device_arn: ARN of a specific device to use.
                        If None, the first available Android device is selected.
        """
        super().__init__(device_id)

        settings = get_settings()
        self.project_arn = project_arn or settings.device.aws_device_farm_project_arn
        self.device_arn = device_arn or settings.device.aws_device_farm_device_arn

        # Populated after session creation
        self._session_arn: Optional[str] = None
        self._appium_url: Optional[str] = None
        self._interactive_url: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # boto3 client (created lazily in connect)
        self._df_client: Any = None

        if not self.project_arn:
            logger.warning(
                "AWS Device Farm project ARN not configured. "
                "Set AWS_DEVICE_FARM_PROJECT_ARN in .env"
            )

    # ── boto3 helpers ─────────────────────────────────────────────

    def _get_boto_client(self) -> Any:
        """Get or create boto3 Device Farm client."""
        if self._df_client is None:
            import boto3

            settings = get_settings()
            kwargs: dict[str, Any] = {"region_name": AWS_DEVICE_FARM_REGION}

            # Allow explicit credentials from env (optional – falls back to
            # default credential chain: env vars, ~/.aws/credentials, IAM role)
            aws_key = settings.device.aws_access_key_id
            aws_secret = settings.device.aws_secret_access_key

            if aws_key and aws_secret:
                kwargs["aws_access_key_id"] = aws_key
                kwargs["aws_secret_access_key"] = aws_secret

            self._df_client = boto3.client("devicefarm", **kwargs)
            logger.debug("Created boto3 devicefarm client", region=AWS_DEVICE_FARM_REGION)

        return self._df_client

    async def _boto_call(self, method: str, **kwargs: Any) -> Any:
        """Run a boto3 call in a thread so we don't block the event loop."""
        client = self._get_boto_client()
        fn = getattr(client, method)
        return await asyncio.to_thread(fn, **kwargs)

    # ── HTTP session for Appium REST API ──────────────────────────

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for Appium calls."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._http_session

    # ── Device listing / selection ────────────────────────────────

    async def _pick_device_arn(self) -> str:
        """
        Pick a device ARN.

        If self.device_arn is set, validate and return it.
        Otherwise list available Android devices and pick the first one.
        """
        if self.device_arn:
            logger.info("Using specified device ARN", device_arn=self.device_arn)
            return self.device_arn

        logger.info("No device ARN specified – listing available Android devices…")
        response = await self._boto_call(
            "list_devices",
            filters=[
                {"attribute": "PLATFORM", "operator": "EQUALS", "values": ["ANDROID"]},
                {"attribute": "AVAILABILITY", "operator": "EQUALS", "values": ["HIGHLY_AVAILABLE"]},
            ],
        )

        devices = response.get("devices", [])
        if not devices:
            # Retry without availability filter
            response = await self._boto_call(
                "list_devices",
                filters=[
                    {"attribute": "PLATFORM", "operator": "EQUALS", "values": ["ANDROID"]},
                ],
            )
            devices = response.get("devices", [])

        if not devices:
            raise RuntimeError(
                "No Android devices available in AWS Device Farm. "
                "Check your Device Farm project and device slots."
            )

        # Prefer devices with remoteAccessEnabled
        remote_devices = [d for d in devices if d.get("remoteAccessEnabled")]
        chosen = remote_devices[0] if remote_devices else devices[0]

        self.device_arn = chosen["arn"]
        logger.info(
            "Selected device",
            name=chosen.get("name"),
            model=chosen.get("model"),
            os=chosen.get("os"),
            arn=self.device_arn,
        )
        return self.device_arn

    # ── CloudDevice interface ─────────────────────────────────────

    async def connect(self) -> bool:
        """
        Create a remote access session and wait until it's running.

        Returns:
            True if session started and Appium endpoint is available.
        """
        self.state = DeviceState.CONNECTING
        logger.info("Creating AWS Device Farm remote access session…")

        try:
            device_arn = await self._pick_device_arn()

            # Create remote access session
            create_resp = await self._boto_call(
                "create_remote_access_session",
                projectArn=self.project_arn,
                deviceArn=device_arn,
                name="android-ai-agent-session",
                configuration={
                    "billingMethod": "METERED",
                },
                interactionMode="NO_VIDEO",
            )

            session = create_resp.get("remoteAccessSession", {})
            self._session_arn = session.get("arn")

            if not self._session_arn:
                logger.error("No session ARN in create response", response=create_resp)
                self.state = DeviceState.ERROR
                return False

            logger.info(
                "Remote access session created, waiting for RUNNING status…",
                session_arn=self._session_arn,
            )

            # Poll until RUNNING
            for attempt in range(MAX_POLL_ATTEMPTS):
                get_resp = await self._boto_call(
                    "get_remote_access_session",
                    arn=self._session_arn,
                )
                session = get_resp.get("remoteAccessSession", {})
                status = session.get("status", "UNKNOWN")

                logger.debug(
                    "Session poll",
                    attempt=attempt + 1,
                    status=status,
                )

                if status == "RUNNING":
                    break
                elif status in ("COMPLETED", "ERRORED", "STOPPED"):
                    msg = session.get("message", "")
                    logger.error(
                        "Session failed to start",
                        status=status,
                        message=msg,
                        result=session.get("result"),
                    )
                    self.state = DeviceState.ERROR
                    return False

                await asyncio.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.error("Timed out waiting for session to start")
                self.state = DeviceState.ERROR
                # Best-effort cleanup
                try:
                    await self._boto_call(
                        "stop_remote_access_session", arn=self._session_arn
                    )
                except Exception:
                    pass
                return False

            # Extract endpoints
            endpoint = session.get("endpoint", "")
            self._appium_url = endpoint.rstrip("/") if endpoint else None
            self._interactive_url = session.get("deviceUdid", "")

            if not self._appium_url:
                logger.error("No Appium endpoint returned by session")
                self.state = DeviceState.ERROR
                return False

            # Build DeviceInfo from session's device metadata
            device_info = session.get("device", {})
            resolution = device_info.get("resolution", {})

            self.info = DeviceInfo(
                device_id=self._session_arn,
                platform="android",
                os_version=device_info.get("os", ""),
                screen_width=resolution.get("width", 1080),
                screen_height=resolution.get("height", 2340),
                model=device_info.get("model", device_info.get("name", "")),
                manufacturer=device_info.get("manufacturer", ""),
            )

            self.state = DeviceState.CONNECTED
            logger.info(
                "Connected to AWS Device Farm device",
                session_arn=self._session_arn,
                model=self.info.model,
                os_version=self.info.os_version,
                screen=f"{self.info.screen_width}x{self.info.screen_height}",
                appium_url=self._appium_url,
            )
            return True

        except Exception as e:
            logger.error("Failed to connect to AWS Device Farm", error=str(e))
            self.state = DeviceState.ERROR
            return False

    async def disconnect(self) -> None:
        """Stop the remote access session and clean up."""
        if self._session_arn:
            try:
                await self._boto_call(
                    "stop_remote_access_session", arn=self._session_arn
                )
                logger.info(
                    "Remote access session stopped",
                    session_arn=self._session_arn,
                )
            except Exception as e:
                logger.warning("Error stopping session", error=str(e))

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        self._session_arn = None
        self._appium_url = None
        self._http_session = None
        self.state = DeviceState.DISCONNECTED
        logger.info("Disconnected from AWS Device Farm")

    # ── Screenshot ────────────────────────────────────────────────

    async def capture_screenshot(self) -> str:
        """
        Capture current screen as base64-encoded PNG via Appium.

        Returns:
            Base64-encoded screenshot string.
        """
        if not self.is_connected or not self._appium_url:
            raise RuntimeError("Device not connected")

        session = await self._get_http_session()
        start_time = time.time()

        async with session.get(f"{self._appium_url}/screenshot") as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Screenshot failed ({response.status}): {error_text}")

            data = await response.json()
            screenshot_b64 = data.get("value", "")

            logger.debug(
                "Screenshot captured",
                duration_ms=int((time.time() - start_time) * 1000),
            )
            return screenshot_b64

    # ── UI hierarchy ──────────────────────────────────────────────

    async def get_ui_hierarchy(self) -> dict[str, Any]:
        """
        Get UI hierarchy via Appium page source (UiAutomator2 XML).

        Returns:
            Dictionary with 'xml' and 'elements' keys.
        """
        if not self.is_connected or not self._appium_url:
            raise RuntimeError("Device not connected")

        session = await self._get_http_session()

        async with session.get(f"{self._appium_url}/source") as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"UI hierarchy failed ({response.status}): {error_text}")

            data = await response.json()
            xml_content = data.get("value", "")

            elements = self._parse_ui_xml(xml_content)
            logger.debug("UI hierarchy retrieved", element_count=len(elements))

            return {"xml": xml_content, "elements": elements}

    def _parse_ui_xml(self, xml_content: str) -> list[dict[str, Any]]:
        """Parse Appium page source XML into element list."""
        from xml.etree import ElementTree

        elements: list[dict[str, Any]] = []
        try:
            root = ElementTree.fromstring(xml_content)
            self._extract_elements(root, elements, 0)
        except ElementTree.ParseError as e:
            logger.warning("Failed to parse UI XML", error=str(e))
        return elements

    def _extract_elements(
        self,
        node: Any,
        elements: list[dict[str, Any]],
        index: int,
    ) -> int:
        """Recursively extract meaningful elements from XML tree."""
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
            index = self._extract_elements(child, elements, index)

        return index

    @staticmethod
    def _parse_bounds(bounds_str: str) -> dict[str, int]:
        """Parse bounds string '[x1,y1][x2,y2]' → coordinate dict."""
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return {
                "left": x1, "top": y1, "right": x2, "bottom": y2,
                "center_x": (x1 + x2) // 2, "center_y": (y1 + y2) // 2,
                "width": x2 - x1, "height": y2 - y1,
            }
        return {
            "left": 0, "top": 0, "right": 0, "bottom": 0,
            "center_x": 0, "center_y": 0, "width": 0, "height": 0,
        }

    # ── Touch actions ─────────────────────────────────────────────

    async def tap(self, x: int, y: int) -> ActionResult:
        """Perform a tap at (x, y) using W3C touch actions."""
        return await self._perform_touch_action([
            {"type": "pointerMove", "duration": 0, "x": x, "y": y},
            {"type": "pointerDown", "button": 0},
            {"type": "pause", "duration": 100},
            {"type": "pointerUp", "button": 0},
        ])

    async def long_press(
        self, x: int, y: int, duration_ms: int = 1000
    ) -> ActionResult:
        """Perform a long press at (x, y)."""
        return await self._perform_touch_action([
            {"type": "pointerMove", "duration": 0, "x": x, "y": y},
            {"type": "pointerDown", "button": 0},
            {"type": "pause", "duration": duration_ms},
            {"type": "pointerUp", "button": 0},
        ])

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> ActionResult:
        """Perform a swipe gesture."""
        return await self._perform_touch_action([
            {"type": "pointerMove", "duration": 0, "x": start_x, "y": start_y},
            {"type": "pointerDown", "button": 0},
            {"type": "pointerMove", "duration": duration_ms, "x": end_x, "y": end_y},
            {"type": "pointerUp", "button": 0},
        ])

    async def type_text(self, text: str) -> ActionResult:
        """Type text into the currently focused element."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        try:
            session = await self._get_http_session()

            # Get the active element
            element_id = None
            async with session.get(f"{self._appium_url}/element/active") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    element_id = data.get("value", {}).get("ELEMENT")

            if element_id:
                async with session.post(
                    f"{self._appium_url}/element/{element_id}/value",
                    json={"text": text},
                ) as resp:
                    if resp.status == 200:
                        return ActionResult(success=True)
                    error_text = await resp.text()
                    return ActionResult(success=False, error=error_text)
            else:
                # Fallback: send keys directly
                async with session.post(
                    f"{self._appium_url}/keys",
                    json={"value": list(text)},
                ) as resp:
                    if resp.status == 200:
                        return ActionResult(success=True)
                    error_text = await resp.text()
                    return ActionResult(success=False, error=error_text)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def press_key(self, key_code: str) -> ActionResult:
        """Press a system key (back, home, enter, etc.)."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        key_map = {
            "KEYCODE_BACK": 4, "KEYCODE_HOME": 3, "KEYCODE_MENU": 82,
            "KEYCODE_ENTER": 66, "KEYCODE_SEARCH": 84, "KEYCODE_DEL": 67,
            "KEYCODE_TAB": 61, "KEYCODE_SPACE": 62,
            "KEYCODE_POWER": 26, "KEYCODE_APP_SWITCH": 187,
            "KEYCODE_VOLUME_UP": 24, "KEYCODE_VOLUME_DOWN": 25,
            "back": 4, "home": 3, "menu": 82, "enter": 66,
        }

        # Normalize
        android_key = key_map.get(key_code) or key_map.get(key_code.upper())
        if android_key is None:
            return ActionResult(success=False, error=f"Unknown key code: {key_code}")

        try:
            session = await self._get_http_session()
            async with session.post(
                f"{self._appium_url}/appium/device/press_keycode",
                json={"keycode": android_key},
            ) as resp:
                if resp.status == 200:
                    return ActionResult(success=True)
                error_text = await resp.text()
                return ActionResult(success=False, error=error_text)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def launch_app(self, package_name: str) -> ActionResult:
        """Launch an app by package name."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        try:
            session = await self._get_http_session()
            async with session.post(
                f"{self._appium_url}/appium/device/activate_app",
                json={"bundleId": package_name},
            ) as resp:
                if resp.status == 200:
                    return ActionResult(success=True)
                error_text = await resp.text()
                return ActionResult(success=False, error=error_text)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def get_current_app(self) -> str:
        """Get the currently focused app's package name."""
        if not self.is_connected or not self._appium_url:
            return ""

        try:
            session = await self._get_http_session()
            async with session.get(
                f"{self._appium_url}/appium/device/current_package"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("value", "")
        except Exception as e:
            logger.warning("Failed to get current app", error=str(e))
        return ""

    # ── Helpers ───────────────────────────────────────────────────

    async def _perform_touch_action(self, actions: list[dict]) -> ActionResult:
        """Execute W3C touch actions via Appium endpoint."""
        if not self.is_connected or not self._appium_url:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            session = await self._get_http_session()

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
            ) as resp:
                duration_ms = int((time.time() - start_time) * 1000)
                if resp.status == 200:
                    return ActionResult(success=True, duration_ms=duration_ms)
                error_text = await resp.text()
                return ActionResult(success=False, error=error_text, duration_ms=duration_ms)

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    # ── Utility: list devices (for setup / debugging) ─────────────

    async def list_available_devices(self) -> list[dict[str, Any]]:
        """
        List available Android devices in Device Farm.

        Returns a list of dicts with keys: arn, name, model, os,
        manufacturer, resolution, availability, remoteAccessEnabled.

        Useful for the setup script or choosing a device_arn.
        """
        response = await self._boto_call(
            "list_devices",
            filters=[
                {"attribute": "PLATFORM", "operator": "EQUALS", "values": ["ANDROID"]},
            ],
        )

        devices = []
        for d in response.get("devices", []):
            resolution = d.get("resolution", {})
            devices.append({
                "arn": d.get("arn", ""),
                "name": d.get("name", ""),
                "model": d.get("model", ""),
                "os": d.get("os", ""),
                "manufacturer": d.get("manufacturer", ""),
                "resolution": f"{resolution.get('width', 0)}x{resolution.get('height', 0)}",
                "availability": d.get("availability", ""),
                "remoteAccessEnabled": d.get("remoteAccessEnabled", False),
            })

        return devices

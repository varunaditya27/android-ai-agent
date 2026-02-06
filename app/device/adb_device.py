"""
ADB Local Device Client
=======================

Implementation of device control for local Android emulator via ADB.
This is a FREE alternative to paid cloud device services.

ADB (Android Debug Bridge) is included with Android SDK and allows:
- Screenshot capture
- UI hierarchy (accessibility tree) retrieval via uiautomator
- Touch and gesture input
- App launching and key presses

Prerequisites:
    1. Android SDK installed with platform-tools (adb)
    2. Android Emulator running OR physical device connected via USB
    3. ADB available in PATH or ANDROID_HOME set

Usage:
    from app.device import ADBDevice

    device = ADBDevice()  # Uses default emulator
    await device.connect()
    screenshot = await device.capture_screenshot()
    await device.tap(500, 300)
"""

import asyncio
import base64
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from app.device.cloud_provider import (
    ActionResult,
    CloudDevice,
    DeviceInfo,
    DeviceState,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ADBDevice(CloudDevice):
    """
    Local Android device control via ADB.

    Provides full device control using Android Debug Bridge:
    - Works with Android Emulator (free)
    - Works with physical devices via USB
    - No API keys or paid services required

    This implementation uses subprocess to call ADB commands,
    making it lightweight and dependency-free.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        adb_path: Optional[str] = None,
    ) -> None:
        """
        Initialize ADB device client.

        Args:
            device_id: Optional device serial (from 'adb devices').
                       If None, uses the first available device.
            adb_path: Optional path to adb executable.
                      If None, searches PATH and ANDROID_HOME.
        """
        super().__init__(device_id)

        self.adb_path = adb_path or self._find_adb()
        self._device_serial: Optional[str] = device_id

        if not self.adb_path:
            logger.warning(
                "ADB not found. Please install Android SDK platform-tools "
                "and ensure 'adb' is in PATH or set ANDROID_HOME."
            )

    def _find_adb(self) -> Optional[str]:
        """Find ADB executable in system."""
        # Try PATH first
        adb_in_path = shutil.which("adb")
        if adb_in_path:
            return adb_in_path

        # Try ANDROID_HOME
        import os
        android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
        if android_home:
            adb_candidates = [
                Path(android_home) / "platform-tools" / "adb",
                Path(android_home) / "platform-tools" / "adb.exe",
            ]
            for candidate in adb_candidates:
                if candidate.exists():
                    return str(candidate)

        # Common installation paths
        common_paths = [
            Path.home() / "Android" / "Sdk" / "platform-tools" / "adb",
            Path("/usr/local/android-sdk/platform-tools/adb"),
            Path("/opt/android-sdk/platform-tools/adb"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    async def _run_adb(
        self,
        *args: str,
        timeout: float = 30.0,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run an ADB command asynchronously.

        Args:
            *args: ADB command arguments.
            timeout: Command timeout in seconds.
            capture_output: Whether to capture stdout/stderr.

        Returns:
            CompletedProcess with command result.

        Raises:
            RuntimeError: If ADB is not available.
        """
        if not self.adb_path:
            raise RuntimeError("ADB not found. Please install Android SDK platform-tools.")

        cmd = [self.adb_path]

        # Add device serial if specified
        if self._device_serial:
            cmd.extend(["-s", self._device_serial])

        cmd.extend(args)

        logger.debug("Running ADB command", cmd=" ".join(cmd))

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=capture_output,
                timeout=timeout,
                text=True,
            )
            return result
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ADB command timed out after {timeout}s") from e

    async def _run_adb_bytes(
        self,
        *args: str,
        timeout: float = 30.0,
    ) -> bytes:
        """Run ADB command and return raw bytes (for screenshots)."""
        if not self.adb_path:
            raise RuntimeError("ADB not found")

        cmd = [self.adb_path]
        if self._device_serial:
            cmd.extend(["-s", self._device_serial])
        cmd.extend(args)

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ADB command timed out") from e

    async def connect(self) -> bool:
        """
        Connect to an Android device/emulator via ADB.

        Returns:
            True if connection successful.
        """
        self.state = DeviceState.CONNECTING
        logger.info("Connecting to ADB device", device_id=self._device_serial)

        try:
            # Get list of connected devices
            result = await self._run_adb("devices", "-l")

            if result.returncode != 0:
                logger.error("Failed to get device list", error=result.stderr)
                self.state = DeviceState.ERROR
                return False

            # Parse device list
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            devices = []

            for line in lines:
                if line.strip() and "device" in line:
                    parts = line.split()
                    serial = parts[0]
                    # Extract model from device info
                    model = ""
                    for part in parts:
                        if part.startswith("model:"):
                            model = part.split(":")[1]
                    devices.append({"serial": serial, "model": model})

            if not devices:
                logger.error("No Android devices found. Start an emulator or connect a device.")
                self.state = DeviceState.ERROR
                return False

            # Use specified device or first available
            if self._device_serial:
                matching = [d for d in devices if d["serial"] == self._device_serial]
                if not matching:
                    logger.error(
                        "Specified device not found",
                        device_id=self._device_serial,
                        available=[d["serial"] for d in devices],
                    )
                    self.state = DeviceState.ERROR
                    return False
                device = matching[0]
            else:
                device = devices[0]
                self._device_serial = device["serial"]

            # Get device properties
            screen_size = await self._get_screen_size()
            android_version = await self._get_android_version()

            self.info = DeviceInfo(
                device_id=self._device_serial,
                platform="android",
                os_version=android_version,
                screen_width=screen_size[0],
                screen_height=screen_size[1],
                model=device.get("model", "Unknown"),
                manufacturer=await self._get_device_property("ro.product.manufacturer"),
            )

            self.state = DeviceState.CONNECTED
            logger.info(
                "Connected to ADB device",
                serial=self._device_serial,
                model=self.info.model,
                android_version=android_version,
                screen_size=f"{self.info.screen_width}x{self.info.screen_height}",
            )
            return True

        except Exception as e:
            logger.error("Failed to connect to ADB device", error=str(e))
            self.state = DeviceState.ERROR
            return False

    async def _get_screen_size(self) -> tuple[int, int]:
        """Get device screen dimensions."""
        result = await self._run_adb("shell", "wm", "size")
        if result.returncode == 0:
            # Parse "Physical size: 1080x2340"
            match = re.search(r"(\d+)x(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        return 1080, 2340  # Default fallback

    async def _get_android_version(self) -> str:
        """Get Android OS version."""
        return await self._get_device_property("ro.build.version.release")

    async def _get_device_property(self, prop: str) -> str:
        """Get a device property via getprop."""
        result = await self._run_adb("shell", "getprop", prop)
        if result.returncode == 0:
            return result.stdout.strip()
        return ""

    async def disconnect(self) -> None:
        """Disconnect from the device (cleanup)."""
        self._device_serial = None
        self.state = DeviceState.DISCONNECTED
        logger.info("Disconnected from ADB device")

    async def capture_screenshot(self) -> str:
        """
        Capture current screen as base64-encoded PNG.

        Uses 'adb exec-out screencap -p' for fast screenshot capture.

        Returns:
            Base64-encoded screenshot string.

        Raises:
            RuntimeError: If device not connected or screenshot fails.
        """
        if not self.is_connected:
            raise RuntimeError("Device not connected")

        start_time = time.time()

        try:
            # Use exec-out for direct binary output (faster)
            screenshot_bytes = await self._run_adb_bytes(
                "exec-out", "screencap", "-p"
            )

            if not screenshot_bytes or len(screenshot_bytes) < 1000:
                raise RuntimeError("Screenshot appears empty or corrupted")

            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            logger.debug(
                "Screenshot captured",
                size_kb=len(screenshot_bytes) / 1024,
                duration_ms=int((time.time() - start_time) * 1000),
            )

            return screenshot_b64

        except Exception as e:
            logger.error("Screenshot failed", error=str(e))
            raise RuntimeError(f"Screenshot capture failed: {e}") from e

    async def get_ui_hierarchy(self) -> dict[str, Any]:
        """
        Get the UI accessibility tree/hierarchy.

        Uses 'uiautomator dump' to get XML representation of the UI,
        then parses it into a dictionary structure.

        Returns:
            Dictionary representing the UI tree structure.
        """
        if not self.is_connected:
            raise RuntimeError("Device not connected")

        try:
            # Dump UI hierarchy to device
            await self._run_adb(
                "shell", "uiautomator", "dump", "/sdcard/ui_dump.xml"
            )

            # Pull the XML content
            result = await self._run_adb(
                "shell", "cat", "/sdcard/ui_dump.xml"
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to read UI dump: {result.stderr}")

            xml_content = result.stdout

            # Clean up temp file on device
            await self._run_adb("shell", "rm", "/sdcard/ui_dump.xml")

            # Parse XML to dict
            return self._parse_ui_xml(xml_content)

        except Exception as e:
            logger.error("Failed to get UI hierarchy", error=str(e))
            # Return empty hierarchy on failure
            return {"elements": []}

    def _parse_ui_xml(self, xml_content: str) -> dict[str, Any]:
        """Parse uiautomator XML dump into dictionary."""
        try:
            root = ET.fromstring(xml_content)
            return {"elements": self._parse_node(root)}
        except ET.ParseError as e:
            logger.error("Failed to parse UI XML", error=str(e))
            return {"elements": []}

    def _parse_node(self, element: ET.Element) -> list[dict[str, Any]]:
        """Recursively parse XML nodes."""
        nodes = []

        for node in element.iter("node"):
            # Extract bounds "[x1,y1][x2,y2]"
            bounds_str = node.get("bounds", "[0,0][0,0]")
            bounds_match = re.findall(r"\[(\d+),(\d+)\]", bounds_str)
            left = int(bounds_match[0][0]) if bounds_match else 0
            top = int(bounds_match[0][1]) if bounds_match else 0
            right = int(bounds_match[1][0]) if len(bounds_match) > 1 else 0
            bottom = int(bounds_match[1][1]) if len(bounds_match) > 1 else 0
            bounds = {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "center_x": (left + right) // 2,
                "center_y": (top + bottom) // 2,
                "width": right - left,
                "height": bottom - top,
            }

            node_data = {
                "class": node.get("class", ""),
                "text": node.get("text", ""),
                "content_desc": node.get("content-desc", ""),
                "resource_id": node.get("resource-id", ""),
                "package": node.get("package", ""),
                "clickable": node.get("clickable", "false") == "true",
                "focusable": node.get("focusable", "false") == "true",
                "enabled": node.get("enabled", "true") == "true",
                "scrollable": node.get("scrollable", "false") == "true",
                "long_clickable": node.get("long-clickable", "false") == "true",
                "checkable": node.get("checkable", "false") == "true",
                "checked": node.get("checked", "false") == "true",
                "focused": node.get("focused", "false") == "true",
                "bounds": bounds,
            }

            nodes.append(node_data)

        return nodes

    async def tap(self, x: int, y: int) -> ActionResult:
        """
        Perform a tap at coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            result = await self._run_adb("shell", "input", "tap", str(x), str(y))

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Tap failed: {result.stderr}",
                    duration_ms=int((time.time() - start_time) * 1000),
                )

            logger.debug("Tap performed", x=x, y=y)
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def long_press(
        self, x: int, y: int, duration_ms: int = 1000
    ) -> ActionResult:
        """
        Perform a long press at coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            duration_ms: Duration of press in milliseconds.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            # Long press via swipe with same start/end coordinates
            result = await self._run_adb(
                "shell", "input", "swipe",
                str(x), str(y), str(x), str(y), str(duration_ms)
            )

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Long press failed: {result.stderr}",
                )

            logger.debug("Long press performed", x=x, y=y, duration_ms=duration_ms)
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> ActionResult:
        """
        Perform a swipe gesture.

        Args:
            start_x: Starting X coordinate.
            start_y: Starting Y coordinate.
            end_x: Ending X coordinate.
            end_y: Ending Y coordinate.
            duration_ms: Duration of swipe in milliseconds.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            result = await self._run_adb(
                "shell", "input", "swipe",
                str(start_x), str(start_y),
                str(end_x), str(end_y),
                str(duration_ms)
            )

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Swipe failed: {result.stderr}",
                )

            logger.debug(
                "Swipe performed",
                start=(start_x, start_y),
                end=(end_x, end_y),
            )
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def type_text(self, text: str) -> ActionResult:
        """
        Type text into the currently focused element.

        Args:
            text: Text to type.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            # Escape special characters for shell
            # Replace spaces with %s (ADB input text format)
            escaped_text = text.replace(" ", "%s")
            escaped_text = escaped_text.replace("'", "\\'")
            escaped_text = escaped_text.replace('"', '\\"')
            escaped_text = escaped_text.replace("&", "\\&")
            escaped_text = escaped_text.replace("<", "\\<")
            escaped_text = escaped_text.replace(">", "\\>")
            escaped_text = escaped_text.replace("|", "\\|")

            result = await self._run_adb("shell", "input", "text", escaped_text)

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Type failed: {result.stderr}",
                )

            logger.debug("Text typed", length=len(text))
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def press_key(self, key_code: str) -> ActionResult:
        """
        Press a system key (back, home, etc.).

        Args:
            key_code: Android key code name (e.g., KEYCODE_BACK, KEYCODE_HOME).

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        # Map common key names to key codes
        key_map = {
            "back": "KEYCODE_BACK",
            "home": "KEYCODE_HOME",
            "menu": "KEYCODE_MENU",
            "enter": "KEYCODE_ENTER",
            "delete": "KEYCODE_DEL",
            "power": "KEYCODE_POWER",
            "volume_up": "KEYCODE_VOLUME_UP",
            "volume_down": "KEYCODE_VOLUME_DOWN",
            "tab": "KEYCODE_TAB",
            "space": "KEYCODE_SPACE",
            "recent": "KEYCODE_APP_SWITCH",
        }

        # Normalize key code
        normalized_key = key_code.lower().replace("keycode_", "")
        actual_key = key_map.get(normalized_key, key_code)

        # Ensure KEYCODE_ prefix
        if not actual_key.startswith("KEYCODE_"):
            actual_key = f"KEYCODE_{actual_key.upper()}"

        try:
            result = await self._run_adb("shell", "input", "keyevent", actual_key)

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Key press failed: {result.stderr}",
                )

            logger.debug("Key pressed", key=actual_key)
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def launch_app(self, package_name: str) -> ActionResult:
        """
        Launch an application by package name.

        Args:
            package_name: Android package name (e.g., com.google.android.youtube).

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            # Use monkey to launch app (works without knowing activity name)
            result = await self._run_adb(
                "shell", "monkey", "-p", package_name,
                "-c", "android.intent.category.LAUNCHER", "1"
            )

            if result.returncode != 0 or "No activities found" in result.stdout:
                # Fallback: try am start with package
                result = await self._run_adb(
                    "shell", "am", "start",
                    "-n", f"{package_name}/.MainActivity"
                )

            # Wait for app to start
            await asyncio.sleep(0.5)

            logger.debug("App launched", package=package_name)
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def get_current_app(self) -> str:
        """
        Get the currently focused app's package name.

        Returns:
            Package name of the foreground app.
        """
        if not self.is_connected:
            return ""

        try:
            result = await self._run_adb(
                "shell", "dumpsys", "window", "displays",
                "|", "grep", "-E", "'mCurrentFocus|mFocusedApp'"
            )

            # Try alternative method
            result = await self._run_adb(
                "shell", "dumpsys", "activity", "activities",
            )

            # Parse for current app
            for line in result.stdout.split("\n"):
                if "mResumedActivity" in line or "topResumedActivity" in line:
                    match = re.search(r"([a-zA-Z0-9_.]+)/", line)
                    if match:
                        return match.group(1)

            return ""

        except Exception as e:
            logger.error("Failed to get current app", error=str(e))
            return ""

    async def install_app(self, apk_path: str) -> ActionResult:
        """
        Install an APK on the device.

        Args:
            apk_path: Path to the APK file.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        start_time = time.time()

        try:
            result = await self._run_adb("install", "-r", apk_path, timeout=120)

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Install failed: {result.stderr}",
                )

            logger.info("APK installed", path=apk_path)
            return ActionResult(
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def clear_app_data(self, package_name: str) -> ActionResult:
        """
        Clear app data and cache.

        Args:
            package_name: Package name of the app.

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.is_connected:
            return ActionResult(success=False, error="Device not connected")

        try:
            result = await self._run_adb("shell", "pm", "clear", package_name)

            if result.returncode != 0:
                return ActionResult(
                    success=False,
                    error=f"Clear data failed: {result.stderr}",
                )

            logger.debug("App data cleared", package=package_name)
            return ActionResult(success=True)

        except Exception as e:
            return ActionResult(success=False, error=str(e))


def get_available_emulators() -> list[str]:
    """
    Get list of available Android emulators.

    Returns:
        List of emulator AVD names.
    """
    emulator_path = shutil.which("emulator")
    if not emulator_path:
        return []

    try:
        result = subprocess.run(
            [emulator_path, "-list-avds"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [avd.strip() for avd in result.stdout.strip().split("\n") if avd.strip()]
    except Exception:
        pass

    return []


async def start_emulator(avd_name: str) -> bool:
    """
    Start an Android emulator.

    Args:
        avd_name: Name of the AVD to start.

    Returns:
        True if emulator started successfully.
    """
    emulator_path = shutil.which("emulator")
    if not emulator_path:
        logger.error("Emulator not found in PATH")
        return False

    try:
        # Start emulator in background
        subprocess.Popen(
            [emulator_path, "-avd", avd_name, "-no-snapshot-load"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for device to be ready
        logger.info("Waiting for emulator to boot...", avd=avd_name)

        for _ in range(60):  # Wait up to 60 seconds
            await asyncio.sleep(1)
            result = subprocess.run(
                ["adb", "shell", "getprop", "sys.boot_completed"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip() == "1":
                logger.info("Emulator booted successfully", avd=avd_name)
                return True

        logger.error("Emulator boot timeout")
        return False

    except Exception as e:
        logger.error("Failed to start emulator", error=str(e))
        return False
"""
Cloud Device Provider Abstraction
=================================

Abstract base class defining the interface for cloud device providers.
Supports operations like screenshot capture, UI hierarchy, and input actions.

Usage:
    from app.device import create_cloud_device

    device = await create_cloud_device("limrun", device_id="pixel-6")
    await device.connect()
    screenshot = await device.capture_screenshot()
    await device.tap(500, 300)
    await device.disconnect()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class DeviceState(Enum):
    """State of the cloud device connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DeviceInfo:
    """
    Information about a connected device.

    Attributes:
        device_id: Unique identifier for the device.
        platform: Operating system (android).
        os_version: Android version string.
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        model: Device model name.
        manufacturer: Device manufacturer.
    """

    device_id: str
    platform: str = "android"
    os_version: str = ""
    screen_width: int = 1080
    screen_height: int = 2340
    model: str = ""
    manufacturer: str = ""


@dataclass
class ActionResult:
    """
    Result of a device action.

    Attributes:
        success: Whether the action succeeded.
        error: Error message if failed.
        duration_ms: How long the action took.
    """

    success: bool
    error: Optional[str] = None
    duration_ms: int = 0


class CloudDevice(ABC):
    """
    Abstract base class for cloud device providers.

    All cloud device implementations (Limrun, BrowserStack, etc.)
    must implement this interface.
    """

    def __init__(self, device_id: Optional[str] = None) -> None:
        """
        Initialize cloud device.

        Args:
            device_id: Optional specific device to connect to.
        """
        self.device_id = device_id
        self.state = DeviceState.DISCONNECTED
        self.info: Optional[DeviceInfo] = None

    @property
    def is_connected(self) -> bool:
        """Check if device is currently connected."""
        return self.state == DeviceState.CONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the cloud device.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the device and release resources."""
        pass

    @abstractmethod
    async def capture_screenshot(self) -> str:
        """
        Capture the current screen.

        Returns:
            Base64-encoded PNG screenshot.
        """
        pass

    @abstractmethod
    async def get_ui_hierarchy(self) -> dict[str, Any]:
        """
        Get the UI accessibility tree/hierarchy.

        Returns:
            Dictionary representing the UI tree structure.
        """
        pass

    @abstractmethod
    async def tap(self, x: int, y: int) -> ActionResult:
        """
        Perform a tap at coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            ActionResult indicating success/failure.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def type_text(self, text: str) -> ActionResult:
        """
        Type text into the currently focused element.

        Args:
            text: Text to type.

        Returns:
            ActionResult indicating success/failure.
        """
        pass

    @abstractmethod
    async def press_key(self, key_code: str) -> ActionResult:
        """
        Press a system key (back, home, etc.).

        Args:
            key_code: Android key code name (KEYCODE_BACK, etc.).

        Returns:
            ActionResult indicating success/failure.
        """
        pass

    @abstractmethod
    async def launch_app(self, package_name: str) -> ActionResult:
        """
        Launch an application by package name.

        Args:
            package_name: Android package name (e.g., com.google.android.youtube).

        Returns:
            ActionResult indicating success/failure.
        """
        pass

    @abstractmethod
    async def get_current_app(self) -> str:
        """
        Get the currently focused app's package name.

        Returns:
            Package name of the foreground app.
        """
        pass

    async def swipe_direction(
        self, direction: str, distance_percent: float = 0.5
    ) -> ActionResult:
        """
        Swipe in a direction (up, down, left, right).

        Args:
            direction: Direction to swipe.
            distance_percent: Percentage of screen to swipe (0.0-1.0).

        Returns:
            ActionResult indicating success/failure.
        """
        if not self.info:
            return ActionResult(success=False, error="Device not connected")

        width = self.info.screen_width
        height = self.info.screen_height
        center_x = width // 2
        center_y = height // 2

        direction = direction.lower()
        distance_x = int(width * distance_percent / 2)
        distance_y = int(height * distance_percent / 2)

        if direction == "up":
            return await self.swipe(center_x, center_y + distance_y, center_x, center_y - distance_y)
        elif direction == "down":
            return await self.swipe(center_x, center_y - distance_y, center_x, center_y + distance_y)
        elif direction == "left":
            return await self.swipe(center_x + distance_x, center_y, center_x - distance_x, center_y)
        elif direction == "right":
            return await self.swipe(center_x - distance_x, center_y, center_x + distance_x, center_y)
        else:
            return ActionResult(success=False, error=f"Unknown direction: {direction}")


async def create_cloud_device(
    provider: str,
    device_id: Optional[str] = None,
    **kwargs: Any,
) -> CloudDevice:
    """
    Factory function to create device instances.

    Args:
        provider: Provider name ('adb', 'local', 'limrun', 'browserstack').
        device_id: Optional specific device ID.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured CloudDevice instance.

    Raises:
        ValueError: If provider is not supported.

    Note:
        'adb' and 'local' are FREE options using Android Emulator.
        'limrun' and 'browserstack' are paid cloud services.
    """
    from app.device.adb_device import ADBDevice
    from app.device.limrun_client import LimrunDevice
    from app.device.browserstack import BrowserStackDevice

    provider = provider.lower()

    if provider in ("adb", "local", "emulator"):
        # FREE option: Local emulator via ADB
        return ADBDevice(device_id=device_id, **kwargs)
    elif provider == "limrun":
        return LimrunDevice(device_id=device_id, **kwargs)
    elif provider == "browserstack":
        return BrowserStackDevice(device_id=device_id, **kwargs)
    else:
        raise ValueError(
            f"Unsupported device provider: {provider}. "
            f"Use 'adb' (free), 'limrun', or 'browserstack'."
        )


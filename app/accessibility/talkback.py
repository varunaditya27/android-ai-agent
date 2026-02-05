"""
TalkBack Integration Module
===========================

Provides integration with Android's TalkBack screen reader.

TalkBack is Android's built-in screen reader that provides
spoken feedback for blind and visually impaired users.
This module helps control TalkBack and work alongside it.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TalkBackGesture(Enum):
    """TalkBack gesture types."""

    # Single finger gestures
    SWIPE_RIGHT = "swipe_right"  # Next item
    SWIPE_LEFT = "swipe_left"  # Previous item
    SWIPE_UP = "swipe_up"  # Next reading control option
    SWIPE_DOWN = "swipe_down"  # Previous reading control option
    TAP = "tap"  # Activate
    DOUBLE_TAP = "double_tap"  # Activate focused item

    # Two finger gestures
    TWO_FINGER_TAP = "two_finger_tap"  # Pause/resume speech
    TWO_FINGER_SWIPE_UP = "two_finger_swipe_up"  # Read from top
    TWO_FINGER_SWIPE_DOWN = "two_finger_swipe_down"  # Read from current position

    # Three finger gestures
    THREE_FINGER_TAP = "three_finger_tap"  # Read from top
    THREE_FINGER_SWIPE_LEFT = "three_finger_swipe_left"  # Previous screen
    THREE_FINGER_SWIPE_RIGHT = "three_finger_swipe_right"  # Next screen

    # Special gestures
    EXPLORE_BY_TOUCH = "explore_by_touch"  # Touch to explore mode
    SCROLL_FORWARD = "scroll_forward"  # Scroll forward
    SCROLL_BACKWARD = "scroll_backward"  # Scroll backward


@dataclass
class TalkBackSettings:
    """
    TalkBack configuration settings.

    Attributes:
        enabled: Whether TalkBack is enabled.
        speech_rate: Speed of speech (percentage, 100 is normal).
        pitch: Pitch of speech (percentage, 100 is normal).
        volume: Volume level (0-100).
        verbosity: Level of detail (minimal, default, custom).
        speak_passwords: Whether to speak password characters.
        vibrate_on_keypress: Whether to vibrate on key presses.
        sound_feedback: Whether to play sounds for actions.
    """

    enabled: bool = True
    speech_rate: int = 100
    pitch: int = 100
    volume: int = 100
    verbosity: str = "default"
    speak_passwords: bool = False
    vibrate_on_keypress: bool = True
    sound_feedback: bool = True


class TalkBackController:
    """
    Controller for TalkBack integration.

    Provides methods to control TalkBack behavior and
    perform TalkBack-specific gestures.
    """

    def __init__(
        self,
        device: CloudDevice,
        settings: Optional[TalkBackSettings] = None,
    ) -> None:
        """
        Initialize TalkBack controller.

        Args:
            device: Cloud device to control.
            settings: TalkBack settings.
        """
        self.device = device
        self.settings = settings or TalkBackSettings()
        self._is_enabled: Optional[bool] = None

        logger.info("TalkBackController initialized")

    async def is_enabled(self) -> bool:
        """
        Check if TalkBack is currently enabled.

        Returns:
            True if TalkBack is enabled.
        """
        try:
            # Check via settings
            result = await self.device.execute_shell(
                "settings get secure enabled_accessibility_services"
            )
            self._is_enabled = "talkback" in result.lower()
            return self._is_enabled
        except Exception as e:
            logger.error("Failed to check TalkBack status", error=str(e))
            return False

    async def enable(self) -> bool:
        """
        Enable TalkBack on the device.

        Returns:
            True if successfully enabled.
        """
        try:
            # Enable TalkBack via settings
            await self.device.execute_shell(
                "settings put secure enabled_accessibility_services "
                "com.google.android.marvin.talkback/com.google.android.marvin.talkback.TalkBackService"
            )
            await self.device.execute_shell(
                "settings put secure accessibility_enabled 1"
            )
            self._is_enabled = True
            logger.info("TalkBack enabled")
            return True
        except Exception as e:
            logger.error("Failed to enable TalkBack", error=str(e))
            return False

    async def disable(self) -> bool:
        """
        Disable TalkBack on the device.

        Returns:
            True if successfully disabled.
        """
        try:
            await self.device.execute_shell(
                "settings put secure enabled_accessibility_services \"\""
            )
            await self.device.execute_shell(
                "settings put secure accessibility_enabled 0"
            )
            self._is_enabled = False
            logger.info("TalkBack disabled")
            return True
        except Exception as e:
            logger.error("Failed to disable TalkBack", error=str(e))
            return False

    async def perform_gesture(self, gesture: TalkBackGesture) -> bool:
        """
        Perform a TalkBack gesture.

        Args:
            gesture: The gesture to perform.

        Returns:
            True if gesture was performed.
        """
        try:
            result = await self._execute_gesture(gesture)
            logger.debug("TalkBack gesture performed", gesture=gesture.name)
            return result
        except Exception as e:
            logger.error("Failed to perform gesture", gesture=gesture.name, error=str(e))
            return False

    async def _execute_gesture(self, gesture: TalkBackGesture) -> bool:
        """Execute a specific TalkBack gesture."""
        # Map gestures to device actions
        screen_info = await self.device.get_screen_info()
        width = screen_info.get("width", 1080)
        height = screen_info.get("height", 2400)

        center_x = width // 2
        center_y = height // 2

        if gesture == TalkBackGesture.SWIPE_RIGHT:
            return await self._swipe(center_x - 200, center_y, center_x + 200, center_y)
        elif gesture == TalkBackGesture.SWIPE_LEFT:
            return await self._swipe(center_x + 200, center_y, center_x - 200, center_y)
        elif gesture == TalkBackGesture.SWIPE_UP:
            return await self._swipe(center_x, center_y + 200, center_x, center_y - 200)
        elif gesture == TalkBackGesture.SWIPE_DOWN:
            return await self._swipe(center_x, center_y - 200, center_x, center_y + 200)
        elif gesture == TalkBackGesture.DOUBLE_TAP:
            result = await self.device.tap(center_x, center_y)
            if result.success:
                result = await self.device.tap(center_x, center_y)
            return result.success
        else:
            logger.warning("Gesture not implemented", gesture=gesture.name)
            return False

    async def _swipe(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Perform a swipe gesture."""
        result = await self.device.swipe(x1, y1, x2, y2, duration_ms=300)
        return result.success

    async def navigate_next(self) -> bool:
        """Navigate to next element."""
        return await self.perform_gesture(TalkBackGesture.SWIPE_RIGHT)

    async def navigate_previous(self) -> bool:
        """Navigate to previous element."""
        return await self.perform_gesture(TalkBackGesture.SWIPE_LEFT)

    async def activate(self) -> bool:
        """Activate the currently focused element."""
        return await self.perform_gesture(TalkBackGesture.DOUBLE_TAP)

    async def read_screen(self) -> bool:
        """Read the entire screen from top."""
        return await self.perform_gesture(TalkBackGesture.THREE_FINGER_TAP)

    async def pause_speech(self) -> bool:
        """Pause or resume TalkBack speech."""
        return await self.perform_gesture(TalkBackGesture.TWO_FINGER_TAP)

    async def set_speech_rate(self, rate: int) -> bool:
        """
        Set TalkBack speech rate.

        Args:
            rate: Speech rate percentage (50-300).

        Returns:
            True if successfully set.
        """
        try:
            rate = max(50, min(300, rate))
            # TalkBack stores rate as a float
            rate_float = rate / 100.0
            await self.device.execute_shell(
                f"settings put secure tts_default_rate {rate_float}"
            )
            self.settings.speech_rate = rate
            logger.info("TalkBack speech rate set", rate=rate)
            return True
        except Exception as e:
            logger.error("Failed to set speech rate", error=str(e))
            return False

    async def get_focused_element(self) -> Optional[dict[str, Any]]:
        """
        Get information about the currently focused element.

        Returns:
            Element info dict or None.
        """
        try:
            hierarchy = await self.device.get_ui_hierarchy()
            # Parse to find focused element
            from lxml import etree

            root = etree.fromstring(hierarchy.encode())
            focused = root.xpath("//*[@focused='true']")

            if focused:
                elem = focused[0]
                return {
                    "text": elem.get("text", ""),
                    "content_desc": elem.get("content-desc", ""),
                    "class": elem.get("class", ""),
                    "resource_id": elem.get("resource-id", ""),
                    "bounds": elem.get("bounds", ""),
                }
            return None
        except Exception as e:
            logger.error("Failed to get focused element", error=str(e))
            return None

    async def announce(self, text: str) -> bool:
        """
        Make TalkBack announce specific text.

        Args:
            text: Text to announce.

        Returns:
            True if announcement was triggered.
        """
        try:
            # Use accessibility announcement via intent
            await self.device.execute_shell(
                f'am broadcast -a android.accessibilityservice.AccessibilityService.ACTION_ANNOUNCEMENT '
                f'-e text "{text}"'
            )
            logger.debug("TalkBack announcement triggered", text=text[:50])
            return True
        except Exception as e:
            logger.error("Failed to trigger announcement", error=str(e))
            return False


def describe_element_for_talkback(
    element_type: str,
    text: str,
    content_desc: str,
    state: dict[str, bool],
) -> str:
    """
    Create a TalkBack-style description for an element.

    Args:
        element_type: Element class name.
        text: Element text.
        content_desc: Content description.
        state: Element state (checked, enabled, etc.).

    Returns:
        TalkBack-style description string.
    """
    parts = []

    # Primary text
    if text:
        parts.append(text)
    elif content_desc:
        parts.append(content_desc)

    # Element role
    role_map = {
        "android.widget.Button": "button",
        "android.widget.EditText": "edit text",
        "android.widget.CheckBox": "checkbox",
        "android.widget.RadioButton": "radio button",
        "android.widget.Switch": "switch",
        "android.widget.ImageButton": "button",
        "android.widget.SeekBar": "slider",
        "android.widget.Spinner": "dropdown",
        "android.widget.ToggleButton": "toggle",
    }

    role = role_map.get(element_type, "")
    if role:
        parts.append(role)

    # State
    if state.get("checked"):
        parts.append("checked" if state["checked"] else "not checked")

    if not state.get("enabled", True):
        parts.append("disabled")

    return ", ".join(parts)

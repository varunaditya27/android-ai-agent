"""
TalkBack Integration — Device-side Screen Reader Control
========================================================

Controls the Android TalkBack screen reader via ADB shell commands,
allowing the agent to:
- Enable / disable TalkBack on the device
- Configure speech rate, high-contrast text, font scale
- Perform TalkBack gestures (swipe to navigate, double-tap to activate)
- Read the currently focused element from the accessibility tree

All device communication goes through ``CloudDevice.execute_shell()``,
which is implemented by ``ADBDevice``.

Usage::

    from app.accessibility.talkback import TalkBackController

    tb = TalkBackController(device)
    await tb.enable()
    await tb.set_speech_rate(150)
    await tb.navigate_next()
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TalkBack gestures
# ---------------------------------------------------------------------------


class TalkBackGesture(Enum):
    """Standard TalkBack gestures mapped to swipe actions."""

    SWIPE_RIGHT = "swipe_right"           # Next element
    SWIPE_LEFT = "swipe_left"             # Previous element
    SWIPE_UP = "swipe_up"                 # Change setting / scroll
    SWIPE_DOWN = "swipe_down"             # Change setting / scroll
    DOUBLE_TAP = "double_tap"             # Activate element
    TWO_FINGER_TAP = "two_finger_tap"     # Pause / resume speech
    THREE_FINGER_TAP = "three_finger_tap"  # Read from top


# ---------------------------------------------------------------------------
# Settings DTO
# ---------------------------------------------------------------------------


@dataclass
class TalkBackSettings:
    """Desired TalkBack configuration."""

    enabled: bool = False
    speech_rate: int = 100          # percentage (50-300)
    high_contrast: bool = False
    large_text: bool = False
    font_scale: float = 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class TalkBackController:
    """
    Controls Android TalkBack and related display accessibility features
    via ``adb shell`` commands.
    """

    def __init__(
        self,
        device: CloudDevice,
        settings: Optional[TalkBackSettings] = None,
    ) -> None:
        self.device = device
        self.settings = settings or TalkBackSettings()
        self._is_enabled = False

        logger.info("TalkBackController initialized")

    # ------------------------------------------------------------------
    # TalkBack enable / disable
    # ------------------------------------------------------------------

    async def is_enabled(self) -> bool:
        """Check if TalkBack is currently enabled on the device."""
        try:
            result = await self.device.execute_shell(
                "settings get secure enabled_accessibility_services"
            )
            self._is_enabled = "talkback" in result.lower()
            return self._is_enabled
        except Exception as e:
            logger.error("Failed to check TalkBack status", error=str(e))
            return False

    async def enable(self) -> bool:
        """Enable TalkBack on the device."""
        try:
            await self.device.execute_shell(
                "settings put secure enabled_accessibility_services "
                "com.google.android.marvin.talkback/"
                "com.google.android.marvin.talkback.TalkBackService"
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
        """Disable TalkBack on the device."""
        try:
            await self.device.execute_shell(
                'settings put secure enabled_accessibility_services ""'
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

    # ------------------------------------------------------------------
    # Display / accessibility settings
    # ------------------------------------------------------------------

    async def set_high_contrast(self, enabled: bool) -> bool:
        """Enable or disable high-contrast text."""
        try:
            val = "1" if enabled else "0"
            await self.device.execute_shell(
                f"settings put secure high_text_contrast_enabled {val}"
            )
            self.settings.high_contrast = enabled
            logger.info("High contrast text", enabled=enabled)
            return True
        except Exception as e:
            logger.error("Failed to set high contrast", error=str(e))
            return False

    async def set_font_scale(self, scale: float) -> bool:
        """Set the system font scale (1.0 = default, 1.3 = large)."""
        try:
            scale = max(0.85, min(2.0, scale))
            await self.device.execute_shell(
                f"settings put system font_scale {scale}"
            )
            self.settings.font_scale = scale
            logger.info("Font scale set", scale=scale)
            return True
        except Exception as e:
            logger.error("Failed to set font scale", error=str(e))
            return False

    async def set_speech_rate(self, rate: int) -> bool:
        """Set TalkBack TTS speech rate (percentage 50-300)."""
        try:
            rate = max(50, min(300, rate))
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

    # ------------------------------------------------------------------
    # TalkBack gestures (simulated via device swipe / tap)
    # ------------------------------------------------------------------

    async def perform_gesture(self, gesture: TalkBackGesture) -> bool:
        """Perform a TalkBack gesture on the device."""
        try:
            result = await self._execute_gesture(gesture)
            logger.debug("TalkBack gesture performed", gesture=gesture.name)
            return result
        except Exception as e:
            logger.error(
                "Failed to perform gesture",
                gesture=gesture.name,
                error=str(e),
            )
            return False

    async def _execute_gesture(self, gesture: TalkBackGesture) -> bool:
        """Map a gesture enum to a device action."""
        info = self.device.info
        width = info.screen_width if info else 1080
        height = info.screen_height if info else 2400

        cx = width // 2
        cy = height // 2

        if gesture == TalkBackGesture.SWIPE_RIGHT:
            return await self._swipe(cx - 200, cy, cx + 200, cy)
        elif gesture == TalkBackGesture.SWIPE_LEFT:
            return await self._swipe(cx + 200, cy, cx - 200, cy)
        elif gesture == TalkBackGesture.SWIPE_UP:
            return await self._swipe(cx, cy + 200, cx, cy - 200)
        elif gesture == TalkBackGesture.SWIPE_DOWN:
            return await self._swipe(cx, cy - 200, cx, cy + 200)
        elif gesture == TalkBackGesture.DOUBLE_TAP:
            r = await self.device.tap(cx, cy)
            if r.success:
                r = await self.device.tap(cx, cy)
            return r.success
        else:
            logger.warning("Gesture not implemented", gesture=gesture.name)
            return False

    async def _swipe(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> bool:
        result = await self.device.swipe(
            x1, y1, x2, y2, duration_ms=300
        )
        return result.success

    # Convenience navigation helpers

    async def navigate_next(self) -> bool:
        """Navigate to the next element (swipe right)."""
        return await self.perform_gesture(TalkBackGesture.SWIPE_RIGHT)

    async def navigate_previous(self) -> bool:
        """Navigate to the previous element (swipe left)."""
        return await self.perform_gesture(TalkBackGesture.SWIPE_LEFT)

    async def activate(self) -> bool:
        """Activate the currently focused element (double tap)."""
        return await self.perform_gesture(TalkBackGesture.DOUBLE_TAP)

    # ------------------------------------------------------------------
    # Read focused element from UI hierarchy
    # ------------------------------------------------------------------

    async def get_focused_element(self) -> Optional[dict[str, Any]]:
        """Get the currently focused element from the accessibility tree."""
        try:
            hierarchy = await self.device.get_ui_hierarchy()
            if isinstance(hierarchy, dict):
                elements = hierarchy.get("elements", [])
                for elem in elements:
                    if elem.get("focused"):
                        return {
                            "text": elem.get("text", ""),
                            "content_desc": elem.get("content_desc", ""),
                            "class": elem.get("class", ""),
                            "resource_id": elem.get("resource_id", ""),
                            "bounds": elem.get("bounds", ""),
                        }
            return None
        except Exception as e:
            logger.error("Failed to get focused element", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Apply all settings at once (called during device setup)
    # ------------------------------------------------------------------

    async def apply_settings(self) -> None:
        """
        Apply the current ``TalkBackSettings`` to the device.

        Called once during session setup to configure device-side
        accessibility based on user preferences.
        """
        if self.settings.enabled:
            await self.enable()
        if self.settings.high_contrast:
            await self.set_high_contrast(True)
        if self.settings.large_text:
            await self.set_font_scale(1.3)
        if self.settings.speech_rate != 100:
            await self.set_speech_rate(self.settings.speech_rate)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def describe_element_for_talkback(
    element_type: str,
    text: str,
    content_desc: str,
    state: dict[str, bool],
) -> str:
    """
    Create a TalkBack-style spoken description for a UI element.

    Args:
        element_type: Full Android class name.
        text: Element visible text.
        content_desc: Content description attribute.
        state: Dict of boolean states (checked, enabled, ...).

    Returns:
        Natural-language description string.
    """
    parts: list[str] = []

    if text:
        parts.append(text)
    elif content_desc:
        parts.append(content_desc)

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

    if state.get("checked"):
        parts.append("checked" if state["checked"] else "not checked")
    if not state.get("enabled", True):
        parts.append("disabled")

    return ", ".join(parts)

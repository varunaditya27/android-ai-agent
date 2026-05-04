"""
Haptic Feedback — Device-side Vibration for Blind Users
=======================================================

Triggers vibration patterns on the Android device via ADB shell commands
to give blind users tactile feedback about action results.

Patterns are defined as lists of ``(vibrate_ms, pause_ms)`` tuples.
The controller auto-detects the best vibrator command for the device.

Usage::

    from app.accessibility.haptics import HapticsController, HapticPattern

    haptics = HapticsController(device)
    await haptics.success()
    await haptics.error()
    await haptics.action_feedback("Tap", True)
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & pattern definitions
# ---------------------------------------------------------------------------


class VibrationIntensity(Enum):
    """Vibration intensity level."""

    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"


class HapticPattern(Enum):
    """Named haptic patterns for common events."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    CLICK = "click"
    LONG_PRESS = "long_press"
    PROGRESS = "progress"
    INPUT_REQUIRED = "input_required"
    NOTIFICATION = "notification"
    DOUBLE_TAP = "double_tap"
    SCROLL = "scroll"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"


# Each pattern is a list of (vibrate_ms, pause_ms) tuples.
HAPTIC_PATTERNS: dict[HapticPattern, list[tuple[int, int]]] = {
    HapticPattern.SUCCESS: [(100, 50), (100, 0)],
    HapticPattern.ERROR: [(300, 100), (300, 100), (300, 0)],
    HapticPattern.WARNING: [(200, 100), (200, 0)],
    HapticPattern.CLICK: [(50, 0)],
    HapticPattern.LONG_PRESS: [(150, 0)],
    HapticPattern.PROGRESS: [(80, 80), (80, 0)],
    HapticPattern.INPUT_REQUIRED: [
        (100, 50),
        (100, 50),
        (100, 50),
        (100, 0),
    ],
    HapticPattern.NOTIFICATION: [(150, 80), (150, 0)],
    HapticPattern.DOUBLE_TAP: [(40, 30), (40, 0)],
    HapticPattern.SCROLL: [(30, 0)],
    HapticPattern.TASK_COMPLETE: [(100, 80), (100, 80), (200, 0)],
    HapticPattern.TASK_FAILED: [(400, 150), (400, 0)],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HapticsConfig:
    """Haptic-feedback configuration."""

    enabled: bool = True
    intensity: VibrationIntensity = VibrationIntensity.MEDIUM


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class HapticsController:
    """
    Triggers vibration patterns on the Android device.

    Auto-detects the best available vibrator command on first use:
    1. ``cmd vibrator_manager vibrate`` (Android 12+)
    2. ``cmd vibrator vibrate`` (older Android)
    3. ``input keyevent 229`` (last resort)
    """

    _INTENSITY_MULTIPLIERS = {
        VibrationIntensity.LIGHT: 0.5,
        VibrationIntensity.MEDIUM: 1.0,
        VibrationIntensity.STRONG: 1.5,
    }

    def __init__(
        self,
        device: CloudDevice,
        config: Optional[HapticsConfig] = None,
    ) -> None:
        self.device = device
        self.config = config or HapticsConfig()
        self._vibrator_cmd: Optional[str] = None

        logger.info(
            "HapticsController initialized",
            enabled=self.config.enabled,
            intensity=self.config.intensity.value,
        )

    # ------------------------------------------------------------------
    # Low-level vibration
    # ------------------------------------------------------------------

    async def _detect_vibrator(self) -> str:
        """Auto-detect which vibrator command the device supports."""
        if self._vibrator_cmd:
            return self._vibrator_cmd

        # Try modern command first (Android 12+)
        try:
            await self.device.execute_shell(
                "cmd vibrator_manager vibrate 1 -f"
            )
            self._vibrator_cmd = "cmd vibrator_manager vibrate"
            logger.debug("Using vibrator_manager command")
            return self._vibrator_cmd
        except Exception:
            pass

        # Fallback: older API
        try:
            await self.device.execute_shell("cmd vibrator vibrate 1")
            self._vibrator_cmd = "cmd vibrator vibrate"
            logger.debug("Using legacy vibrator command")
            return self._vibrator_cmd
        except Exception:
            pass

        # Last resort: input keyevent
        self._vibrator_cmd = "input keyevent"
        logger.debug("Falling back to input keyevent for vibration")
        return self._vibrator_cmd

    async def _vibrate_once(self, duration_ms: int) -> None:
        """Vibrate the device for *duration_ms* milliseconds."""
        cmd = await self._detect_vibrator()

        if cmd == "input keyevent":
            # KEYCODE_CONTACTS (229) triggers a short haptic buzz
            await self.device.execute_shell("input keyevent 229")
        else:
            await self.device.execute_shell(f"{cmd} {duration_ms}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def vibrate(
        self,
        pattern: HapticPattern,
        intensity: Optional[VibrationIntensity] = None,
    ) -> bool:
        """
        Play a predefined haptic pattern.

        Args:
            pattern: Pattern to play.
            intensity: Override default intensity.

        Returns:
            True if vibration was triggered.
        """
        if not self.config.enabled:
            return False

        intensity = intensity or self.config.intensity
        multiplier = self._INTENSITY_MULTIPLIERS[intensity]

        try:
            pattern_def = HAPTIC_PATTERNS.get(pattern, [(50, 0)])

            for vib_ms, pause_ms in pattern_def:
                adjusted = max(10, int(vib_ms * multiplier))
                await self._vibrate_once(adjusted)
                if pause_ms > 0:
                    await asyncio.sleep(pause_ms / 1000.0)

            logger.debug(
                "Haptic pattern played",
                pattern=pattern.name,
                intensity=intensity.value,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to play haptic pattern",
                pattern=pattern.name,
                error=str(e),
            )
            return False

    async def vibrate_ms(self, duration_ms: int) -> bool:
        """Vibrate for a specific duration in milliseconds."""
        if not self.config.enabled:
            return False
        try:
            await self._vibrate_once(duration_ms)
            return True
        except Exception as e:
            logger.error("Failed to vibrate", error=str(e))
            return False

    # Convenience shortcuts

    async def click(self) -> bool:
        return await self.vibrate(HapticPattern.CLICK)

    async def success(self) -> bool:
        return await self.vibrate(HapticPattern.SUCCESS)

    async def error(self) -> bool:
        return await self.vibrate(HapticPattern.ERROR)

    async def warning(self) -> bool:
        return await self.vibrate(HapticPattern.WARNING)

    async def input_required(self) -> bool:
        return await self.vibrate(HapticPattern.INPUT_REQUIRED)

    async def progress(self) -> bool:
        return await self.vibrate(HapticPattern.PROGRESS)

    async def task_complete(self) -> bool:
        return await self.vibrate(HapticPattern.TASK_COMPLETE)

    async def task_failed(self) -> bool:
        return await self.vibrate(HapticPattern.TASK_FAILED)

    async def action_feedback(
        self, action_type: str, success: bool
    ) -> bool:
        """
        Provide haptic feedback for an action result.

        Maps the action type to an appropriate vibration pattern.
        """
        if not self.config.enabled:
            return False

        if not success:
            return await self.vibrate(HapticPattern.ERROR)

        action_patterns = {
            "Tap": HapticPattern.CLICK,
            "LongPress": HapticPattern.LONG_PRESS,
            "DoubleTap": HapticPattern.DOUBLE_TAP,
            "Swipe": HapticPattern.SCROLL,
            "Type": HapticPattern.CLICK,
            "PressKey": HapticPattern.CLICK,
            "Launch": HapticPattern.SUCCESS,
            "Back": HapticPattern.CLICK,
            "Home": HapticPattern.CLICK,
        }

        pat = action_patterns.get(action_type, HapticPattern.CLICK)
        return await self.vibrate(pat)

    # Configuration helpers

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled
        logger.info(
            "Haptics " + ("enabled" if enabled else "disabled")
        )

    def set_intensity(self, intensity: VibrationIntensity) -> None:
        self.config.intensity = intensity
        logger.info("Haptics intensity set", intensity=intensity.value)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def create_custom_pattern(
    pulses: list[int],
    pauses: list[int],
) -> list[tuple[int, int]]:
    """
    Create a custom haptic pattern from parallel lists.

    Args:
        pulses: Vibration durations in ms.
        pauses: Pause durations in ms.

    Returns:
        Pattern definition for ``HapticsController.vibrate()``.
    """
    if len(pulses) != len(pauses):
        if len(pulses) == len(pauses) + 1:
            pauses = pauses + [0]
        else:
            raise ValueError(
                "Pulses and pauses must have matching lengths"
            )
    return list(zip(pulses, pauses))

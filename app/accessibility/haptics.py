"""
Haptic Feedback Module
======================

Provides haptic (vibration) feedback for actions and events.

Haptic feedback helps blind users confirm actions without
relying on visual cues. Different patterns convey different
meanings (success, error, progress, etc.).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VibrationIntensity(Enum):
    """Intensity levels for vibration."""

    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"


class HapticPattern(Enum):
    """
    Predefined haptic patterns for different events.

    Each pattern has a specific meaning:
    - SUCCESS: Action completed successfully
    - ERROR: Action failed
    - WARNING: Something needs attention
    - CLICK: Simple button click
    - LONG_PRESS: Long press confirmation
    - PROGRESS: Step completion in multi-step task
    - INPUT_REQUIRED: User input needed
    - NOTIFICATION: New notification/message
    """

    SUCCESS = auto()
    ERROR = auto()
    WARNING = auto()
    CLICK = auto()
    LONG_PRESS = auto()
    PROGRESS = auto()
    INPUT_REQUIRED = auto()
    NOTIFICATION = auto()
    DOUBLE_TAP = auto()
    SCROLL = auto()


# Vibration pattern definitions
# Format: list of (duration_ms, pause_ms) tuples
HAPTIC_PATTERNS: dict[HapticPattern, list[tuple[int, int]]] = {
    HapticPattern.SUCCESS: [(100, 50), (100, 0)],  # Two quick bursts
    HapticPattern.ERROR: [(300, 100), (300, 100), (300, 0)],  # Three long bursts
    HapticPattern.WARNING: [(200, 100), (200, 0)],  # Two medium bursts
    HapticPattern.CLICK: [(50, 0)],  # Single short tap
    HapticPattern.LONG_PRESS: [(200, 0)],  # Single long pulse
    HapticPattern.PROGRESS: [(50, 50), (50, 0)],  # Two very short taps
    HapticPattern.INPUT_REQUIRED: [(100, 50), (100, 50), (100, 50), (100, 0)],  # Four taps
    HapticPattern.NOTIFICATION: [(150, 100), (150, 0)],  # Two medium taps
    HapticPattern.DOUBLE_TAP: [(30, 30), (30, 0)],  # Quick double tap feel
    HapticPattern.SCROLL: [(20, 0)],  # Very light scroll feedback
}


@dataclass
class HapticsConfig:
    """
    Configuration for haptic feedback.

    Attributes:
        enabled: Whether haptics are enabled.
        intensity: Default vibration intensity.
        respect_system_settings: Honor device vibration settings.
    """

    enabled: bool = True
    intensity: VibrationIntensity = VibrationIntensity.MEDIUM
    respect_system_settings: bool = True


class HapticsController:
    """
    Controller for haptic feedback.

    Provides methods to trigger various haptic patterns
    for different events and actions.
    """

    def __init__(
        self,
        device: CloudDevice,
        config: Optional[HapticsConfig] = None,
    ) -> None:
        """
        Initialize haptics controller.

        Args:
            device: Cloud device to control.
            config: Haptics configuration.
        """
        self.device = device
        self.config = config or HapticsConfig()

        # Intensity multipliers
        self._intensity_multipliers = {
            VibrationIntensity.LIGHT: 0.5,
            VibrationIntensity.MEDIUM: 1.0,
            VibrationIntensity.STRONG: 1.5,
        }

        logger.info(
            "HapticsController initialized",
            enabled=self.config.enabled,
            intensity=self.config.intensity.value,
        )

    async def vibrate(
        self,
        pattern: HapticPattern,
        intensity: Optional[VibrationIntensity] = None,
    ) -> bool:
        """
        Trigger a haptic pattern.

        Args:
            pattern: The pattern to play.
            intensity: Override default intensity.

        Returns:
            True if vibration was triggered.
        """
        if not self.config.enabled:
            return False

        intensity = intensity or self.config.intensity
        multiplier = self._intensity_multipliers[intensity]

        try:
            # Get pattern definition
            pattern_def = HAPTIC_PATTERNS.get(pattern, [(50, 0)])

            # Build vibration command
            durations = []
            for duration_ms, pause_ms in pattern_def:
                adjusted_duration = int(duration_ms * multiplier)
                durations.append(adjusted_duration)
                if pause_ms > 0:
                    durations.append(pause_ms)

            # Create pattern string for Android
            # Format: duration1,pause1,duration2,pause2,...
            pattern_str = ",".join(str(d) for d in durations)

            # Execute vibration via shell
            await self.device.execute_shell(
                f"cmd vibrator vibrate -f {pattern_str}"
            )

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

    async def vibrate_ms(
        self,
        duration_ms: int,
        intensity: Optional[VibrationIntensity] = None,
    ) -> bool:
        """
        Vibrate for a specific duration.

        Args:
            duration_ms: Duration in milliseconds.
            intensity: Vibration intensity.

        Returns:
            True if vibration was triggered.
        """
        if not self.config.enabled:
            return False

        intensity = intensity or self.config.intensity
        multiplier = self._intensity_multipliers[intensity]
        adjusted_duration = int(duration_ms * multiplier)

        try:
            await self.device.execute_shell(
                f"cmd vibrator vibrate {adjusted_duration}"
            )
            return True
        except Exception as e:
            logger.error("Failed to vibrate", error=str(e))
            return False

    async def click(self) -> bool:
        """Trigger click feedback."""
        return await self.vibrate(HapticPattern.CLICK)

    async def success(self) -> bool:
        """Trigger success feedback."""
        return await self.vibrate(HapticPattern.SUCCESS)

    async def error(self) -> bool:
        """Trigger error feedback."""
        return await self.vibrate(HapticPattern.ERROR)

    async def warning(self) -> bool:
        """Trigger warning feedback."""
        return await self.vibrate(HapticPattern.WARNING)

    async def input_required(self) -> bool:
        """Trigger input required feedback."""
        return await self.vibrate(HapticPattern.INPUT_REQUIRED)

    async def progress(self) -> bool:
        """Trigger progress feedback."""
        return await self.vibrate(HapticPattern.PROGRESS)

    async def action_feedback(self, action_type: str, success: bool) -> bool:
        """
        Provide haptic feedback for an action result.

        Args:
            action_type: Type of action performed.
            success: Whether action succeeded.

        Returns:
            True if feedback was provided.
        """
        if not self.config.enabled:
            return False

        # Map actions to patterns
        action_patterns = {
            "Tap": HapticPattern.CLICK,
            "LongPress": HapticPattern.LONG_PRESS,
            "DoubleTap": HapticPattern.DOUBLE_TAP,
            "Swipe": HapticPattern.SCROLL,
            "Type": HapticPattern.CLICK,
            "Launch": HapticPattern.SUCCESS,
            "Back": HapticPattern.CLICK,
            "Home": HapticPattern.CLICK,
        }

        if not success:
            return await self.vibrate(HapticPattern.ERROR)

        pattern = action_patterns.get(action_type, HapticPattern.CLICK)
        return await self.vibrate(pattern)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable haptic feedback."""
        self.config.enabled = enabled
        logger.info("Haptics enabled" if enabled else "Haptics disabled")

    def set_intensity(self, intensity: VibrationIntensity) -> None:
        """Set default vibration intensity."""
        self.config.intensity = intensity
        logger.info("Haptics intensity set", intensity=intensity.value)


def create_custom_pattern(pulses: list[int], pauses: list[int]) -> list[tuple[int, int]]:
    """
    Create a custom haptic pattern.

    Args:
        pulses: List of pulse durations in ms.
        pauses: List of pause durations in ms.

    Returns:
        Pattern definition for use with HapticsController.
    """
    if len(pulses) != len(pauses):
        if len(pulses) == len(pauses) + 1:
            pauses = pauses + [0]
        else:
            raise ValueError("Pulses and pauses must have matching lengths")

    return list(zip(pulses, pauses))

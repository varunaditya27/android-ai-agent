"""
Accessibility Manager — Unified Entry Point
============================================

Coordinates the Announcer (host TTS), TalkBackController (device screen
reader), and HapticsController (device vibration) into a single facade
used by the agent loop and demo runner.

The manager is configured from ``AccessibilitySettings`` in
``app.config`` and provides high-level methods like:

- ``on_step(step_result)``  — called after each agent step
- ``on_task_start(task)``   — called when a new task begins
- ``on_task_complete(...)`` — called when a task finishes
- ``on_input_required(prompt)`` — called when user input is needed

Usage::

    from app.accessibility.manager import AccessibilityManager

    mgr = AccessibilityManager(device, settings)
    await mgr.setup()                   # configure device-side a11y
    await mgr.on_task_start("Open YouTube")
    await mgr.on_step(step_result)
    await mgr.on_task_complete(True, "Done")
"""

from typing import Optional

from app.accessibility.announcer import (
    Announcer,
    AnnouncementConfig,
)
from app.accessibility.haptics import (
    HapticsConfig,
    HapticsController,
)
from app.accessibility.talkback import (
    TalkBackController,
    TalkBackSettings,
)
from app.device.cloud_provider import CloudDevice
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AccessibilityManager:
    """
    Unified accessibility controller.

    Wraps Announcer, TalkBackController, and HapticsController.
    All methods are safe to call even when accessibility is disabled —
    they become no-ops.
    """

    def __init__(
        self,
        device: Optional[CloudDevice] = None,
        enable_tts: bool = True,
        tts_rate: int = 200,
        tts_volume: float = 1.0,
        enable_haptics: bool = True,
        enable_talkback: bool = False,
        high_contrast: bool = False,
        large_text: bool = False,
        screen_reader_mode: bool = True,
    ) -> None:
        """
        Initialise the manager.

        Args:
            device: CloudDevice for TalkBack / haptics (can be None for
                    host-only TTS).
            enable_tts: Enable host-side text-to-speech.
            tts_rate: Words-per-minute for pyttsx3.
            tts_volume: Volume 0.0–1.0.
            enable_haptics: Enable device vibration feedback.
            enable_talkback: Enable TalkBack on the device.
            high_contrast: Enable high-contrast text on device.
            large_text: Enable large text (1.3× font scale) on device.
            screen_reader_mode: Strip emojis and format output for
                                desktop screen readers.
        """
        self._device = device

        # Announcer (host-side TTS)
        self.announcer = Announcer(
            config=AnnouncementConfig(
                enabled=enable_tts,
                speech_rate=tts_rate,
                volume=tts_volume,
                screen_reader_mode=screen_reader_mode,
            )
        )

        # TalkBack (device-side)
        self.talkback: Optional[TalkBackController] = None
        if device:
            self.talkback = TalkBackController(
                device,
                settings=TalkBackSettings(
                    enabled=enable_talkback,
                    high_contrast=high_contrast,
                    large_text=large_text,
                ),
            )

        # Haptics (device-side)
        self.haptics: Optional[HapticsController] = None
        if device:
            self.haptics = HapticsController(
                device,
                config=HapticsConfig(enabled=enable_haptics),
            )

        logger.info(
            "AccessibilityManager initialized",
            tts=enable_tts,
            haptics=enable_haptics,
            talkback=enable_talkback,
        )

    # ------------------------------------------------------------------
    # Device setup (called once after device.connect())
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """
        Apply device-side accessibility settings.

        Call this after the device is connected to configure TalkBack,
        high contrast, large text, etc.
        """
        if self.talkback:
            try:
                await self.talkback.apply_settings()
            except Exception as e:
                logger.warning(
                    "Failed to apply TalkBack settings", error=str(e)
                )

    # ------------------------------------------------------------------
    # Agent-loop callbacks
    # ------------------------------------------------------------------

    async def on_task_start(self, task: str) -> None:
        """Announce that a new task has started."""
        await self.announcer.announce_task_start(task)

    async def on_step(
        self,
        step_num: int,
        max_steps: int,
        action_type: str,
        success: bool,
        thinking: str = "",
        error: Optional[str] = None,
        finished: bool = False,
        result_message: str = "",
    ) -> None:
        """
        Called after each agent step.

        Provides both TTS and haptic feedback.
        """
        # TTS announcement
        if finished:
            await self.announcer.announce_completion(True, result_message)
        elif error:
            short_error = error[:100] if len(error) > 100 else error
            await self.announcer.announce_error(short_error, recoverable=True)
        else:
            # Announce step progress with the action taken
            await self.announcer.announce_progress(
                step_num, max_steps, action_type
            )

        # Haptic feedback
        if self.haptics:
            try:
                if finished:
                    await self.haptics.task_complete()
                elif error:
                    await self.haptics.error()
                else:
                    await self.haptics.action_feedback(action_type, success)
            except Exception as e:
                logger.debug("Haptic feedback failed", error=str(e))

    async def on_task_complete(
        self, success: bool, message: str
    ) -> None:
        """Announce task completion/failure."""
        await self.announcer.announce_completion(success, message)
        if self.haptics:
            try:
                if success:
                    await self.haptics.task_complete()
                else:
                    await self.haptics.task_failed()
            except Exception:
                pass

    async def on_input_required(self, prompt: str) -> None:
        """Announce that user input is needed."""
        await self.announcer.announce_input_required(prompt)
        if self.haptics:
            try:
                await self.haptics.input_required()
            except Exception:
                pass

    async def on_rate_limit(self, wait_seconds: float) -> None:
        """Announce a rate-limit wait."""
        await self.announcer.announce_rate_limit(wait_seconds)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop all accessibility services."""
        self.announcer.stop()

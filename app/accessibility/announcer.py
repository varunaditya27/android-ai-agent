"""
Voice Announcement Module
=========================

Provides voice feedback to users through text-to-speech.

This module handles generating spoken feedback about:
- Action confirmations
- Screen state descriptions
- Error messages
- Progress updates

The announcements are designed to be concise and informative
for screen reader users.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from queue import PriorityQueue

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnnouncementPriority(Enum):
    """Priority levels for announcements."""

    LOW = 3  # Background info, can be interrupted
    NORMAL = 2  # Standard announcements
    HIGH = 1  # Important, interrupt current speech
    CRITICAL = 0  # Urgent, must be heard immediately


@dataclass(order=True)
class Announcement:
    """
    A single announcement to be spoken.

    Attributes:
        priority: How urgent this announcement is.
        text: The text to speak.
        timestamp: When the announcement was created.
        force: Whether to force-interrupt current speech.
    """

    priority: int
    text: str = field(compare=False)
    timestamp: float = field(compare=False, default=0.0)
    force: bool = field(compare=False, default=False)


@dataclass
class AnnouncementConfig:
    """
    Configuration for the announcer.

    Attributes:
        enabled: Whether announcements are enabled.
        speech_rate: Speed of speech (0.5-2.0, 1.0 is normal).
        pitch: Pitch of speech (0.5-2.0, 1.0 is normal).
        language: Language code (e.g., 'en-US').
        max_queue_size: Maximum queued announcements.
        duplicate_timeout: Ignore duplicates within this time (seconds).
    """

    enabled: bool = True
    speech_rate: float = 1.0
    pitch: float = 1.0
    language: str = "en-US"
    max_queue_size: int = 10
    duplicate_timeout: float = 2.0


class Announcer:
    """
    Manages voice announcements for the agent.

    Handles queuing, deduplication, and delivery of
    spoken feedback to users.
    """

    def __init__(
        self,
        config: Optional[AnnouncementConfig] = None,
        speech_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the announcer.

        Args:
            config: Announcer configuration.
            speech_callback: Function to call for actual TTS.
        """
        self.config = config or AnnouncementConfig()
        self.speech_callback = speech_callback

        # Queue for announcements (priority queue)
        self._queue: PriorityQueue[Announcement] = PriorityQueue(
            maxsize=self.config.max_queue_size
        )

        # Track recent announcements for deduplication
        self._recent: dict[str, float] = {}

        # State
        self._is_speaking = False
        self._current_announcement: Optional[Announcement] = None

        logger.info(
            "Announcer initialized",
            enabled=self.config.enabled,
            speech_rate=self.config.speech_rate,
        )

    async def announce(
        self,
        text: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        force: bool = False,
    ) -> bool:
        """
        Queue an announcement to be spoken.

        Args:
            text: The text to speak.
            priority: Announcement priority.
            force: Whether to force-interrupt current speech.

        Returns:
            True if announcement was queued.
        """
        if not self.config.enabled:
            return False

        # Check for duplicates
        current_time = asyncio.get_event_loop().time()
        if text in self._recent:
            last_time = self._recent[text]
            if current_time - last_time < self.config.duplicate_timeout:
                logger.debug("Skipping duplicate announcement", text=text[:50])
                return False

        # Create announcement
        announcement = Announcement(
            priority=priority.value,
            text=text,
            timestamp=current_time,
            force=force,
        )

        # Try to add to queue
        try:
            self._queue.put_nowait(announcement)
            self._recent[text] = current_time
            logger.debug(
                "Announcement queued",
                text=text[:50],
                priority=priority.name,
            )
            return True
        except asyncio.QueueFull:
            logger.warning("Announcement queue full, dropping", text=text[:50])
            return False

    async def announce_action(self, action_type: str, success: bool, details: str = "") -> None:
        """
        Announce the result of an action.

        Args:
            action_type: Type of action performed.
            success: Whether action succeeded.
            details: Additional details.
        """
        if success:
            text = f"{action_type} completed"
            if details:
                text += f": {details}"
            await self.announce(text, AnnouncementPriority.NORMAL)
        else:
            text = f"{action_type} failed"
            if details:
                text += f": {details}"
            await self.announce(text, AnnouncementPriority.HIGH)

    async def announce_screen(self, app_name: str, screen_summary: str) -> None:
        """
        Announce current screen state.

        Args:
            app_name: Name of current app.
            screen_summary: Brief description of screen.
        """
        text = f"{app_name}. {screen_summary}"
        await self.announce(text, AnnouncementPriority.NORMAL)

    async def announce_progress(self, step: int, total_steps: int, current_action: str) -> None:
        """
        Announce task progress.

        Args:
            step: Current step number.
            total_steps: Total expected steps (0 if unknown).
            current_action: What's currently being done.
        """
        if total_steps > 0:
            text = f"Step {step} of {total_steps}: {current_action}"
        else:
            text = f"Step {step}: {current_action}"
        await self.announce(text, AnnouncementPriority.LOW)

    async def announce_error(self, error: str, recoverable: bool = True) -> None:
        """
        Announce an error.

        Args:
            error: Error description.
            recoverable: Whether the error is recoverable.
        """
        priority = AnnouncementPriority.HIGH if recoverable else AnnouncementPriority.CRITICAL
        text = f"Error: {error}"
        if recoverable:
            text += ". Retrying."
        await self.announce(text, priority)

    async def announce_input_required(self, prompt: str) -> None:
        """
        Announce that user input is needed.

        Args:
            prompt: The input prompt.
        """
        await self.announce(
            f"Input required: {prompt}",
            AnnouncementPriority.CRITICAL,
            force=True,
        )

    async def announce_completion(self, success: bool, message: str) -> None:
        """
        Announce task completion.

        Args:
            success: Whether task succeeded.
            message: Completion message.
        """
        if success:
            text = f"Task completed: {message}"
        else:
            text = f"Task failed: {message}"
        await self.announce(text, AnnouncementPriority.CRITICAL, force=True)

    async def process_queue(self) -> None:
        """Process the announcement queue."""
        while not self._queue.empty():
            try:
                announcement = self._queue.get_nowait()

                # Deliver via callback
                if self.speech_callback:
                    self.speech_callback(announcement.text)

                logger.debug("Announcement delivered", text=announcement.text[:50])

                # Brief pause between announcements
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error("Error processing announcement", error=str(e))

    def clear_queue(self) -> None:
        """Clear all pending announcements."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        logger.debug("Announcement queue cleared")

    def stop(self) -> None:
        """Stop current speech and clear queue."""
        self._is_speaking = False
        self._current_announcement = None
        self.clear_queue()


def format_element_for_speech(
    element_type: str,
    text: str,
    properties: dict[str, Any],
) -> str:
    """
    Format a UI element for speech output.

    Args:
        element_type: Type of element (Button, EditText, etc.).
        text: Element text content.
        properties: Element properties.

    Returns:
        Speech-friendly description.
    """
    # Map element types to natural speech
    type_map = {
        "Button": "button",
        "EditText": "text field",
        "TextView": "text",
        "ImageView": "image",
        "ImageButton": "image button",
        "CheckBox": "checkbox",
        "RadioButton": "radio button",
        "Switch": "switch",
        "SeekBar": "slider",
        "ProgressBar": "progress",
        "RecyclerView": "list",
        "ScrollView": "scrollable area",
    }

    spoken_type = type_map.get(element_type, element_type.lower())

    parts = []

    # Text content
    if text:
        parts.append(text)

    # Type
    parts.append(spoken_type)

    # State
    if properties.get("checked"):
        parts.append("checked")
    elif properties.get("checkable"):
        parts.append("not checked")

    if not properties.get("enabled", True):
        parts.append("disabled")

    if properties.get("focused"):
        parts.append("focused")

    return ", ".join(parts)

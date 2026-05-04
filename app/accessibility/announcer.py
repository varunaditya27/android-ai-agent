"""
Voice Announcer — Host-side TTS for Blind Users
================================================

Provides real text-to-speech output via **pyttsx3** so that blind users
hear what the agent is doing through the host computer's speakers.

Key design decisions:
- pyttsx3 runs synchronously, so we call it via ``asyncio.to_thread()``
  to avoid blocking the event loop.
- Deduplication prevents repeating the same announcement within a timeout.
- ``screen_reader_mode`` strips emojis and formats text for NVDA / JAWS
  / VoiceOver compatibility.

Usage::

    from app.accessibility.announcer import Announcer, AnnouncementPriority

    announcer = Announcer()
    await announcer.speak("Task started: Open YouTube")
    await announcer.announce_action("Tap", success=True, details="Search button")
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue
from typing import Any, Callable, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class AnnouncementPriority(IntEnum):
    """Priority levels — lower numeric value = spoken first."""

    LOW = 30
    NORMAL = 20
    HIGH = 10
    CRITICAL = 0


@dataclass(order=True)
class Announcement:
    """A queued speech item, ordered by priority then timestamp."""

    priority: int
    timestamp: float = field(compare=True)
    text: str = field(compare=False)
    force: bool = field(default=False, compare=False)


@dataclass
class AnnouncementConfig:
    """Configuration for the announcer."""

    enabled: bool = True
    speech_rate: int = 200
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    max_queue_size: int = 50
    duplicate_timeout: float = 3.0
    screen_reader_mode: bool = True


# ---------------------------------------------------------------------------
# Emoji / special-char stripping for screen readers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\u200d"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)


def _clean_for_speech(text: str) -> str:
    """Strip emojis and normalise whitespace for TTS / screen readers."""
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Announcer
# ---------------------------------------------------------------------------


class Announcer:
    """
    Voice announcer with real TTS output.

    Provides:
    - ``speak(text)`` — immediate TTS on the host via pyttsx3.
    - High-level helpers (``announce_action``, ``announce_progress``, etc.)
      that format text before speaking.
    - Deduplication to prevent repeated announcements.
    """

    def __init__(
        self,
        config: Optional[AnnouncementConfig] = None,
        speech_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or AnnouncementConfig()
        self.speech_callback = speech_callback

        # TTS engine (lazy-initialised on first use)
        self._tts_engine = None

        # Queue for announcements
        self._queue: PriorityQueue[Announcement] = PriorityQueue(
            maxsize=self.config.max_queue_size
        )

        # Track recent announcements for dedup
        self._recent: dict[str, float] = {}

        # State
        self._is_speaking = False
        self._current_announcement: Optional[Announcement] = None

        logger.info(
            "Announcer initialized",
            enabled=self.config.enabled,
            speech_rate=self.config.speech_rate,
        )

    # ------------------------------------------------------------------
    # TTS engine management
    # ------------------------------------------------------------------

    def _get_tts_engine(self):
        """Lazily create pyttsx3 engine."""
        if self._tts_engine is None:
            try:
                import pyttsx3

                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", self.config.speech_rate)
                self._tts_engine.setProperty("volume", self.config.volume)
                logger.info("pyttsx3 TTS engine initialized")
            except Exception as e:
                logger.warning(
                    "pyttsx3 not available, TTS disabled", error=str(e)
                )
                self._tts_engine = None
        return self._tts_engine

    def _speak_sync(self, text: str) -> None:
        """Synchronous TTS — called via ``asyncio.to_thread``."""
        engine = self._get_tts_engine()
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.warning("TTS speak failed", error=str(e))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(
        self,
        text: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
    ) -> bool:
        """
        Speak text out loud via host TTS.

        Also prints a ``[Agent]`` prefixed line to stdout so that
        desktop screen readers (NVDA, JAWS, VoiceOver) pick it up.

        Args:
            text: Text to speak.
            priority: Announcement priority.

        Returns:
            True if announcement was delivered.
        """
        if not self.config.enabled:
            return False

        clean = (
            _clean_for_speech(text)
            if self.config.screen_reader_mode
            else text
        )
        if not clean:
            return False

        # Print for screen-reader software (NVDA etc. reads stdout)
        print(f"[Agent] {clean}")

        # Deduplicate
        now = time.time()
        if clean in self._recent:
            if now - self._recent[clean] < self.config.duplicate_timeout:
                return False
        self._recent[clean] = now

        # External callback (for tests / WebSocket forwarding)
        if self.speech_callback:
            self.speech_callback(clean)

        # TTS via pyttsx3 on a worker thread
        try:
            await asyncio.to_thread(self._speak_sync, clean)
        except Exception as e:
            logger.debug("TTS unavailable", error=str(e))

        return True

    # Alias for backward compatibility
    async def announce(
        self,
        text: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        force: bool = False,
    ) -> bool:
        """Queue/speak an announcement."""
        return await self.speak(text, priority)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def announce_action(
        self, action_type: str, success: bool, details: str = ""
    ) -> None:
        """Announce the result of an action."""
        if success:
            text = f"{action_type} completed"
            if details:
                text += f": {details}"
            await self.speak(text, AnnouncementPriority.NORMAL)
        else:
            text = f"{action_type} failed"
            if details:
                text += f": {details}"
            await self.speak(text, AnnouncementPriority.HIGH)

    async def announce_screen(
        self, app_name: str, screen_summary: str
    ) -> None:
        """Announce current screen state."""
        await self.speak(
            f"{app_name}. {screen_summary}", AnnouncementPriority.NORMAL
        )

    async def announce_progress(
        self, step: int, total_steps: int, current_action: str
    ) -> None:
        """Announce task progress."""
        if total_steps > 0:
            text = f"Step {step} of {total_steps}: {current_action}"
        else:
            text = f"Step {step}: {current_action}"
        await self.speak(text, AnnouncementPriority.LOW)

    async def announce_error(
        self, error: str, recoverable: bool = True
    ) -> None:
        """Announce an error."""
        priority = (
            AnnouncementPriority.HIGH
            if recoverable
            else AnnouncementPriority.CRITICAL
        )
        text = f"Error: {error}"
        if recoverable:
            text += ". Retrying."
        await self.speak(text, priority)

    async def announce_input_required(self, prompt: str) -> None:
        """Announce that user input is needed."""
        await self.speak(
            f"Input required: {prompt}", AnnouncementPriority.CRITICAL
        )

    async def announce_completion(
        self, success: bool, message: str
    ) -> None:
        """Announce task completion."""
        if success:
            text = f"Task completed: {message}"
        else:
            text = f"Task failed: {message}"
        await self.speak(text, AnnouncementPriority.CRITICAL)

    async def announce_task_start(self, task: str) -> None:
        """Announce that a new task has started."""
        await self.speak(
            f"Starting task: {task}", AnnouncementPriority.HIGH
        )

    async def announce_rate_limit(self, wait_seconds: float) -> None:
        """Announce rate-limit wait."""
        await self.speak(
            f"Rate limited. Waiting {int(wait_seconds)} seconds.",
            AnnouncementPriority.NORMAL,
        )

    # ------------------------------------------------------------------
    # Queue processing (for batch / WebSocket)
    # ------------------------------------------------------------------

    async def process_queue(self) -> None:
        """Process queued announcements."""
        while not self._queue.empty():
            try:
                announcement = self._queue.get_nowait()
                if self.speech_callback:
                    self.speech_callback(announcement.text)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(
                    "Error processing announcement", error=str(e)
                )

    def clear_queue(self) -> None:
        """Clear all pending announcements."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop current speech and clear queue."""
        self._is_speaking = False
        self._current_announcement = None
        self.clear_queue()
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def format_element_for_speech(
    element_type: str,
    text: str,
    properties: dict[str, Any],
) -> str:
    """
    Format a UI element for speech output.

    Maps Android widget types to natural-language role names
    and appends state information (checked, disabled, etc.).
    """
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
    parts: list[str] = []

    if text:
        parts.append(text)

    parts.append(spoken_type)

    if properties.get("checked"):
        parts.append("checked")
    elif properties.get("checkable"):
        parts.append("not checked")

    if not properties.get("enabled", True):
        parts.append("disabled")

    if properties.get("focused"):
        parts.append("focused")

    return ", ".join(parts)

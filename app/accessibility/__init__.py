"""
Accessibility Module
====================

Accessibility features for blind users, including:
    - announcer: Voice announcements and speech feedback (pyttsx3 TTS)
    - talkback: TalkBack integration and control via ADB
    - haptics: Haptic feedback patterns via device vibration

These modules ensure the agent provides appropriate audio, speech,
and tactile feedback to users relying on screen readers and
assistive technology.
"""

from app.accessibility.announcer import (
    Announcer,
    AnnouncementConfig,
    AnnouncementPriority,
)
from app.accessibility.talkback import (
    TalkBackController,
    TalkBackGesture,
    TalkBackSettings,
)
from app.accessibility.haptics import (
    HapticsController,
    HapticPattern,
    HapticsConfig,
    VibrationIntensity,
)

__all__ = [
    "Announcer",
    "AnnouncementConfig",
    "AnnouncementPriority",
    "TalkBackController",
    "TalkBackGesture",
    "TalkBackSettings",
    "HapticsController",
    "HapticPattern",
    "HapticsConfig",
    "VibrationIntensity",
]

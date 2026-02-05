"""
Accessibility Module
====================

Accessibility features for blind users, including:
    - announcer: Voice announcements and speech feedback
    - talkback: TalkBack integration and control
    - haptics: Haptic feedback patterns

These modules ensure the agent provides appropriate feedback
to users relying on screen readers and assistive technology.
"""

from app.accessibility.announcer import (
    Announcer,
    AnnouncementPriority,
    AnnouncementConfig,
)
from app.accessibility.talkback import (
    TalkBackController,
    TalkBackGesture,
    TalkBackSettings,
)
from app.accessibility.haptics import (
    HapticsController,
    HapticPattern,
    VibrationIntensity,
)

__all__ = [
    "Announcer",
    "AnnouncementPriority",
    "AnnouncementConfig",
    "TalkBackController",
    "TalkBackGesture",
    "TalkBackSettings",
    "HapticsController",
    "HapticPattern",
    "VibrationIntensity",
]

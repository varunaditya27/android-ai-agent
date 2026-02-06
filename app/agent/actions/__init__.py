"""
Agent Actions Module
====================

Action implementations for the Android AI Agent.

This package contains:
    - handler: Action dispatcher/executor
    - tap: Tap/click actions
    - swipe: Swipe and scroll actions
    - type_text: Text input actions
    - launch_app: App launching
    - system: System actions (back, home, etc.)
"""

from app.agent.actions.handler import ActionHandler, ActionExecutionResult
from app.agent.actions.tap import tap_element, tap_coordinates, find_tap_target
from app.agent.actions.swipe import swipe, scroll_up, scroll_down, SwipeDirection
from app.agent.actions.type_text import type_text, type_in_element
from app.agent.actions.launch_app import launch_app, resolve_package_name
from app.agent.actions.system import SystemActions, press_back, press_home

__all__ = [
    "ActionHandler",
    "ActionExecutionResult",
    "tap_element",
    "tap_coordinates",
    "find_tap_target",
    "swipe",
    "scroll_up",
    "scroll_down",
    "SwipeDirection",
    "type_text",
    "type_in_element",
    "launch_app",
    "resolve_package_name",
    "SystemActions",
    "press_back",
    "press_home",
]
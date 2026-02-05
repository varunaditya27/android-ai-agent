"""
Perception Module
=================

UI perception and understanding for Android screens.

This package contains:
    - ui_parser: Parse and format UI accessibility tree
    - element_detector: Hybrid element detection (tree + vision)
    - auth_detector: Login/authentication screen detection
    - ocr: Text recognition utilities
"""

from app.perception.ui_parser import UIElement, UIParser, ScreenType
from app.perception.element_detector import ElementDetector
from app.perception.auth_detector import AuthDetector, AuthType, AuthScreen

__all__ = [
    "UIElement",
    "UIParser",
    "ScreenType",
    "ElementDetector",
    "AuthDetector",
    "AuthType",
    "AuthScreen",
]

"""
Tap Action
==========

Tap/click action implementation with element targeting.
"""

from dataclasses import dataclass
from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.perception.ui_parser import UIElement


@dataclass
class TapTarget:
    """Target for a tap action."""

    x: int
    y: int
    element: Optional[UIElement] = None
    description: str = ""


async def tap_element(
    device: CloudDevice,
    element: UIElement,
) -> bool:
    """
    Tap on a UI element.

    Args:
        device: Cloud device to tap on.
        element: Element to tap.

    Returns:
        True if tap succeeded.
    """
    result = await device.tap(element.center_x, element.center_y)
    return result.success


async def tap_coordinates(
    device: CloudDevice,
    x: int,
    y: int,
) -> bool:
    """
    Tap at specific coordinates.

    Args:
        device: Cloud device to tap on.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        True if tap succeeded.
    """
    result = await device.tap(x, y)
    return result.success


async def double_tap(
    device: CloudDevice,
    x: int,
    y: int,
    delay_ms: int = 100,
) -> bool:
    """
    Perform a double tap.

    Args:
        device: Cloud device.
        x: X coordinate.
        y: Y coordinate.
        delay_ms: Delay between taps.

    Returns:
        True if both taps succeeded.
    """
    import asyncio

    result1 = await device.tap(x, y)
    if not result1.success:
        return False

    await asyncio.sleep(delay_ms / 1000)

    result2 = await device.tap(x, y)
    return result2.success


def find_tap_target(
    elements: list[UIElement],
    text: Optional[str] = None,
    element_type: Optional[str] = None,
    index: Optional[int] = None,
) -> Optional[TapTarget]:
    """
    Find a tap target by various criteria.

    Args:
        elements: List of UI elements.
        text: Text to match (partial).
        element_type: Element type to match.
        index: Direct element index.

    Returns:
        TapTarget if found, None otherwise.
    """
    if index is not None:
        for elem in elements:
            if elem.index == index and elem.clickable:
                return TapTarget(
                    x=elem.center_x,
                    y=elem.center_y,
                    element=elem,
                    description=elem.display_text,
                )
        return None

    for elem in elements:
        if not elem.clickable:
            continue

        # Match by text
        if text:
            elem_text = (elem.text + " " + elem.content_desc).lower()
            if text.lower() in elem_text:
                return TapTarget(
                    x=elem.center_x,
                    y=elem.center_y,
                    element=elem,
                    description=elem.display_text,
                )

        # Match by type
        if element_type and element_type.lower() == elem.element_type.lower():
            return TapTarget(
                x=elem.center_x,
                y=elem.center_y,
                element=elem,
                description=elem.display_text,
            )

    return None

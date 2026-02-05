"""
Swipe Action
============

Swipe and scroll action implementations.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

from app.device.cloud_provider import CloudDevice


class SwipeDirection(Enum):
    """Swipe direction constants."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass
class SwipeParams:
    """Parameters for a swipe gesture."""

    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration_ms: int = 300


def calculate_swipe_coords(
    direction: SwipeDirection,
    screen_width: int,
    screen_height: int,
    distance_percent: float = 0.5,
) -> SwipeParams:
    """
    Calculate swipe coordinates for a direction.

    Args:
        direction: Direction to swipe.
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        distance_percent: Percentage of screen to swipe.

    Returns:
        SwipeParams with calculated coordinates.
    """
    center_x = screen_width // 2
    center_y = screen_height // 2

    distance_x = int(screen_width * distance_percent / 2)
    distance_y = int(screen_height * distance_percent / 2)

    if direction == SwipeDirection.UP:
        return SwipeParams(
            start_x=center_x,
            start_y=center_y + distance_y,
            end_x=center_x,
            end_y=center_y - distance_y,
        )
    elif direction == SwipeDirection.DOWN:
        return SwipeParams(
            start_x=center_x,
            start_y=center_y - distance_y,
            end_x=center_x,
            end_y=center_y + distance_y,
        )
    elif direction == SwipeDirection.LEFT:
        return SwipeParams(
            start_x=center_x + distance_x,
            start_y=center_y,
            end_x=center_x - distance_x,
            end_y=center_y,
        )
    elif direction == SwipeDirection.RIGHT:
        return SwipeParams(
            start_x=center_x - distance_x,
            start_y=center_y,
            end_x=center_x + distance_x,
            end_y=center_y,
        )

    # Default to down
    return SwipeParams(
        start_x=center_x,
        start_y=center_y - distance_y,
        end_x=center_x,
        end_y=center_y + distance_y,
    )


async def swipe(
    device: CloudDevice,
    direction: SwipeDirection,
    distance_percent: float = 0.5,
) -> bool:
    """
    Perform a swipe in a direction.

    Args:
        device: Cloud device.
        direction: Direction to swipe.
        distance_percent: Percentage of screen to swipe.

    Returns:
        True if swipe succeeded.
    """
    if not device.info:
        return False

    params = calculate_swipe_coords(
        direction=direction,
        screen_width=device.info.screen_width,
        screen_height=device.info.screen_height,
        distance_percent=distance_percent,
    )

    result = await device.swipe(
        start_x=params.start_x,
        start_y=params.start_y,
        end_x=params.end_x,
        end_y=params.end_y,
        duration_ms=params.duration_ms,
    )

    return result.success


async def scroll_up(device: CloudDevice, distance: float = 0.4) -> bool:
    """Scroll up (swipe down gesture)."""
    return await swipe(device, SwipeDirection.DOWN, distance)


async def scroll_down(device: CloudDevice, distance: float = 0.4) -> bool:
    """Scroll down (swipe up gesture)."""
    return await swipe(device, SwipeDirection.UP, distance)


async def scroll_to_element(
    device: CloudDevice,
    target_text: str,
    max_scrolls: int = 5,
    get_elements_fn=None,
) -> bool:
    """
    Scroll until an element with target text is visible.

    Args:
        device: Cloud device.
        target_text: Text to find.
        max_scrolls: Maximum scroll attempts.
        get_elements_fn: Async function to get current elements.

    Returns:
        True if element found.
    """
    if not get_elements_fn:
        return False

    for _ in range(max_scrolls):
        elements = await get_elements_fn()

        # Check if target is visible
        target_lower = target_text.lower()
        for elem in elements:
            elem_text = (elem.text + " " + elem.content_desc).lower()
            if target_lower in elem_text:
                return True

        # Scroll down to see more
        await scroll_down(device)

    return False


def parse_direction(direction_str: str) -> SwipeDirection:
    """
    Parse direction string to SwipeDirection.

    Args:
        direction_str: Direction string (up, down, left, right).

    Returns:
        SwipeDirection enum value.
    """
    direction_map = {
        "up": SwipeDirection.UP,
        "down": SwipeDirection.DOWN,
        "left": SwipeDirection.LEFT,
        "right": SwipeDirection.RIGHT,
    }

    return direction_map.get(direction_str.lower(), SwipeDirection.DOWN)

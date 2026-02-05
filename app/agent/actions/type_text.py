"""
Type Text Action
================

Text input action implementation with special character handling.
"""

import asyncio
from typing import Optional

from app.device.cloud_provider import CloudDevice
from app.perception.ui_parser import UIElement


async def type_text(
    device: CloudDevice,
    text: str,
    clear_first: bool = False,
) -> bool:
    """
    Type text into the focused element.

    Args:
        device: Cloud device.
        text: Text to type.
        clear_first: Whether to clear existing text first.

    Returns:
        True if successful.
    """
    if clear_first:
        # Select all and delete
        await device.press_key("KEYCODE_MOVE_HOME")
        await asyncio.sleep(0.1)
        # Note: This is a simplification. Real implementation might need
        # to handle text clearing differently based on the device.

    result = await device.type_text(text)
    return result.success


async def type_in_element(
    device: CloudDevice,
    element: UIElement,
    text: str,
    clear_first: bool = False,
) -> bool:
    """
    Tap on an element and type text into it.

    Args:
        device: Cloud device.
        element: Element to type into.
        text: Text to type.
        clear_first: Whether to clear existing text.

    Returns:
        True if successful.
    """
    # First tap to focus the element
    tap_result = await device.tap(element.center_x, element.center_y)
    if not tap_result.success:
        return False

    # Wait for keyboard to appear
    await asyncio.sleep(0.5)

    # Type the text
    return await type_text(device, text, clear_first)


async def type_with_enter(
    device: CloudDevice,
    text: str,
) -> bool:
    """
    Type text and press enter.

    Useful for search fields that submit on enter.

    Args:
        device: Cloud device.
        text: Text to type.

    Returns:
        True if successful.
    """
    type_result = await device.type_text(text)
    if not type_result.success:
        return False

    await asyncio.sleep(0.2)

    enter_result = await device.press_key("KEYCODE_ENTER")
    return enter_result.success


def find_input_element(
    elements: list[UIElement],
    hint_text: Optional[str] = None,
) -> Optional[UIElement]:
    """
    Find an input/editable element.

    Args:
        elements: List of UI elements.
        hint_text: Optional hint text to match.

    Returns:
        Matching UIElement or None.
    """
    editable_elements = [e for e in elements if e.editable and e.enabled]

    if not editable_elements:
        return None

    if hint_text:
        hint_lower = hint_text.lower()
        for elem in editable_elements:
            elem_text = (elem.text + " " + elem.content_desc).lower()
            if hint_lower in elem_text:
                return elem

    # Return first editable element if no hint match
    return editable_elements[0] if editable_elements else None


def is_password_field(element: UIElement) -> bool:
    """
    Check if an element is a password field.

    Args:
        element: Element to check.

    Returns:
        True if it appears to be a password field.
    """
    text = (element.text + " " + element.content_desc + " " + element.resource_id).lower()
    password_indicators = ["password", "passcode", "pin", "secret"]
    return any(indicator in text for indicator in password_indicators)


def is_email_field(element: UIElement) -> bool:
    """
    Check if an element is an email field.

    Args:
        element: Element to check.

    Returns:
        True if it appears to be an email field.
    """
    text = (element.text + " " + element.content_desc + " " + element.resource_id).lower()
    email_indicators = ["email", "e-mail", "mail"]
    return any(indicator in text for indicator in email_indicators)

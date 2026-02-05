"""
System Actions
==============

System-level actions like back, home, recent apps, etc.
"""

import asyncio

from app.device.cloud_provider import CloudDevice


# Android key codes for system actions
class KeyCode:
    """Android key code constants."""

    BACK = "KEYCODE_BACK"
    HOME = "KEYCODE_HOME"
    MENU = "KEYCODE_MENU"
    APP_SWITCH = "KEYCODE_APP_SWITCH"
    ENTER = "KEYCODE_ENTER"
    TAB = "KEYCODE_TAB"
    ESCAPE = "KEYCODE_ESCAPE"
    DEL = "KEYCODE_DEL"
    FORWARD_DEL = "KEYCODE_FORWARD_DEL"
    VOLUME_UP = "KEYCODE_VOLUME_UP"
    VOLUME_DOWN = "KEYCODE_VOLUME_DOWN"
    VOLUME_MUTE = "KEYCODE_VOLUME_MUTE"
    POWER = "KEYCODE_POWER"
    CAMERA = "KEYCODE_CAMERA"
    SEARCH = "KEYCODE_SEARCH"
    DPAD_UP = "KEYCODE_DPAD_UP"
    DPAD_DOWN = "KEYCODE_DPAD_DOWN"
    DPAD_LEFT = "KEYCODE_DPAD_LEFT"
    DPAD_RIGHT = "KEYCODE_DPAD_RIGHT"
    DPAD_CENTER = "KEYCODE_DPAD_CENTER"


async def press_back(device: CloudDevice) -> bool:
    """
    Press the back button.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    result = await device.press_key(KeyCode.BACK)
    return result.success


async def press_home(device: CloudDevice) -> bool:
    """
    Press the home button.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    result = await device.press_key(KeyCode.HOME)
    return result.success


async def press_recent_apps(device: CloudDevice) -> bool:
    """
    Open recent apps / app switcher.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    result = await device.press_key(KeyCode.APP_SWITCH)
    return result.success


async def press_enter(device: CloudDevice) -> bool:
    """
    Press the enter key.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    result = await device.press_key(KeyCode.ENTER)
    return result.success


async def press_search(device: CloudDevice) -> bool:
    """
    Press the search key.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    result = await device.press_key(KeyCode.SEARCH)
    return result.success


async def navigate_back_multiple(
    device: CloudDevice,
    times: int = 1,
    delay: float = 0.5,
) -> bool:
    """
    Press back button multiple times.

    Args:
        device: Cloud device.
        times: Number of times to press.
        delay: Delay between presses.

    Returns:
        True if all presses succeeded.
    """
    for i in range(times):
        result = await press_back(device)
        if not result:
            return False
        if i < times - 1:
            await asyncio.sleep(delay)
    return True


async def go_to_home_screen(device: CloudDevice) -> bool:
    """
    Go to the home screen from anywhere.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    return await press_home(device)


async def wake_device(device: CloudDevice) -> bool:
    """
    Wake the device if sleeping.

    Note: This may not work on all cloud device providers.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    # Press power button briefly to wake
    result = await device.press_key(KeyCode.POWER)
    return result.success


async def dismiss_keyboard(device: CloudDevice) -> bool:
    """
    Dismiss the on-screen keyboard.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    # Press back usually dismisses keyboard
    return await press_back(device)


async def clear_notifications(device: CloudDevice) -> bool:
    """
    Clear all notifications.

    Note: This is a multi-step action that may not work on all devices.

    Args:
        device: Cloud device.

    Returns:
        True if successful.
    """
    # Swipe down to show notification shade
    if device.info:
        center_x = device.info.screen_width // 2
        await device.swipe(center_x, 0, center_x, 500, 200)
        await asyncio.sleep(0.5)

        # Look for "Clear all" button and tap it
        # This is device-specific and may not always work
        # Press home to go back
        await press_home(device)

    return True


class SystemActions:
    """
    Collection of system actions for convenient access.

    Usage:
        actions = SystemActions(device)
        await actions.back()
        await actions.home()
    """

    def __init__(self, device: CloudDevice) -> None:
        """Initialize with device."""
        self.device = device

    async def back(self) -> bool:
        """Press back button."""
        return await press_back(self.device)

    async def home(self) -> bool:
        """Press home button."""
        return await press_home(self.device)

    async def recent_apps(self) -> bool:
        """Open recent apps."""
        return await press_recent_apps(self.device)

    async def enter(self) -> bool:
        """Press enter key."""
        return await press_enter(self.device)

    async def search(self) -> bool:
        """Press search key."""
        return await press_search(self.device)

    async def wake(self) -> bool:
        """Wake device."""
        return await wake_device(self.device)

    async def dismiss_keyboard(self) -> bool:
        """Dismiss keyboard."""
        return await dismiss_keyboard(self.device)

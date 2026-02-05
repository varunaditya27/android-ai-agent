"""
Device Integration Module
=========================

Device abstraction layer supporting local emulator and cloud providers.

This package contains:
    - cloud_provider: Abstract base class for device control
    - adb_device: FREE local Android emulator/device via ADB
    - limrun_client: Limrun cloud device implementation (paid)
    - browserstack: BrowserStack cloud device implementation (paid)
    - screenshot: Screenshot capture and processing utilities

Recommended for FREE usage:
    Use 'adb' provider with Android Studio Emulator
"""

from app.device.cloud_provider import CloudDevice, DeviceInfo, create_cloud_device
from app.device.adb_device import ADBDevice, get_available_emulators, start_emulator
from app.device.limrun_client import LimrunDevice
from app.device.browserstack import BrowserStackDevice

__all__ = [
    "CloudDevice",
    "DeviceInfo",
    "create_cloud_device",
    "ADBDevice",
    "get_available_emulators",
    "start_emulator",
    "LimrunDevice",
    "BrowserStackDevice",
]

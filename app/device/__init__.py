"""
Device Integration Module
=========================

Device abstraction layer supporting local emulator and AWS Device Farm.

This package contains:
    - cloud_provider: Abstract base class for device control
    - adb_device: FREE local Android emulator/device via ADB
    - aws_device_farm: AWS Device Farm cloud device implementation
    - screenshot: Screenshot capture and processing utilities

Recommended for FREE usage:
    Use 'adb' provider with Android Studio Emulator

For cloud devices:
    Use 'aws_device_farm' with an AWS Device Farm project
"""

from app.device.cloud_provider import CloudDevice, DeviceInfo, create_cloud_device
from app.device.adb_device import ADBDevice, get_available_emulators, start_emulator
from app.device.aws_device_farm import AWSDeviceFarmDevice

__all__ = [
    "CloudDevice",
    "DeviceInfo",
    "create_cloud_device",
    "ADBDevice",
    "get_available_emulators",
    "start_emulator",
    "AWSDeviceFarmDevice",
]

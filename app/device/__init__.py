"""
Device Integration Module
=========================

Cloud device abstraction layer supporting multiple providers.

This package contains:
    - cloud_provider: Abstract base class for cloud devices
    - limrun_client: Limrun cloud device implementation
    - browserstack: BrowserStack cloud device implementation
    - screenshot: Screenshot capture and processing utilities
"""

from app.device.cloud_provider import CloudDevice, DeviceInfo, create_cloud_device
from app.device.limrun_client import LimrunDevice
from app.device.browserstack import BrowserStackDevice

__all__ = [
    "CloudDevice",
    "DeviceInfo",
    "create_cloud_device",
    "LimrunDevice",
    "BrowserStackDevice",
]

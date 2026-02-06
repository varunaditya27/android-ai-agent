#!/usr/bin/env python3
"""
Cloud Device Setup Script
=========================

Interactive script to set up and test cloud device connections.

Usage:
    python scripts/setup_cloud_device.py

This script will:
1. Check for required environment variables
2. Test connection to the cloud device provider
3. Allocate a test device
4. Capture a test screenshot
5. Release the device
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.device.cloud_provider import create_cloud_device
from app.utils.logger import setup_logging, get_logger


async def main() -> int:
    """Main setup function."""
    # Setup logging
    setup_logging(level="INFO", json_logs=False)
    logger = get_logger(__name__)

    print("=" * 60)
    print("Android AI Agent - Cloud Device Setup")
    print("=" * 60)
    print()

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        print("\nMake sure you have a .env file with the required configuration.")
        print("Copy .env.example to .env and fill in your API keys.")
        return 1

    # Check required settings
    print("Checking configuration...")

    if not settings.cloud_device.api_key:
        print("❌ CLOUD_DEVICE_API_KEY is not set")
        return 1
    print(f"✓ Cloud device API key configured")

    if not settings.cloud_device.base_url:
        print("⚠ CLOUD_DEVICE_BASE_URL not set, using default")

    print(f"✓ Provider: {settings.cloud_device.provider}")
    print()

    # Create device client
    print("Creating cloud device client...")
    try:
        device = create_cloud_device(
            provider=settings.cloud_device.provider,
            api_key=settings.cloud_device.api_key,
            base_url=settings.cloud_device.base_url or "",
        )
        print("✓ Device client created")
    except Exception as e:
        print(f"❌ Failed to create device client: {e}")
        return 1

    print()

    # Allocate device
    print("Allocating test device...")
    try:
        device_info = await device.allocate()
        print("✓ Device allocated!")
        print(f"  Device ID: {device_info.device_id}")
        print(f"  Device Name: {device_info.device_name}")
        print(f"  OS Version: {device_info.os_version}")
        print(f"  Screen: {device_info.screen_width}x{device_info.screen_height}")
    except Exception as e:
        print(f"❌ Failed to allocate device: {e}")
        return 1

    print()

    # Test screenshot
    print("Capturing test screenshot...")
    try:
        screenshot = await device.capture_screenshot()
        print(f"✓ Screenshot captured ({len(screenshot)} bytes)")
    except Exception as e:
        print(f"❌ Failed to capture screenshot: {e}")

    print()

    # Test UI hierarchy
    print("Getting UI hierarchy...")
    try:
        hierarchy = await device.get_ui_hierarchy()
        print(f"✓ UI hierarchy retrieved ({len(hierarchy)} chars)")
    except Exception as e:
        print(f"❌ Failed to get UI hierarchy: {e}")

    print()

    # Test tap action
    print("Testing tap action (center of screen)...")
    try:
        screen_info = await device.get_screen_info()
        cx = screen_info.get("width", 1080) // 2
        cy = screen_info.get("height", 2400) // 2
        result = await device.tap(cx, cy)
        if result.success:
            print(f"✓ Tap successful at ({cx}, {cy})")
        else:
            print(f"⚠ Tap returned: {result.message}")
    except Exception as e:
        print(f"❌ Failed to tap: {e}")

    print()

    # Release device
    print("Releasing device...")
    try:
        await device.release()
        print("✓ Device released")
    except Exception as e:
        print(f"❌ Failed to release device: {e}")

    print()
    print("=" * 60)
    print("Setup complete! Your cloud device connection is working.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

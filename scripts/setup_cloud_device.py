#!/usr/bin/env python3
"""
AWS Device Farm Setup & Test Script
====================================

Interactive script to verify AWS Device Farm configuration,
list available devices, and run a quick smoke test.

Usage:
    python scripts/setup_cloud_device.py

This script will:
1. Validate AWS credentials and Device Farm project ARN
2. List available Android devices
3. Optionally create a test remote access session
4. Capture a test screenshot
5. Stop the session
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.device.aws_device_farm import AWSDeviceFarmDevice
from app.utils.logger import setup_logging, get_logger


async def main() -> int:
    """Main setup function."""
    setup_logging(level="INFO", json_logs=False)
    logger = get_logger(__name__)

    print("=" * 60)
    print("Android AI Agent — AWS Device Farm Setup")
    print("=" * 60)
    print()

    # ── Load settings ─────────────────────────────────────────────
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        print("\nMake sure you have a .env file with required configuration.")
        print("See AWS_SETUP.md for details.")
        return 1

    # ── Validate configuration ────────────────────────────────────
    print("Checking configuration…")

    project_arn = settings.device.aws_device_farm_project_arn
    if not project_arn:
        print("❌ AWS_DEVICE_FARM_PROJECT_ARN is not set in .env")
        print("   See AWS_SETUP.md for how to create a Device Farm project.")
        return 1
    print(f"  ✓ Project ARN: {project_arn[:60]}…")

    device_arn = settings.device.aws_device_farm_device_arn
    if device_arn:
        print(f"  ✓ Device ARN:  {device_arn[:60]}…")
    else:
        print("  ℹ  No specific device ARN set — will auto-select")

    # Check AWS credentials
    aws_key = settings.device.aws_access_key_id
    if aws_key:
        print(f"  ✓ AWS access key: {aws_key[:4]}****")
    else:
        print("  ℹ  No explicit AWS credentials — using default credential chain")
        print("     (env vars AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or IAM role)")

    print()

    # ── Create device client ─────────────────────────────────────
    print("Creating AWS Device Farm client…")
    try:
        device = AWSDeviceFarmDevice(
            project_arn=project_arn,
            device_arn=device_arn or None,
        )
        print("  ✓ Client created")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return 1

    # ── List available devices ────────────────────────────────────
    print()
    print("Listing available Android devices…")
    try:
        devices = await device.list_available_devices()
        if not devices:
            print("  ⚠  No Android devices found.")
            print("     Make sure your AWS account has Device Farm access.")
            return 1

        print(f"  Found {len(devices)} Android device(s):\n")
        print(f"  {'#':<4} {'Name':<30} {'OS':<8} {'Resolution':<14} {'Available':<12} {'Remote'}")
        print(f"  {'─'*4} {'─'*30} {'─'*8} {'─'*14} {'─'*12} {'─'*6}")

        for i, d in enumerate(devices[:20], 1):
            remote = "✓" if d["remoteAccessEnabled"] else "✗"
            print(
                f"  {i:<4} {d['name'][:29]:<30} {d['os']:<8} "
                f"{d['resolution']:<14} {d['availability']:<12} {remote}"
            )

        if len(devices) > 20:
            print(f"\n  … and {len(devices) - 20} more")

    except Exception as e:
        print(f"❌ Failed to list devices: {e}")
        print("   Check your AWS credentials and permissions.")
        return 1

    # ── Optional: smoke test ──────────────────────────────────────
    print()
    try:
        answer = input("Run a quick connection smoke test? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer != "y":
        print("\nSetup verification complete (skipped smoke test).")
        print("Your AWS Device Farm configuration looks good! ✓")
        return 0

    print()
    print("Starting remote access session (this may take 1-3 minutes)…")

    try:
        connected = await device.connect()
        if not connected:
            print("❌ Failed to connect to device")
            return 1

        info = device.info
        print(f"  ✓ Connected to {info.model if info else 'device'}")
        if info:
            print(f"    Android {info.os_version}")
            print(f"    Screen: {info.screen_width}x{info.screen_height}")

        # Screenshot
        print()
        print("  Capturing test screenshot…")
        try:
            screenshot = await device.capture_screenshot()
            size_kb = len(screenshot) * 3 // 4 // 1024  # approx decoded size
            print(f"  ✓ Screenshot captured (~{size_kb} KB)")
        except Exception as e:
            print(f"  ⚠  Screenshot failed: {e}")

        # UI hierarchy
        print("  Getting UI hierarchy…")
        try:
            hierarchy = await device.get_ui_hierarchy()
            elem_count = len(hierarchy.get("elements", []))
            print(f"  ✓ UI hierarchy: {elem_count} elements")
        except Exception as e:
            print(f"  ⚠  UI hierarchy failed: {e}")

        # Tap center
        if info:
            cx, cy = info.screen_width // 2, info.screen_height // 2
            print(f"  Tapping center ({cx}, {cy})…")
            result = await device.tap(cx, cy)
            print(f"  {'✓' if result.success else '⚠'} Tap {'succeeded' if result.success else 'failed'}")

    except Exception as e:
        print(f"❌ Smoke test error: {e}")
    finally:
        print()
        print("  Stopping remote access session…")
        try:
            await device.disconnect()
            print("  ✓ Session stopped")
        except Exception as e:
            print(f"  ⚠  Cleanup error: {e}")

    print()
    print("=" * 60)
    print("Setup complete! AWS Device Farm is working. ✓")
    print("=" * 60)
    print()
    print("To use AWS Device Farm with the agent, set in .env:")
    print("  DEVICE_PROVIDER=aws_device_farm")
    print(f"  AWS_DEVICE_FARM_PROJECT_ARN={project_arn}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

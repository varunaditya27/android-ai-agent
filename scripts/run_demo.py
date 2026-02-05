#!/usr/bin/env python3
"""
Demo Script
===========

Interactive demo of the Android AI Agent.

This demo uses:
- Google Gemini for LLM (FREE tier available)
- Local Android Emulator via ADB (FREE)

Prerequisites:
    1. Get a Gemini API key: https://aistudio.google.com/apikey
    2. Install Android Studio and create an emulator
    3. Start the emulator before running this script

Usage:
    python scripts/run_demo.py

    # With specific task
    python scripts/run_demo.py --task "Open YouTube and search for music"

    # Non-interactive mode
    python scripts/run_demo.py --task "Open Settings" --no-interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import ReActAgent, AgentConfig, StepResult
from app.config import get_settings
from app.device import create_cloud_device, get_available_emulators
from app.llm.client import LLMClient
from app.llm.models import LLMConfig
from app.utils.logger import setup_logging, get_logger


class DemoRunner:
    """Interactive demo runner."""

    def __init__(self, settings):
        """Initialize demo runner."""
        self.settings = settings
        self.logger = get_logger(__name__)
        self.device = None
        self.agent = None

    async def setup(self) -> bool:
        """Set up device and agent."""
        print("\n🚀 Setting up Android AI Agent...")
        print("   Using FREE tools: Google Gemini + Local Emulator\n")

        # Create device client (ADB for local emulator)
        try:
            provider = self.settings.device.device_provider
            print(f"📱 Device provider: {provider}")
            
            if provider in ("adb", "local", "emulator"):
                # Check for available emulators
                emulators = get_available_emulators()
                if emulators:
                    print(f"   Available emulators: {', '.join(emulators)}")
                
            self.device = await create_cloud_device(
                provider=provider,
                device_id=self.settings.device.adb_device_serial or None,
            )

            # Connect to device
            print("📱 Connecting to device...")
            connected = await self.device.connect()
            
            if not connected:
                print("❌ Failed to connect to device")
                print("\n💡 Tips:")
                print("   - Make sure Android Emulator is running")
                print("   - Or connect a physical device via USB with USB debugging enabled")
                print("   - Run 'adb devices' to check connected devices")
                return False
                
            print(f"   ✓ Connected to {self.device.info.model if self.device.info else 'device'}")
            if self.device.info:
                print(f"   ✓ Android {self.device.info.os_version}")
                print(f"   ✓ Screen: {self.device.info.screen_width}x{self.device.info.screen_height}")

        except Exception as e:
            print(f"❌ Failed to setup device: {e}")
            print("\n💡 Make sure:")
            print("   - Android SDK is installed (adb in PATH)")
            print("   - Emulator is running or device is connected")
            return False

        # Create LLM client (Gemini)
        try:
            llm_config = LLMConfig(
                api_key=self.settings.llm.gemini_api_key,
                model=self.settings.llm.llm_model,
                max_output_tokens=self.settings.llm.llm_max_output_tokens,
                temperature=self.settings.llm.llm_temperature,
                top_p=self.settings.llm.llm_top_p,
                top_k=self.settings.llm.llm_top_k,
            )
            llm_client = LLMClient(llm_config)
            print(f"🤖 Using Gemini model: {self.settings.llm.llm_model}")

        except Exception as e:
            print(f"❌ Failed to setup LLM: {e}")
            print("\n💡 Make sure:")
            print("   - GEMINI_API_KEY is set in .env file")
            print("   - Get free API key: https://aistudio.google.com/apikey")
            return False

        # Create agent
        self.agent = ReActAgent(
            llm_client=llm_client,
            device=self.device,
            config=AgentConfig(
                max_steps=self.settings.agent.max_steps,
                verbose=True,
            ),
            on_step=self._on_step,
            on_input_required=self._on_input_required,
        )

        print("✓ Agent ready!\n")
        return True

    async def cleanup(self):
        """Clean up resources."""
        if self.device:
            print("\n📱 Disconnecting device...")
            await self.device.disconnect()
            print("✓ Device disconnected")

    def _on_step(self, step: StepResult):
        """Callback for each step."""
        status = "✓" if step.success else "✗"
        print(f"\n{status} Step: {step.action_type}")
        
        thinking_display = step.thinking[:100] + "..." if len(step.thinking) > 100 else step.thinking
        print(f"   💭 {thinking_display}")

        if step.error:
            print(f"   ❌ Error: {step.error}")

        if step.finished:
            print(f"\n🎉 Task finished: {step.action_message}")

    def _on_input_required(self, prompt: str) -> str:
        """Handle input requests (for authentication flows)."""
        print(f"\n⌨️  Input required: {prompt}")
        print("   (This is used for entering credentials, OTPs, etc.)")
        return input("   Enter value: ")

    async def run_task(self, task: str) -> bool:
        """Run a task."""
        print(f"\n📋 Task: {task}")
        print("-" * 50)

        try:
            result = await self.agent.run(task)

            print("\n" + "=" * 50)
            if result.success:
                print("✅ TASK COMPLETED SUCCESSFULLY")
                print(f"   Result: {result.result}")
            else:
                print("❌ TASK FAILED")
                print(f"   Error: {result.error}")

            print(f"   Steps: {result.steps_taken}")
            print(f"   Duration: {result.duration_seconds:.1f}s")
            print("=" * 50)

            return result.success

        except Exception as e:
            print(f"\n❌ Task execution error: {e}")
            self.logger.exception("Task failed", error=str(e))
            return False


async def interactive_mode(runner: DemoRunner):
    """Run in interactive mode."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Type your tasks or commands:")
    print("  - Type a task description to execute it")
    print("  - Type 'exit' or 'quit' to stop")
    print("  - Type 'help' for example tasks")
    print("=" * 50)

    example_tasks = [
        "Open YouTube and search for relaxing music",
        "Open Settings and turn on WiFi",
        "Open Chrome and go to google.com",
        "Open the Calculator app",
        "Take a screenshot",
        "Open the Camera app",
    ]

    while True:
        try:
            task = input("\n🎯 Enter task: ").strip()

            if not task:
                continue

            if task.lower() in ("exit", "quit", "q"):
                print("Goodbye! 👋")
                break

            if task.lower() == "help":
                print("\nExample tasks:")
                for i, t in enumerate(example_tasks, 1):
                    print(f"  {i}. {t}")
                print("\nTip: Just type a number to run that task!")
                continue

            # Check for numbered shortcut
            if task.isdigit():
                idx = int(task) - 1
                if 0 <= idx < len(example_tasks):
                    task = example_tasks[idx]
                    print(f"Running: {task}")
                else:
                    print("Invalid number. Type 'help' for examples.")
                    continue

            await runner.run_task(task)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Android AI Agent Demo (FREE - uses Gemini + Local Emulator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py
  python run_demo.py --task "Open YouTube"
  python run_demo.py --task "Search for news" --no-interactive

Setup (all FREE):
  1. Get Gemini API key: https://aistudio.google.com/apikey
  2. Install Android Studio and create an emulator
  3. Start the emulator
  4. Run this script!
        """,
    )
    parser.add_argument(
        "--task",
        help="Task to execute (skips interactive prompt)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run single task and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level, json_logs=False)

    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║           Android AI Agent - Demo                     ║
    ║   AI-powered mobile automation for accessibility      ║
    ║                                                       ║
    ║   🆓 100% FREE: Gemini API + Local Emulator          ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Copied .env.example to .env")
        print("   2. Set GEMINI_API_KEY in .env")
        print("\nGet a FREE API key: https://aistudio.google.com/apikey")
        return 1

    # Check for API key
    if not settings.llm.gemini_api_key or settings.llm.gemini_api_key == "your-gemini-api-key":
        print("❌ GEMINI_API_KEY not configured")
        print("\n💡 Steps to fix:")
        print("   1. Go to https://aistudio.google.com/apikey")
        print("   2. Create a FREE API key")
        print("   3. Set GEMINI_API_KEY in your .env file")
        return 1

    # Create runner
    runner = DemoRunner(settings)

    try:
        # Setup
        if not await runner.setup():
            return 1

        # Run mode
        if args.task:
            # Single task mode
            success = await runner.run_task(args.task)

            if not args.no_interactive:
                # Continue to interactive mode
                await interactive_mode(runner)

            return 0 if success else 1
        else:
            # Interactive mode
            await interactive_mode(runner)
            return 0

    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)

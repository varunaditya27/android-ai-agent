#!/usr/bin/env python3
"""
Demo Script
===========

Interactive demo of the Android AI Agent.

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
from app.device.cloud_provider import create_cloud_device
from app.llm.client import LLMClient
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

        # Create device client
        try:
            self.device = create_cloud_device(
                provider=self.settings.cloud_device.provider,
                api_key=self.settings.cloud_device.api_key,
                base_url=self.settings.cloud_device.base_url or "",
            )

            # Allocate device
            print("📱 Allocating cloud device...")
            device_info = await self.device.allocate()
            print(f"   ✓ Connected to {device_info.device_name}")

        except Exception as e:
            print(f"❌ Failed to setup device: {e}")
            return False

        # Create LLM client
        try:
            llm_client = LLMClient(
                api_key=self.settings.llm.api_key,
                model=self.settings.llm.model_name,
                base_url=self.settings.llm.api_base,
            )
            print(f"🤖 Using model: {self.settings.llm.model_name}")

        except Exception as e:
            print(f"❌ Failed to setup LLM: {e}")
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
            print("\n📱 Releasing device...")
            await self.device.release()
            print("✓ Device released")

    def _on_step(self, step: StepResult):
        """Callback for each step."""
        status = "✓" if step.success else "✗"
        print(f"\n{status} Step: {step.action_type}")
        print(f"   💭 {step.thinking[:100]}..." if len(step.thinking) > 100 else f"   💭 {step.thinking}")

        if step.error:
            print(f"   ❌ Error: {step.error}")

        if step.finished:
            print(f"\n🎉 Task finished: {step.action_message}")

    def _on_input_required(self, prompt: str) -> str:
        """Handle input requests."""
        print(f"\n⌨️  Input required: {prompt}")
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
        "Send a text message to John saying 'Hello!'",
        "Check the weather for today",
        "Open Instagram and like the first post",
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
        description="Android AI Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py
  python run_demo.py --task "Open YouTube"
  python run_demo.py --task "Search for news" --no-interactive
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
    ╚═══════════════════════════════════════════════════════╝
    """)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure you have configured your .env file.")
        return 1

    # Check for API keys
    if not settings.llm.api_key:
        print("❌ LLM_API_KEY not configured")
        return 1

    if not settings.cloud_device.api_key:
        print("❌ CLOUD_DEVICE_API_KEY not configured")
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

#!/usr/bin/env python3
"""
Demo Script
===========

Interactive demo of the Android AI Agent.

This demo supports two LLM providers:
- **Groq** (recommended, FREE): Llama 4 Scout vision — 1,000 req/day
- **Gemini**: Google AI with optional key rotation

Prerequisites:
    1. Set LLM_PROVIDER=groq (or gemini) in .env
    2. For Groq: set GROQ_API_KEY in .env (get free → https://console.groq.com/keys)
       For Gemini: set GEMINI_API_KEY or GEMINI_API_KEYS in .env
    3. Start an Android emulator OR configure AWS Device Farm

Usage:
    python scripts/run_demo.py

    # With specific task
    python scripts/run_demo.py --task "Open YouTube and search for music"

    # Non-interactive mode
    python scripts/run_demo.py --task "Open Settings" --no-interactive
"""

import argparse
import asyncio
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import ReActAgent, AgentConfig, StepResult
from app.config import get_settings
from app.device import create_cloud_device, get_available_emulators
from app.llm.client import LLMClient, RateLimitError
from app.llm.groq_client import GroqLLMClient
from app.llm.groq_client import RateLimitError as GroqRateLimitError
from app.llm.key_rotator import ApiKeyRotator
from app.llm.models import LLMConfig
from app.accessibility.manager import AccessibilityManager
from app.utils.logger import setup_logging, get_logger


# ── Graceful shutdown ──────────────────────────────────────────────
_shutdown_requested = False


def _handle_signal(sig, frame):
    """Handle SIGINT / SIGTERM cleanly."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second interrupt → force exit
        sys.exit(1)
    _shutdown_requested = True
    print(f"\n\nInterrupted -- shutting down gracefully.")


class DemoRunner:
    """Interactive demo runner with robust error handling."""

    def __init__(self, settings):
        """Initialize demo runner."""
        self.settings = settings
        self.logger = get_logger(__name__)
        self.device = None
        self.agent = None
        self.llm_client = None
        self.a11y: Optional[AccessibilityManager] = None

    @property
    def _sr(self) -> bool:
        """True when screen-reader mode is active."""
        return (
            self.a11y is not None
            and self.a11y.announcer.config.screen_reader_mode
        )

    async def setup(self) -> bool:
        """Set up device and agent.  Returns True on success."""
        # Note: _sr is False here because a11y isn't created yet.
        # Screen-reader output begins after the Accessibility block below.
        print("\nSetting up Android AI Agent...")
        print("   Using FREE tools: Google Gemini + Local Emulator\n")

        # ── Device ────────────────────────────────────────────────
        try:
            provider = self.settings.device.device_provider
            print(f"Device provider: {provider}")

            if provider in ("adb", "local", "emulator"):
                try:
                    emulators = get_available_emulators()
                    if emulators:
                        print(f"   Available emulators: {', '.join(emulators)}")
                except Exception:
                    pass  # non-critical
            elif provider == "aws_device_farm":
                print("   Using AWS Device Farm (session may take 1-3 min to start)")

            self.device = await create_cloud_device(
                provider=provider,
                device_id=self.settings.device.adb_device_serial or None,
            )

            print("Connecting to device...")
            connected = await self.device.connect()

            if not connected:
                print("FAILED: Could not connect to device")
                print("\nTips:")
                print("   - Make sure Android Emulator is running")
                print("   - Or connect a physical device via USB with USB debugging enabled")
                print("   - Run 'adb devices' to check connected devices")
                return False

            info = self.device.info
            print(f"   ✓ Connected to {info.model if info else 'device'}")
            if info:
                print(f"   ✓ Android {info.os_version}")
                print(f"   ✓ Screen: {info.screen_width}x{info.screen_height}")

        except FileNotFoundError:
            print("FAILED: 'adb' command not found")
            print("\nInstall Android SDK platform-tools and ensure 'adb' is in your PATH.")
            return False
        except Exception as e:
            print(f"FAILED: Device setup error: {e}")
            if provider == "aws_device_farm":
                print("\nMake sure:")
                print("   - AWS credentials are configured (aws configure)")
                print("   - AWS_DEVICE_FARM_PROJECT_ARN is set in .env")
                print("   - Your AWS account has Device Farm access")
            else:
                print("\nMake sure:")
                print("   - Android SDK is installed (adb in PATH)")
                print("   - Emulator is running or device is connected")
                print("   - Run 'adb devices' to check connected devices")
            return False

        # ── LLM ───────────────────────────────────────────────────
        try:
            provider = self.settings.llm.llm_provider

            if provider == "groq":
                # Groq: Llama 4 Scout vision (free, 1000 RPD)
                groq_key = self.settings.llm.groq_api_key
                groq_model = self.settings.llm.groq_model
                llm_config = LLMConfig(
                    api_key=groq_key,
                    model=groq_model,
                    max_output_tokens=self.settings.llm.llm_max_output_tokens,
                    temperature=self.settings.llm.llm_temperature,
                    top_p=self.settings.llm.llm_top_p,
                    top_k=self.settings.llm.llm_top_k,
                )
                self.llm_client = GroqLLMClient(llm_config)
                print(f"LLM: Groq model: {groq_model}")
                print(f"   Free tier: 1,000 requests/day, 30 req/min")
                print(f"   Max output tokens: {self.settings.llm.llm_max_output_tokens}")
            else:
                # Gemini: Google AI (with key rotation)
                all_api_keys = self.settings.llm.get_all_api_keys()
                rotator = ApiKeyRotator(all_api_keys) if len(all_api_keys) > 1 else None

                llm_config = LLMConfig(
                    api_key=all_api_keys[0],
                    model=self.settings.llm.llm_model,
                    max_output_tokens=self.settings.llm.llm_max_output_tokens,
                    temperature=self.settings.llm.llm_temperature,
                    top_p=self.settings.llm.llm_top_p,
                    top_k=self.settings.llm.llm_top_k,
                )
                self.llm_client = LLMClient(llm_config, key_rotator=rotator)
                print(f"LLM: Gemini model: {self.settings.llm.llm_model}")
                print(f"   Max output tokens: {self.settings.llm.llm_max_output_tokens}")
                if rotator:
                    print(f"   Key rotation enabled: {rotator.total_keys} API keys")
                else:
                    print("   Using single API key")
        except ValueError as e:
            print(f"FAILED: LLM configuration error: {e}")
            if self.settings.llm.llm_provider == "groq":
                print("\nMake sure GROQ_API_KEY is set in your .env file.")
                print("   Get a free key → https://console.groq.com/keys")
            else:
                print("\nMake sure GEMINI_API_KEY or GEMINI_API_KEYS is set in your .env file.")
                print("   Get a free key: https://aistudio.google.com/apikey")
            return False
        except Exception as e:
            print(f"FAILED: LLM setup error: {e}")
            return False

        # ── Accessibility ──────────────────────────────────────────
        a11y_cfg = self.settings.accessibility
        screen_reader = a11y_cfg.screen_reader_mode if a11y_cfg.enable_accessibility else False
        self.a11y = AccessibilityManager(
            device=self.device,
            enable_tts=a11y_cfg.enable_tts and a11y_cfg.enable_accessibility,
            tts_rate=a11y_cfg.tts_rate,
            tts_volume=a11y_cfg.tts_volume,
            enable_haptics=a11y_cfg.enable_haptics and a11y_cfg.enable_accessibility,
            enable_talkback=a11y_cfg.enable_talkback and a11y_cfg.enable_accessibility,
            high_contrast=a11y_cfg.enable_high_contrast,
            large_text=a11y_cfg.enable_large_text,
            screen_reader_mode=screen_reader,
        )
        await self.a11y.setup()

        if a11y_cfg.enable_accessibility:
            features = []
            if a11y_cfg.enable_tts:
                features.append("TTS")
            if a11y_cfg.enable_haptics:
                features.append("Haptics")
            if a11y_cfg.enable_talkback:
                features.append("TalkBack")
            if screen_reader:
                features.append("Screen-reader output")
            sr_prefix = "" if screen_reader else "   "
            if screen_reader:
                print(f"Accessibility enabled: {', '.join(features)}")
            else:
                print(f"♿ Accessibility enabled: {', '.join(features)}")

        # ── Agent ─────────────────────────────────────────────────
        self.agent = ReActAgent(
            llm_client=self.llm_client,
            device=self.device,
            config=AgentConfig(
                max_steps=self.settings.agent.max_steps,
                verbose=True,
                min_step_interval=self.settings.agent.min_step_interval,
                rate_limit_max_retries=self.settings.agent.rate_limit_max_retries,
                enable_vision=self.settings.agent.enable_vision,
                enable_accessibility_tree=self.settings.agent.enable_accessibility_tree,
            ),
            on_step=self._on_step,
            on_input_required=self._on_input_required,
            accessibility=self.a11y,
        )

        if screen_reader:
            print("Agent ready.\n")
        else:
            print("✓ Agent ready!\n")
        return True

    async def cleanup(self):
        """Clean up resources."""
        if self.device:
            print("\nDisconnecting device...")
            try:
                await self.device.disconnect()
                print("Done. Device disconnected.")
            except Exception:
                pass  # best-effort

    def _on_step(self, step: StepResult):
        """Display each step to the user."""
        sr = (
            self.a11y is not None
            and self.a11y.announcer.config.screen_reader_mode
        )

        # Rate-limit wait notifications
        if step.action_type == "WAIT_RATE_LIMIT":
            if sr:
                print(f"\n   {step.error}")
            else:
                print(f"\n   ⏳ {step.error}")
            return

        status = "OK" if step.success else "FAIL"
        if not sr:
            status = "✓" if step.success else "✗"
        print(f"\n{status} Step: {step.action_type}")

        if step.thinking:
            thinking_display = (
                step.thinking[:120] + "…" if len(step.thinking) > 120 else step.thinking
            )
            if sr:
                print(f"   Thinking: {thinking_display}")
            else:
                print(f"   💭 {thinking_display}")

        if step.error:
            # Shorten huge API error dumps for readability
            short_error = step.error
            if len(short_error) > 200:
                short_error = short_error[:200] + "…"
            if sr:
                print(f"   Error: {short_error}")
            else:
                print(f"   ❌ Error: {short_error}")

        if step.finished:
            if sr:
                print(f"\nTask finished: {step.action_message}")
            else:
                print(f"\n🎉 Task finished: {step.action_message}")

    def _on_input_required(self, prompt: str) -> str:
        """Handle input requests (for authentication flows)."""
        sr = (
            self.a11y is not None
            and self.a11y.announcer.config.screen_reader_mode
        )
        if sr:
            print(f"\n   Input required: {prompt}")
            print("   (Used for entering credentials, OTPs, etc.)")
        else:
            print(f"\n⌨️  Input required: {prompt}")
            print("   (This is used for entering credentials, OTPs, etc.)")
        return input("   Enter value: ")

    async def run_task(self, task: str) -> bool:
        """Run a single task with full error handling."""
        sr = (
            self.a11y is not None
            and self.a11y.announcer.config.screen_reader_mode
        )

        if sr:
            print(f"\nTask: {task}")
            print("-" * 50)
        else:
            print(f"\n📋 Task: {task}")
            print("─" * 50)

        try:
            result = await self.agent.run(task)

            print("\n" + "=" * 50)
            if result.success:
                if sr:
                    print("TASK COMPLETED SUCCESSFULLY")
                else:
                    print("✅ TASK COMPLETED SUCCESSFULLY")
                print(f"   Result: {result.result}")
            else:
                if sr:
                    print("TASK FAILED")
                else:
                    print("❌ TASK FAILED")
                error_display = result.error or "Unknown error"
                if len(error_display) > 200:
                    error_display = error_display[:200] + "…"
                print(f"   Error: {error_display}")

            print(f"   Steps: {result.steps_taken}")
            print(f"   API Calls: {result.api_calls}")
            print(f"   Duration: {result.duration_seconds:.1f}s")
            print("=" * 50)

            return result.success

        except (RateLimitError, GroqRateLimitError) as e:
            provider_name = "Groq" if self.settings.llm.llm_provider == "groq" else "Gemini"
            print(f"\nRate limited by {provider_name} API.")
            print(f"   The API asks to wait ~{round(e.retry_after)}s.")
            print("   You can retry the task in a moment, or use a different API key.")
            return False

        except KeyboardInterrupt:
            print("\n\nTask interrupted by user.")
            return False

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            self.logger.error("Task failed", error=str(e))
            if self.settings.server.debug:
                traceback.print_exc()
            return False


async def interactive_mode(runner: DemoRunner):
    """Run in interactive mode with a prompt loop."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Type your tasks or commands:")
    print("  - Type a task description to execute it")
    print("  - Type 'exit' or 'quit' to stop")
    print("  - Type 'help' for example tasks")
    print("=" * 50)

    example_tasks = [
        "Open ChatGPT and ask what's the capital of France",
        "Open YouTube and search for relaxing music",
        "Open Gmail and check my latest email",
        "Open Chrome and go to google.com and search for AI news",
        "Open Settings and check the battery level",
        "Open the Calculator app and compute 127 * 43",
    ]

    while True:
        try:
            task = input("\nEnter task: ").strip()

            if not task:
                continue

            if task.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if task.lower() == "help":
                print("\nExample tasks:")
                for i, t in enumerate(example_tasks, 1):
                    print(f"  {i}. {t}")
                print("\nTip: Just type a number to run that task!")
                continue

            # Numbered shortcut
            if task.isdigit():
                idx = int(task) - 1
                if 0 <= idx < len(example_tasks):
                    task = example_tasks[idx]
                    print(f"Running: {task}")
                else:
                    print("Invalid number. Type 'help' for examples.")
                    continue

            await runner.run_task(task)

        except (KeyboardInterrupt, EOFError):
            print("\n\nInterrupted. Goodbye!")
            break


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Android AI Agent Demo — Groq / Gemini + ADB / AWS Device Farm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py
  python run_demo.py --task "Open YouTube"
  python run_demo.py --task "Search for news" --no-interactive

Setup:
  1. Set LLM_PROVIDER=groq (recommended) or LLM_PROVIDER=gemini in .env
  2a. (Groq)   Get free key: https://console.groq.com/keys → set GROQ_API_KEY
  2b. (Gemini) Get free key: https://aistudio.google.com/apikey → set GEMINI_API_KEY
  3a. (Local)  Start an Android emulator  — DEVICE_PROVIDER=adb
  3b. (Cloud)  Configure AWS Device Farm — DEVICE_PROVIDER=aws_device_farm
  4. Run this script!
        """,
    )
    parser.add_argument("--task", help="Task to execute (skips interactive prompt)")
    parser.add_argument(
        "--no-interactive", action="store_true", help="Run single task and exit"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level, json_logs=False)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Print banner
    print(
        """
    ╔════════════════════════════════════════════════════════╗
    ║                Android AI Agent — Demo                 ║
    ║    AI-powered mobile automation for accessibility      ║
    ║                                                        ║
    ║      Groq / Gemini  •  ADB / AWS Device Farm           ║
    ╚════════════════════════════════════════════════════════╝
    """
    )

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("\nMake sure you have:")
        print("   1. Copied .env.example to .env")
        print("   2. Set GEMINI_API_KEY in .env")
        print("\nGet a FREE API key: https://aistudio.google.com/apikey")
        return 1

    # Validate API key(s) based on selected provider
    llm_provider = settings.llm.llm_provider
    placeholder_keys = ("your-gemini-api-key", "your-groq-api-key", "")

    if llm_provider == "groq":
        groq_key = settings.llm.groq_api_key
        if not groq_key or groq_key in placeholder_keys:
            print("No Groq API key configured")
            print("\nSteps to fix:")
            print("   1. Go to https://console.groq.com/keys")
            print("   2. Create a FREE API key")
            print("   3. Set GROQ_API_KEY in your .env file")
            print("   4. (Optional) Set LLM_PROVIDER=groq in .env")
            return 1
    else:
        all_keys = settings.llm.get_all_api_keys()
        valid_keys = [k for k in all_keys if k not in placeholder_keys]
        if not valid_keys:
            print("No Gemini API key configured")
            print("\nSteps to fix:")
            print("   1. Go to https://aistudio.google.com/apikey")
            print("   2. Create a FREE API key")
            print("   3. Set GEMINI_API_KEY (single) or GEMINI_API_KEYS (comma-separated) in .env")
            print("   Tip: Keys from different Google Cloud projects multiply your quota!")
            return 1

    # Create & run
    runner = DemoRunner(settings)

    try:
        if not await runner.setup():
            return 1

        if args.task:
            success = await runner.run_task(args.task)
            if not args.no_interactive:
                await interactive_mode(runner)
            return 0 if success else 1
        else:
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
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
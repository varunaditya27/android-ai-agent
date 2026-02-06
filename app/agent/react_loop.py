"""
ReAct Loop Implementation
=========================

Core ReAct (Reasoning + Acting) agent loop for Android automation.

The agent follows this cycle:
1. Observe: Capture screenshot and UI hierarchy
2. Think: LLM analyzes state and decides action
3. Act: Execute the chosen action
4. Repeat until task complete or max steps reached

Usage:
    from app.agent import ReActAgent, AgentConfig

    agent = ReActAgent(
        llm_client=llm,
        device=device,
        config=AgentConfig(max_steps=30),
    )

    result = await agent.run("Open YouTube and search for music")
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.agent.actions.handler import ActionHandler, ActionExecutionResult
from app.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agent.state import AgentState, TaskStatus
from app.device.cloud_provider import CloudDevice
from app.device.screenshot import resize_for_llm
from app.llm.response_parser import ActionType, parse_response, format_action_for_log
from app.perception.auth_detector import AuthDetector
from app.perception.element_detector import ElementDetector
from app.perception.ui_parser import UIParser
from app.utils.logger import get_logger

logger = get_logger(__name__)

# LLM error types — imported from both clients to keep the loop provider-agnostic.
# Both Gemini (client.py) and Groq (groq_client.py) define compatible
# LLMError and RateLimitError classes with the same interface.
try:
    from app.llm.client import LLMError, RateLimitError
except ImportError:
    # Fallback if gemini client isn't available
    from app.llm.groq_client import LLMError, RateLimitError  # type: ignore[assignment]


@dataclass
class AgentConfig:
    """
    Configuration for the ReAct agent.

    Attributes:
        max_steps: Maximum steps before giving up.
        step_timeout: Timeout for each step in seconds.
        max_consecutive_errors: Errors before failing.
        screenshot_quality: Screenshot quality (1-100).
        enable_vision: Use vision for element detection.
        enable_accessibility_tree: Use accessibility tree.
        action_delay: Delay between actions in seconds.
        verbose: Enable verbose logging.
    """

    max_steps: int = 30
    step_timeout: float = 30.0
    max_consecutive_errors: int = 5
    screenshot_quality: int = 85
    enable_vision: bool = False  # disabled: rely on accessibility tree only (saves 1 API call/step)
    enable_accessibility_tree: bool = True
    action_delay: float = 0.5
    verbose: bool = True
    min_step_interval: float = 3.0  # 3s = safe for Groq 30 RPM; increase to 12s for Gemini 5 RPM
    rate_limit_max_retries: int = 5  # retries per step on rate-limit


@dataclass
class StepResult:
    """
    Result of a single ReAct step.

    Attributes:
        success: Whether the step succeeded.
        finished: Whether the task is complete.
        thinking: Agent's reasoning.
        action_type: Type of action taken.
        action_message: Result message from action.
        requires_input: Whether user input is needed.
        input_prompt: Prompt for user input.
        error: Error message if failed.
    """

    success: bool
    finished: bool = False
    thinking: str = ""
    action_type: str = ""
    action_message: str = ""
    requires_input: bool = False
    input_prompt: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TaskResult:
    """
    Final result of task execution.

    Attributes:
        success: Whether the task completed successfully.
        result: The final result/answer.
        steps_taken: Number of steps executed.
        duration_seconds: Total time taken.
        error: Error message if failed.
        history: Full step history.
        api_calls: Number of LLM API calls made.
    """

    success: bool
    result: str
    steps_taken: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    history: list[dict] = field(default_factory=list)
    api_calls: int = 0


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for mobile automation.

    Combines LLM reasoning with device actions to accomplish
    user tasks through iterative observation and action.
    """

    def __init__(
        self,
        llm_client,
        device: CloudDevice,
        config: Optional[AgentConfig] = None,
        on_step: Optional[Callable[[StepResult], None]] = None,
        on_input_required: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize the ReAct agent.

        Args:
            llm_client: LLM client for reasoning (LLMClient or GroqLLMClient).
                        Must implement complete_with_vision(), get_api_call_count(),
                        and reset_api_call_count().
            device: Cloud device to control.
            config: Agent configuration.
            on_step: Callback for each step completion.
            on_input_required: Callback for user input requests.
        """
        self.llm = llm_client
        self.device = device
        self.config = config or AgentConfig()

        # Initialize components
        self.ui_parser = UIParser()
        self.element_detector = ElementDetector(llm_client, self.ui_parser)
        self.auth_detector = AuthDetector()
        self.action_handler = ActionHandler(
            device,
            action_delay=self.config.action_delay,
        )

        # Callbacks
        self.on_step = on_step
        self.on_input_required = on_input_required

        # State
        self.state = AgentState()

        # Rate-limit tracking
        self._last_llm_call_time: float = 0.0

        logger.info(
            "ReActAgent initialized",
            max_steps=self.config.max_steps,
            enable_vision=self.config.enable_vision,
        )

    async def run(self, task: str) -> TaskResult:
        """
        Execute a task using the ReAct loop.

        Args:
            task: Natural language task description.

        Returns:
            TaskResult with outcome and details.
        """
        logger.info("Starting task", task=task)
        self.state.start_task(task)
        
        # Reset API call counter for this task
        self.llm.reset_api_call_count()

        try:
            while self.state.current_step < self.config.max_steps:
                # Check for too many consecutive errors
                if self.state.error_count >= self.config.max_consecutive_errors:
                    error_msg = f"Too many consecutive errors ({self.state.error_count})"
                    logger.error(error_msg)
                    self.state.fail(error_msg)
                    break

                # --- Rate-limit pacing: wait between LLM calls ---
                now = time.time()
                elapsed = now - self._last_llm_call_time
                if elapsed < self.config.min_step_interval:
                    wait = self.config.min_step_interval - elapsed
                    await asyncio.sleep(wait)
                self._last_llm_call_time = time.time()

                # Execute one step (with rate-limit retry)
                step_result = await self._execute_step_with_retry()

                # Notify callback
                if self.on_step:
                    self.on_step(step_result)

                # Check if task is finished
                if step_result.finished:
                    break

                # Handle input requests
                if step_result.requires_input:
                    if self.on_input_required:
                        user_input = self.on_input_required(step_result.input_prompt or "")
                        self.state.provide_input(user_input)

                        # Type the input
                        type_result = await self.device.type_text(user_input)
                        if not type_result.success:
                            logger.warning("Failed to type user input")
                        else:
                            # Auto-press Enter to submit (works for most password/OTP fields)
                            await asyncio.sleep(0.3)
                            enter_result = await self.device.press_key("KEYCODE_ENTER")
                            if enter_result.success:
                                logger.info("Pressed Enter after input")
                                # Wait for screen to update after submit
                                await asyncio.sleep(1.5)
                            else:
                                # If Enter doesn't work, set flag so next step knows to find submit button
                                self.state.input_needs_submit = True
                                # Try to hide keyboard so submit button becomes visible
                                await self.device.press_key("KEYCODE_BACK")
                                await asyncio.sleep(0.5)
                    else:
                        # No input handler - fail the task
                        self.state.fail("User input required but no handler provided")
                        break

                # Brief pause between steps
                await asyncio.sleep(0.1)

            # Task completed or max steps reached
            if self.state.status == TaskStatus.RUNNING:
                self.state.fail("Maximum steps reached without completion")

        except asyncio.CancelledError:
            self.state.cancel()
            logger.info("Task cancelled")
        except Exception as e:
            logger.exception("Task failed with exception", error=str(e))
            self.state.fail(str(e))

        # Log API call count
        api_calls = self.llm.get_api_call_count()
        logger.info("Task completed", api_calls_made=api_calls, steps_taken=self.state.current_step)

        # Build result
        return TaskResult(
            success=self.state.status == TaskStatus.COMPLETED,
            result=self.state.result or "",
            steps_taken=self.state.current_step,
            duration_seconds=self.state.duration_seconds,
            error=self.state.result if self.state.status == TaskStatus.FAILED else None,
            history=[step.to_dict() for step in self.state.history],
            api_calls=api_calls,
        )

    async def _execute_step_with_retry(self) -> StepResult:
        """
        Execute a step with automatic retry on rate-limit errors.

        Wraps _execute_step to catch RateLimitError, wait, and retry
        up to config.rate_limit_max_retries times before giving up.
        """
        for attempt in range(1, self.config.rate_limit_max_retries + 1):
            result = await self._execute_step()

            # If the step error is a rate-limit, wait and retry
            if (
                not result.success
                and result.error
                and ("RESOURCE_EXHAUSTED" in result.error or "429" in result.error)
            ):
                if attempt < self.config.rate_limit_max_retries:
                    # Try to extract delay from the error message
                    import re as _re
                    delay = 30.0
                    match = _re.search(r"retry in ([\d.]+)s", result.error, _re.IGNORECASE)
                    if match:
                        try:
                            delay = float(match.group(1))
                        except ValueError:
                            pass

                    logger.warning(
                        "Rate limited during step, waiting before retry",
                        attempt=attempt,
                        max_retries=self.config.rate_limit_max_retries,
                        wait_seconds=round(delay, 1),
                    )
                    # Notify callback so the user sees the wait
                    if self.on_step:
                        wait_result = StepResult(
                            success=False,
                            finished=False,
                            thinking=f"Rate limited — waiting {round(delay)}s before retrying…",
                            action_type="WAIT_RATE_LIMIT",
                            error=f"Rate limited (attempt {attempt}/{self.config.rate_limit_max_retries}). Waiting {round(delay)}s…",
                        )
                        self.on_step(wait_result)
                    await asyncio.sleep(delay)
                    # Update the LLM call timestamp
                    self._last_llm_call_time = time.time()
                    continue

            return result

        # All retries exhausted – return the last (failed) result
        return result

    async def _execute_step(self) -> StepResult:
        """
        Execute a single ReAct step.

        Returns:
            StepResult with step outcome.
        """
        step_start = time.time()

        try:
            # 1. OBSERVE: Capture current state
            screenshot_b64 = await self.device.capture_screenshot()
            ui_hierarchy = await self.device.get_ui_hierarchy()

            # Resize screenshot for LLM
            screenshot_for_llm = resize_for_llm(screenshot_b64)

            # Detect UI elements from accessibility tree only (no extra LLM call)
            elements = await self.element_detector.detect_elements(
                screenshot_for_llm,
                ui_hierarchy,
                use_vision_fallback=False,
            )

            self.state.last_screenshot = screenshot_b64
            self.state.last_elements = elements

            # Get current app
            self.state.current_app = await self.device.get_current_app()

            # Check for auth screens
            auth_screen = self.auth_detector.detect_auth(elements)

            # 2. THINK: Ask LLM for next action
            # Detect loops before building prompt
            loop_warning = self.state.detect_action_loop(lookback=5)
            additional_context = self._build_additional_context(auth_screen, loop_warning)
            
            user_prompt = build_user_prompt(
                task=self.state.task,
                elements=elements,
                history_summary=self.state.get_history_summary(),
                current_app=self.state.current_app,
                additional_context=additional_context,
                progress_status=self.state.progress_status,
                current_step=self.state.current_step + 1,
                max_steps=self.config.max_steps,
            )

            llm_response = await self.llm.complete_with_vision(
                prompt=user_prompt,
                image_data=screenshot_for_llm,
                system_prompt=SYSTEM_PROMPT,
            )

            # Parse response
            parsed = parse_response(llm_response.content)

            if self.config.verbose:
                logger.debug(
                    "LLM response",
                    thinking=parsed.thinking[:100] + "..." if len(parsed.thinking) > 100 else parsed.thinking,
                    action=format_action_for_log(parsed.action),
                )

            # 3. ACT: Execute the action
            action_result = await self.action_handler.execute(parsed.action, elements)

            # Calculate step duration
            duration_ms = int((time.time() - step_start) * 1000)

            # Record step
            self.state.record_step(
                thinking=parsed.thinking,
                action_type=parsed.action.action_type.name,
                action_params=parsed.action.params,
                success=action_result.success,
                error=action_result.error,
                screenshot_b64=screenshot_b64,
                duration_ms=duration_ms,
            )

            # Check for task completion
            if parsed.action.action_type == ActionType.FINISH:
                result_message = parsed.action.params.get("message", "Task completed")
                self.state.complete(result_message)

                return StepResult(
                    success=True,
                    finished=True,
                    thinking=parsed.thinking,
                    action_type=parsed.action.action_type.name,
                    action_message=result_message,
                )

            # Check for input request
            if action_result.requires_input:
                self.state.request_input(action_result.input_prompt or "")

                return StepResult(
                    success=True,
                    finished=False,
                    thinking=parsed.thinking,
                    action_type=parsed.action.action_type.name,
                    action_message=action_result.message,
                    requires_input=True,
                    input_prompt=action_result.input_prompt,
                )

            return StepResult(
                success=action_result.success,
                finished=False,
                thinking=parsed.thinking,
                action_type=parsed.action.action_type.name,
                action_message=action_result.message,
                error=action_result.error,
            )

        except asyncio.TimeoutError:
            logger.warning("Step timed out", timeout=self.config.step_timeout)
            return StepResult(
                success=False,
                finished=False,
                error="Step timed out",
            )
        except Exception as e:
            logger.exception("Step failed", error=str(e))
            
            # Record failed step to increment step counter and prevent infinite loops
            # Note: screenshot_b64 might not be defined if it failed during capture
            img_b64 = screenshot_b64 if 'screenshot_b64' in locals() else None
            
            self.state.record_step(
                thinking="Step execution encountered a system error.",
                action_type="SYSTEM_ERROR",
                action_params={},
                success=False,
                error=f"System Error: {str(e)}",
                screenshot_b64=img_b64,
            )
            
            return StepResult(
                success=False,
                finished=False,
                error=str(e),
            )

    def _build_additional_context(self, auth_screen: Any, loop_warning: Optional[str] = None) -> str:
        """Build additional context for the prompt.
        
        Args:
            auth_screen: Detected authentication screen info.
            loop_warning: Warning message if agent is in a loop.
            
        Returns:
            Formatted context string with warnings and hints.
        """
        context_parts = []

        # HIGHEST PRIORITY: Loop detection
        if loop_warning:
            context_parts.append(f"🚨 {loop_warning}")

        # HIGH PRIORITY: Submit button after credentials
        if self.state.input_needs_submit:
            context_parts.append(
                "⚠️ IMPORTANT: Credentials were just entered. Now you MUST find and tap the submit/continue/next button to proceed.\n"
                "Look for buttons with text like: 'Next', 'Continue', 'Submit', 'Sign in', 'Login', 'Done', or arrow icons."
            )
            self.state.input_needs_submit = False  # Reset flag

        # Authentication detection
        if auth_screen:
            auth_type_hints = {
                "LOGIN": "login screen with email and password fields",
                "PASSWORD_ONLY": "password entry screen",
                "OTP": "OTP/verification code screen",
                "TWO_FACTOR": "two-factor authentication screen",
                "REGISTER": "registration/signup screen",
            }
            hint = auth_type_hints.get(auth_screen.auth_type.name, "authentication screen")
            context_parts.append(
                f"🔐 AUTHENTICATION DETECTED: This is a {hint}.\n"
                f"CRITICAL: You MUST use RequestInput to ask the user for credentials.\n"
                f"NEVER type passwords, PINs, or codes directly - they must come from the user via RequestInput.\n"
                f"Example: do(action=\"RequestInput\", prompt=\"Please enter your password\")"
            )

        # Recent errors
        if self.state.error_count >= 2:
            context_parts.append(
                f"⚠️ WARNING: {self.state.error_count} consecutive errors. "
                "Your current approach is not working. Try a completely different strategy."
            )
        elif self.state.error_count > 0:
            context_parts.append(
                f"ℹ️ Note: {self.state.error_count} recent error. Consider adjusting your approach."
            )

        return "\n\n".join(context_parts) if context_parts else ""

    def get_state(self) -> dict[str, Any]:
        """Get current agent state as dictionary."""
        return self.state.to_dict()

    async def provide_input(self, value: str) -> None:
        """
        Provide requested user input.

        Args:
            value: The input value from user.
        """
        self.state.provide_input(value)

    def cancel(self) -> None:
        """Cancel the current task."""
        self.state.cancel()
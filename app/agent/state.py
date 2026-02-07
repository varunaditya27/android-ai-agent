"""
Agent State Management
======================

Tracks agent state across multi-step task execution.

Maintains:
- Task context and goal
- Action history
- Current UI state
- Error tracking
- Progress indicators
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


class TaskStatus(Enum):
    """Status of the current task."""

    PENDING = auto()
    RUNNING = auto()
    WAITING_INPUT = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class StepRecord:
    """
    Record of a single agent step.

    Attributes:
        step_number: Step number in the sequence.
        timestamp: When the step occurred.
        thinking: Agent's reasoning for this step.
        action_type: Type of action taken.
        action_params: Parameters of the action.
        success: Whether the action succeeded.
        error: Error message if failed.
        screenshot_b64: Optional screenshot after action.
        duration_ms: How long the step took.
    """

    step_number: int
    timestamp: datetime
    thinking: str
    action_type: str
    action_params: dict[str, Any]
    success: bool
    error: Optional[str] = None
    screenshot_b64: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp.isoformat(),
            "thinking": self.thinking,
            "action_type": self.action_type,
            "action_params": self.action_params,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentState:
    """
    Maintains the state of an agent during task execution.

    Tracks the entire context needed for the agent to make decisions
    and provides history for debugging and analysis.

    Attributes:
        task: The original user task/goal.
        status: Current task status.
        current_step: Current step number.
        history: List of step records.
        current_app: Currently active app package name.
        last_screenshot: Most recent screenshot (base64).
        last_elements: Most recent UI elements.
        error_count: Number of consecutive errors.
        input_required: Whether user input is currently needed.
        input_prompt: Prompt for required input.
        result: Final task result if completed.
        started_at: When the task started.
        completed_at: When the task completed.
    """

    task: str = ""
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    history: list[StepRecord] = field(default_factory=list)
    current_app: str = ""
    last_screenshot: Optional[str] = None
    last_elements: list[Any] = field(default_factory=list)
    error_count: int = 0
    input_required: bool = False
    input_prompt: Optional[str] = None
    input_needs_submit: bool = False  # Flag: credentials entered but need to tap submit button
    progress_status: str = ""  # Track what has been accomplished toward the goal
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _screen_fingerprints: list[str] = field(default_factory=list)  # per-step screen hashes

    def start_task(self, task: str) -> None:
        """
        Initialize state for a new task.

        Args:
            task: The user's task description.
        """
        self.task = task
        self.status = TaskStatus.RUNNING
        self.current_step = 0
        self.history = []
        self.error_count = 0
        self.input_required = False
        self.input_prompt = None
        self.progress_status = "No progress yet."
        self.result = None
        self.started_at = datetime.now()
        self.completed_at = None
        self._screen_fingerprints = []

    def record_step(
        self,
        thinking: str,
        action_type: str,
        action_params: dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        screenshot_b64: Optional[str] = None,
        duration_ms: int = 0,
    ) -> StepRecord:
        """
        Record a completed step.

        Args:
            thinking: Agent's reasoning.
            action_type: Type of action taken.
            action_params: Action parameters.
            success: Whether action succeeded.
            error: Error message if failed.
            screenshot_b64: Screenshot after action.
            duration_ms: Step duration.

        Returns:
            The created StepRecord.
        """
        self.current_step += 1

        record = StepRecord(
            step_number=self.current_step,
            timestamp=datetime.now(),
            thinking=thinking,
            action_type=action_type,
            action_params=action_params,
            success=success,
            error=error,
            screenshot_b64=screenshot_b64,
            duration_ms=duration_ms,
        )

        self.history.append(record)

        # Track consecutive errors
        if success:
            self.error_count = 0
        else:
            self.error_count += 1

        # Update screenshot
        if screenshot_b64:
            self.last_screenshot = screenshot_b64

        return record

    def request_input(self, prompt: str) -> None:
        """
        Set state to waiting for user input.

        Args:
            prompt: Prompt to show the user.
        """
        self.status = TaskStatus.WAITING_INPUT
        self.input_required = True
        self.input_prompt = prompt

    def provide_input(self, value: str) -> None:
        """
        Provide requested input and resume.

        Args:
            value: The input value from user.
        """
        self.status = TaskStatus.RUNNING
        self.input_required = False
        self.input_prompt = None
        self.metadata["last_input"] = value

    def complete(self, result: str) -> None:
        """
        Mark task as completed.

        Args:
            result: The final result message.
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """
        Mark task as failed.

        Args:
            error: The error message.
        """
        self.status = TaskStatus.FAILED
        self.result = error
        self.completed_at = datetime.now()

    def cancel(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """Check if task has finished (success, failure, or cancelled)."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        if not self.started_at:
            return 0.0

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def last_step(self) -> Optional[StepRecord]:
        """Get the most recent step record."""
        return self.history[-1] if self.history else None

    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps."""
        if not self.history:
            return 0.0
        successes = sum(1 for step in self.history if step.success)
        return successes / len(self.history)

    def get_recent_history(self, n: int = 5) -> list[StepRecord]:
        """
        Get the N most recent steps.

        Args:
            n: Number of steps to return.

        Returns:
            List of recent StepRecords.
        """
        return self.history[-n:] if self.history else []

    def get_history_summary(self) -> str:
        """
        Get a text summary of recent history for LLM context.

        Returns:
            Formatted history string.
        """
        if not self.history:
            return "No previous actions taken."

        lines = ["Recent action history:"]
        # Show last 12 steps for better context — helps LLM understand
        # how it got to the current screen and avoid repeating mistakes
        for step in self.get_recent_history(12):
            status = "Successful" if step.success else "Failed"
            # Include action parameters for clarity
            action_desc = f"{step.action_type}"
            if step.action_params:
                # Show key params (app, element_id, text preview, direction)
                if "app" in step.action_params:
                    action_desc += f" app={step.action_params['app']}"
                if "element_id" in step.action_params:
                    action_desc += f" element={step.action_params['element_id']}"
                if "text" in step.action_params:
                    text_preview = step.action_params['text'][:30]
                    action_desc += f" text='{text_preview}...'"
                if "direction" in step.action_params:
                    action_desc += f" direction={step.action_params['direction']}"
            
            lines.append(
                f"  Step {step.step_number}: Action: {action_desc} | Outcome: {status}"
            )
            if step.error:
                # Show enough of the error for the LLM to understand the problem
                error_preview = step.error[:120]
                if len(step.error) > 120:
                    error_preview += "..."
                lines.append(f"    Error: {error_preview}")

        return "\n".join(lines)

    def record_screen_fingerprint(self, elements: list[Any]) -> None:
        """Record a fingerprint of the current screen's element set.
        
        Call this each step with the detected UI elements so
        detect_action_loop can check for repeated screens.
        """
        # Build a fingerprint from element texts and types (order-independent)
        parts = []
        for elem in elements:
            text = getattr(elem, 'display_text', '') or ''
            etype = getattr(elem, 'element_type', '') or ''
            parts.append(f"{etype}:{text}")
        parts.sort()
        self._screen_fingerprints.append("|".join(parts))

    def _action_signature(self, step: "StepRecord") -> str:
        """Create a signature string from action type + key params for loop comparison."""
        sig = step.action_type
        if step.action_params:
            # Normalize: treat both 'element' and 'element_id' as the same key
            el_id = step.action_params.get("element_id") or step.action_params.get("element")
            if el_id is not None:
                sig += f":el={el_id}"
            elif "x" in step.action_params and "y" in step.action_params:
                sig += f":xy=({step.action_params['x']},{step.action_params['y']})"
            if "app" in step.action_params:
                sig += f":app={step.action_params['app']}"
            if "direction" in step.action_params:
                sig += f":dir={step.action_params['direction']}"
        return sig

    def detect_action_loop(self, lookback: int = 6) -> Optional[str]:
        """
        Detect if agent is repeating similar actions in a loop.

        Uses action *signatures* (type + params like element_id) so that
        tapping two different elements in alternation is caught even though
        both are TAP actions.

        Args:
            lookback: Number of recent steps to check.

        Returns:
            Warning message if loop detected, None otherwise.
        """
        if len(self.history) < 3:
            return None

        recent = self.get_recent_history(lookback)
        signatures = [self._action_signature(step) for step in recent]

        # --- 1. Repeated failures on same action (most specific, check first) ---
        recent_failures = [s for s in recent if not s.success]
        if len(recent_failures) >= 3:
            failure_sigs = [self._action_signature(s) for s in recent_failures]
            if len(set(failure_sigs)) == 1:
                return (
                    f"LOOP DETECTED: Action '{failure_sigs[0]}' has failed "
                    f"{len(recent_failures)} times. Strongly consider a "
                    f"completely different action or navigation path."
                )

        # --- 2. Ping-pong: alternating between 2 specific actions (A-B-A-B) ---
        if len(signatures) >= 4:
            if (
                signatures[-1] == signatures[-3]
                and signatures[-2] == signatures[-4]
                and signatures[-1] != signatures[-2]
            ):
                return (
                    f"LOOP DETECTED: You are alternating between '{signatures[-1]}' "
                    f"and '{signatures[-2]}' without making progress. "
                    f"STOP repeating these two actions. Carefully re-read ALL the "
                    f"elements on the current screen and pick a DIFFERENT element "
                    f"or action that actually moves you toward the goal."
                )

        # --- 3. Exact same action (type+params) repeated 3+ times ---
        if len(signatures) >= 3 and len(set(signatures[-3:])) == 1:
            return (
                f"LOOP DETECTED: You have performed the EXACT same action "
                f"'{signatures[-1]}' {sum(1 for s in signatures if s == signatures[-1])} times. "
                f"This action is NOT making progress. You MUST try a completely "
                f"different element or approach. Look at ALL elements on the screen "
                f"— especially ones you have NOT interacted with yet."
            )

        # --- 4. Small action vocabulary over many steps (stuck in a rut) ---
        if len(signatures) >= 5:
            unique_sigs = set(signatures)
            if len(unique_sigs) <= 2:
                return (
                    f"LOOP DETECTED: Over the last {len(signatures)} steps you have only used "
                    f"{len(unique_sigs)} distinct action(s): {', '.join(unique_sigs)}. "
                    f"You are stuck. STOP and think about what OTHER elements or "
                    f"approaches could achieve the goal. Read every element on screen carefully."
                )

        # --- 5. Same screen appearing repeatedly (screen fingerprint) ---
        if len(self._screen_fingerprints) >= 4:
            recent_fps = self._screen_fingerprints[-6:]  # last 6 screen fingerprints
            # Count how many times the most recent screen appeared
            current_fp = recent_fps[-1]
            occurrences = sum(1 for fp in recent_fps if fp == current_fp)
            if occurrences >= 3 and current_fp:  # same screen seen 3+ times in last 6 steps
                return (
                    f"LOOP DETECTED: You are seeing the SAME SCREEN for the {occurrences}th time. "
                    f"Your actions keep bringing you back to this exact screen. "
                    f"The approach you're using is fundamentally wrong. You MUST: \n"
                    f"1. Identify what this screen is actually asking you to do\n"
                    f"2. Fulfill that requirement (e.g., tap 'Add an email address' instead of dismissing the dialog)\n"
                    f"3. If no element on this screen helps, try going Back or Home"
                )

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "task": self.task,
            "status": self.status.name,
            "current_step": self.current_step,
            "history": [step.to_dict() for step in self.history],
            "current_app": self.current_app,
            "error_count": self.error_count,
            "input_required": self.input_required,
            "input_prompt": self.input_prompt,
            "progress_status": self.progress_status,
            "result": self.result,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }

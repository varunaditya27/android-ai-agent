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
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
        self.result = None
        self.started_at = datetime.now()
        self.completed_at = None

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

        lines = ["Recent actions:"]
        for step in self.get_recent_history(3):
            status = "✓" if step.success else "✗"
            lines.append(
                f"  {status} Step {step.step_number}: {step.action_type} - {step.thinking[:50]}..."
            )

        return "\n".join(lines)

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
            "result": self.result,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }

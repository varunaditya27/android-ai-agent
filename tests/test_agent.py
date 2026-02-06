"""
Tests for ReAct Agent Core
==========================

Tests for:
- AgentState management
- ReAct loop execution
- Task completion and failure handling
- Input request handling
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.state import AgentState, StepRecord, TaskStatus
from app.agent.prompts import (
    build_user_prompt,
    format_elements_for_prompt,
    build_auth_prompt,
)
from app.agent.react_loop import ReActAgent, AgentConfig, StepResult, TaskResult
from app.llm.response_parser import ActionType, ParsedAction


class TestAgentState:
    """Tests for AgentState class."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentState()
        assert state.task == ""
        assert state.status == TaskStatus.PENDING
        assert state.current_step == 0
        assert state.history == []
        assert state.error_count == 0
        assert not state.is_running
        assert not state.is_finished

    def test_start_task(self):
        """Test starting a new task."""
        state = AgentState()
        state.start_task("Open YouTube")

        assert state.task == "Open YouTube"
        assert state.status == TaskStatus.RUNNING
        assert state.current_step == 0
        assert state.started_at is not None
        assert state.is_running
        assert not state.is_finished

    def test_record_step_success(self):
        """Test recording a successful step."""
        state = AgentState()
        state.start_task("Test task")

        record = state.record_step(
            thinking="Found the button",
            action_type="Tap",
            action_params={"element_id": 5},
            success=True,
        )

        assert state.current_step == 1
        assert len(state.history) == 1
        assert record.step_number == 1
        assert record.success is True
        assert record.action_type == "Tap"
        assert state.error_count == 0

    def test_record_step_failure(self):
        """Test recording a failed step."""
        state = AgentState()
        state.start_task("Test task")

        state.record_step(
            thinking="Trying to tap",
            action_type="Tap",
            action_params={"element_id": 5},
            success=False,
            error="Element not found",
        )

        assert state.error_count == 1

        # Multiple failures
        state.record_step(
            thinking="Retry",
            action_type="Tap",
            action_params={"element_id": 5},
            success=False,
            error="Element not found",
        )

        assert state.error_count == 2

    def test_error_count_reset_on_success(self):
        """Test error count resets after success."""
        state = AgentState()
        state.start_task("Test task")

        # Fail twice
        state.record_step("", "Tap", {}, False, "Error")
        state.record_step("", "Tap", {}, False, "Error")
        assert state.error_count == 2

        # Succeed once
        state.record_step("", "Tap", {}, True)
        assert state.error_count == 0

    def test_complete_task(self):
        """Test task completion."""
        state = AgentState()
        state.start_task("Test task")
        state.complete("Task done successfully")

        assert state.status == TaskStatus.COMPLETED
        assert state.result == "Task done successfully"
        assert state.completed_at is not None
        assert state.is_finished
        assert not state.is_running

    def test_fail_task(self):
        """Test task failure."""
        state = AgentState()
        state.start_task("Test task")
        state.fail("Something went wrong")

        assert state.status == TaskStatus.FAILED
        assert state.result == "Something went wrong"
        assert state.is_finished

    def test_cancel_task(self):
        """Test task cancellation."""
        state = AgentState()
        state.start_task("Test task")
        state.cancel()

        assert state.status == TaskStatus.CANCELLED
        assert state.is_finished

    def test_request_input(self):
        """Test requesting user input."""
        state = AgentState()
        state.start_task("Test task")
        state.request_input("Enter your password")

        assert state.status == TaskStatus.WAITING_INPUT
        assert state.input_required is True
        assert state.input_prompt == "Enter your password"

    def test_provide_input(self):
        """Test providing user input."""
        state = AgentState()
        state.start_task("Test task")
        state.request_input("Enter password")
        state.provide_input("secret123")

        assert state.status == TaskStatus.RUNNING
        assert state.input_required is False
        assert state.metadata.get("last_input") == "secret123"

    def test_get_recent_history(self):
        """Test getting recent history."""
        state = AgentState()
        state.start_task("Test task")

        for i in range(10):
            state.record_step(f"Step {i}", "Tap", {"i": i}, True)

        recent = state.get_recent_history(3)
        assert len(recent) == 3
        assert recent[0].step_number == 8
        assert recent[2].step_number == 10

    def test_success_rate(self):
        """Test success rate calculation."""
        state = AgentState()
        state.start_task("Test task")

        # 3 successes, 1 failure
        state.record_step("", "Tap", {}, True)
        state.record_step("", "Tap", {}, True)
        state.record_step("", "Tap", {}, False)
        state.record_step("", "Tap", {}, True)

        assert state.success_rate == 0.75

    def test_to_dict(self):
        """Test serialization to dict."""
        state = AgentState()
        state.start_task("Test task")
        state.record_step("Thinking", "Tap", {"x": 100}, True)

        data = state.to_dict()

        assert data["task"] == "Test task"
        assert data["status"] == "RUNNING"
        assert data["current_step"] == 1
        assert len(data["history"]) == 1


class TestPrompts:
    """Tests for prompt building functions."""

    def test_build_user_prompt_basic(self):
        """Test basic user prompt building."""
        from app.perception.ui_parser import UIElement

        elements = [
            UIElement(
                index=0,
                element_type="Button",
                display_text="Submit",
                clickable=True,
                bounds=(100, 100, 200, 150),
            ),
        ]

        prompt = build_user_prompt(
            task="Click the submit button",
            elements=elements,
        )

        assert "Click the submit button" in prompt
        assert "Submit" in prompt
        assert "Button" in prompt

    def test_format_elements_for_prompt(self):
        """Test element formatting."""
        from app.perception.ui_parser import UIElement

        elements = [
            UIElement(
                index=0,
                element_type="Button",
                display_text="Submit",
                clickable=True,
                editable=False,
                bounds=(100, 100, 200, 150),
            ),
            UIElement(
                index=1,
                element_type="EditText",
                display_text="",
                clickable=True,
                editable=True,
                bounds=(100, 200, 500, 250),
            ),
        ]

        formatted = format_elements_for_prompt(elements)

        assert "[0]" in formatted
        assert "[1]" in formatted
        assert "clickable" in formatted
        assert "editable/input" in formatted

    def test_format_elements_empty(self):
        """Test empty elements formatting."""
        formatted = format_elements_for_prompt([])
        assert "No interactive elements" in formatted

    def test_build_auth_prompt(self):
        """Test auth prompt building."""
        prompt = build_auth_prompt("login", "email")
        assert "email" in prompt.lower()

        prompt = build_auth_prompt("otp", "otp")
        assert "verification" in prompt.lower() or "code" in prompt.lower()


class TestReActAgent:
    """Tests for ReActAgent class."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_device, mock_llm_client):
        """Test agent initialization."""
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=10),
        )

        assert agent.config.max_steps == 10
        assert agent.state.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_agent_run_simple_task(self, mock_device, mock_llm_client):
        """Test running a simple task."""
        # Setup LLM to return finish action
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=MagicMock(
                content='<think>Task is done.</think>\n<answer>finish(message="Done!")</answer>'
            )
        )

        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=5),
        )

        result = await agent.run("Test task")

        assert result.success is True
        assert result.result == "Done!"
        assert result.steps_taken >= 1

    @pytest.mark.asyncio
    async def test_agent_max_steps_reached(self, mock_device, mock_llm_client):
        """Test agent stops at max steps."""
        # Setup LLM to always return tap action (never finishes)
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=MagicMock(
                content='<think>Tapping button.</think>\n<answer>do(action="Tap", element_id=0)</answer>'
            )
        )

        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=3),
        )

        result = await agent.run("Endless task")

        assert result.success is False
        assert "Maximum steps" in result.error or "Maximum steps" in result.result
        assert result.steps_taken == 3

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_device, mock_llm_client):
        """Test agent handles errors gracefully."""
        # Setup device to fail
        mock_device.tap = AsyncMock(
            return_value=MagicMock(success=False, message="", error="Element not found")
        )

        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=MagicMock(
                content='<think>Tapping.</think>\n<answer>do(action="Tap", element_id=0)</answer>'
            )
        )

        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=10, max_consecutive_errors=3),
        )

        result = await agent.run("Failing task")

        assert result.success is False
        assert "error" in result.error.lower() or "error" in result.result.lower()

    @pytest.mark.asyncio
    async def test_agent_callback(self, mock_device, mock_llm_client):
        """Test step callback is called."""
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=MagicMock(
                content='<think>Done.</think>\n<answer>finish(message="Complete")</answer>'
            )
        )

        callback_calls = []

        def on_step(step: StepResult):
            callback_calls.append(step)

        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=5),
            on_step=on_step,
        )

        await agent.run("Test task")

        assert len(callback_calls) >= 1
        assert callback_calls[0].finished is True

    @pytest.mark.asyncio
    async def test_agent_cancel(self, mock_device, mock_llm_client):
        """Test agent cancellation."""
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=10),
        )

        agent.state.start_task("Test")
        agent.cancel()

        assert agent.state.status == TaskStatus.CANCELLED


class TestStepRecord:
    """Tests for StepRecord class."""

    def test_step_record_creation(self):
        """Test StepRecord creation."""
        record = StepRecord(
            step_number=1,
            timestamp=datetime.now(),
            thinking="Found button",
            action_type="Tap",
            action_params={"element_id": 5},
            success=True,
            duration_ms=150,
        )

        assert record.step_number == 1
        assert record.action_type == "Tap"
        assert record.success is True

    def test_step_record_to_dict(self):
        """Test StepRecord serialization."""
        record = StepRecord(
            step_number=1,
            timestamp=datetime.now(),
            thinking="Test",
            action_type="Tap",
            action_params={"x": 100},
            success=True,
        )

        data = record.to_dict()

        assert data["step_number"] == 1
        assert data["action_type"] == "Tap"
        assert data["action_params"] == {"x": 100}
        assert "timestamp" in data

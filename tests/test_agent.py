"""
Tests for ReAct Agent Core
==========================

Comprehensive tests for:
- AgentState lifecycle (start, record, complete, fail, cancel, input)
- StepRecord serialisation
- History summary and progress tracking
- Loop detection (repeated actions, alternating patterns, repeated failures)
- Prompt building (build_user_prompt, format_elements_for_prompt, build_auth_prompt)
- ReActAgent run flow, max steps, error handling, cancellation
- Response parser (parse_response, parse_action, format_action_for_log)
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.state import AgentState, StepRecord, TaskStatus
from app.agent.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_elements_for_prompt,
    build_auth_prompt,
    build_error_recovery_prompt,
)
from app.agent.react_loop import ReActAgent, AgentConfig, StepResult, TaskResult
from app.llm.response_parser import (
    ActionType,
    ParsedAction,
    ParsedResponse,
    parse_response,
    parse_action,
    format_action_for_log,
)
from app.perception.ui_parser import UIElement
from tests.conftest import _make_bounds


# ===================================================================
# AgentState tests
# ===================================================================


class TestAgentState:
    def test_initial_state(self):
        state = AgentState()
        assert state.task == ""
        assert state.status == TaskStatus.PENDING
        assert state.current_step == 0
        assert state.history == []
        assert state.error_count == 0
        assert not state.is_running
        assert not state.is_finished

    def test_start_task(self):
        state = AgentState()
        state.start_task("Open YouTube")
        assert state.task == "Open YouTube"
        assert state.status == TaskStatus.RUNNING
        assert state.is_running
        assert state.started_at is not None
        assert state.progress_status == "No progress yet."

    def test_record_step_success(self):
        state = AgentState()
        state.start_task("test")
        rec = state.record_step("found btn", "Tap", {"element_id": 5}, True)
        assert state.current_step == 1
        assert len(state.history) == 1
        assert rec.step_number == 1
        assert rec.success is True
        assert state.error_count == 0

    def test_record_step_failure(self):
        state = AgentState()
        state.start_task("test")
        state.record_step("try", "Tap", {}, False, error="not found")
        assert state.error_count == 1
        state.record_step("retry", "Tap", {}, False, error="still not found")
        assert state.error_count == 2

    def test_error_count_resets_on_success(self):
        state = AgentState()
        state.start_task("test")
        state.record_step("", "Tap", {}, False, error="e1")
        state.record_step("", "Tap", {}, False, error="e2")
        assert state.error_count == 2
        state.record_step("", "Tap", {}, True)
        assert state.error_count == 0

    def test_complete(self):
        state = AgentState()
        state.start_task("test")
        state.complete("Done!")
        assert state.status == TaskStatus.COMPLETED
        assert state.result == "Done!"
        assert state.completed_at is not None
        assert state.is_finished
        assert not state.is_running

    def test_fail(self):
        state = AgentState()
        state.start_task("test")
        state.fail("boom")
        assert state.status == TaskStatus.FAILED
        assert state.result == "boom"
        assert state.is_finished

    def test_cancel(self):
        state = AgentState()
        state.start_task("test")
        state.cancel()
        assert state.status == TaskStatus.CANCELLED
        assert state.is_finished

    def test_request_and_provide_input(self):
        state = AgentState()
        state.start_task("test")
        state.request_input("Enter password")
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.input_required is True
        assert state.input_prompt == "Enter password"

        state.provide_input("secret123")
        assert state.status == TaskStatus.RUNNING
        assert state.input_required is False
        assert state.metadata.get("last_input") == "secret123"

    def test_get_recent_history(self):
        state = AgentState()
        state.start_task("test")
        for i in range(10):
            state.record_step(f"Step {i}", "Tap", {"i": i}, True)
        recent = state.get_recent_history(3)
        assert len(recent) == 3
        assert recent[0].step_number == 8
        assert recent[-1].step_number == 10

    def test_success_rate(self):
        state = AgentState()
        state.start_task("test")
        state.record_step("", "Tap", {}, True)
        state.record_step("", "Tap", {}, True)
        state.record_step("", "Tap", {}, False)
        state.record_step("", "Tap", {}, True)
        assert state.success_rate == 0.75

    def test_success_rate_empty(self):
        state = AgentState()
        assert state.success_rate == 0.0

    def test_duration_seconds(self):
        state = AgentState()
        state.start_task("test")
        assert state.duration_seconds >= 0.0

    def test_to_dict(self):
        state = AgentState()
        state.start_task("test")
        state.record_step("Think", "Tap", {"x": 100}, True)
        d = state.to_dict()
        assert d["task"] == "test"
        assert d["status"] == "RUNNING"
        assert d["current_step"] == 1
        assert len(d["history"]) == 1

    def test_input_needs_submit_flag(self):
        state = AgentState()
        state.start_task("test")
        state.input_needs_submit = True
        assert state.input_needs_submit is True


# ===================================================================
# Loop detection tests
# ===================================================================


class TestLoopDetection:
    def _state_with_actions(self, action_types: list[str], all_success: bool = True) -> AgentState:
        state = AgentState()
        state.start_task("test")
        for at in action_types:
            state.record_step("", at, {}, all_success)
        return state

    def test_no_loop_too_few_steps(self):
        state = self._state_with_actions(["Tap", "Swipe"])
        assert state.detect_action_loop() is None

    def test_repeated_same_action(self):
        state = self._state_with_actions(["Tap", "Tap", "Tap", "Tap"])
        warning = state.detect_action_loop()
        assert warning is not None
        assert "repeated" in warning.lower() or "same action" in warning.lower()

    def test_alternating_pattern(self):
        state = self._state_with_actions(["Tap", "Back", "Tap", "Back"])
        warning = state.detect_action_loop()
        assert warning is not None
        assert "alternating" in warning.lower()

    def test_repeated_failures(self):
        state = AgentState()
        state.start_task("test")
        # Use varied action types so we don't trigger the "repeated same action" check first
        for action in ["Tap", "Swipe", "Type", "Launch"]:
            state.record_step("", action, {}, False, error="fail")
        warning = state.detect_action_loop()
        # The repeated failures check requires same action type failing 3+ times
        # With varied actions, no loop should be detected
        # Let's test with same action type failing
        state2 = AgentState()
        state2.start_task("test")
        # First a success to avoid triggering "repeated same action"
        state2.record_step("", "Swipe", {}, True)
        # Then 3 failures of same type
        for _ in range(3):
            state2.record_step("", "Tap", {}, False, error="fail")
        warning2 = state2.detect_action_loop()
        assert warning2 is not None
        assert "failed" in warning2.lower() or "fail" in warning2.lower()

    def test_no_loop_varied_actions(self):
        state = self._state_with_actions(["Tap", "Type", "Swipe", "Tap", "Launch"])
        assert state.detect_action_loop() is None


# ===================================================================
# History summary tests
# ===================================================================


class TestHistorySummary:
    def test_empty_history(self):
        state = AgentState()
        state.start_task("test")
        summary = state.get_history_summary()
        assert "No previous" in summary

    def test_summary_includes_action_params(self):
        state = AgentState()
        state.start_task("test")
        state.record_step("", "Launch", {"app": "YouTube"}, True)
        state.record_step("", "Tap", {"element_id": 5}, True)
        summary = state.get_history_summary()
        assert "YouTube" in summary
        assert "element=5" in summary


# ===================================================================
# StepRecord tests
# ===================================================================


class TestStepRecord:
    def test_creation(self):
        rec = StepRecord(
            step_number=1,
            timestamp=datetime.now(),
            thinking="Found button",
            action_type="Tap",
            action_params={"element_id": 5},
            success=True,
            duration_ms=150,
        )
        assert rec.step_number == 1
        assert rec.success is True

    def test_to_dict(self):
        rec = StepRecord(
            step_number=1,
            timestamp=datetime.now(),
            thinking="Test",
            action_type="Tap",
            action_params={"x": 100},
            success=True,
        )
        d = rec.to_dict()
        assert d["step_number"] == 1
        assert d["action_type"] == "Tap"
        assert "timestamp" in d
        # screenshot_b64 should NOT be in serialised dict (save space)
        assert "screenshot_b64" not in d


# ===================================================================
# Prompt tests
# ===================================================================


class TestPrompts:
    def test_system_prompt_contains_key_rules(self):
        assert "RequestInput" in SYSTEM_PROMPT
        assert "password" in SYSTEM_PROMPT.lower()
        assert "<think>" in SYSTEM_PROMPT
        assert "<answer>" in SYSTEM_PROMPT

    def test_build_user_prompt_basic(self):
        elem = UIElement(
            index=0,
            element_class="android.widget.Button",
            text="Submit",
            bounds=_make_bounds(100, 100, 200, 150),
            clickable=True,
        )
        prompt = build_user_prompt(task="Click Submit", elements=[elem])
        assert "Click Submit" in prompt
        assert "Submit" in prompt
        assert "PRIMARY GOAL" in prompt

    def test_build_user_prompt_step_counter(self):
        prompt = build_user_prompt(
            task="test", elements=[], current_step=5, max_steps=30
        )
        assert "5/30" in prompt

    def test_build_user_prompt_progress(self):
        prompt = build_user_prompt(
            task="test", elements=[], progress_status="Opened YouTube"
        )
        assert "Opened YouTube" in prompt

    def test_build_user_prompt_caps_elements_at_40(self):
        elems = [
            UIElement(
                index=i,
                element_class="android.widget.Button",
                text=f"Btn{i}",
                bounds=_make_bounds(0, i * 50, 100, i * 50 + 40),
                clickable=True,
            )
            for i in range(60)
        ]
        prompt = build_user_prompt(task="test", elements=elems)
        # Only first 40 interactive elements should appear
        assert "[39]" in prompt
        assert "[59]" not in prompt

    def test_format_elements_empty(self):
        assert "No interactive" in format_elements_for_prompt([])

    def test_format_elements_properties(self):
        elem = UIElement(
            index=0,
            element_class="android.widget.EditText",
            text="Search",
            bounds=_make_bounds(0, 0, 100, 50),
            clickable=True,
            editable=True,
        )
        fmt = format_elements_for_prompt([elem])
        assert "[0]" in fmt
        assert "clickable" in fmt
        assert "editable" in fmt.lower()

    def test_build_auth_prompt(self):
        assert "email" in build_auth_prompt("login", "email").lower()
        assert "password" in build_auth_prompt("login", "password").lower()
        assert "verification" in build_auth_prompt("otp", "otp").lower() or \
               "code" in build_auth_prompt("otp", "otp").lower()

    def test_build_error_recovery_prompt(self):
        prompt = build_error_recovery_prompt(
            error="Element not found", failed_action="Tap", retry_count=2
        )
        assert "Element not found" in prompt
        assert "Tap" in prompt


# ===================================================================
# Response parser tests
# ===================================================================


class TestResponseParser:
    def test_parse_tap_element(self):
        resp = '<think>Need to tap</think>\n<answer>do(action="Tap", element_id=5)</answer>'
        parsed = parse_response(resp)
        assert parsed.thinking == "Need to tap"
        assert parsed.action.action_type == ActionType.TAP
        assert parsed.action.params["element_id"] == 5

    def test_parse_tap_coordinates(self):
        resp = '<think>Tap coord</think>\n<answer>do(action="Tap", x=500, y=300)</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.TAP
        assert parsed.action.params["x"] == 500
        assert parsed.action.params["y"] == 300

    def test_parse_type(self):
        resp = '<think>Type text</think>\n<answer>do(action="Type", text="hello world")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.TYPE
        assert parsed.action.params["text"] == "hello world"

    def test_parse_launch(self):
        resp = '<think>Open app</think>\n<answer>do(action="Launch", app="YouTube")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.LAUNCH
        assert parsed.action.params["app"] == "YouTube"

    def test_parse_swipe(self):
        resp = '<think>Scroll</think>\n<answer>do(action="Swipe", direction="up")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.SWIPE
        assert parsed.action.params["direction"] == "up"

    def test_parse_back(self):
        resp = '<think>Go back</think>\n<answer>do(action="Back")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.BACK

    def test_parse_home(self):
        resp = '<think>Home</think>\n<answer>do(action="Home")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.HOME

    def test_parse_wait(self):
        resp = '<think>Wait</think>\n<answer>do(action="Wait", seconds=5)</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.WAIT
        assert parsed.action.params["seconds"] == 5

    def test_parse_finish(self):
        resp = '<think>Done</think>\n<answer>finish(message="Task completed")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.FINISH
        assert parsed.action.params["message"] == "Task completed"
        assert parsed.action.is_terminal is True

    def test_parse_request_input(self):
        resp = '<think>Need password</think>\n<answer>do(action="RequestInput", prompt="Enter password")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.REQUEST_INPUT
        assert parsed.action.requires_input is True
        assert parsed.action.params["prompt"] == "Enter password"

    def test_parse_unknown_action(self):
        resp = '<think>??</think>\n<answer>do(action="FlyToMoon")</answer>'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.UNKNOWN

    def test_parse_empty_response(self):
        parsed = parse_response("")
        assert parsed.action.action_type == ActionType.UNKNOWN

    def test_parse_missing_tags(self):
        """Fallback: action pattern without <answer> tags."""
        resp = 'do(action="Tap", element_id=3)'
        parsed = parse_response(resp)
        assert parsed.action.action_type == ActionType.TAP
        assert parsed.action.params["element_id"] == 3

    def test_parse_action_aliases(self):
        """Click → TAP, Open → LAUNCH, Input → TYPE."""
        assert parse_action('do(action="Click", element_id=1)').action_type == ActionType.TAP
        assert parse_action('do(action="Open", app="Maps")').action_type == ActionType.LAUNCH
        assert parse_action('do(action="Input", text="hi")').action_type == ActionType.TYPE

    def test_format_action_for_log_tap(self):
        action = ParsedAction(ActionType.TAP, {"element_id": 5})
        assert "#5" in format_action_for_log(action)

    def test_format_action_for_log_finish(self):
        action = ParsedAction(ActionType.FINISH, {"message": "All done"})
        assert "All done" in format_action_for_log(action)

    def test_format_action_for_log_type(self):
        action = ParsedAction(ActionType.TYPE, {"text": "short"})
        assert "short" in format_action_for_log(action)


# ===================================================================
# ReActAgent tests
# ===================================================================


class TestReActAgent:
    @pytest.mark.asyncio
    async def test_init(self, mock_device, mock_llm_client):
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=10),
        )
        assert agent.config.max_steps == 10
        assert agent.state.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_run_finish(self, mock_device, mock_llm_client):
        """Agent should finish when LLM returns a finish action."""
        from app.llm.models import LLMResponse
        
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=LLMResponse(
                content='<think>Done.</think>\n<answer>finish(message="All good")</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
        )
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=5, min_step_interval=0),
        )
        result = await agent.run("Test")
        assert result.success is True
        assert result.result == "All good"
        assert result.steps_taken >= 1

    @pytest.mark.asyncio
    async def test_max_steps_reached(self, mock_device, mock_llm_client):
        """Agent should fail gracefully when max steps reached."""
        from app.llm.models import LLMResponse
        
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=LLMResponse(
                content='<think>Tapping.</think>\n<answer>do(action="Tap", element_id=0)</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
        )
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=3, min_step_interval=0),
        )
        result = await agent.run("Endless")
        assert result.success is False
        assert result.steps_taken == 3

    @pytest.mark.asyncio
    async def test_consecutive_errors_abort(self, mock_device, mock_llm_client):
        """Agent should abort after too many consecutive errors."""
        from app.llm.models import LLMResponse
        from app.device.cloud_provider import ActionResult
        
        mock_device.tap = AsyncMock(
            return_value=ActionResult(success=False, error="Element not found")
        )
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=LLMResponse(
                content='<think>Tap.</think>\n<answer>do(action="Tap", element_id=0)</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
        )
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(
                max_steps=20,
                min_step_interval=0,
            ),
        )
        result = await agent.run("Failing task")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_cancel(self, mock_device, mock_llm_client):
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=10),
        )
        agent.state.start_task("Test")
        agent.cancel()
        assert agent.state.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_step_callback(self, mock_device, mock_llm_client):
        """on_step callback should fire for each step."""
        from app.llm.models import LLMResponse
        
        mock_llm_client.complete_with_vision = AsyncMock(
            return_value=LLMResponse(
                content='<think>Done.</think>\n<answer>finish(message="OK")</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
        )
        calls: list[StepResult] = []
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=5, min_step_interval=0),
            on_step=lambda s: calls.append(s),
        )
        await agent.run("Test")
        assert len(calls) >= 1
        assert calls[-1].finished is True

    @pytest.mark.asyncio
    async def test_get_state(self, mock_device, mock_llm_client):
        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
        )
        d = agent.get_state()
        assert "status" in d
        assert "task" in d

    @pytest.mark.asyncio
    async def test_input_handling(self, mock_device, mock_llm_client):
        """When LLM requests input, on_input_required should be called."""
        from app.llm.models import LLMResponse
        
        responses = [
            LLMResponse(
                content='<think>Need pw</think>\n<answer>do(action="RequestInput", prompt="Enter password")</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            ),
            LLMResponse(
                content='<think>Done</think>\n<answer>finish(message="Logged in")</answer>',
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            ),
        ]
        mock_llm_client.complete_with_vision = AsyncMock(side_effect=responses)

        agent = ReActAgent(
            llm_client=mock_llm_client,
            device=mock_device,
            config=AgentConfig(max_steps=5, min_step_interval=0),
            on_input_required=lambda prompt: "mypassword",
        )
        result = await agent.run("Login to Gmail")
        assert result.success is True
        mock_device.type_text.assert_called_with("mypassword")


# ===================================================================
# TaskResult tests
# ===================================================================


class TestTaskResult:
    def test_success_result(self):
        r = TaskResult(success=True, result="Done", steps_taken=5, duration_seconds=10.5)
        assert r.success is True
        assert r.steps_taken == 5

    def test_failure_result(self):
        r = TaskResult(success=False, result="", error="timeout")
        assert r.success is False
        assert r.error == "timeout"

"""
Agent Module
============

Core ReAct agent implementation for Android automation.

This package contains:
    - react_loop: Main ReAct (Reasoning + Acting) loop
    - state: Agent state management
    - prompts: System prompts for the LLM
    - actions/: Action implementations
"""

from app.agent.react_loop import ReActAgent, AgentConfig, StepResult, TaskResult
from app.agent.state import AgentState, StepRecord
from app.agent.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "ReActAgent",
    "AgentConfig",
    "StepResult",
    "TaskResult",
    "AgentState",
    "StepRecord",
    "SYSTEM_PROMPT",
    "build_user_prompt",
]

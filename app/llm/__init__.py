"""
LLM Integration Module
======================

Provides LLM client and response parsing for the Android AI Agent.

This package contains:
    - client: OpenAI-compatible LLM client with vision support
    - response_parser: Parse <think>/<answer> formatted responses
    - models: Model configuration and data classes
"""

from app.llm.client import LLMClient
from app.llm.models import LLMConfig, ModelType
from app.llm.response_parser import ActionResult, parse_action, parse_response

__all__ = [
    "LLMClient",
    "LLMConfig",
    "ModelType",
    "parse_response",
    "parse_action",
    "ActionResult",
]

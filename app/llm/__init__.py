"""
LLM Integration Module
======================

Provides Google Gemini LLM client and response parsing for the Android AI Agent.

This package contains:
    - client: Gemini LLM client with vision/multimodal support
    - response_parser: Parse <think>/<answer> formatted responses
    - models: Model configuration and data classes

Uses google-genai SDK for Google AI's Gemini models.
"""

from app.llm.client import LLMClient, LLMError, RateLimitError, encode_image_to_base64
from app.llm.models import LLMConfig, ModelType, LLMResponse, Message, ImageContent
from app.llm.response_parser import ParsedAction, ActionResult, parse_action, parse_response

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "RateLimitError",
    "LLMResponse",
    "ModelType",
    "Message",
    "ImageContent",
    "parse_response",
    "parse_action",
    "ParsedAction",
    "ActionResult",  # Backwards compatibility alias
    "encode_image_to_base64",
]
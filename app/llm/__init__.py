"""
LLM Integration Module
======================

Provides LLM clients (Gemini and Groq) and response parsing for the Android AI Agent.

This package contains:
    - client: Gemini LLM client with vision/multimodal support
    - groq_client: Groq LLM client with Llama 4 Scout vision support
    - response_parser: Parse <think>/<answer> formatted responses
    - models: Model configuration and data classes
    - key_rotator: Circular API key rotation for rate-limit resilience

Supports two providers:
    - ``groq``: FREE Llama 4 Scout via Groq (1,000 RPD, recommended)
    - ``gemini``: Google Gemini via google-genai SDK
"""

from app.llm.client import LLMClient, LLMError, RateLimitError, encode_image_to_base64
from app.llm.groq_client import GroqLLMClient
from app.llm.key_rotator import ApiKeyRotator
from app.llm.models import LLMConfig, ModelType, LLMResponse, Message, ImageContent
from app.llm.response_parser import ParsedAction, ActionResult, parse_action, parse_response

__all__ = [
    "LLMClient",
    "GroqLLMClient",
    "LLMConfig",
    "LLMError",
    "RateLimitError",
    "LLMResponse",
    "ModelType",
    "ApiKeyRotator",
    "Message",
    "ImageContent",
    "parse_response",
    "parse_action",
    "ParsedAction",
    "ActionResult",  # Backwards compatibility alias
    "encode_image_to_base64",
]
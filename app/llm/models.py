"""
LLM Model Configuration
=======================

Data classes and configuration for LLM models.
Uses Google Gemini as the primary LLM provider.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Supported Gemini model types."""

    # Gemini 2.0 models (latest)
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_2_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_FLASH_EXP = "gemini-2.0-flash-exp"
    
    # Gemini 1.5 models (stable)
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_15_FLASH_8B = "gemini-1.5-flash-8b"

    @property
    def supports_vision(self) -> bool:
        """Check if the model supports vision/image inputs."""
        # All Gemini models support multimodal vision input
        return True

    @property
    def max_output_tokens(self) -> int:
        """Get default max output tokens for the model."""
        token_limits = {
            ModelType.GEMINI_2_FLASH: 8192,
            ModelType.GEMINI_2_FLASH_LITE: 8192,
            ModelType.GEMINI_2_FLASH_EXP: 8192,
            ModelType.GEMINI_15_PRO: 8192,
            ModelType.GEMINI_15_FLASH: 8192,
            ModelType.GEMINI_15_FLASH_8B: 8192,
        }
        return token_limits.get(self, 8192)
    
    @property
    def context_window(self) -> int:
        """Get max context window size for the model."""
        context_limits = {
            ModelType.GEMINI_2_FLASH: 1_048_576,  # 1M tokens
            ModelType.GEMINI_2_FLASH_LITE: 1_048_576,
            ModelType.GEMINI_2_FLASH_EXP: 1_048_576,
            ModelType.GEMINI_15_PRO: 2_097_152,  # 2M tokens
            ModelType.GEMINI_15_FLASH: 1_048_576,
            ModelType.GEMINI_15_FLASH_8B: 1_048_576,
        }
        return context_limits.get(self, 1_048_576)


@dataclass
class LLMConfig:
    """
    Configuration for Google Gemini LLM client.

    Attributes:
        model: The Gemini model identifier string.
        api_key: Google AI API key for authentication.
        max_output_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-2.0).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on failure.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
    """

    model: str = "gemini-2.0-flash"
    api_key: str = ""
    max_output_tokens: int = 8192
    temperature: float = 0.1
    timeout: float = 60.0
    max_retries: int = 3
    top_p: float = 0.95
    top_k: int = 40

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("Google AI API key is required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class Message:
    """
    A message in the conversation.

    Attributes:
        role: The role (user, model).
        content: The message content (text or structured).
    """

    role: str  # 'user' or 'model' for Gemini
    content: str | list[dict]

    def to_gemini_format(self) -> dict:
        """Convert to Gemini API format."""
        return {"role": self.role, "parts": [{"text": self.content}] if isinstance(self.content, str) else self.content}


@dataclass
class ImageContent:
    """
    Image content for vision models.

    Attributes:
        image_data: Base64-encoded image data.
        media_type: MIME type (e.g., image/png, image/jpeg).
    """

    image_data: str
    media_type: str = "image/png"

    def to_gemini_format(self) -> dict:
        """Convert to Gemini vision API format."""
        return {
            "inline_data": {
                "mime_type": self.media_type,
                "data": self.image_data,
            }
        }


@dataclass
class LLMResponse:
    """
    Response from Gemini LLM API.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        usage: Token usage statistics.
        finish_reason: Why generation stopped.
    """

    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None

    @property
    def prompt_tokens(self) -> int:
        """Get number of prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get number of completion tokens used."""
        return self.usage.get("candidates_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)

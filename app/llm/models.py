"""
LLM Model Configuration
=======================

Data classes and configuration for LLM models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Supported LLM model types."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    GEMINI_PRO = "gemini-1.5-pro"
    GEMINI_FLASH = "gemini-1.5-flash"

    @property
    def supports_vision(self) -> bool:
        """Check if the model supports vision/image inputs."""
        vision_models = {
            ModelType.GPT_4O,
            ModelType.GPT_4O_MINI,
            ModelType.GPT_4_TURBO,
            ModelType.CLAUDE_3_OPUS,
            ModelType.CLAUDE_3_SONNET,
            ModelType.CLAUDE_3_HAIKU,
            ModelType.GEMINI_PRO,
            ModelType.GEMINI_FLASH,
        }
        return self in vision_models

    @property
    def max_tokens(self) -> int:
        """Get default max tokens for the model."""
        token_limits = {
            ModelType.GPT_4O: 4096,
            ModelType.GPT_4O_MINI: 4096,
            ModelType.GPT_4_TURBO: 4096,
            ModelType.CLAUDE_3_OPUS: 4096,
            ModelType.CLAUDE_3_SONNET: 4096,
            ModelType.CLAUDE_3_HAIKU: 4096,
            ModelType.GEMINI_PRO: 8192,
            ModelType.GEMINI_FLASH: 8192,
        }
        return token_limits.get(self, 4096)


@dataclass
class LLMConfig:
    """
    Configuration for LLM client.

    Attributes:
        model: The model identifier string.
        api_key: API key for authentication.
        api_base: Base URL for the API endpoint.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-1.0).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on failure.
    """

    model: str = "gpt-4o"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class Message:
    """
    A message in the conversation.

    Attributes:
        role: The role (system, user, assistant).
        content: The message content (text or structured).
    """

    role: str
    content: str | list[dict]

    def to_dict(self) -> dict:
        """Convert to dictionary for API request."""
        return {"role": self.role, "content": self.content}


@dataclass
class ImageContent:
    """
    Image content for vision models.

    Attributes:
        image_data: Base64-encoded image data.
        media_type: MIME type (e.g., image/png, image/jpeg).
        detail: Detail level for processing (low, high, auto).
    """

    image_data: str
    media_type: str = "image/png"
    detail: str = "high"

    def to_openai_format(self) -> dict:
        """Convert to OpenAI vision API format."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.media_type};base64,{self.image_data}",
                "detail": self.detail,
            },
        }


@dataclass
class LLMResponse:
    """
    Response from LLM API.

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
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)

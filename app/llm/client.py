"""
LLM Client
==========

OpenAI-compatible LLM client with vision support.
Handles API communication, retries, and error handling.

Usage:
    from app.llm import LLMClient, LLMConfig

    config = LLMConfig(api_key="sk-...", model="gpt-4o")
    client = LLMClient(config)

    # Text completion
    response = await client.complete("What is the capital of France?")

    # Vision completion
    response = await client.complete_with_vision(
        prompt="What do you see?",
        image_data=base64_screenshot,
    )
"""

import asyncio
import base64
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

from app.llm.models import ImageContent, LLMConfig, LLMResponse, Message
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Async LLM client with vision support.

    Supports OpenAI, Azure OpenAI, and other OpenAI-compatible APIs.
    Includes automatic retries, timeout handling, and structured logging.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration with API credentials.
        """
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=httpx.Timeout(config.timeout),
            max_retries=config.max_retries,
        )
        logger.info(
            "LLM client initialized",
            model=config.model,
            base_url=config.api_base,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[Message]] = None,
    ) -> LLMResponse:
        """
        Generate a text completion.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system message.
            history: Optional conversation history.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[m.to_dict() for m in messages],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
            )

        except RateLimitError as e:
            logger.warning("Rate limit hit, backing off", error=str(e))
            raise LLMError(f"Rate limit exceeded: {e}") from e
        except APIConnectionError as e:
            logger.error("API connection error", error=str(e))
            raise LLMError(f"Connection error: {e}") from e
        except APIError as e:
            logger.error("API error", error=str(e), status_code=e.status_code)
            raise LLMError(f"API error: {e}") from e

    async def complete_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/png",
        detail: str = "high",
    ) -> LLMResponse:
        """
        Generate a completion with vision/image input.

        Args:
            prompt: The user prompt/question about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system message.
            media_type: Image MIME type (default: image/png).
            detail: Detail level - 'low', 'high', or 'auto'.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
        """
        image_content = ImageContent(
            image_data=image_data,
            media_type=media_type,
            detail=detail,
        )

        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with text and image
        user_content: list[dict] = [
            {"type": "text", "text": prompt},
            image_content.to_openai_format(),
        ]

        messages.append({"role": "user", "content": user_content})

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            logger.debug(
                "Vision completion successful",
                model=response.model,
                tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
            )

        except RateLimitError as e:
            logger.warning("Rate limit hit, backing off", error=str(e))
            raise LLMError(f"Rate limit exceeded: {e}") from e
        except APIConnectionError as e:
            logger.error("API connection error", error=str(e))
            raise LLMError(f"Connection error: {e}") from e
        except APIError as e:
            logger.error("API error", error=str(e), status_code=e.status_code)
            raise LLMError(f"API error: {e}") from e

    async def stream_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[Message]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a text completion token by token.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system message.
            history: Optional conversation history.

        Yields:
            String tokens as they are generated.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[m.to_dict() for m in messages],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except RateLimitError as e:
            logger.warning("Rate limit hit during streaming", error=str(e))
            raise LLMError(f"Rate limit exceeded: {e}") from e
        except APIConnectionError as e:
            logger.error("Connection error during streaming", error=str(e))
            raise LLMError(f"Connection error: {e}") from e
        except APIError as e:
            logger.error("API error during streaming", error=str(e))
            raise LLMError(f"API error: {e}") from e

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[list[Message]],
    ) -> list[Message]:
        """Build the message list for the API request."""
        messages: list[Message] = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(Message(role="user", content=prompt))

        return messages

    async def close(self) -> None:
        """Close the client and release resources."""
        await self._client.close()
        logger.debug("LLM client closed")


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")

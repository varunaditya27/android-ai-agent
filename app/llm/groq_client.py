"""
Groq LLM Client
================

Groq LLM client with vision (multimodal) support via Llama 4 Scout.
Handles API communication, retries, key rotation, and error handling.

Uses the official ``groq`` Python SDK, which follows the OpenAI-compatible
chat completions format.

Groq free-tier for ``meta-llama/llama-4-scout-17b-16e-instruct``:
    - 1,000 requests / day
    - 30 requests / minute
    - 30,000 tokens / minute

Vision input: base64-encoded images via ``image_url`` with data-URI scheme.

Usage:
    from app.llm.groq_client import GroqLLMClient
    from app.llm.models import LLMConfig

    config = LLMConfig(
        api_key="gsk_...",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    client = GroqLLMClient(config)

    # Text completion
    response = await client.complete("What is the capital of France?")

    # Vision completion (screenshot analysis)
    response = await client.complete_with_vision(
        prompt="What do you see on this Android screen?",
        image_data=base64_screenshot,
    )
"""

import asyncio
import re
import time
from typing import AsyncIterator, Optional

from app.llm.key_rotator import ApiKeyRotator
from app.llm.models import LLMConfig, LLMResponse, Message
from app.utils.logger import get_logger

# Lazy-guard imports from the groq SDK so the module can be loaded even
# when groq is not installed (useful for Gemini-only setups).  Storing
# them at *module* level makes them patchable in tests via
# ``@patch("app.llm.groq_client.Groq")``.
try:
    from groq import (
        Groq,
        APIStatusError as GroqAPIStatusError,
        RateLimitError as GroqRateLimitError,
        APIConnectionError as GroqAPIConnectionError,
    )
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    GroqAPIStatusError = None  # type: ignore[assignment,misc]
    GroqRateLimitError = None  # type: ignore[assignment,misc]
    GroqAPIConnectionError = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

# Rate-limit defaults
_DEFAULT_RETRY_DELAY = 30.0  # seconds
_MAX_RETRIES_RATE_LIMIT = 5
_BACKOFF_MULTIPLIER = 1.5

# Default Groq vision model
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when the API returns a 429 rate-limit error."""

    def __init__(self, message: str, retry_after: float = _DEFAULT_RETRY_DELAY):
        super().__init__(message)
        self.retry_after = retry_after


def _extract_retry_delay(error: Exception, attempt: int = 1) -> float:
    """Extract retry delay from a Groq 429 error.

    Groq returns a ``retry-after`` header and sometimes embeds delay
    information in the error message.  Falls back to exponential backoff.
    """
    error_str = str(error)
    # Groq format: "Please try again in 1m0.123s" or "try again in 27.5s"
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", error_str, re.IGNORECASE)
    if match:
        try:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        except ValueError:
            pass
    # Fallback: also try plain "retry in Xs"
    match2 = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            pass
    # Exponential backoff: 30s, 45s, 67.5s …
    return _DEFAULT_RETRY_DELAY * (_BACKOFF_MULTIPLIER ** (attempt - 1))


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True if the error is a 429 / rate-limit error."""
    error_str = str(error).lower()
    return "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str


class GroqLLMClient:
    """
    Async Groq LLM client with vision support and key rotation.

    Uses Groq's official Python SDK (OpenAI-compatible) for
    Llama 4 Scout vision model. Includes automatic retries,
    key rotation on rate-limit, timeout handling, and structured logging.
    """

    def __init__(
        self,
        config: LLMConfig,
        key_rotator: Optional[ApiKeyRotator] = None,
    ) -> None:
        """
        Initialize the Groq LLM client.

        Args:
            config: LLM configuration with API credentials.
            key_rotator: Optional key rotator for multi-key rotation.
        """
        # Lazy import to avoid hard dependency if groq is not installed
        if Groq is None:
            raise ImportError(
                "The 'groq' package is required for GroqLLMClient. "
                "Install it with: pip install groq"
            )

        self.config = config
        self._key_rotator = key_rotator

        # Resolve the initial API key
        if self._key_rotator:
            active_key = self._key_rotator.get_key()
        else:
            active_key = config.api_key

        self._active_key = active_key
        self._client = Groq(api_key=active_key)

        # Counter for API calls
        self.api_call_count = 0

        logger.info(
            "Groq LLM client initialized",
            model=config.model,
            key_rotation_enabled=key_rotator is not None,
            total_keys=key_rotator.total_keys if key_rotator else 1,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate_key_on_rate_limit(self, cooldown: float) -> None:
        """
        Rotate to the next API key and rebuild the Groq client.

        Called when a 429 rate-limit error is caught.
        If no rotator is configured this is a no-op.
        """
        if not self._key_rotator:
            return

        new_key = self._key_rotator.report_rate_limit(cooldown=cooldown)
        if new_key != self._active_key:
            self._active_key = new_key
            self._client = Groq(api_key=new_key)
            logger.info("Switched to new Groq API key after rate limit")

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[Message]] = None,
    ) -> list[dict]:
        """Build the messages list for Groq chat completions API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            history: Optional conversation history.

        Returns:
            List of message dicts in OpenAI-compatible format.
        """
        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for msg in history:
                role = msg.role
                if role == "model":
                    role = "assistant"  # Groq uses OpenAI format
                elif role == "system":
                    continue  # Already handled above
                messages.append({"role": role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_vision_messages(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> list[dict]:
        """Build messages with image content for Groq vision API.

        Per Groq docs, images are passed as ``image_url`` content parts
        using the ``data:`` URI scheme for base64-encoded images.

        Args:
            prompt: Text prompt about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system instruction.
            media_type: Image MIME type (default: image/jpeg).

        Returns:
            List of message dicts with multimodal content.
        """
        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build multimodal user message per Groq vision docs:
        # https://console.groq.com/docs/vision
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        messages.append({"role": "user", "content": user_content})
        return messages

    async def _call_api(self, messages: list[dict]) -> LLMResponse:
        """Execute a Groq chat completion with retry logic.

        Args:
            messages: Messages list in OpenAI-compatible format.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
            RateLimitError: After exhausting all retries on 429.
        """
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES_RATE_LIMIT + 1):
            try:
                # Increment API call counter
                self.api_call_count += 1
                if self._key_rotator:
                    self._key_rotator.record_call()

                # Run blocking SDK call in a thread
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stream=False,
                )

                # Extract text content
                text_content = ""
                if response.choices and response.choices[0].message.content:
                    text_content = response.choices[0].message.content

                # Extract usage metadata
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "candidates_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    }

                # Extract finish reason
                finish_reason = None
                if response.choices and response.choices[0].finish_reason:
                    finish_reason = response.choices[0].finish_reason

                return LLMResponse(
                    content=text_content,
                    model=self.config.model,
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except GroqRateLimitError as e:
                last_error = e
                delay = _extract_retry_delay(e, attempt)
                logger.warning(
                    "Rate limited by Groq API",
                    attempt=attempt,
                    max_attempts=_MAX_RETRIES_RATE_LIMIT,
                    retry_after_seconds=round(delay, 1),
                )
                if attempt < _MAX_RETRIES_RATE_LIMIT:
                    self._rotate_key_on_rate_limit(cooldown=delay)
                    if self._key_rotator and self._key_rotator.total_keys > 1:
                        await asyncio.sleep(1.0)
                    else:
                        await asyncio.sleep(delay)
                    continue
                raise RateLimitError(str(e), retry_after=delay) from e

            except GroqAPIStatusError as e:
                last_error = e
                # Check if it's actually a rate-limit with a non-standard status
                if _is_rate_limit_error(e):
                    delay = _extract_retry_delay(e, attempt)
                    logger.warning(
                        "Rate limited by Groq API (status error)",
                        attempt=attempt,
                        status_code=e.status_code,
                    )
                    if attempt < _MAX_RETRIES_RATE_LIMIT:
                        self._rotate_key_on_rate_limit(cooldown=delay)
                        if self._key_rotator and self._key_rotator.total_keys > 1:
                            await asyncio.sleep(1.0)
                        else:
                            await asyncio.sleep(delay)
                        continue
                    raise RateLimitError(str(e), retry_after=delay) from e
                logger.error("Groq API error", error=str(e), status_code=e.status_code)
                raise LLMError(f"API error: {e}") from e

            except GroqAPIConnectionError as e:
                last_error = e
                logger.error("Groq API connection error", error=str(e))
                raise LLMError(f"Connection error: {e}") from e

            except Exception as e:
                logger.error("Unexpected error during Groq API call", error=str(e))
                raise LLMError(f"Unexpected error: {e}") from e

        # Should not reach here, but just in case
        raise LLMError(f"Failed after {_MAX_RETRIES_RATE_LIMIT} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Public API — matches LLMClient interface
    # ------------------------------------------------------------------

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
            system_prompt: Optional system instruction.
            history: Optional conversation history.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_messages(prompt, system_prompt, history)
        return await self._call_api(messages)

    async def complete_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> LLMResponse:
        """
        Generate a completion with vision/image input.

        Uses Groq's Llama 4 Scout multimodal model to analyze images.
        Images are sent as base64 data-URIs per Groq's official docs.

        Args:
            prompt: The user prompt about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system instruction.
            media_type: Image MIME type (default: image/jpeg).

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_vision_messages(prompt, image_data, system_prompt, media_type)
        return await self._call_api(messages)

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
            system_prompt: Optional system instruction.
            history: Optional conversation history.

        Yields:
            String tokens as they are generated.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            def _stream():
                return self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stream=True,
                )

            stream = await asyncio.to_thread(_stream)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Error during Groq streaming", error=str(e))
            raise LLMError(f"Streaming error: {e}") from e

    async def stream_completion_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> AsyncIterator[str]:
        """
        Stream a vision completion token by token.

        Args:
            prompt: The user prompt about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system instruction.
            media_type: Image MIME type.

        Yields:
            String tokens as they are generated.

        Raises:
            LLMError: On API or network errors.
        """
        messages = self._build_vision_messages(prompt, image_data, system_prompt, media_type)

        try:
            def _stream():
                return self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stream=True,
                )

            stream = await asyncio.to_thread(_stream)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Error during Groq vision streaming", error=str(e))
            raise LLMError(f"Streaming error: {e}") from e

    def get_api_call_count(self) -> int:
        """Get the total number of API calls made."""
        return self.api_call_count

    def reset_api_call_count(self) -> None:
        """Reset the API call counter."""
        self.api_call_count = 0

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client:
            self._client.close()
        logger.debug("Groq LLM client closed")

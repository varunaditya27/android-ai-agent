"""
Gemini LLM Client
=================

Google Gemini LLM client with vision (multimodal) support.
Handles API communication, retries, key rotation, and error handling.

Uses the official google-genai SDK for Python.
When multiple API keys are provided (from different Google Cloud projects),
the client automatically rotates to the next key on rate-limit errors,
effectively multiplying throughput.

Usage:
    from app.llm import LLMClient, LLMConfig

    config = LLMConfig(api_key="your-gemini-api-key", model="gemini-2.5-flash")
    client = LLMClient(config)

    # With key rotation (multiple keys from different projects):
    from app.llm.key_rotator import ApiKeyRotator
    rotator = ApiKeyRotator(["key1", "key2", "key3"])
    client = LLMClient(config, key_rotator=rotator)

    # Text completion
    response = await client.complete("What is the capital of France?")

    # Vision completion
    response = await client.complete_with_vision(
        prompt="What do you see in this screenshot?",
        image_data=base64_screenshot,
    )
"""

import asyncio
import base64
import re
import time
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from app.llm.key_rotator import ApiKeyRotator
from app.llm.models import ImageContent, LLMConfig, LLMResponse, Message
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Rate-limit defaults
_DEFAULT_RETRY_DELAY = 30.0  # seconds
_MAX_RETRIES_RATE_LIMIT = 5
_BACKOFF_MULTIPLIER = 1.5


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when the API returns a 429 rate-limit / quota error."""

    def __init__(self, message: str, retry_after: float = _DEFAULT_RETRY_DELAY):
        super().__init__(message)
        self.retry_after = retry_after


def _extract_retry_delay(error: Exception, attempt: int = 1) -> float:
    """Extract the recommended retry delay from a Gemini 429 error.

    Parses strings like 'Please retry in 27.085023799s.' from the error
    message.  Falls back to exponential backoff if parsing fails.
    """
    error_str = str(error)
    match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Exponential backoff: 30s, 45s, 67.5s, 101s, 152s...
    return _DEFAULT_RETRY_DELAY * (_BACKOFF_MULTIPLIER ** (attempt - 1))


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True if the error is a 429 / RESOURCE_EXHAUSTED error."""
    error_str = str(error)
    return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str


class LLMClient:
    """
    Async Gemini LLM client with vision support and key rotation.

    Uses Google's official genai SDK for Gemini API access.
    Includes automatic retries, key rotation on rate-limit,
    timeout handling, and structured logging.
    """

    def __init__(
        self,
        config: LLMConfig,
        key_rotator: Optional[ApiKeyRotator] = None,
    ) -> None:
        """
        Initialize the Gemini LLM client.

        Args:
            config: LLM configuration with API credentials.
            key_rotator: Optional key rotator for multi-key rotation.
                         If provided, the rotator's current key is used
                         instead of ``config.api_key``.
        """
        self.config = config
        self._key_rotator = key_rotator

        # Resolve the initial API key
        if self._key_rotator:
            active_key = self._key_rotator.get_key()
        else:
            active_key = config.api_key

        self._active_key = active_key
        self._client = genai.Client(api_key=active_key)

        # Counter for API calls
        self.api_call_count = 0

        logger.info(
            "Gemini LLM client initialized",
            model=config.model,
            key_rotation_enabled=key_rotator is not None,
            total_keys=key_rotator.total_keys if key_rotator else 1,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate_key_on_rate_limit(self, cooldown: float) -> None:
        """
        Rotate to the next API key and rebuild the underlying genai client.

        Called when a 429 / RESOURCE_EXHAUSTED error is caught.
        If no rotator is configured this is a no-op.
        """
        if not self._key_rotator:
            return

        new_key = self._key_rotator.report_rate_limit(cooldown=cooldown)
        if new_key != self._active_key:
            self._active_key = new_key
            self._client = genai.Client(api_key=new_key)
            logger.info("Switched to new API key after rate limit")

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
        contents = self._build_contents(prompt, history)
        
        # Build config with system instruction if provided
        config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=system_prompt if system_prompt else None,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES_RATE_LIMIT + 1):
            try:
                # Increment API call counter
                self.api_call_count += 1
                if self._key_rotator:
                    self._key_rotator.record_call()
                
                # Use async method for non-blocking operation
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )

                # Extract text from response
                text_content = ""
                if response.text:
                    text_content = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    text_content = response.candidates[0].content.parts[0].text

                # Extract usage metadata
                usage = {}
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                        "candidates_tokens": response.usage_metadata.candidates_token_count or 0,
                        "total_tokens": response.usage_metadata.total_token_count or 0,
                    }

                # Extract finish reason
                finish_reason = None
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason = str(response.candidates[0].finish_reason)

                return LLMResponse(
                    content=text_content,
                    model=self.config.model,
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except (APIError, ClientError) as e:
                last_error = e
                if _is_rate_limit_error(e):
                    delay = _extract_retry_delay(e, attempt)
                    logger.warning(
                        "Rate limited by Gemini API",
                        attempt=attempt,
                        max_attempts=_MAX_RETRIES_RATE_LIMIT,
                        retry_after_seconds=round(delay, 1),
                    )
                    if attempt < _MAX_RETRIES_RATE_LIMIT:
                        # Rotate to a fresh key (may avoid the sleep entirely)
                        self._rotate_key_on_rate_limit(cooldown=delay)
                        if self._key_rotator and self._key_rotator.total_keys > 1:
                            # New key from a different project – retry immediately
                            await asyncio.sleep(1.0)
                        else:
                            # Single key – must wait for cooldown
                            await asyncio.sleep(delay)
                        continue
                    raise RateLimitError(str(e), retry_after=delay) from e
                logger.error("Gemini API error", error=str(e))
                raise LLMError(f"API error: {e}") from e
            except Exception as e:
                logger.error("Unexpected error", error=str(e))
                raise LLMError(f"Unexpected error: {e}") from e

        # Should not reach here, but just in case
        raise LLMError(f"Failed after {_MAX_RETRIES_RATE_LIMIT} attempts: {last_error}")

    async def complete_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> LLMResponse:
        """
        Generate a completion with vision/image input.

        Gemini natively supports multimodal inputs with images.

        Args:
            prompt: The user prompt/question about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system instruction.
            media_type: Image MIME type (default: image/png).

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: On API or network errors.
        """
        # Build multimodal content with image and text
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_data),
            mime_type=media_type,
        )
        text_part = types.Part.from_text(text=prompt)
        
        contents = [image_part, text_part]

        # Build config with system instruction if provided
        config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=system_prompt if system_prompt else None,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES_RATE_LIMIT + 1):
            try:
                # Increment API call counter
                self.api_call_count += 1
                if self._key_rotator:
                    self._key_rotator.record_call()
                
                # Use async method for non-blocking operation
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )

                # Extract text from response
                text_content = ""
                if response.text:
                    text_content = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_content += part.text

                # Extract usage metadata
                usage = {}
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                        "candidates_tokens": response.usage_metadata.candidates_token_count or 0,
                        "total_tokens": response.usage_metadata.total_token_count or 0,
                    }

                logger.debug(
                    "Vision completion successful",
                    model=self.config.model,
                    tokens=usage.get("total_tokens", 0),
                )

                # Extract finish reason
                finish_reason = None
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason = str(response.candidates[0].finish_reason)

                return LLMResponse(
                    content=text_content,
                    model=self.config.model,
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except (APIError, ClientError) as e:
                last_error = e
                if _is_rate_limit_error(e):
                    delay = _extract_retry_delay(e, attempt)
                    logger.warning(
                        "Rate limited by Gemini API (vision)",
                        attempt=attempt,
                        max_attempts=_MAX_RETRIES_RATE_LIMIT,
                        retry_after_seconds=round(delay, 1),
                    )
                    if attempt < _MAX_RETRIES_RATE_LIMIT:
                        # Rotate to a fresh key (may avoid the sleep entirely)
                        self._rotate_key_on_rate_limit(cooldown=delay)
                        if self._key_rotator and self._key_rotator.total_keys > 1:
                            # New key from a different project – retry immediately
                            await asyncio.sleep(1.0)
                        else:
                            # Single key – must wait for cooldown
                            await asyncio.sleep(delay)
                        continue
                    raise RateLimitError(str(e), retry_after=delay) from e
                logger.error("Gemini API error", error=str(e))
                raise LLMError(f"API error: {e}") from e
            except Exception as e:
                logger.error("Unexpected error during vision completion", error=str(e))
                raise LLMError(f"Unexpected error: {e}") from e

        # Should not reach here, but just in case
        raise LLMError(f"Failed after {_MAX_RETRIES_RATE_LIMIT} attempts: {last_error}")

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
        contents = self._build_contents(prompt, history)
        
        # Build config with system instruction if provided
        config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            # Get streaming response in a thread to avoid blocking
            def _stream():
                return list(self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                ))
            
            chunks = await asyncio.to_thread(_stream)
            
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text

        except APIError as e:
            logger.error("Gemini API error during streaming", error=str(e))
            raise LLMError(f"API error: {e}") from e
        except ClientError as e:
            logger.error("Gemini client error during streaming", error=str(e))
            raise LLMError(f"Client error: {e}") from e
        except Exception as e:
            logger.error("Error during streaming", error=str(e))
            raise LLMError(f"Streaming error: {e}") from e

    async def stream_completion_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/png",
    ) -> AsyncIterator[str]:
        """
        Stream a vision completion token by token.

        Args:
            prompt: The user prompt/question about the image.
            image_data: Base64-encoded image data.
            system_prompt: Optional system instruction.
            media_type: Image MIME type.

        Yields:
            String tokens as they are generated.

        Raises:
            LLMError: On API or network errors.
        """
        # Build multimodal content
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_data),
            mime_type=media_type,
        )
        text_part = types.Part.from_text(text=prompt)
        contents = [image_part, text_part]

        config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            def _stream():
                return list(self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                ))
            
            chunks = await asyncio.to_thread(_stream)
            
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text

        except APIError as e:
            logger.error("Gemini API error during vision streaming", error=str(e))
            raise LLMError(f"API error: {e}") from e
        except Exception as e:
            logger.error("Error during vision streaming", error=str(e))
            raise LLMError(f"Streaming error: {e}") from e

    def _build_contents(
        self,
        prompt: str,
        history: Optional[list[Message]],
    ) -> list:
        """Build the contents list for the API request."""
        contents = []

        if history:
            for msg in history:
                # Convert to Gemini format: 'user' or 'model'
                role = msg.role
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    # System messages are handled via system_instruction
                    continue
                
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content if isinstance(msg.content, str) else str(msg.content))],
                    )
                )

        # Add current user prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        return contents

    def get_api_call_count(self) -> int:
        """Get the total number of API calls made."""
        return self.api_call_count

    def reset_api_call_count(self) -> None:
        """Reset the API call counter."""
        self.api_call_count = 0

    async def close(self) -> None:
        """Close the client and release resources."""
        # The google-genai client doesn't require explicit cleanup
        logger.debug("Gemini LLM client closed")


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_base64_to_bytes(base64_string: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        base64_string: Base64-encoded string.

    Returns:
        Decoded bytes.
    """
    return base64.b64decode(base64_string)
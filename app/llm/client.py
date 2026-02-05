"""
Gemini LLM Client
=================

Google Gemini LLM client with vision (multimodal) support.
Handles API communication, retries, and error handling.

Uses the official google-genai SDK for Python.

Usage:
    from app.llm import LLMClient, LLMConfig

    config = LLMConfig(api_key="your-gemini-api-key", model="gemini-2.0-flash")
    client = LLMClient(config)

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
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from app.llm.models import ImageContent, LLMConfig, LLMResponse, Message
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class LLMClient:
    """
    Async Gemini LLM client with vision support.

    Uses Google's official genai SDK for Gemini API access.
    Includes automatic retries, timeout handling, and structured logging.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the Gemini LLM client.

        Args:
            config: LLM configuration with API credentials.
        """
        self.config = config
        
        # Initialize the Google GenAI client
        self._client = genai.Client(api_key=config.api_key)
        
        logger.info(
            "Gemini LLM client initialized",
            model=config.model,
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

        try:
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

        except APIError as e:
            logger.error("Gemini API error", error=str(e))
            raise LLMError(f"API error: {e}") from e
        except ClientError as e:
            logger.error("Gemini client error", error=str(e))
            raise LLMError(f"Client error: {e}") from e
        except Exception as e:
            logger.error("Unexpected error", error=str(e))
            raise LLMError(f"Unexpected error: {e}") from e

    async def complete_with_vision(
        self,
        prompt: str,
        image_data: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/png",
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
        text_part = types.Part.from_text(prompt)
        
        contents = [image_part, text_part]

        # Build config with system instruction if provided
        config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
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

        except APIError as e:
            logger.error("Gemini API error", error=str(e))
            raise LLMError(f"API error: {e}") from e
        except ClientError as e:
            logger.error("Gemini client error", error=str(e))
            raise LLMError(f"Client error: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during vision completion", error=str(e))
            raise LLMError(f"Unexpected error: {e}") from e

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
        text_part = types.Part.from_text(prompt)
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
                        parts=[types.Part.from_text(msg.content if isinstance(msg.content, str) else str(msg.content))],
                    )
                )

        # Add current user prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(prompt)],
            )
        )

        return contents

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

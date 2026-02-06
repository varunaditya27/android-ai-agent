"""
Tests for Groq LLM Client
==========================

Comprehensive tests for:
- GroqLLMClient initialization (single key, with rotator)
- Message building (text, vision, system prompt, history)
- Text completion (success, rate-limit retry, API errors)
- Vision completion (success, base64 encoding, media types)
- Rate-limit handling (retry, backoff, key rotation)
- Streaming (text, vision)
- API call counting and reset
- Error extraction and classification
- Client close
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Ensure test env vars are set
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("GROQ_API_KEY", "gsk_test-groq-key-for-testing")

from app.llm.groq_client import (
    GroqLLMClient,
    LLMError,
    RateLimitError,
    _extract_retry_delay,
    _is_rate_limit_error,
    _DEFAULT_RETRY_DELAY,
    _BACKOFF_MULTIPLIER,
    _MAX_RETRIES_RATE_LIMIT,
    DEFAULT_GROQ_MODEL,
)
from app.llm.models import LLMConfig, LLMResponse, Message
from app.llm.key_rotator import ApiKeyRotator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(api_key: str = "gsk_test_key_123", model: str = DEFAULT_GROQ_MODEL) -> LLMConfig:
    """Create a valid LLMConfig for tests."""
    return LLMConfig(api_key=api_key, model=model)


def _mock_groq_response(content: str = "Hello", prompt_tokens: int = 10,
                         completion_tokens: int = 5, finish_reason: str = "stop"):
    """Create a mock Groq chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mock_groq_stream_chunks(texts: list[str]):
    """Create mock streaming chunks."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


class TestErrorHelpers:
    """Tests for _extract_retry_delay and _is_rate_limit_error."""

    def test_extract_retry_delay_groq_format_minutes_and_seconds(self):
        err = Exception("Rate limit reached. Please try again in 1m30.5s")
        assert _extract_retry_delay(err) == 90.5

    def test_extract_retry_delay_groq_format_seconds_only(self):
        err = Exception("Please try again in 27.5s")
        assert _extract_retry_delay(err) == 27.5

    def test_extract_retry_delay_gemini_format(self):
        err = Exception("Please retry in 12.3s")
        assert _extract_retry_delay(err) == 12.3

    def test_extract_retry_delay_fallback_attempt_1(self):
        err = Exception("Some unknown error format")
        assert _extract_retry_delay(err, attempt=1) == _DEFAULT_RETRY_DELAY

    def test_extract_retry_delay_fallback_attempt_3(self):
        err = Exception("Unknown")
        expected = _DEFAULT_RETRY_DELAY * (_BACKOFF_MULTIPLIER ** 2)
        assert _extract_retry_delay(err, attempt=3) == pytest.approx(expected)

    def test_is_rate_limit_error_429(self):
        assert _is_rate_limit_error(Exception("Error 429: Too many requests"))

    def test_is_rate_limit_error_rate_limit(self):
        assert _is_rate_limit_error(Exception("rate_limit_exceeded"))

    def test_is_rate_limit_error_case_insensitive(self):
        assert _is_rate_limit_error(Exception("Rate Limit Exceeded"))

    def test_is_rate_limit_error_false(self):
        assert not _is_rate_limit_error(Exception("Internal server error 500"))


# ---------------------------------------------------------------------------
# GroqLLMClient init
# ---------------------------------------------------------------------------


class TestGroqLLMClientInit:
    """Tests for GroqLLMClient initialization."""

    @patch("app.llm.groq_client.Groq")
    def test_init_with_single_key(self, mock_groq_cls):
        """Client initializes with a single API key."""
        mock_groq_cls.return_value = MagicMock()
        config = _make_config()
        client = GroqLLMClient(config)

        assert client.config == config
        assert client._active_key == "gsk_test_key_123"
        assert client._key_rotator is None
        assert client.api_call_count == 0
        mock_groq_cls.assert_called_once_with(api_key="gsk_test_key_123")

    @patch("app.llm.groq_client.Groq")
    def test_init_with_key_rotator(self, mock_groq_cls):
        """Client uses the rotator's key when provided."""
        mock_groq_cls.return_value = MagicMock()
        rotator = ApiKeyRotator(["key_a", "key_b", "key_c"])
        config = _make_config()

        client = GroqLLMClient(config, key_rotator=rotator)

        assert client._key_rotator is rotator
        assert client._active_key == "key_a"
        mock_groq_cls.assert_called_once_with(api_key="key_a")

    def test_init_missing_groq_package(self):
        """ImportError when groq package is not installed."""
        config = _make_config()
        with patch("app.llm.groq_client.Groq", None):
            with pytest.raises(ImportError, match="groq"):
                GroqLLMClient(config)


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


class TestMessageBuilding:
    """Tests for _build_messages and _build_vision_messages."""

    @patch("app.llm.groq_client.Groq")
    def test_build_messages_basic(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        messages = client._build_messages("Hello")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    @patch("app.llm.groq_client.Groq")
    def test_build_messages_with_system_prompt(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        messages = client._build_messages("Hello", system_prompt="You are helpful")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @patch("app.llm.groq_client.Groq")
    def test_build_messages_with_history(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        history = [
            Message(role="user", content="First question"),
            Message(role="model", content="First answer"),  # 'model' → 'assistant'
        ]
        messages = client._build_messages("Second question", history=history)

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "First question"}
        assert messages[1] == {"role": "assistant", "content": "First answer"}
        assert messages[2] == {"role": "user", "content": "Second question"}

    @patch("app.llm.groq_client.Groq")
    def test_build_messages_skips_system_in_history(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        history = [
            Message(role="system", content="system msg"),
            Message(role="user", content="user msg"),
        ]
        messages = client._build_messages("new msg", history=history)
        # System in history should be skipped (only explicit system_prompt used)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "user"

    @patch("app.llm.groq_client.Groq")
    def test_build_vision_messages(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        messages = client._build_vision_messages(
            prompt="What's in this image?",
            image_data="base64data",
            media_type="image/jpeg",
        )

        assert len(messages) == 1
        user_msg = messages[0]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2

        img_part = user_msg["content"][0]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"] == "data:image/jpeg;base64,base64data"

        text_part = user_msg["content"][1]
        assert text_part["type"] == "text"
        assert text_part["text"] == "What's in this image?"

    @patch("app.llm.groq_client.Groq")
    def test_build_vision_messages_with_system_prompt(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        messages = client._build_vision_messages(
            prompt="Describe",
            image_data="abc",
            system_prompt="You are a UI analyzer",
        )

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a UI analyzer"}
        assert messages[1]["role"] == "user"

    @patch("app.llm.groq_client.Groq")
    def test_build_vision_messages_png_type(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())

        messages = client._build_vision_messages(
            prompt="Describe", image_data="pngdata", media_type="image/png"
        )
        url = messages[0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Text completion
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for GroqLLMClient.complete()."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_complete_success(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Paris is the capital"
        )

        client = GroqLLMClient(_make_config())
        response = await client.complete("What is the capital of France?")

        assert isinstance(response, LLMResponse)
        assert response.content == "Paris is the capital"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["candidates_tokens"] == 5
        assert response.finish_reason == "stop"
        assert client.api_call_count == 1

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_complete_with_system_prompt(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response("OK")

        client = GroqLLMClient(_make_config())
        await client.complete("Hello", system_prompt="Be concise")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_complete_empty_response(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        response.choices[0].finish_reason = "stop"
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        client = GroqLLMClient(_make_config())
        result = await client.complete("test")

        assert result.content == ""
        assert result.usage == {}


# ---------------------------------------------------------------------------
# Vision completion
# ---------------------------------------------------------------------------


class TestCompleteWithVision:
    """Tests for GroqLLMClient.complete_with_vision()."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_vision_success(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "I see a login screen with email and password fields"
        )

        client = GroqLLMClient(_make_config())
        response = await client.complete_with_vision(
            prompt="What UI elements are on this screen?",
            image_data="iVBORw0KGgoAAAA",
            system_prompt="You are a UI analyzer",
        )

        assert response.content == "I see a login screen with email and password fields"
        assert client.api_call_count == 1

        # Verify the message structure
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        # Image part should be first (per Groq docs)
        assert user_msg["content"][0]["type"] == "image_url"
        assert "data:image/jpeg;base64,iVBORw0KGgoAAAA" in user_msg["content"][0]["image_url"]["url"]

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_vision_png_media_type(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response("PNG image")

        client = GroqLLMClient(_make_config())
        await client.complete_with_vision(
            prompt="Describe", image_data="abc", media_type="image/png"
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        url = messages[0]["content"][0]["image_url"]["url"]
        assert url == "data:image/png;base64,abc"

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_vision_without_system_prompt(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response("result")

        client = GroqLLMClient(_make_config())
        await client.complete_with_vision(prompt="test", image_data="abc")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # No system message, just user
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------


class TestRateLimitHandling:
    """Tests for rate-limit retries and key rotation."""

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("app.llm.groq_client.Groq")
    async def test_rate_limit_retry_then_success(self, mock_groq_cls, mock_sleep):
        """Rate-limit on first call, succeeds on second."""
        from groq import RateLimitError as GroqRateLimit

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        rate_err = GroqRateLimit(
            message="Rate limit reached. Please try again in 5.0s",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_client.chat.completions.create.side_effect = [
            rate_err,
            _mock_groq_response("Success after retry"),
        ]

        client = GroqLLMClient(_make_config())
        response = await client.complete("test")

        assert response.content == "Success after retry"
        assert client.api_call_count == 2
        # Should sleep for the parsed delay (5.0s for single key)
        mock_sleep.assert_called()

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("app.llm.groq_client.Groq")
    async def test_rate_limit_with_key_rotation(self, mock_groq_cls, mock_sleep):
        """Rate-limit triggers key rotation when rotator is available."""
        from groq import RateLimitError as GroqRateLimit

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        rate_err = GroqRateLimit(
            message="Rate limit reached. Please try again in 10s",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_client.chat.completions.create.side_effect = [
            rate_err,
            _mock_groq_response("OK"),
        ]

        rotator = ApiKeyRotator(["key1", "key2"])
        client = GroqLLMClient(_make_config(api_key="key1"), key_rotator=rotator)
        response = await client.complete("test")

        assert response.content == "OK"
        # With multi-key rotation, should sleep 1.0s (not full delay)
        mock_sleep.assert_called_with(1.0)

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("app.llm.groq_client.Groq")
    async def test_rate_limit_exhausts_retries(self, mock_groq_cls, mock_sleep):
        """All retries exhausted → raises RateLimitError."""
        from groq import RateLimitError as GroqRateLimit

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        rate_err = GroqRateLimit(
            message="Rate limit",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = rate_err

        client = GroqLLMClient(_make_config())

        with pytest.raises(RateLimitError):
            await client.complete("test")

        assert client.api_call_count == _MAX_RETRIES_RATE_LIMIT


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------


class TestAPIErrorHandling:
    """Tests for non-rate-limit API errors."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_api_status_error(self, mock_groq_cls):
        """Non-rate-limit API error raises LLMError."""
        from groq import APIStatusError

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        api_err = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = api_err

        client = GroqLLMClient(_make_config())

        with pytest.raises(LLMError, match="API error"):
            await client.complete("test")

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_connection_error(self, mock_groq_cls):
        """Connection error raises LLMError."""
        from groq import APIConnectionError

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        conn_err = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create.side_effect = conn_err

        client = GroqLLMClient(_make_config())

        with pytest.raises(LLMError, match="Connection error"):
            await client.complete("test")

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_unexpected_error(self, mock_groq_cls):
        """Random exception raises LLMError."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Boom")

        client = GroqLLMClient(_make_config())

        with pytest.raises(LLMError, match="Unexpected error"):
            await client.complete("test")


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------


class TestKeyRotation:
    """Tests for _rotate_key_on_rate_limit."""

    @patch("app.llm.groq_client.Groq")
    def test_rotate_no_op_without_rotator(self, mock_groq_cls):
        mock_groq_cls.return_value = MagicMock()
        client = GroqLLMClient(_make_config())
        # Should not raise
        client._rotate_key_on_rate_limit(30.0)

    @patch("app.llm.groq_client.Groq")
    def test_rotate_rebuilds_client(self, mock_groq_cls):
        mock_instance = MagicMock()
        mock_groq_cls.return_value = mock_instance

        rotator = ApiKeyRotator(["key_a", "key_b"])
        client = GroqLLMClient(_make_config(api_key="key_a"), key_rotator=rotator)

        # Initially using key_a
        assert client._active_key == "key_a"

        # Rotate
        client._rotate_key_on_rate_limit(60.0)

        # Should have switched to key_b
        assert client._active_key == "key_b"
        # Groq client should have been re-created
        assert mock_groq_cls.call_count == 2


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Tests for stream_completion and stream_completion_with_vision."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_stream_completion(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        chunks = _mock_groq_stream_chunks(["Hello", " ", "World"])
        mock_client.chat.completions.create.return_value = chunks

        client = GroqLLMClient(_make_config())
        tokens = []
        async for token in client.stream_completion("Hi"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_stream_completion_error(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Stream fail")

        client = GroqLLMClient(_make_config())

        with pytest.raises(LLMError, match="Streaming error"):
            async for _ in client.stream_completion("test"):
                pass

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_stream_vision_completion(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        chunks = _mock_groq_stream_chunks(["I see", " a button"])
        mock_client.chat.completions.create.return_value = chunks

        client = GroqLLMClient(_make_config())
        tokens = []
        async for token in client.stream_completion_with_vision("Describe", "imgdata"):
            tokens.append(token)

        assert tokens == ["I see", " a button"]


# ---------------------------------------------------------------------------
# API call counting
# ---------------------------------------------------------------------------


class TestAPICallCounting:
    """Tests for get_api_call_count and reset_api_call_count."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_call_count_increments(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response("OK")

        client = GroqLLMClient(_make_config())
        assert client.get_api_call_count() == 0

        await client.complete("a")
        assert client.get_api_call_count() == 1

        await client.complete_with_vision("b", "img")
        assert client.get_api_call_count() == 2

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_call_count_reset(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_groq_response("OK")

        client = GroqLLMClient(_make_config())
        await client.complete("a")
        assert client.get_api_call_count() == 1

        client.reset_api_call_count()
        assert client.get_api_call_count() == 0


# ---------------------------------------------------------------------------
# Client close
# ---------------------------------------------------------------------------


class TestClientClose:
    """Tests for close()."""

    @pytest.mark.asyncio
    @patch("app.llm.groq_client.Groq")
    async def test_close(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        client = GroqLLMClient(_make_config())
        await client.close()

        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config.py LLM provider settings."""

    def test_llm_settings_groq_defaults(self):
        """LLMSettings defaults to groq provider."""
        from app.config import LLMSettings
        settings = LLMSettings()
        assert settings.llm_provider == "groq"
        assert settings.groq_model == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_get_active_api_key_groq(self):
        from app.config import LLMSettings
        settings = LLMSettings()
        key = settings.get_active_api_key()
        # Should return the groq key (from env)
        assert key == os.environ.get("GROQ_API_KEY", "")

    def test_get_active_model_groq(self):
        from app.config import LLMSettings
        settings = LLMSettings()
        model = settings.get_active_model()
        assert model == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_get_active_model_gemini(self):
        from app.config import LLMSettings
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            settings = LLMSettings()
            model = settings.get_active_model()
            assert model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# ModelType enum
# ---------------------------------------------------------------------------


class TestModelType:
    """Tests for the updated ModelType enum."""

    def test_groq_models_exist(self):
        from app.llm.models import ModelType
        assert ModelType.GROQ_LLAMA4_SCOUT.value == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert ModelType.GROQ_LLAMA4_MAVERICK.value == "meta-llama/llama-4-maverick-17b-128e-instruct"

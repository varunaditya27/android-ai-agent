"""
Tests for LLM Models
=====================

Comprehensive tests for:
- LLMConfig defaults and validation
- ModelType enum
- Message dataclass
- ImageContent
- LLMResponse
"""

import pytest

from app.llm.models import (
    LLMConfig,
    ModelType,
    Message,
    ImageContent,
    LLMResponse,
)


class TestModelType:
    def test_gemini_25_flash(self):
        assert ModelType.GEMINI_25_FLASH.value == "gemini-2.5-flash"

    def test_gemini_25_pro(self):
        assert ModelType.GEMINI_25_PRO.value == "gemini-2.5-pro"

    def test_gemini_2_flash(self):
        assert ModelType.GEMINI_2_FLASH.value == "gemini-2.0-flash"

    def test_enum_members_exist(self):
        members = list(ModelType)
        assert len(members) >= 3

    def test_value_is_string(self):
        for m in ModelType:
            assert isinstance(m.value, str)

    def test_supports_vision(self):
        for m in ModelType:
            assert m.supports_vision is True

    def test_max_output_tokens(self):
        tokens = ModelType.GEMINI_2_FLASH.max_output_tokens
        assert tokens == 8192

    def test_context_window(self):
        window = ModelType.GEMINI_2_FLASH.context_window
        assert window == 1_048_576


class TestLLMConfig:
    def test_defaults_require_api_key(self):
        """LLMConfig requires api_key - empty raises ValueError."""
        with pytest.raises(ValueError, match="API key"):
            LLMConfig()

    def test_with_api_key(self):
        cfg = LLMConfig(api_key="test-key-123")
        assert cfg.max_output_tokens == 2048
        assert cfg.temperature == 0.1
        assert cfg.model == "gemini-2.5-flash"

    def test_custom_values(self):
        cfg = LLMConfig(
            model="gemini-2.0-flash",
            api_key="test-key",
            max_output_tokens=4096,
            temperature=0.5,
        )
        assert cfg.max_output_tokens == 4096
        assert cfg.temperature == 0.5
        assert cfg.model == "gemini-2.0-flash"

    def test_invalid_temperature_high(self):
        with pytest.raises(ValueError, match="Temperature"):
            LLMConfig(api_key="key", temperature=3.0)

    def test_invalid_temperature_low(self):
        with pytest.raises(ValueError, match="Temperature"):
            LLMConfig(api_key="key", temperature=-1.0)

    def test_invalid_max_output_tokens(self):
        with pytest.raises(ValueError, match="max_output_tokens"):
            LLMConfig(api_key="key", max_output_tokens=0)

    def test_invalid_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            LLMConfig(api_key="key", top_p=1.5)

    def test_top_k_default(self):
        cfg = LLMConfig(api_key="key")
        assert cfg.top_k == 40


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_model_role(self):
        msg = Message(role="model", content="Sure!")
        assert msg.role == "model"

    def test_to_gemini_format_text(self):
        msg = Message(role="user", content="Hello")
        fmt = msg.to_gemini_format()
        assert fmt["role"] == "user"
        assert fmt["parts"] == [{"text": "Hello"}]

    def test_to_gemini_format_parts(self):
        parts = [{"text": "Hello"}, {"inline_data": {"data": "abc"}}]
        msg = Message(role="user", content=parts)
        fmt = msg.to_gemini_format()
        assert fmt["parts"] == parts

    def test_content_list(self):
        msg = Message(role="user", content=[{"text": "hi"}])
        assert isinstance(msg.content, list)


class TestImageContent:
    def test_creation(self):
        img = ImageContent(image_data="base64data", media_type="image/png")
        assert img.image_data == "base64data"
        assert img.media_type == "image/png"

    def test_jpeg(self):
        img = ImageContent(image_data="jpegdata", media_type="image/jpeg")
        assert img.media_type == "image/jpeg"

    def test_default_media_type(self):
        img = ImageContent(image_data="data")
        assert img.media_type == "image/png"

    def test_to_gemini_format(self):
        img = ImageContent(image_data="abc123", media_type="image/jpeg")
        fmt = img.to_gemini_format()
        assert fmt == {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": "abc123",
            }
        }


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(content="Hello", model="gemini-2.5-flash")
        assert resp.content == "Hello"
        assert resp.model == "gemini-2.5-flash"

    def test_empty_content(self):
        resp = LLMResponse(content="", model="gemini-2.5-flash")
        assert resp.content == ""

    def test_usage_tokens_from_dict(self):
        resp = LLMResponse(
            content="Hi",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "candidates_tokens": 5, "total_tokens": 15},
        )
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15

    def test_usage_defaults_to_zero(self):
        resp = LLMResponse(content="Hi", model="gemini-2.5-flash")
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0

    def test_finish_reason(self):
        resp = LLMResponse(
            content="Hi",
            model="gemini-2.5-flash",
            finish_reason="STOP",
        )
        assert resp.finish_reason == "STOP"

    def test_finish_reason_default_none(self):
        resp = LLMResponse(content="Hi", model="gemini-2.5-flash")
        assert resp.finish_reason is None

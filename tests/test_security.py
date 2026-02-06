"""
Tests for Security Utilities
=============================

Comprehensive tests for:
- SecureString masking and comparison
- mask_sensitive for various input types
- sanitize_for_logging
- generate_session_id
- validate_input_safe
"""

import pytest

from app.utils.security import (
    SecureString,
    mask_sensitive,
    sanitize_for_logging,
    generate_session_id,
    validate_input_safe,
)


class TestSecureString:
    def test_str_masked(self):
        s = SecureString("password123")
        assert "password123" not in str(s)
        assert str(s) == "********"

    def test_repr_masked(self):
        s = SecureString("password123")
        assert "password123" not in repr(s)

    def test_get_secret(self):
        s = SecureString("secret")
        assert s.get_secret() == "secret"

    def test_len(self):
        s = SecureString("abcde")
        assert len(s) == 5

    def test_bool_true(self):
        assert bool(SecureString("x"))

    def test_bool_false(self):
        assert not bool(SecureString(""))

    def test_equality_same(self):
        a = SecureString("hello")
        b = SecureString("hello")
        assert a == b

    def test_equality_different(self):
        a = SecureString("hello")
        b = SecureString("world")
        assert a != b

    def test_equality_with_str(self):
        s = SecureString("test")
        assert s == "test"
        assert s != "other"

    def test_hash_consistent(self):
        a = SecureString("hello")
        b = SecureString("hello")
        assert hash(a) == hash(b)

    def test_clear(self):
        s = SecureString("secret")
        s.clear()
        assert s.get_secret() == ""
        assert len(s) == 0

    def test_type_error(self):
        with pytest.raises(TypeError):
            SecureString(12345)


class TestMaskSensitive:
    def test_basic(self):
        result = mask_sensitive("password123")
        assert result == "pas********"

    def test_email(self):
        result = mask_sensitive("user@example.com")
        assert result == "use***@example.com"

    def test_email_short_local(self):
        result = mask_sensitive("ab@example.com")
        assert "@example.com" in result

    def test_short_string(self):
        result = mask_sensitive("ab")
        assert result == "**"

    def test_empty(self):
        assert mask_sensitive("") == ""

    def test_custom_visible_chars(self):
        result = mask_sensitive("password123", visible_chars=5)
        assert result == "passw********"


class TestSanitizeForLogging:
    def test_masks_password(self):
        data = {"username": "john", "password": "secret123"}
        result = sanitize_for_logging(data)
        assert result["username"] == "john"
        assert "secret123" not in result["password"]

    def test_masks_api_key(self):
        data = {"api_key": "abc-123-xyz"}
        result = sanitize_for_logging(data)
        assert "abc-123-xyz" not in str(result["api_key"])

    def test_masks_token(self):
        data = {"auth_token": "bearer-xyz"}
        result = sanitize_for_logging(data)
        assert "bearer-xyz" not in str(result["auth_token"])

    def test_preserves_safe_keys(self):
        data = {"name": "John", "age": 30}
        result = sanitize_for_logging(data)
        assert result["name"] == "John"
        assert result["age"] == 30

    def test_handles_nested_dicts(self):
        data = {"config": {"secret_key": "abc"}}
        result = sanitize_for_logging(data)
        assert "abc" not in str(result["config"]["secret_key"])

    def test_handles_secure_string_safe_key(self):
        """SecureString values on non-sensitive keys still get str() conversion."""
        data = {"token": SecureString("mytoken")}
        result = sanitize_for_logging(data)
        # "token" matches sensitive pattern, so it gets str(SecureString) = '*' * min(7, 8) = '*******'
        assert result["token"] == "*******"
        assert "mytoken" not in str(result["token"])

    def test_secure_string_nonsensitive_key(self):
        """SecureString on a non-sensitive key is still converted to str()."""
        data = {"name": SecureString("alice")}
        result = sanitize_for_logging(data)
        assert result["name"] == "*****"


class TestGenerateSessionId:
    def test_returns_string(self):
        sid = generate_session_id()
        assert isinstance(sid, str)
        assert len(sid) > 20

    def test_unique(self):
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100


class TestValidateInputSafe:
    def test_normal_input(self):
        assert validate_input_safe("Hello World") == "Hello World"

    def test_max_length(self):
        with pytest.raises(ValueError):
            validate_input_safe("a" * 10001)

    def test_custom_max_length(self):
        with pytest.raises(ValueError):
            validate_input_safe("hello", max_length=3)

    def test_strips_null_bytes(self):
        result = validate_input_safe("hello\x00world")
        assert "\x00" not in result

    def test_preserves_newlines(self):
        result = validate_input_safe("line1\nline2")
        assert "\n" in result

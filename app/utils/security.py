"""
Security Utilities
==================

Secure handling of sensitive data like credentials.
Ensures passwords and API keys are never logged or exposed.

Usage:
    from app.utils.security import SecureString, mask_sensitive

    password = SecureString("my-secret-password")
    print(password)  # Outputs: ********
    password.get_secret()  # Returns actual value

    masked = mask_sensitive("user@example.com")  # u***@example.com
"""

import gc
import re
import secrets
from typing import Any


class SecureString:
    """
    A string wrapper that prevents accidental exposure of sensitive values.

    The actual value is never revealed in:
    - String representation (__str__, __repr__)
    - Logging
    - Error messages
    - JSON serialization

    Usage:
        password = SecureString("secret123")
        print(password)  # "********"
        actual = password.get_secret()  # "secret123"
    """

    __slots__ = ("_secret_value", "_length")

    def __init__(self, value: str) -> None:
        """
        Initialize with a secret value.

        Args:
            value: The sensitive string to protect.
        """
        if not isinstance(value, str):
            raise TypeError("SecureString value must be a string")
        self._secret_value = value
        self._length = len(value)

    def get_secret(self) -> str:
        """
        Get the actual secret value.

        Returns:
            The unmasked secret string.

        Warning:
            Only use this when you actually need the value.
            Never log or print the result.
        """
        return self._secret_value

    def __str__(self) -> str:
        """Return masked representation."""
        return "*" * min(self._length, 8)

    def __repr__(self) -> str:
        """Return masked representation for debugging."""
        return f"SecureString('{self}')"

    def __len__(self) -> int:
        """Return the length of the secret."""
        return self._length

    def __bool__(self) -> bool:
        """Return True if the secret is not empty."""
        return self._length > 0

    def __eq__(self, other: Any) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        if isinstance(other, SecureString):
            return secrets.compare_digest(self._secret_value, other._secret_value)
        if isinstance(other, str):
            return secrets.compare_digest(self._secret_value, other)
        return False

    def __hash__(self) -> int:
        """Hash the secret value."""
        return hash(self._secret_value)

    def clear(self) -> None:
        """
        Clear the secret from memory.

        Call this when you're done with the secret to minimize
        exposure time in memory.
        """
        self._secret_value = ""
        self._length = 0
        gc.collect()

    def __del__(self) -> None:
        """Clear on deletion."""
        self.clear()


def mask_sensitive(value: str, visible_chars: int = 3) -> str:
    """
    Mask a sensitive string, showing only first few characters.

    Args:
        value: The string to mask.
        visible_chars: Number of characters to show at start.

    Returns:
        Masked string with asterisks.

    Examples:
        >>> mask_sensitive("password123")
        'pas********'
        >>> mask_sensitive("user@example.com")
        'use***@example.com'
    """
    if not value:
        return ""

    # For emails, preserve domain
    if "@" in value:
        local, domain = value.split("@", 1)
        if len(local) <= visible_chars:
            return f"{local[0]}***@{domain}"
        return f"{local[:visible_chars]}***@{domain}"

    # For other strings
    if len(value) <= visible_chars:
        return "*" * len(value)
    return f"{value[:visible_chars]}{'*' * 8}"


def sanitize_for_logging(data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a dictionary for safe logging.

    Masks common sensitive fields like passwords, tokens, and keys.

    Args:
        data: Dictionary that may contain sensitive data.

    Returns:
        New dictionary with sensitive values masked.
    """
    sensitive_patterns = [
        r"password",
        r"secret",
        r"token",
        r"api[_-]?key",
        r"access[_-]?key",
        r"auth",
        r"credential",
        r"private",
        r"otp",
        r"pin",
    ]
    pattern = re.compile("|".join(sensitive_patterns), re.IGNORECASE)

    result = {}
    for key, value in data.items():
        if pattern.search(key):
            if isinstance(value, str):
                result[key] = mask_sensitive(value)
            elif isinstance(value, SecureString):
                result[key] = str(value)
            else:
                result[key] = "********"
        elif isinstance(value, dict):
            result[key] = sanitize_for_logging(value)
        elif isinstance(value, SecureString):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def generate_session_id() -> str:
    """
    Generate a cryptographically secure session ID.

    Returns:
        A URL-safe random string suitable for session identification.
    """
    return secrets.token_urlsafe(32)


def validate_input_safe(value: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize user input.

    Args:
        value: The input string to validate.
        max_length: Maximum allowed length.

    Returns:
        The sanitized input string.

    Raises:
        ValueError: If input exceeds maximum length or contains invalid chars.
    """
    if len(value) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")

    # Remove null bytes and other control characters (except newlines/tabs)
    sanitized = "".join(
        char for char in value if char.isprintable() or char in "\n\t\r"
    )

    return sanitized

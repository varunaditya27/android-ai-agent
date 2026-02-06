"""
Tests for API Key Rotator & LLMClient Key Rotation Integration
===============================================================

Comprehensive tests for:
- ApiKeyRotator: init, validation, deduplication
- Circular rotation: round-robin get_key
- Cooldown: report_rate_limit marks cooling, get_key skips cooling keys
- All-cooling fallback: picks key with shortest remaining cooldown
- Stats: get_stats, record_call, reset
- Thread-safety (basic concurrency smoke test)
- LLMClient integration: _rotate_key_on_rate_limit rebuilds genai.Client
- Config: get_all_api_keys merges keys properly
"""

import os
import time
import threading

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.llm.key_rotator import ApiKeyRotator


# ===================================================================
# ApiKeyRotator: Initialization & Validation
# ===================================================================


class TestApiKeyRotatorInit:
    def test_single_key(self):
        rotator = ApiKeyRotator(["key1"])
        assert rotator.total_keys == 1
        assert rotator.get_key() == "key1"

    def test_multiple_keys(self):
        rotator = ApiKeyRotator(["key1", "key2", "key3"])
        assert rotator.total_keys == 3
        assert rotator.get_key() == "key1"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one API key"):
            ApiKeyRotator([])

    def test_all_empty_strings_raises(self):
        with pytest.raises(ValueError, match="No valid API keys"):
            ApiKeyRotator(["", "  ", ""])

    def test_deduplication(self):
        rotator = ApiKeyRotator(["key1", "key2", "key1", "key3", "key2"])
        assert rotator.total_keys == 3

    def test_strips_whitespace(self):
        rotator = ApiKeyRotator(["  key1  ", "key2\t"])
        assert rotator.get_key() == "key1"
        assert rotator.total_keys == 2

    def test_filters_empty_strings(self):
        rotator = ApiKeyRotator(["", "key1", "", "key2", ""])
        assert rotator.total_keys == 2

    def test_custom_cooldown(self):
        rotator = ApiKeyRotator(["key1"], default_cooldown=120.0)
        assert rotator._default_cooldown == 120.0

    def test_preserves_order(self):
        keys = ["alpha", "beta", "gamma", "delta"]
        rotator = ApiKeyRotator(keys)
        assert rotator.get_key() == "alpha"


# ===================================================================
# Circular Rotation: get_key & report_rate_limit
# ===================================================================


class TestCircularRotation:
    def test_initial_key_is_first(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        assert rotator.get_key() == "a"

    def test_rotate_advances_to_next(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        new_key = rotator.report_rate_limit(cooldown=60.0)
        assert new_key == "b"
        assert rotator.get_key() == "b"

    def test_rotate_wraps_around(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        rotator.report_rate_limit(cooldown=60.0)  # a → b
        rotator.report_rate_limit(cooldown=60.0)  # b → c
        rotator.report_rate_limit(cooldown=60.0)  # c → a (but a is cooling)
        # a, b, c are all cooling; should return the one with shortest cooldown
        key = rotator.get_key()
        assert key in ("a", "b", "c")

    def test_rotate_returns_non_cooling_key(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        rotator.report_rate_limit(cooldown=60.0)  # a → b
        assert rotator.get_key() == "b"
        rotator.report_rate_limit(cooldown=60.0)  # b → c
        assert rotator.get_key() == "c"

    def test_single_key_rotates_to_self(self):
        rotator = ApiKeyRotator(["solo"])
        new_key = rotator.report_rate_limit(cooldown=60.0)
        assert new_key == "solo"

    def test_multiple_rotations_cycle(self):
        rotator = ApiKeyRotator(["a", "b"])
        # Use very short cooldown so keys un-cool quickly
        rotator.report_rate_limit(cooldown=0.001)
        time.sleep(0.01)  # Let cooldown expire
        assert rotator.get_key() in ("a", "b")


# ===================================================================
# Cooldown Tracking
# ===================================================================


class TestCooldownTracking:
    def test_cooling_key_is_skipped(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        rotator.report_rate_limit(cooldown=60.0)  # a is now cooling
        assert rotator.get_key() == "b"

    def test_expired_cooldown_key_is_available(self):
        rotator = ApiKeyRotator(["a", "b"])
        rotator.report_rate_limit(cooldown=0.01)  # a cools for 10ms
        time.sleep(0.02)  # Wait for cooldown to expire
        # Now a is available again; current index is on b
        assert rotator.get_key() == "b"  # stays on b (current index)

    def test_custom_cooldown_per_call(self):
        rotator = ApiKeyRotator(["a", "b"], default_cooldown=120.0)
        rotator.report_rate_limit(cooldown=5.0)  # Override with 5s
        stats = rotator.get_stats()
        assert stats[0]["is_cooling"]  # a should be cooling

    def test_all_keys_cooling_returns_shortest_remaining(self):
        rotator = ApiKeyRotator(["a", "b", "c"])
        # Cool a for 10s, b for 20s, c for 30s
        rotator.report_rate_limit(cooldown=10.0)   # a cooling 10s → moves to b
        rotator.report_rate_limit(cooldown=20.0)   # b cooling 20s → moves to c
        rotator.report_rate_limit(cooldown=30.0)   # c cooling 30s → all cooling
        
        # All are cooling; get_key should return the one expiring soonest (a)
        key = rotator.get_key()
        assert key == "a"

    def test_rate_limit_counter_increments(self):
        rotator = ApiKeyRotator(["a", "b"])
        rotator.report_rate_limit(cooldown=60.0)
        rotator.report_rate_limit(cooldown=60.0)
        stats = rotator.get_stats()
        assert stats[0]["total_rate_limits"] == 1
        assert stats[1]["total_rate_limits"] == 1


# ===================================================================
# Call Tracking & Stats
# ===================================================================


class TestCallTrackingAndStats:
    def test_record_call_increments_counter(self):
        rotator = ApiKeyRotator(["a", "b"])
        rotator.record_call()
        rotator.record_call()
        rotator.record_call()
        stats = rotator.get_stats()
        assert stats[0]["total_calls"] == 3
        assert stats[1]["total_calls"] == 0

    def test_record_call_tracks_per_key(self):
        rotator = ApiKeyRotator(["a", "b"])
        rotator.record_call()  # call on a
        rotator.report_rate_limit(cooldown=0.001)  # move to b
        rotator.record_call()  # call on b
        rotator.record_call()  # call on b
        stats = rotator.get_stats()
        assert stats[0]["total_calls"] == 1
        assert stats[1]["total_calls"] == 2

    def test_get_stats_structure(self):
        rotator = ApiKeyRotator(["a", "b"])
        stats = rotator.get_stats()
        assert len(stats) == 2
        for s in stats:
            assert "index" in s
            assert "total_calls" in s
            assert "total_rate_limits" in s
            assert "is_cooling" in s
            assert "remaining_cooldown" in s

    def test_reset_clears_everything(self):
        rotator = ApiKeyRotator(["a", "b"])
        rotator.record_call()
        rotator.report_rate_limit(cooldown=60.0)
        rotator.record_call()
        
        rotator.reset()
        stats = rotator.get_stats()
        for s in stats:
            assert s["total_calls"] == 0
            assert s["total_rate_limits"] == 0
            assert not s["is_cooling"]


# ===================================================================
# Thread Safety (Smoke Test)
# ===================================================================


class TestThreadSafety:
    def test_concurrent_rotations(self):
        """Ensure no crashes under concurrent access."""
        rotator = ApiKeyRotator(["a", "b", "c", "d", "e"])
        errors = []

        def worker(n):
            try:
                for _ in range(50):
                    rotator.get_key()
                    rotator.record_call()
                    if n % 3 == 0:
                        rotator.report_rate_limit(cooldown=0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent access: {errors}"
        # All keys should have some stats
        stats = rotator.get_stats()
        total_calls = sum(s["total_calls"] for s in stats)
        assert total_calls > 0


# ===================================================================
# LLMClient Key Rotation Integration
# ===================================================================


class TestLLMClientKeyRotation:
    """Test that LLMClient correctly interacts with ApiKeyRotator."""

    def test_client_init_with_rotator(self):
        """LLMClient uses rotator's key instead of config.api_key."""
        from app.llm.models import LLMConfig

        rotator = ApiKeyRotator(["rotator-key-1", "rotator-key-2"])
        config = LLMConfig(api_key="config-key-fallback")

        with patch("app.llm.client.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from app.llm.client import LLMClient
            client = LLMClient(config, key_rotator=rotator)
            # Should use rotator's current key, not config.api_key
            mock_genai.Client.assert_called_with(api_key="rotator-key-1")
            assert client._active_key == "rotator-key-1"

    def test_client_init_without_rotator(self):
        """LLMClient uses config.api_key when no rotator."""
        from app.llm.models import LLMConfig

        config = LLMConfig(api_key="direct-key")

        with patch("app.llm.client.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from app.llm.client import LLMClient
            client = LLMClient(config)
            mock_genai.Client.assert_called_with(api_key="direct-key")
            assert client._key_rotator is None

    def test_rotate_key_on_rate_limit_rebuilds_client(self):
        """_rotate_key_on_rate_limit swaps to new key and rebuilds genai.Client."""
        from app.llm.models import LLMConfig

        rotator = ApiKeyRotator(["key-a", "key-b"])
        config = LLMConfig(api_key="key-a")

        with patch("app.llm.client.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from app.llm.client import LLMClient
            client = LLMClient(config, key_rotator=rotator)

            # Initial client built with key-a
            assert mock_genai.Client.call_count == 1

            # Simulate rate limit rotation
            client._rotate_key_on_rate_limit(cooldown=60.0)

            # Should rebuild client with key-b
            assert mock_genai.Client.call_count == 2
            mock_genai.Client.assert_called_with(api_key="key-b")
            assert client._active_key == "key-b"

    def test_rotate_key_noop_without_rotator(self):
        """_rotate_key_on_rate_limit is a no-op when no rotator."""
        from app.llm.models import LLMConfig

        config = LLMConfig(api_key="solo-key")

        with patch("app.llm.client.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from app.llm.client import LLMClient
            client = LLMClient(config)

            client._rotate_key_on_rate_limit(cooldown=30.0)

            # Only the initial Client call, no rebuild
            assert mock_genai.Client.call_count == 1

    def test_rotate_key_same_key_no_rebuild(self):
        """If rotator returns the same key (single key), don't rebuild."""
        from app.llm.models import LLMConfig

        rotator = ApiKeyRotator(["only-key"])
        config = LLMConfig(api_key="only-key")

        with patch("app.llm.client.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from app.llm.client import LLMClient
            client = LLMClient(config, key_rotator=rotator)

            client._rotate_key_on_rate_limit(cooldown=60.0)

            # Same key returned – no rebuild needed
            assert mock_genai.Client.call_count == 1


# ===================================================================
# Config: get_all_api_keys
# ===================================================================


class TestConfigGetAllApiKeys:
    """Test LLMSettings.get_all_api_keys() merges keys correctly."""

    def test_only_single_key(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="single", gemini_api_keys="")
        assert s.get_all_api_keys() == ["single"]

    def test_comma_separated_keys(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="fallback", gemini_api_keys="a,b,c")
        keys = s.get_all_api_keys()
        assert keys == ["a", "b", "c", "fallback"]

    def test_deduplicates(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="a", gemini_api_keys="a,b,a")
        keys = s.get_all_api_keys()
        assert keys == ["a", "b"]

    def test_strips_whitespace(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="x", gemini_api_keys=" a , b , c ")
        keys = s.get_all_api_keys()
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys

    def test_empty_strings_filtered(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="x", gemini_api_keys=",,,x,,,")
        keys = s.get_all_api_keys()
        assert keys == ["x"]

    def test_single_key_used_as_fallback(self):
        from app.config import LLMSettings

        s = LLMSettings(gemini_api_key="main", gemini_api_keys="k1,k2")
        keys = s.get_all_api_keys()
        # main should be at the end (appended only if not already present)
        assert keys == ["k1", "k2", "main"]

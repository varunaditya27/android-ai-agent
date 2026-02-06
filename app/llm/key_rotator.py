"""
API Key Rotator
===============

Thread-safe circular API key rotator for Gemini API.

When a key hits a rate limit (429 / RESOURCE_EXHAUSTED), the rotator
marks it as cooling-down and advances to the next available key.

Keys from *different* Google Cloud projects each have independent quotas,
so rotating across projects effectively multiplies throughput.

Usage:
    rotator = ApiKeyRotator(["key1", "key2", "key3"])
    key = rotator.get_key()          # → "key1"
    rotator.report_rate_limit()      # marks "key1" as cooling, advances
    key = rotator.get_key()          # → "key2"
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Default cooldown when a key is rate-limited (seconds).
_DEFAULT_COOLDOWN = 60.0


@dataclass
class _KeyState:
    """Internal per-key tracking state."""

    key: str
    index: int
    total_calls: int = 0
    total_rate_limits: int = 0
    cooldown_until: float = 0.0  # Unix timestamp when cooldown expires

    @property
    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def remaining_cooldown(self) -> float:
        return max(0.0, self.cooldown_until - time.time())


class ApiKeyRotator:
    """
    Thread-safe circular API key rotator with per-key cooldown.

    Features:
    - Round-robin rotation across keys
    - Per-key cooldown tracking on rate-limit
    - Automatic skip of cooling keys
    - Falls back to the key with the shortest remaining cooldown
      if ALL keys are cooling simultaneously
    - Thread-safe via a reentrant lock
    """

    def __init__(
        self,
        api_keys: list[str],
        default_cooldown: float = _DEFAULT_COOLDOWN,
    ) -> None:
        if not api_keys:
            raise ValueError("At least one API key is required")

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_keys: list[str] = []
        for k in api_keys:
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                unique_keys.append(k)

        if not unique_keys:
            raise ValueError("No valid API keys provided (all empty or duplicates)")

        self._keys: list[_KeyState] = [
            _KeyState(key=k, index=i) for i, k in enumerate(unique_keys)
        ]
        self._current_index = 0
        self._default_cooldown = default_cooldown
        self._lock = threading.RLock()

        logger.info(
            "ApiKeyRotator initialized",
            total_keys=len(self._keys),
            cooldown_seconds=default_cooldown,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """
        Return the current active API key.

        If the current key is cooling down, advance to the next available
        key. If ALL keys are cooling, return the one whose cooldown
        expires soonest.
        """
        with self._lock:
            # Try to find a non-cooling key starting from current index
            for _ in range(len(self._keys)):
                ks = self._keys[self._current_index]
                if not ks.is_cooling:
                    return ks.key
                # Current key is cooling – try the next one
                self._current_index = (self._current_index + 1) % len(self._keys)

            # ALL keys are cooling – pick the one that expires soonest
            best = min(self._keys, key=lambda k: k.cooldown_until)
            self._current_index = best.index
            logger.warning(
                "All API keys are cooling down, using key with shortest remaining cooldown",
                key_index=best.index,
                remaining_seconds=round(best.remaining_cooldown, 1),
            )
            return best.key

    def report_rate_limit(self, cooldown: Optional[float] = None) -> str:
        """
        Mark the current key as rate-limited and rotate to the next.

        Args:
            cooldown: Seconds to cool down this key.
                      Defaults to ``_DEFAULT_COOLDOWN``.

        Returns:
            The NEW active key after rotation.
        """
        cd = cooldown if cooldown is not None else self._default_cooldown
        with self._lock:
            ks = self._keys[self._current_index]
            ks.cooldown_until = time.time() + cd
            ks.total_rate_limits += 1

            old_index = self._current_index
            # Advance to next key
            self._current_index = (self._current_index + 1) % len(self._keys)

            new_key = self.get_key()  # may skip further cooling keys

            logger.info(
                "API key rotated due to rate limit",
                old_key_index=old_index,
                new_key_index=self._current_index,
                cooldown_seconds=round(cd, 1),
                total_keys=len(self._keys),
            )
            return new_key

    def record_call(self) -> None:
        """Increment the call counter for the current key."""
        with self._lock:
            self._keys[self._current_index].total_calls += 1

    def get_stats(self) -> list[dict]:
        """Return per-key statistics (for logging / debugging)."""
        with self._lock:
            return [
                {
                    "index": ks.index,
                    "total_calls": ks.total_calls,
                    "total_rate_limits": ks.total_rate_limits,
                    "is_cooling": ks.is_cooling,
                    "remaining_cooldown": round(ks.remaining_cooldown, 1),
                }
                for ks in self._keys
            ]

    def reset(self) -> None:
        """Clear all cooldowns and counters (useful in tests)."""
        with self._lock:
            for ks in self._keys:
                ks.cooldown_until = 0.0
                ks.total_calls = 0
                ks.total_rate_limits = 0
            self._current_index = 0

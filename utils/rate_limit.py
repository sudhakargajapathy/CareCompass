"""Simple in-memory rate limiting for user actions."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    """Tracks events per key and enforces a sliding window limit."""

    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = {}

    def check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Check if a key is within the rate limit.

        Args:
            key: Identifier for the caller (e.g., username)
            limit: Max events allowed in the window
            window_seconds: Window size in seconds

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()
        window = self._events.setdefault(key, deque())

        while window and now - window[0] > window_seconds:
            window.popleft()

        if len(window) >= limit:
            retry_after = int(window_seconds - (now - window[0]))
            return False, max(retry_after, 1)

        window.append(now)
        return True, 0


def get_rate_limiter() -> RateLimiter:
    """Get singleton rate limiter instance."""
    if not hasattr(get_rate_limiter, "_instance"):
        get_rate_limiter._instance = RateLimiter()

    return get_rate_limiter._instance

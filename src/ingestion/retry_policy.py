"""Configurable exponential backoff for ingestion tasks."""

from __future__ import annotations

import random


def compute_backoff_delay(
    retry_count: int,
    base_delay: float = 10.0,
    max_delay: float = 300.0,
    jitter: bool = True,
) -> float:
    """Return a bounded exponential delay in seconds.

    ``retry_count`` is one-based from the task queue perspective.  Jitter is
    deliberately bounded to the calculated delay so a retry never exceeds the
    configured maximum.
    """

    exponent = max(0, int(retry_count) - 1)
    delay = min(float(max_delay), float(base_delay) * (2 ** exponent))
    if jitter and delay > 0:
        delay = random.uniform(delay * 0.5, delay)
    return max(0.0, min(float(max_delay), delay))


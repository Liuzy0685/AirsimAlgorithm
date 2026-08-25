"""ConsecutiveInvalidTracker — ROUND 3.3.

Counts consecutive failures and signals when a threshold is reached.
Resets to zero on success.  Used by sector_smoke_test.py to implement
the max_consecutive_invalid safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsecutiveInvalidTracker:
    """Track consecutive invalid frames and signal when to stop.

    Attributes:
        threshold: Maximum number of consecutive failures before stopping.
        count: Current consecutive failure count.
    """

    threshold: int
    count: int = 0

    def record_failure(self) -> int:
        """Record one invalid frame. Returns the new count."""
        self.count += 1
        return self.count

    def record_success(self) -> None:
        """Record one valid frame — resets the counter to zero."""
        self.count = 0

    @property
    def should_stop(self) -> bool:
        """True when the consecutive failure count has reached the threshold."""
        return self.count >= self.threshold

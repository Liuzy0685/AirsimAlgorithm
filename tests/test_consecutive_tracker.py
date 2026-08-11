"""ROUND 3.3: ConsecutiveInvalidTracker unit tests."""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.consecutive_tracker import ConsecutiveInvalidTracker


class TestConsecutiveInvalidTracker:
    def test_initial_count_zero(self):
        t = ConsecutiveInvalidTracker(threshold=3)
        assert t.count == 0
        assert not t.should_stop

    def test_three_failures_stops_at_threshold_3(self):
        t = ConsecutiveInvalidTracker(threshold=3)
        t.record_failure()
        assert not t.should_stop
        t.record_failure()
        assert not t.should_stop
        cnt = t.record_failure()
        assert cnt == 3
        assert t.should_stop

    def test_two_failures_then_success_resets_count(self):
        t = ConsecutiveInvalidTracker(threshold=3)
        t.record_failure()  # count=1
        t.record_failure()  # count=2
        t.record_success()  # count=0
        assert t.count == 0
        assert not t.should_stop

    def test_failure_after_reset_starts_from_one(self):
        t = ConsecutiveInvalidTracker(threshold=3)
        t.record_failure()
        t.record_success()
        cnt = t.record_failure()
        assert cnt == 1
        assert not t.should_stop

    def test_success_does_not_increment(self):
        t = ConsecutiveInvalidTracker(threshold=3)
        t.record_success()
        assert t.count == 0
        t.record_success()
        assert t.count == 0

    def test_threshold_1_stops_immediately(self):
        t = ConsecutiveInvalidTracker(threshold=1)
        cnt = t.record_failure()
        assert cnt == 1
        assert t.should_stop

    def test_threshold_10(self):
        t = ConsecutiveInvalidTracker(threshold=10)
        for i in range(9):
            t.record_failure()
            assert not t.should_stop, f"should not stop at {i+1}"
        t.record_failure()
        assert t.should_stop

    def test_record_failure_returns_count(self):
        t = ConsecutiveInvalidTracker(threshold=5)
        assert t.record_failure() == 1
        assert t.record_failure() == 2
        t.record_success()
        assert t.record_failure() == 1

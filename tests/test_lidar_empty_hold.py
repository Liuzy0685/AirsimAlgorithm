"""Round 9: LiDAR empty-frame health state machine — deterministic tests.

A FRESH timestamp with an empty point cloud is a *transient dropout*, not an
immediate mission abort and not "free space".  These tests pin down the four
LiDAR states (VALID_NONEMPTY / FRESH_EMPTY / STALE / INVALID), the empty-frame
safety hold, the persistent-empty fail-safe, and the invariant that the
intentional zero-velocity hold never feeds a fake stuck / recovery / bypass /
rejoin episode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from sensors.lidar_reader import LidarReader


# ---------------------------------------------------------------------------
# AutomaticMode minimal constructor (mirrors test_bypass_and_bounds.py).
# ---------------------------------------------------------------------------

def _make_minimal_auto():
    from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams

    session = MagicMock()
    session.client = MagicMock()
    session.adapter = MagicMock()
    session.vehicle_name = "Drone1"

    return AutomaticMode(
        session,
        params=AutomaticModeParams(
            target_z_ned=-2.0,
            max_flight_duration_s=0.2,
        ),
        cli_overrides={"planner_mode": "reactive"},
    )


def _empty_frame(ts_ns=1000, point_count=0):
    """A LidarFrame-like object carrying only the fields the empty-hold path reads."""
    return SimpleNamespace(raw_timestamp_ns=ts_ns, point_count=point_count)


# ---------------------------------------------------------------------------
# LiDAR reader mock helpers (mirrors test_lidar_parsing.py).
# ---------------------------------------------------------------------------

def _make_mock_lidar_data(point_cloud_flat, time_stamp=1785762195084642816):
    data = MagicMock()
    data.point_cloud = point_cloud_flat
    data.time_stamp = time_stamp
    data.pose = MagicMock()
    data.pose.position = MagicMock()
    data.pose.position.x_val = 0.2
    data.pose.position.y_val = 0.0
    data.pose.position.z_val = 0.0
    data.pose.orientation = MagicMock()
    data.pose.orientation.w_val = 1.0
    data.pose.orientation.x_val = 0.0
    data.pose.orientation.y_val = 0.0
    data.pose.orientation.z_val = 0.0
    return data


def _make_reader(frame_timeout_seconds=0.5, monotonic_clock=None, stale_poll_threshold=5):
    mock_client = MagicMock()
    adapter = MagicMock(spec=AirSimClientAdapter)
    adapter.vehicle_name = "Drone1"
    adapter.lidar_name = "LidarSensor1"
    adapter.get_raw_client.return_value = mock_client
    return LidarReader(
        adapter,
        frame_timeout_seconds=frame_timeout_seconds,
        stale_poll_threshold=stale_poll_threshold,
        monotonic_clock=monotonic_clock,
    )


def _read_once(reader, raw):
    reader._adapter.get_raw_client().getLidarData.return_value = raw
    return reader.read()


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        self._t += delta


# ---------------------------------------------------------------------------
# Test 1: transient empty frame → no termination, safety hold, no fake recovery
# ---------------------------------------------------------------------------

class TestTransientEmptyHold:
    def test_single_empty_returns_no_termination_and_sets_hold(self):
        auto = _make_minimal_auto()
        lf = _empty_frame(ts_ns=1000)

        term = auto._handle_lidar_empty_frame(
            lf, 100.0, (0.0, 0.0, -2.0), (0.25, 0.0, 0.0),
        )

        assert term is None  # caller must hold, NOT abort
        assert auto._lidar_consecutive_empty == 1
        assert auto._lidar_empty_frames_total == 1
        assert auto._lidar_empty_hold_active is True
        assert auto._last_frame_empty_hold is True
        assert auto._lidar_prev_ts_ns == 1000

    def test_empty_then_valid_does_not_fake_recovery(self):
        """VALID_NONEMPTY → FRESH_EMPTY → VALID_NONEMPTY: the hold re-anchors
        the stuck detector, so the following valid frame is NOT misread as a
        recovery trigger."""
        auto = _make_minimal_auto()

        # One transient empty frame (fresh timestamp).
        term = auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=1), 100.0, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )
        assert term is None

        # The hold reset the stuck detector; a single stationary valid frame is
        # the start of a fresh window, not "stuck"/"needs recovery".
        d = auto._recovery.update(100.05, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0), 0.0)
        assert not d.needs_recovery
        assert not d.is_stuck
        assert d.window_size_frames == 1


# ---------------------------------------------------------------------------
# Test 2: multiple FRESH_EMPTY within grace → continuous hold, no termination
# ---------------------------------------------------------------------------

class TestEmptyWithinGrace:
    def test_multiple_empty_within_grace_no_termination(self):
        auto = _make_minimal_auto()
        now = 100.0
        for i in range(3):  # 3 empty frames, 0.2 s apart → 0.4 s < grace 1.0 s
            term = auto._handle_lidar_empty_frame(
                _empty_frame(ts_ns=1000 + i), now, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
            )
            assert term is None
            now += 0.2

        assert auto._lidar_consecutive_empty == 3
        assert auto._lidar_empty_max_consecutive == 3
        assert auto._lidar_empty_frames_total == 3
        assert auto._lidar_empty_hold_active is True


# ---------------------------------------------------------------------------
# Test 3: persistent empty → lidar_invalid:persistent_empty
# ---------------------------------------------------------------------------

class TestPersistentEmpty:
    def test_persistent_empty_by_frame_count(self):
        auto = _make_minimal_auto()
        now = 100.0
        term = None
        for i in range(5):  # max_consecutive_empty_frames = 5
            term = auto._handle_lidar_empty_frame(
                _empty_frame(ts_ns=1000 + i), now, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
            )
            now += 0.05  # keep total elapsed well below grace (0.25 s < 1.0 s)
        assert term == "lidar_invalid:persistent_empty"
        assert auto._lidar_consecutive_empty == 5

    def test_persistent_empty_by_duration(self):
        auto = _make_minimal_auto()
        term1 = auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=1), 100.0, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )
        assert term1 is None
        # Only 2 consecutive frames, but elapsed 1.5 s >= empty_grace_s 1.0 s.
        term2 = auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=2), 101.5, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )
        assert term2 == "lidar_invalid:persistent_empty"


# ---------------------------------------------------------------------------
# Test 4: frozen timestamp + empty buffer → STALE (never "empty")
# ---------------------------------------------------------------------------

class TestFrozenTimestampStaleNotEmpty:
    def test_frozen_timestamp_empty_buffer_becomes_stale(self):
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        # Baseline valid frame establishes the timestamp.
        first = _read_once(reader, _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100))
        assert first.frame_valid is True

        # Empty buffer with the SAME frozen timestamp, repeated past the
        # poll-count + timeout thresholds → must be STALE, not "empty".
        final = None
        for _ in range(6):
            clock.advance(0.6)
            final = _read_once(reader, _make_mock_lidar_data([], time_stamp=100))

        assert final.frame_valid is False
        assert final.invalid_reason == "stale"


# ---------------------------------------------------------------------------
# Test 5: malformed / NaN → INVALID (distinct from empty/stale)
# ---------------------------------------------------------------------------

class TestMalformedAndBadValuesInvalid:
    def test_malformed_length_not_divisible_by_3(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data([1.0, 2.0, 3.0, 4.0])  # length 4
        frame = reader._build_frame(raw, reader._clock())
        assert frame.frame_valid is False
        assert frame.invalid_reason == "malformed"

    def test_nan_points_classified_bad_values(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data([1.0, 2.0, 3.0, 4.0, float("nan"), 6.0])
        frame = reader._build_frame(raw, reader._clock())
        assert frame.frame_valid is False
        assert frame.invalid_reason == "bad_values"


# ---------------------------------------------------------------------------
# Test 6: empty hold must not accumulate stuck / recovery / bypass / rejoin
# ---------------------------------------------------------------------------

class TestEmptyHoldNoFakeEpisodes:
    def test_empty_hold_resets_stuck_and_progress_accumulators(self):
        auto = _make_minimal_auto()

        # Pre-fill the stuck detector with real stationary history.
        for i in range(30):
            auto._recovery.update(100.0 + i * 0.1, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0), 0.0)
        assert len(auto._recovery._window) > 10

        auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=1000), 200.0, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )

        # Stuck detector cleared → a fresh stationary frame is not "stuck".
        assert len(auto._recovery._window) == 0
        d = auto._recovery.update(200.05, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0), 0.0)
        assert not d.needs_recovery
        assert d.window_size_frames == 1

        # Progress watchdog re-anchored at the hold position/time.
        assert auto._progress_watchdog._start_time == 200.0
        assert auto._progress_watchdog._start_position == (0.0, 0.0)

    def test_empty_hold_creates_no_bypass_or_rejoin_episode(self):
        auto = _make_minimal_auto()

        for i in range(3):
            auto._handle_lidar_empty_frame(
                _empty_frame(ts_ns=1000 + i), 100.0 + i * 0.1,
                (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
            )

        assert auto._bypass.active is False
        assert auto._rejoin.active is False


# ---------------------------------------------------------------------------
# Test 7: post_empty_recovery does not fake a recovery_enter
# ---------------------------------------------------------------------------

class TestPostEmptyRecoveryNoFakeEnter:
    def test_post_empty_recovery_decision_is_not_needs_recovery(self):
        auto = _make_minimal_auto()

        # Transient empty burst (2 frames, within grace).
        auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=1), 100.0, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )
        auto._handle_lidar_empty_frame(
            _empty_frame(ts_ns=2), 100.1, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0),
        )
        # The empty→valid marker is armed for the post_empty_recovery log.
        assert auto._last_frame_empty_hold is True

        # After the hold ends, a single valid frame must NOT read as a spurious
        # recovery (which would log needs_recovery=True → fake enter).
        d = auto._recovery.update(100.15, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0), 0.0)
        assert not d.needs_recovery
        assert not d.is_stuck
        assert d.window_size_frames == 1

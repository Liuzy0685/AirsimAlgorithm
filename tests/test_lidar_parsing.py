"""
Unit tests for LiDAR point-cloud parsing logic.

These tests exercise the validation rules in ``LidarReader`` using
**mock** AirSim ``LidarData`` objects and an injected monotonic clock.
No ``airsim`` import required — runs without UE4 or AirSim.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from sensors.lidar_reader import LidarReader


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_lidar_data(point_cloud_flat, time_stamp=1785762195084642816):
    """Build a mock AirSim LidarData with a Vector3r-like pose."""
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
    """Return a LidarReader backed by a mock adapter."""
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
    """Bind ``raw`` to the mock client and perform one full ``read()``.

    Uses ``read()`` (not ``_build_frame``) so the poll-count bookkeeping
    (``_rpc_calls_since_change``) is exercised — staleness now requires BOTH
    a poll-count threshold AND a wall-clock timeout.
    """
    reader._adapter.get_raw_client().getLidarData.return_value = raw
    return reader.read()


# ---------------------------------------------------------------------------
# Controllable monotonic clock for timeout tests
# ---------------------------------------------------------------------------

class _FakeClock:
    """A simple controllable clock for testing timeout logic."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        self._t += delta


# ---------------------------------------------------------------------------
# Tests — reshape / validation
# ---------------------------------------------------------------------------

class TestLidarReshape:
    """Normal array reshape: 1-D → N×3."""

    def test_normal_reshape(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is True
        assert frame.point_count == 2
        assert frame.point_cloud_sensor.shape == (2, 3)
        assert frame.point_cloud_sensor[0, 0] == 1.0
        assert frame.point_cloud_sensor[1, 2] == 6.0

    def test_empty_array(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data([])
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "empty"
        assert frame.point_cloud_sensor.size == 0

    def test_length_not_divisible_by_3(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0, 4.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "malformed"

    def test_contains_nan(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0, 4.0, float("nan"), 6.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "bad_values"

    def test_contains_inf(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0, 4.0, float("inf"), 6.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "bad_values"

    def test_contains_neg_inf(self):
        reader = _make_reader()
        flat = [1.0, 2.0, float("-inf"), 4.0, 5.0, 6.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "bad_values"


# ---------------------------------------------------------------------------
# Timestamp staleness — revised ROUND 2.2 (None sentinel, timestamp=0)
# ---------------------------------------------------------------------------

class TestTimestampStaleness:
    """Timestamp-based staleness detection with poll-count + timeout."""

    def test_timestamp_updates_normally(self):
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        clock.advance(0.1)
        raw2 = _make_mock_lidar_data([4.0, 5.0, 6.0], time_stamp=200)
        frame2 = _read_once(reader, raw2)
        assert frame2.frame_valid is True
        assert reader.consecutive_stale_count == 0

    def test_short_repeat_still_valid(self):
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        clock.advance(0.1)
        raw2 = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame2 = _read_once(reader, raw2)
        assert frame2.frame_valid is True
        assert reader.consecutive_stale_count > 0

    def test_old_age_with_few_polls_still_valid(self):
        """A blocked control thread inflates wall-clock age WITHOUT the
        perception worker actually polling.  This must NOT be reported as a
        LiDAR stall — the exact false-positive that caused the reported run's
        ``lidar_invalid:stale`` termination."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        # Huge wall-clock gap but only ONE repeat poll → 1 < stale_poll_threshold
        clock.advance(2.0)
        raw2 = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame2 = _read_once(reader, raw2)
        assert frame2.frame_valid is True  # NOT stale

    def test_repeat_exceeds_timeout_and_poll_threshold(self):
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        # Many consecutive same-timestamp polls, each past the timeout →
        # poll count ≥ threshold AND age > timeout → stale.
        stale_frame = None
        for _ in range(6):
            clock.advance(0.6)
            stale_frame = _read_once(
                reader, _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
            )
        assert stale_frame.frame_valid is False
        assert stale_frame.invalid_reason == "stale"
        assert stale_frame.point_cloud_sensor.size == 0  # no usable data

    def test_new_timestamp_restores_validity(self):
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        stale_frame = None
        for _ in range(6):
            clock.advance(0.6)
            stale_frame = _read_once(
                reader, _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=100)
            )
        assert stale_frame.frame_valid is False

        clock.advance(0.1)
        raw3 = _make_mock_lidar_data([7.0, 8.0, 9.0], time_stamp=200)
        frame3 = _read_once(reader, raw3)
        assert frame3.frame_valid is True
        assert reader.consecutive_stale_count == 0


class TestTimestampZero:
    """Timestamp=0 handling — ROUND 2.2: None sentinel, not zero-as-sentinel."""

    def test_first_frame_timestamp_zero_allowed(self):
        """First frame with timestamp=0 and valid points is allowed."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
        frame = reader._build_frame(raw, clock())
        assert frame.frame_valid is True
        assert reader.consecutive_stale_count == 0

    def test_timestamp_zero_short_repeat_valid(self):
        """Timestamp=0 repeated briefly is still valid."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
        frame = reader._build_frame(raw, clock())
        assert frame.frame_valid is True

        clock.advance(0.1)
        raw2 = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
        frame2 = reader._build_frame(raw2, clock())
        assert frame2.frame_valid is True

    def test_timestamp_zero_exceeds_timeout_stale(self):
        """Timestamp=0 repeated past poll-count + timeout becomes stale."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        stale_frame = None
        for _ in range(6):
            clock.advance(0.6)
            stale_frame = _read_once(
                reader, _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
            )
        assert stale_frame.frame_valid is False
        assert stale_frame.invalid_reason == "stale"

    def test_timestamp_zero_to_new_value(self):
        """Timestamp changes from 0 to a new value → recovery."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=0)
        frame = reader._build_frame(raw, clock())
        assert frame.frame_valid is True

        clock.advance(0.1)
        raw2 = _make_mock_lidar_data([4.0, 5.0, 6.0], time_stamp=500)
        frame2 = reader._build_frame(raw2, clock())
        assert frame2.frame_valid is True
        assert reader.consecutive_stale_count == 0

    def test_nonzero_timestamp_original_behavior(self):
        """Original timestamp > 0 behavior still works with None sentinel."""
        clock = _FakeClock(1000.0)
        reader = _make_reader(frame_timeout_seconds=0.5, monotonic_clock=clock)

        raw = _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=500)
        frame = _read_once(reader, raw)
        assert frame.frame_valid is True

        stale_frame = None
        for _ in range(6):
            clock.advance(0.6)
            stale_frame = _read_once(
                reader, _make_mock_lidar_data([1.0, 2.0, 3.0], time_stamp=500)
            )
        assert stale_frame.frame_valid is False
        assert stale_frame.invalid_reason == "stale"


# ---------------------------------------------------------------------------
# RPC errors & unknown_error
# ---------------------------------------------------------------------------

class TestRpcError:
    """RPC exceptions produce invalid frames."""

    def test_rpc_error_produces_invalid_frame(self):
        reader = _make_reader()
        reader._adapter.get_raw_client().getLidarData.side_effect = RuntimeError("RPC timeout")

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason == "rpc_error"
        assert frame.point_cloud_sensor.size == 0

    def test_missing_sensor_float_zero(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data(0.0)
        raw.point_cloud = 0.0
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.invalid_reason == "missing_sensor"


class TestUnknownError:
    """Catch-all for unexpected build failures — tested via read()."""

    def test_build_frame_exception_caught(self, monkeypatch):
        reader = _make_reader()
        raw = _make_mock_lidar_data([1.0, 2.0, 3.0])
        reader._adapter.get_raw_client().getLidarData.return_value = raw

        def _raising_build(*args, **kwargs):
            raise TypeError("unexpected conversion error")

        monkeypatch.setattr(reader, "_build_frame", _raising_build)

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason == "unknown_error"

    def test_point_cloud_not_convertible(self):
        reader = _make_reader()

        class _BadArray:
            def __iter__(self):
                raise ValueError("cannot iterate")

        raw = _make_mock_lidar_data(_BadArray())
        reader._adapter.get_raw_client().getLidarData.return_value = raw

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason in ("unknown_error", "empty")

    def test_numpy_conversion_failure(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data(None)
        raw.point_cloud = None
        reader._adapter.get_raw_client().getLidarData.return_value = raw

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason in ("unknown_error", "missing_sensor")

    def test_pose_float_conversion_failure(self, monkeypatch):
        reader = _make_reader()
        raw = _make_mock_lidar_data([1.0, 2.0, 3.0])
        reader._adapter.get_raw_client().getLidarData.return_value = raw

        def _raising_build(r, mono):
            raise KeyError("unexpected dict key failure")

        monkeypatch.setattr(reader, "_build_frame", _raising_build)

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason == "unknown_error"


# ---------------------------------------------------------------------------
# Received monotonic timing — ROUND 2.2 fix
# ---------------------------------------------------------------------------

class TestReceivedMonotonic:
    """verify received_monotonic_seconds is recorded AFTER RPC."""

    def test_received_monotonic_after_rpc(self):
        """Clock must be read AFTER the RPC returns, not before."""
        clock = _FakeClock(1000.0)

        # reader with injected clock
        reader = _make_reader(monotonic_clock=clock)
        raw = _make_mock_lidar_data([1.0, 2.0, 3.0])
        reader._adapter.get_raw_client().getLidarData.return_value = raw

        frame = reader.read()
        # read() gets clock AFTER getLidarData returns
        assert frame.received_monotonic_seconds == 1000.0

    def test_received_monotonic_on_rpc_error(self):
        """Clock is read after catching the RPC exception."""
        clock = _FakeClock(2000.0)
        reader = _make_reader(monotonic_clock=clock)
        reader._adapter.get_raw_client().getLidarData.side_effect = RuntimeError("down")

        frame = reader.read()
        assert frame.frame_valid is False
        assert frame.invalid_reason == "rpc_error"
        assert frame.received_monotonic_seconds == 2000.0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestFrameMetadata:
    """Metadata fields are populated correctly."""

    def test_vehicle_and_lidar_name(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.vehicle_name == "Drone1"
        assert frame.lidar_name == "LidarSensor1"

    def test_received_monotonic_passed_through(self):
        clock = _FakeClock(500.0)
        reader = _make_reader(monotonic_clock=clock)
        flat = [1.0, 2.0, 3.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, clock())

        assert frame.received_monotonic_seconds == 500.0

    def test_sensor_pose_present(self):
        reader = _make_reader()
        flat = [1.0, 2.0, 3.0]
        raw = _make_mock_lidar_data(flat)
        frame = reader._build_frame(raw, reader._clock())

        assert frame.sensor_pose is not None
        assert frame.sensor_pose["position"]["x"] == 0.2

    def test_invalid_frame_has_empty_point_cloud(self):
        reader = _make_reader()
        raw = _make_mock_lidar_data([])
        frame = reader._build_frame(raw, reader._clock())

        assert frame.frame_valid is False
        assert frame.point_cloud_sensor.size == 0
        assert frame.point_cloud_sensor.shape == (0, 3)

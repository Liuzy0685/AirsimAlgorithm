"""
Unit tests for CollisionReader.

Tests collision detection and is_new_collision_event logic using
mock ``CollisionInfo`` objects.  No AirSim import required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from sensors.collision_reader import CollisionReader


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_collision(
    has_collided=False,
    time_stamp=0,
    object_name="",
    object_id=-1,
    impact_point=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
    position=(0.0, 0.0, 0.0),
    penetration_depth=0.0,
):
    """Build a mock AirSim CollisionInfo."""
    info = MagicMock()
    info.has_collided = has_collided
    info.time_stamp = time_stamp
    info.object_name = object_name
    info.object_id = object_id
    info.impact_point = MagicMock()
    info.impact_point.x_val = impact_point[0]
    info.impact_point.y_val = impact_point[1]
    info.impact_point.z_val = impact_point[2]
    info.normal = MagicMock()
    info.normal.x_val = normal[0]
    info.normal.y_val = normal[1]
    info.normal.z_val = normal[2]
    info.position = MagicMock()
    info.position.x_val = position[0]
    info.position.y_val = position[1]
    info.position.z_val = position[2]
    info.penetration_depth = penetration_depth
    return info


def _make_reader():
    """Return a CollisionReader backed by a mock adapter."""
    adapter = MagicMock(spec=AirSimClientAdapter)
    adapter.vehicle_name = "Drone1"
    return CollisionReader(adapter)


def _set_collision(reader, info):
    reader._adapter.get_raw_client().simGetCollisionInfo.return_value = info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoCollision:
    def test_no_collision(self):
        reader = _make_reader()
        raw = _make_mock_collision(has_collided=False, time_stamp=0)
        _set_collision(reader, raw)

        state = reader.read()
        assert state.has_collided is False
        assert state.is_new_collision_event is False


class TestFirstCollision:
    def test_first_collision_is_new(self):
        reader = _make_reader()
        raw = _make_mock_collision(
            has_collided=True,
            time_stamp=1000,
            object_name="Wall_1",
            object_id=42,
        )
        _set_collision(reader, raw)

        state = reader.read()
        assert state.has_collided is True
        assert state.is_new_collision_event is True
        assert state.object_name == "Wall_1"
        assert state.object_id == 42

    def test_first_collision_timestamp_zero_not_new(self):
        """timestamp=0 should NOT be treated as a new collision."""
        reader = _make_reader()
        raw = _make_mock_collision(
            has_collided=True,
            time_stamp=0,
            object_name="Floor",
        )
        _set_collision(reader, raw)

        state = reader.read()
        assert state.has_collided is True
        assert state.is_new_collision_event is False  # ts=0 should not count


class TestRepeatedCollision:
    def test_same_timestamp_not_new(self):
        reader = _make_reader()
        ts_large = 1785766644042440192

        # First read — new collision.
        raw1 = _make_mock_collision(has_collided=True, time_stamp=ts_large)
        _set_collision(reader, raw1)
        state1 = reader.read()
        assert state1.is_new_collision_event is True
        assert isinstance(state1.raw_timestamp, int)
        assert state1.raw_timestamp == ts_large

        # Second read — same timestamp.
        raw2 = _make_mock_collision(has_collided=True, time_stamp=ts_large)
        _set_collision(reader, raw2)
        state2 = reader.read()
        assert state2.has_collided is True
        assert state2.is_new_collision_event is False

    def test_new_timestamp_is_new(self):
        reader = _make_reader()

        # First collision.
        raw1 = _make_mock_collision(has_collided=True, time_stamp=1000)
        _set_collision(reader, raw1)
        state1 = reader.read()
        assert state1.is_new_collision_event is True

        # Second collision — different timestamp.
        raw2 = _make_mock_collision(has_collided=True, time_stamp=2000)
        _set_collision(reader, raw2)
        state2 = reader.read()
        assert state2.is_new_collision_event is True


class TestLargeTimestampPrecision:
    def test_large_timestamp_preserved_as_int(self):
        """Large uint64 timestamps must remain as int, not float."""
        reader = _make_reader()
        ts_large = 1785766644042440192
        raw = _make_mock_collision(has_collided=True, time_stamp=ts_large)
        _set_collision(reader, raw)

        state = reader.read()
        assert isinstance(state.raw_timestamp, int)
        assert state.raw_timestamp == ts_large

    def test_large_timestamp_roundtrip(self):
        """Large timestamp plus small delta should not lose precision."""
        reader = _make_reader()
        ts1 = 1785766644042440192
        ts2 = 1785766644042440193  # +1

        raw1 = _make_mock_collision(has_collided=True, time_stamp=ts1)
        _set_collision(reader, raw1)
        state1 = reader.read()
        assert state1.is_new_collision_event is True
        assert state1.raw_timestamp == ts1

        raw2 = _make_mock_collision(has_collided=True, time_stamp=ts2)
        _set_collision(reader, raw2)
        state2 = reader.read()
        assert state2.is_new_collision_event is True
        assert state2.raw_timestamp == ts2


class TestFieldParsing:
    def test_all_fields_parsed(self):
        reader = _make_reader()
        raw = _make_mock_collision(
            has_collided=True,
            time_stamp=12345,
            object_name="Tree_01",
            object_id=7,
            impact_point=(10.0, -5.0, 2.0),
            normal=(0.0, 1.0, 0.0),
            position=(9.5, -5.2, 2.0),
            penetration_depth=0.15,
        )
        _set_collision(reader, raw)

        state = reader.read()
        assert state.object_name == "Tree_01"
        assert state.object_id == 7
        assert state.impact_point_ned_m == [10.0, -5.0, 2.0]
        assert state.normal_ned == [0.0, 1.0, 0.0]
        assert state.position_ned_m == [9.5, -5.2, 2.0]
        assert state.penetration_depth == 0.15


class TestCollisionPatterns:
    def test_collision_then_clear(self):
        """Collision occurs, then ends (has_collided goes False)."""
        reader = _make_reader()

        raw1 = _make_mock_collision(has_collided=True, time_stamp=500)
        _set_collision(reader, raw1)
        state1 = reader.read()
        assert state1.is_new_collision_event is True

        raw2 = _make_mock_collision(has_collided=False, time_stamp=500)
        _set_collision(reader, raw2)
        state2 = reader.read()
        assert state2.has_collided is False
        assert state2.is_new_collision_event is False

    def test_collision_ends_then_new_collision(self):
        """Collision ends, then a new one occurs."""
        reader = _make_reader()

        raw1 = _make_mock_collision(has_collided=True, time_stamp=100, object_name="A")
        _set_collision(reader, raw1)
        state1 = reader.read()
        assert state1.is_new_collision_event is True

        raw2 = _make_mock_collision(has_collided=False, time_stamp=100)
        _set_collision(reader, raw2)
        state2 = reader.read()
        assert state2.has_collided is False

        raw3 = _make_mock_collision(has_collided=True, time_stamp=200, object_name="B")
        _set_collision(reader, raw3)
        state3 = reader.read()
        assert state3.is_new_collision_event is True
        assert state3.object_name == "B"

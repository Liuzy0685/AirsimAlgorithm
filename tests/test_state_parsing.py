"""
Unit tests for vehicle-state parsing logic.

Tests ``StateReader`` using mock ``MultirotorState`` objects and an
**injected** fake euler converter.  No ``airsim`` import required —
these tests run without UE4, without ``AIRSIM_PYTHONCLIENT_PATH``,
and without ``pip install airsim``.

Key fact verified:
    ``airsim.to_eularian_angles()`` returns ``(pitch, roll, yaw)``.
    Our injected fake simulates this order; the real runtime uses the
    actual ``airsim`` function.


"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from sensors.state_reader import StateReader


# ---------------------------------------------------------------------------
# Fake euler converter — returns (pitch, roll, yaw) just like
# airsim.to_eularian_angles.  No airsim import needed.
# ---------------------------------------------------------------------------

def fake_euler_converter(quat):
    """Simulate airsim.to_eularian_angles() return order.

    Returns (pitch, roll, yaw) — confirmed at airsim/utils.py:80.
    """
    # Identity → all zeros.
    if (quat.w_val == 1.0 and quat.x_val == 0.0
            and quat.y_val == 0.0 and quat.z_val == 0.0):
        return (0.0, 0.0, 0.0)
    # Non-trivial: use a simple atan2-based approximation for testing.
    # This just needs to be deterministic — real correctness is tested
    # at the integration level.
    import math as _m
    w, x, y, z = quat.w_val, quat.x_val, quat.y_val, quat.z_val
    pitch = _m.atan2(2.0 * (w * y - z * x), 1.0 - 2.0 * (y * y + z * z))
    roll = _m.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = _m.asin(2.0 * (w * z + x * y))
    return (pitch, roll, yaw)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_kinematics(pos=(10.0, 20.0, -5.0),
                          vel=(1.0, 0.5, 0.0),
                          ang=(0.01, 0.02, 0.03),
                          quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    """Build a mock KinematicsState."""
    kin = MagicMock()
    kin.position = MagicMock()
    kin.position.x_val, kin.position.y_val, kin.position.z_val = pos
    kin.linear_velocity = MagicMock()
    kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val = vel
    kin.angular_velocity = MagicMock()
    kin.angular_velocity.x_val, kin.angular_velocity.y_val, kin.angular_velocity.z_val = ang
    kin.orientation = MagicMock()
    kin.orientation.x_val, kin.orientation.y_val, kin.orientation.z_val, kin.orientation.w_val = quat_xyzw
    return kin


def _make_mock_state(kinematics=None, timestamp=1234567890, ready=True, can_arm=True, landed_state=0):
    """Build a mock MultirotorState."""
    state = MagicMock()
    state.kinematics_estimated = kinematics or _make_mock_kinematics()
    state.timestamp = timestamp
    state.ready = ready
    state.can_arm = can_arm
    state.landed_state = landed_state
    return state


def _make_reader(euler_converter=fake_euler_converter):
    """Return a StateReader with injected fake converter."""
    adapter = MagicMock(spec=AirSimClientAdapter)
    adapter.vehicle_name = "Drone1"
    return StateReader(adapter, euler_converter=euler_converter)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNedStateParsing:
    """Verify position and velocity come through in NED."""

    def test_position_ned(self):
        reader = _make_reader()
        raw = _make_mock_state(
            kinematics=_make_mock_kinematics(pos=(100.0, -50.0, 10.0))
        )
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.position_ned_m == [100.0, -50.0, 10.0]

    def test_linear_velocity_ned(self):
        reader = _make_reader()
        raw = _make_mock_state(
            kinematics=_make_mock_kinematics(vel=(2.5, 1.0, -0.5))
        )
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.linear_velocity_ned_mps == [2.5, 1.0, -0.5]

    def test_angular_velocity_body(self):
        reader = _make_reader()
        raw = _make_mock_state(
            kinematics=_make_mock_kinematics(ang=(0.0, 0.0, 0.5))
        )
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.angular_velocity_body_radps == [0.0, 0.0, 0.5]

    def test_timestamp(self):
        reader = _make_reader()
        raw = _make_mock_state(timestamp=999888777)
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.timestamp == 999888777

    def test_ready_and_can_arm(self):
        reader = _make_reader()
        raw = _make_mock_state(ready=True, can_arm=False)
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.ready is True
        assert state.can_arm is False


class TestEulerAngles:
    """Verify the euler converter return order is handled correctly."""

    def test_identity_quaternion(self):
        """Identity quaternion → all zeros."""
        reader = _make_reader()
        raw = _make_mock_state(
            kinematics=_make_mock_kinematics(quat_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.roll_rad == 0.0
        assert state.pitch_rad == 0.0
        assert state.yaw_rad == 0.0

    def test_pitch_return_order(self):
        """
        Confirm we unpack (pitch, roll, yaw) correctly.

        We inject a fake converter that returns a known non-symmetric
        tuple to lock in the unpacking order.
        """

        def _ordered_fake(quat):
            return (0.1, 0.2, 0.3)  # pitch, roll, yaw

        reader = _make_reader(euler_converter=_ordered_fake)
        raw = _make_mock_state()
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.pitch_rad == 0.1
        assert state.roll_rad == 0.2
        assert state.yaw_rad == 0.3

    def test_yaw_only(self):
        """90° yaw with level attitude."""

        def _yaw_only_fake(quat):
            return (0.0, 0.0, math.pi / 2)  # pitch, roll, yaw

        reader = _make_reader(euler_converter=_yaw_only_fake)
        raw = _make_mock_state()
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        state = reader.read()
        assert state.pitch_rad == 0.0
        assert state.roll_rad == 0.0
        assert math.isclose(state.yaw_rad, math.pi / 2)

    def test_default_resolves_euler_lazily(self):
        """Reader without explicit converter resolves it at read time."""
        reader = _make_reader(euler_converter=None)
        raw = _make_mock_state()
        reader._adapter.get_raw_client().getMultirotorState.return_value = raw

        # With airsim not importable, this should raise ImportError.
        # We patch sys.modules to simulate missing airsim.
        import builtins
        original_import = builtins.__import__

        def _block_airsim(name, *args, **kwargs):
            if name == "airsim":
                raise ImportError("No module named 'airsim'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _block_airsim
        try:
            state = reader.read()
            # Should have fallen through and the fake converter
            # would have been None — but we explicitly passed None,
            # so it will try `import airsim` which we blocked.
            # Actually: the converter is None, so it enters the
            # `if converter is None` branch and tries `import airsim`.
            # That raises ImportError.
            pytest.fail("Expected ImportError when airsim not available")
        except ImportError:
            pass  # expected
        finally:
            builtins.__import__ = original_import

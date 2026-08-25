"""
Unit tests for velocity controller safety limits and mock API calls.

All tests use a mock adapter and an injected **fake airsim module** —
no ``airsim`` import required.  Runs without UE4 or AirSim.
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

from control.velocity_controller import VelocityController, VelocityCommandRejected


# ---------------------------------------------------------------------------
# Fake airsim module builder
# ---------------------------------------------------------------------------

def _make_fake_airsim():
    """Return a MagicMock that mimics the airsim module surface used by VelocityController."""
    fake = MagicMock()

    # DrivetrainType enum
    fake.DrivetrainType.MaxDegreeOfFreedom = "MaxDegreeOfFreedom"

    # YawMode constructor
    fake.YawMode.side_effect = lambda is_rate=True, yaw_or_rate=0.0: MagicMock(
        is_rate=is_rate,
        yaw_or_rate=yaw_or_rate,
    )

    return fake


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_airsim():
    return _make_fake_airsim()


@pytest.fixture
def controller(fake_airsim):
    """Return a VelocityController with a mock adapter (writable mode)."""
    adapter = MagicMock()
    adapter._readonly = False
    adapter.vehicle_name = "Drone1"
    adapter._assert_writable = lambda: None  # no-op
    return VelocityController(
        adapter=adapter,
        airsim_module=fake_airsim,
        max_horizontal_speed_mps=2.0,
        max_vertical_speed_mps=0.5,
        max_yaw_rate_radps=0.5,
        command_duration_seconds=0.2,
    )


# ---------------------------------------------------------------------------
# Horizontal speed clamp
# ---------------------------------------------------------------------------

class TestHorizontalSpeedClamp:
    def test_within_limit_passes(self, controller):
        vx, vy, vz = controller._validate_velocity(1.0, 1.0, 0.0)
        assert math.isclose(math.sqrt(vx**2 + vy**2), math.sqrt(2.0))

    def test_exceeds_limit_clamped(self, controller):
        vx, vy, vz = controller._validate_velocity(3.0, 4.0, 0.0)
        assert math.isclose(math.sqrt(vx**2 + vy**2), 2.0, rel_tol=1e-9)
        assert math.isclose(vx / vy, 3.0 / 4.0, rel_tol=1e-9)

    def test_negative_components_clamped(self, controller):
        vx, vy, vz = controller._validate_velocity(-6.0, -8.0, 0.0)
        h = math.sqrt(vx**2 + vy**2)
        assert math.isclose(h, 2.0, rel_tol=1e-9)
        assert vx < 0 and vy < 0


# ---------------------------------------------------------------------------
# Vertical speed clamp
# ---------------------------------------------------------------------------

class TestVerticalSpeedClamp:
    def test_within_limit_passes(self, controller):
        _, _, vz = controller._validate_velocity(0.0, 0.0, 0.3)
        assert vz == 0.3

    def test_positive_vz_clamped(self, controller):
        _, _, vz = controller._validate_velocity(0.0, 0.0, 1.0)
        assert vz == 0.5

    def test_negative_vz_clamped(self, controller):
        _, _, vz = controller._validate_velocity(0.0, 0.0, -1.0)
        assert vz == -0.5


# ---------------------------------------------------------------------------
# Body-frame velocity clamp
# ---------------------------------------------------------------------------

class TestBodyFrameClamp:
    def test_body_frame_horizontal_clamp(self, controller):
        vx, vy, vz = controller._validate_velocity(2.0, 2.0, 0.0)
        h = math.sqrt(vx**2 + vy**2)
        assert h <= 2.0


# ---------------------------------------------------------------------------
# Yaw rate clamp (rad/s internal)
# ---------------------------------------------------------------------------

class TestYawRateClamp:
    def test_within_limit_passes(self, controller):
        yr = controller._validate_yaw_rate(0.3)
        assert yr == 0.3

    def test_exceeds_limit_clamped(self, controller):
        yr = controller._validate_yaw_rate(1.0)
        assert yr == 0.5

    def test_negative_clamped(self, controller):
        yr = controller._validate_yaw_rate(-0.8)
        assert yr == -0.5

    def test_none_passes(self, controller):
        yr = controller._validate_yaw_rate(None)
        assert yr is None


# ---------------------------------------------------------------------------
# NaN / inf rejection
# ---------------------------------------------------------------------------

class TestNanInfRejection:
    def test_vx_nan_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_velocity(float("nan"), 0.0, 0.0)

    def test_vy_inf_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_velocity(0.0, float("inf"), 0.0)

    def test_vz_neg_inf_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_velocity(0.0, 0.0, float("-inf"))

    def test_yaw_rate_nan_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_yaw_rate(float("nan"))

    def test_duration_nan_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_duration(float("nan"))

    def test_vehicle_name_required_world(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller.send_velocity_world_ned(0.0, 0.0, 0.0, vehicle_name=None)

    def test_vehicle_name_required_body(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller.send_velocity_body_frd(0.0, 0.0, 0.0, vehicle_name=None)


# ---------------------------------------------------------------------------
# Duration validation
# ---------------------------------------------------------------------------

class TestDurationValidation:
    def test_default_duration_used(self, controller):
        dur = controller._validate_duration(None)
        assert dur == 0.2

    def test_positive_duration_accepted(self, controller):
        dur = controller._validate_duration(1.0)
        assert dur == 1.0

    def test_zero_duration_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_duration(0.0)

    def test_negative_duration_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_duration(-0.1)

    def test_excessive_duration_rejected(self, controller):
        with pytest.raises(VelocityCommandRejected):
            controller._validate_duration(20.0)


# ---------------------------------------------------------------------------
# YawMode unit conversion (rad/s → deg/s at API boundary)
# ---------------------------------------------------------------------------

class TestYawModeConversion:
    """Verify that rad/s is converted to deg/s at the AirSim boundary."""

    def test_yaw_rate_converted_to_degps(self, controller, fake_airsim):
        """0.5 rad/s → ~28.6479 deg/s."""
        yaw_mode = controller._build_yaw_mode_from_radps(0.5)
        assert math.isclose(yaw_mode.yaw_or_rate, math.degrees(0.5), rel_tol=1e-9)

    def test_yaw_mode_is_rate_true(self, controller):
        """YawMode must be a rate mode."""
        yaw_mode = controller._build_yaw_mode_from_radps(0.3)
        assert yaw_mode.is_rate is True

    def test_yaw_rate_none_default(self, controller):
        """None yaw_rate → default YawMode."""
        yaw_mode = controller._build_yaw_mode_from_radps(None)
        assert yaw_mode.yaw_or_rate == 0.0


# ---------------------------------------------------------------------------
# Mock API integration tests — send_velocity_world_ned
# ---------------------------------------------------------------------------

class TestSendVelocityWorldNedMock:
    """Verify that send_velocity_world_ned calls the correct AirSim API."""

    def test_calls_move_by_velocity_async(self, controller):
        client = controller._adapter.get_raw_client()
        controller.send_velocity_world_ned(1.0, 0.5, -0.2, duration=1.0, vehicle_name="Drone1")

        client.moveByVelocityAsync.assert_called_once()
        args, kwargs = client.moveByVelocityAsync.call_args
        assert args[0] == 1.0   # vx
        assert args[1] == 0.5   # vy
        assert args[2] == -0.2  # vz
        assert args[3] == 1.0   # duration
        assert kwargs["vehicle_name"] == "Drone1"

    def test_uses_module_drivetrain_type(self, controller, fake_airsim):
        """DrivetrainType comes from the airsim module, not client instance."""
        client = controller._adapter.get_raw_client()
        controller.send_velocity_world_ned(0.0, 0.0, 0.0, vehicle_name="Drone1")

        _, kwargs = client.moveByVelocityAsync.call_args
        assert kwargs["drivetrain"] == fake_airsim.DrivetrainType.MaxDegreeOfFreedom

    def test_vehicle_name_passed_correctly(self, controller):
        client = controller._adapter.get_raw_client()
        controller.send_velocity_world_ned(0.0, 0.0, 0.0, vehicle_name="MyDrone")

        _, kwargs = client.moveByVelocityAsync.call_args
        assert kwargs["vehicle_name"] == "MyDrone"

    def test_readonly_rejects(self, controller):
        controller._adapter._assert_writable = lambda: (_ for _ in ()).throw(
            RuntimeError("read-only mode")
        )
        with pytest.raises(RuntimeError):
            controller.send_velocity_world_ned(0.0, 0.0, 0.0, vehicle_name="Drone1")

    def test_nan_does_not_call_api(self, controller):
        client = controller._adapter.get_raw_client()
        with pytest.raises(VelocityCommandRejected):
            controller.send_velocity_world_ned(float("nan"), 0.0, 0.0, vehicle_name="Drone1")
        client.moveByVelocityAsync.assert_not_called()

    def test_invalid_duration_does_not_call_api(self, controller):
        client = controller._adapter.get_raw_client()
        with pytest.raises(VelocityCommandRejected):
            controller.send_velocity_world_ned(0.0, 0.0, 0.0, duration=-1.0, vehicle_name="Drone1")
        client.moveByVelocityAsync.assert_not_called()


# ---------------------------------------------------------------------------
# Mock API integration tests — send_velocity_body_frd
# ---------------------------------------------------------------------------

class TestSendVelocityBodyFrdMock:
    """Verify that send_velocity_body_frd calls the correct AirSim API."""

    def test_calls_move_by_velocity_body_frame_async(self, controller):
        client = controller._adapter.get_raw_client()
        controller.send_velocity_body_frd(0.5, -0.3, 0.1, duration=0.5, vehicle_name="Drone1")

        client.moveByVelocityBodyFrameAsync.assert_called_once()
        args, kwargs = client.moveByVelocityBodyFrameAsync.call_args
        assert args[0] == 0.5   # vx
        assert args[1] == -0.3  # vy
        assert args[2] == 0.1   # vz
        assert args[3] == 0.5   # duration

    def test_uses_module_drivetrain_type(self, controller, fake_airsim):
        client = controller._adapter.get_raw_client()
        controller.send_velocity_body_frd(0.0, 0.0, 0.0, vehicle_name="Drone1")

        _, kwargs = client.moveByVelocityBodyFrameAsync.call_args
        assert kwargs["drivetrain"] == fake_airsim.DrivetrainType.MaxDegreeOfFreedom

    def test_yaw_converted_to_degps_in_body_frame(self, controller, fake_airsim):
        """The yaw_mode passed to the API must have deg/s, not rad/s."""
        client = controller._adapter.get_raw_client()
        controller.send_velocity_body_frd(0.0, 0.0, 0.0, vehicle_name="Drone1", yaw_rate=0.5)

        _, kwargs = client.moveByVelocityBodyFrameAsync.call_args
        yaw_mode = kwargs["yaw_mode"]
        assert math.isclose(yaw_mode.yaw_or_rate, math.degrees(0.5), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Async control — hover / takeoff / land use .join()
# ---------------------------------------------------------------------------

class TestControlJoinBehavior:
    """takeoff, hover, land all use .join() to synchronize."""

    def test_takeoff_calls_join(self, controller):
        client = controller._adapter.get_raw_client()
        mock_future = MagicMock()
        client.takeoffAsync.return_value = mock_future

        controller.takeoff(vehicle_name="Drone1")
        mock_future.join.assert_called_once()

    def test_hover_calls_hover_async_join(self, controller):
        """hover() uses hoverAsync().join(), NOT a 0.2 s zero-velocity command."""
        client = controller._adapter.get_raw_client()
        mock_future = MagicMock()
        client.hoverAsync.return_value = mock_future

        controller.hover(vehicle_name="Drone1")
        client.hoverAsync.assert_called_once_with(vehicle_name="Drone1")
        mock_future.join.assert_called_once()
        # Must NOT call moveByVelocityAsync for hover.
        client.moveByVelocityAsync.assert_not_called()

    def test_land_calls_join(self, controller):
        client = controller._adapter.get_raw_client()
        mock_future = MagicMock()
        client.landAsync.return_value = mock_future

        controller.land(vehicle_name="Drone1")
        mock_future.join.assert_called_once()

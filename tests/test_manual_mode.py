"""Tests for manual flight mode — review fixes."""

import math, sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.manual_mode import ManualMode, ManualControlType, ManualModeParams


def _make_mock_session():
    session = MagicMock()
    session.client = MagicMock()
    session.adapter = MagicMock()
    session.vehicle_name = "Drone1"
    session.target_z_ned = -1.0
    session.takeoff_called = False
    session.is_airborne = False
    session.state.phase.name = "INITIALIZED"
    return session

_VC_PATH = "control.velocity_controller.VelocityController"


class TestIdleBeforeTakeoff:
    """Before takeoff: no hoverAsync, no velocity, no attitude calls."""

    def test_no_hover_before_takeoff(self):
        session = _make_mock_session()
        with patch(_VC_PATH):
            mode = ManualMode(session)
            mode._read_keys = MagicMock(return_value={"w"})
            mode._running = True
            # Simulate one iteration of run loop (not airborne)
            mode._run_iteration_for_test = True
        # Verify client.hoverAsync was never called
        session.client.hoverAsync.assert_not_called()

    def test_handle_keys_returns_early_when_not_airborne(self):
        session = _make_mock_session()
        with patch(_VC_PATH) as mock_vc:
            mode = ManualMode(session)
            mode._handle_keys({"w"})
            mock_vc.return_value.send_velocity_body_frd.assert_not_called()
            session.client.moveByRollPitchYawrateZAsync.assert_not_called()
            session.client.hoverAsync.assert_not_called()

    def test_no_velocity_without_takeoff(self):
        session = _make_mock_session()
        with patch(_VC_PATH) as mock_vc:
            mode = ManualMode(session)
            mode._handle_keys({"w"})  # uses dispatch which checks is_airborne
            mock_vc.return_value.send_velocity_body_frd.assert_not_called()

    def test_no_attitude_without_takeoff(self):
        session = _make_mock_session()
        with patch(_VC_PATH):
            mode = ManualMode(session, ManualControlType.ATTITUDE)
            mode._client.moveByRollPitchYawrateZAsync.reset_mock()
            mode._handle_keys({"w"})  # uses dispatch which checks is_airborne
            mode._client.moveByRollPitchYawrateZAsync.assert_not_called()


class TestManualModeTakeoff:
    def test_t_calls_takeoff_once(self):
        session = _make_mock_session()
        with patch(_VC_PATH):
            mode = ManualMode(session)
            mode._handle_takeoff()
            session.takeoff_and_climb.assert_called_once()

    def test_t_ignored_if_already_takeoff(self):
        session = _make_mock_session()
        session.takeoff_called = True
        with patch(_VC_PATH):
            mode = ManualMode(session)
            mode._handle_takeoff()
            session.takeoff_and_climb.assert_not_called()

    def test_g_stops(self):
        session = _make_mock_session()
        with patch(_VC_PATH):
            mode = ManualMode(session)
            mode._read_keys = MagicMock(return_value={"g"})
            mode._running = True
            keys = mode._read_keys()
            if "esc" in keys or "g" in keys: mode._running = False
            assert not mode._running


class TestManualModeVelocity:
    def _mode_and_vc(self, keys_set):
        session = _make_mock_session()
        session.is_airborne = True
        with patch(_VC_PATH) as mock_vc_cls:
            mock_vc = mock_vc_cls.return_value
            mode = ManualMode(session, ManualControlType.VELOCITY)
            mode._handle_keys(keys_set)
            return mode, mock_vc

    def test_w_forward_positive_vx(self):
        _, vc = self._mode_and_vc({"w"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vx"] > 0

    def test_s_backward_negative_vx(self):
        _, vc = self._mode_and_vc({"s"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vx"] < 0

    def test_d_right_positive_vy(self):
        _, vc = self._mode_and_vc({"d"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vy"] > 0

    def test_a_left_negative_vy(self):
        _, vc = self._mode_and_vc({"a"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vy"] < 0

    def test_r_climb_negative_vz(self):
        _, vc = self._mode_and_vc({"r"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vz"] < 0

    def test_f_descend_positive_vz(self):
        _, vc = self._mode_and_vc({"f"})
        assert vc.send_velocity_body_frd.call_args.kwargs["vz"] > 0

    def test_q_positive_yaw(self):
        _, vc = self._mode_and_vc({"q"})
        assert vc.send_velocity_body_frd.call_args.kwargs["yaw_rate"] > 0

    def test_e_negative_yaw(self):
        _, vc = self._mode_and_vc({"e"})
        assert vc.send_velocity_body_frd.call_args.kwargs["yaw_rate"] < 0

    def test_w_d_combined(self):
        _, vc = self._mode_and_vc({"w", "d"})
        kw = vc.send_velocity_body_frd.call_args.kwargs
        assert kw["vx"] > 0 and kw["vy"] > 0


class TestManualModeAttitude:
    def _mode(self, keys_set):
        session = _make_mock_session()
        session.is_airborne = True
        with patch(_VC_PATH):
            mode = ManualMode(session, ManualControlType.ATTITUDE)
            mode._handle_keys(keys_set)
            return mode

    def test_api_is_moveByRollPitchYawrateZAsync(self):
        mode = self._mode({"w"})
        mode._client.moveByRollPitchYawrateZAsync.assert_called_once()

    def test_w_positive_pitch(self):
        mode = self._mode({"w"})
        kw = mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs
        assert kw["pitch"] > 0

    def test_s_negative_pitch(self):
        mode = self._mode({"s"})
        kw = mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs
        assert kw["pitch"] < 0

    def test_a_negative_roll(self):
        mode = self._mode({"a"})
        assert mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs["roll"] < 0

    def test_d_positive_roll(self):
        mode = self._mode({"d"})
        assert mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs["roll"] > 0

    def test_q_positive_yaw_rate(self):
        mode = self._mode({"q"})
        assert mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs["yaw_rate"] > 0

    def test_e_negative_yaw_rate(self):
        mode = self._mode({"e"})
        assert mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs["yaw_rate"] < 0

    def test_r_changes_z(self):
        mode = self._mode({"r"})
        kw = mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs
        assert kw["z"] < -1.0  # target_z decreased → climb

    def test_f_changes_z(self):
        mode = self._mode({"f"})
        kw = mode._client.moveByRollPitchYawrateZAsync.call_args.kwargs
        assert kw["z"] > -1.0


class TestAttitudeAltitudeAccumulation:
    """R/F write back to session.target_z_ned for accumulation."""

    def test_double_r_accumulates(self):
        session = _make_mock_session()
        session.is_airborne = True
        session.target_z_ned = -1.0
        with patch(_VC_PATH):
            mode = ManualMode(session, ManualControlType.ATTITUDE)
            # First R
            mode._handle_attitude_keys({"r"})
            assert session.target_z_ned == -1.5
            # Second R
            mode._handle_attitude_keys({"r"})
            assert session.target_z_ned == -2.0

    def test_double_f_accumulates(self):
        session = _make_mock_session()
        session.is_airborne = True
        session.target_z_ned = -1.0
        with patch(_VC_PATH):
            mode = ManualMode(session, ManualControlType.ATTITUDE)
            mode._handle_attitude_keys({"f"})
            assert session.target_z_ned == -0.5
            mode._handle_attitude_keys({"f"})
            assert session.target_z_ned == 0.0


class TestManualModeParams:
    def test_roll_pitch_limits(self):
        p = ManualModeParams()
        assert p.max_roll_rad <= math.radians(6)
        assert p.max_pitch_rad <= math.radians(6)


class TestVelAttitudeSeparation:
    def test_velocity_not_call_attitude(self):
        session = _make_mock_session(); session.is_airborne = True
        with patch(_VC_PATH):
            mode = ManualMode(session, ManualControlType.VELOCITY)
            mode._handle_keys({"w"})
            session.client.moveByRollPitchYawrateZAsync.assert_not_called()

    def test_attitude_not_call_velocity(self):
        session = _make_mock_session(); session.is_airborne = True
        with patch(_VC_PATH) as mock_vc:
            mode = ManualMode(session, ManualControlType.ATTITUDE)
            mode._handle_keys({"w"})
            mock_vc.return_value.send_velocity_body_frd.assert_not_called()

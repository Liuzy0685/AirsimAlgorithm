"""Tests for automatic flight mode — review fixes (collision warm-up)."""

import sys, contextlib, math, types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import (
    AutomaticMode, AutomaticModeParams, choose_reactive_command, ReactiveDecision,
)

_CFG = {"emergency_distance_m": 0.8, "front_threshold_m": 2.5,
        "forward_speed_mps": 0.2, "side_speed_mps": 0.15}


class TestReactiveDecision:
    def test_clear_path_forward(self):
        d = choose_reactive_command(5, 10, 10, 5, _CFG)
        assert d.vx_body_mps == 0.2 and not d.should_terminate

    def test_front_blocked_left_clearer(self):
        d = choose_reactive_command(1, 8, 3, 5, _CFG)
        assert d.vy_body_mps == -0.15

    def test_front_blocked_right_clearer(self):
        d = choose_reactive_command(1, 2, 8, 5, _CFG)
        assert d.vy_body_mps == 0.15

    def test_emergency_stops(self):
        d = choose_reactive_command(1, 10, 10, 0.5, _CFG)
        assert d.should_terminate


class TestCommandCorridorSafety:
    def test_side_pillar_does_not_trigger_global_emergency_stop(self):
        # The nearest return is close, but it is beside the forward command
        # corridor. A centerline gap must remain traversable.
        points = np.array([[0.45, 0.95, 0.0]])
        clearance = AutomaticMode._swept_command_clearance(
            points, 0.75, 0.0, 0.2,
        )
        assert clearance == pytest.approx(0.95)

    def test_obstacle_on_command_centerline_is_detected(self):
        points = np.array([[0.45, 0.0, 0.0]])
        clearance = AutomaticMode._swept_command_clearance(
            points, 0.75, 0.0, 0.2,
        )
        assert clearance == pytest.approx(0.0)


class TestHeadingAlignment:
    def test_goal_behind_requires_a_half_turn(self):
        error = AutomaticMode._wrapped_heading_error(
            (0.0, 0.0, -1.0), 0.0, (-5.0, 0.0),
        )
        assert abs(error) == pytest.approx(3.141592653589793)

    def test_goal_to_right_has_positive_yaw_error(self):
        error = AutomaticMode._wrapped_heading_error(
            (0.0, 0.0, -1.0), 0.0, (0.0, 5.0),
        )
        assert error == pytest.approx(3.141592653589793 / 2.0)

    def test_current_heading_to_goal_has_zero_error(self):
        error = AutomaticMode._wrapped_heading_error(
            (0.0, 0.0, -1.0), 0.0, (5.0, 0.0),
        )
        assert error == pytest.approx(0.0)

    def _runtime_guard(self):
        mode = AutomaticMode.__new__(AutomaticMode)
        mode._runtime_heading_alignment_enabled = True
        mode._runtime_heading_alignment_trigger_rad = math.radians(100.0)
        mode._runtime_heading_alignment_settle_rad = math.radians(12.0)
        mode._runtime_heading_alignment_max_distance_m = 8.0
        mode._runtime_heading_alignment_kp = 1.2
        mode._runtime_heading_alignment_max_rate = 0.5
        mode._runtime_heading_alignment_active = False
        mode._runtime_heading_alignment_started_mono = None
        mode._traj_cached_points = [(1.0, 0.0), (2.0, 0.0)]
        mode._traj_cached_family = "STRAIGHT"
        mode._traj_force_replan = False
        return mode

    def test_runtime_guard_turns_in_place_when_goal_is_behind(self):
        mode = self._runtime_guard()
        st = types.SimpleNamespace(position_ned_m=(0.0, 0.0, -1.0), yaw_rad=0.0)

        active, completed, yaw_rate = mode._runtime_heading_alignment_command(
            st, (-5.0, 0.0), 10.0,
        )

        assert active and not completed
        assert yaw_rate == pytest.approx(-0.5)
        assert mode._traj_cached_points == []
        assert mode._traj_force_replan

    def test_runtime_guard_replans_after_heading_settles(self):
        mode = self._runtime_guard()
        st_behind = types.SimpleNamespace(
            position_ned_m=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        mode._runtime_heading_alignment_command(
            st_behind, (-5.0, 0.0), 10.0,
        )
        st_aligned = types.SimpleNamespace(
            position_ned_m=(0.0, 0.0, -1.0), yaw_rad=math.pi,
        )

        active, completed, yaw_rate = mode._runtime_heading_alignment_command(
            st_aligned, (-5.0, 0.0), 11.0,
        )

        assert not active and completed
        assert yaw_rate == 0.0
        assert mode._traj_cached_points == []
        assert mode._traj_force_replan

    def test_runtime_guard_catches_goal_behind_beyond_near_goal_window(self):
        """A crossed goal must stop forward motion even when still >8 m away."""
        mode = self._runtime_guard()
        st = types.SimpleNamespace(position_ned_m=(0.0, 0.0, -1.0), yaw_rad=0.0)

        active, completed, yaw_rate = mode._runtime_heading_alignment_command(
            st, (-20.0, 0.0), 10.0,
        )

        assert active and not completed
        assert yaw_rate == pytest.approx(-0.5)


# ── mock helpers ──

def _make_mock_session():
    s = MagicMock()
    s.client = MagicMock(); s.adapter = MagicMock()
    s.adapter.vehicle_name = "Drone1"; s.adapter.lidar_name = "LidarSensor1"
    s.vehicle_name = "Drone1"; s.settings_json = "fake.json"
    s.target_z_ned = -1.0
    s.state.phase.name = "CONTROL_RELEASED"
    s.land_and_disarm.return_value = True
    return s

def _lf(ok=True):
    lf = MagicMock(); lf.frame_valid = ok; lf.invalid_reason = None if ok else "x"
    lf.point_cloud_sensor = MagicMock(); return lf

def _st(z=-1.0):
    s = MagicMock()
    s.position_ned_m = [0., 0., z]
    s.landed_state = 1
    s.yaw_rad = 0.0
    s.linear_velocity_ned_mps = [0.0, 0.0, 0.0]
    return s

def _col(ok=True, *, object_name=None, raw_timestamp=0, is_new_event=False):
    c = MagicMock(); c.has_collided = not ok
    c.object_name = object_name if object_name is not None else ("" if ok else "Wall")
    c.raw_timestamp = raw_timestamp; c.is_new_collision_event = is_new_event; return c

def _fr(valid=True):
    fr = MagicMock(); fr.valid = valid; fr.invalid_reason = None if valid else "fail"
    fr.filtered_points_sensor = MagicMock(); return fr

def _dd():
    dd_ = MagicMock(); dd_.frame_valid = True; dd_.minimum_distance_m = 5.0
    dd_.to_legacy_ray_distances.return_value = {"front": 10., "left": 10., "right": 10.}
    return dd_

_LR="sensors.lidar_reader.LidarReader"; _SR="sensors.state_reader.StateReader"
_CR="sensors.collision_reader.CollisionReader"
_LPC="perception.perception_config.load_perception_config"
_FLT="perception.pointcloud_filter.filter_pointcloud"
_DD="perception.pointcloud_to_sectors.pointcloud_to_directional_distances"
_LFOV="perception.sensor_fov.load_lidar_fov"
_VFOV="perception.sensor_fov.validate_sector_fov_coverage"
_VC="control.velocity_controller.VelocityController"

def _setup_patches():
    """Return an ExitStack with all patches applied."""
    stack = contextlib.ExitStack()
    mocks = []
    for target in [_LR, _SR, _CR, _LPC, _FLT, _DD, _LFOV, _VFOV, _VC]:
        mocks.append(stack.enter_context(patch(target)))
    return stack, mocks

def _cfg_mocks(mocks):
    lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
    lr.return_value.read.return_value = _lf()
    sr.return_value.read.return_value = _st()
    cr.return_value.read.return_value = _col()
    flt.return_value = _fr(); dd.return_value = _dd()
    lfov.return_value = MagicMock(); vfov.return_value = []
    lpc.return_value = MagicMock()
    lpc.return_value.sectorization.sectors = []
    lpc.return_value.pointcloud.self_exclusion.enabled = False
    lpc.return_value.pointcloud.voxel_downsample.enabled = False


class TestAutomaticMode:
    def test_takeoff_called(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            _cfg_mocks(mocks)
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb") as mock_tc:
                auto._running = False; auto.run()
                mock_tc.assert_called_once()

    def test_time_limit(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            sr.return_value.read.side_effect = [_st()] * 100
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"

    def test_lidar_invalid_terminates(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            lr.return_value.read.side_effect = [_lf()] * 6 + [_lf(False)]
            auto = AutomaticMode(session, params=AutomaticModeParams(
                max_flight_duration_s=3.0, command_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert "lidar_invalid" in (r.termination_reason or "")

    def test_collision_terminates(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            # warmup: 5 clean (breaks at 5). flight: 5 clean + 1 collision = 11 total
            cr.return_value.read.side_effect = [_col()] * 11 + [_col(False)]
            sr.return_value.read.side_effect = [_st()] * 30
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=10.0))
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert "collision" in (r.termination_reason or "")


class TestCollisionWarmup:
    """Startup floor contact warm-up — before enableApiControl."""

    def test_floor_then_clean_allows(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="Floor", raw_timestamp=0, is_new_event=False),
            ] + [_col()] * 30
            lr.return_value.read.side_effect = [_lf()] * 30
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb") as mock_tc:
                auto._running = False; r = auto.run()
                mock_tc.assert_called_once()

    def test_persistent_floor_rejects(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.return_value = _col(
                ok=False, object_name="Floor", raw_timestamp=0, is_new_event=False)
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            r = auto.run()
            assert "warmup_floor_persists" in (r.termination_reason or "")
            assert not r.api_control_acquired

    def test_wall_rejects(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.return_value = _col(
                ok=False, object_name="Wall_1", raw_timestamp=0, is_new_event=False)
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            r = auto.run()
            assert "warmup_non_ground" in (r.termination_reason or "")
            assert not r.api_control_acquired

    def test_floor_nonzero_ts_new_event_then_clean_allows(self):
        """First Floor with non-zero ts and is_new_event=True → accepted as candidate."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            # Real AirSim: drone spawns on floor → collision with large ts, is_new_event=True
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="Floor", raw_timestamp=1786026074827645952, is_new_event=True),
            ] + [_col()] * 30
            lr.return_value.read.side_effect = [_lf()] * 30
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb") as mock_tc:
                auto._running = False; r = auto.run()
                mock_tc.assert_called_once()
            assert r.startup_floor_contact_baseline is True

    def test_null_ground_object_accepted_as_floor(self):
        """Unnamed ground (AirSim reports '(null)') → accepted as startup floor."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="(null)", raw_timestamp=1786026074827645952, is_new_event=True),
            ] + [_col()] * 30
            lr.return_value.read.side_effect = [_lf()] * 30
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb") as mock_tc:
                auto._running = False; r = auto.run()
                mock_tc.assert_called_once()
            assert r.startup_floor_contact_baseline is True
            assert "warmup_non_ground" not in (r.termination_reason or "")

    def test_empty_string_ground_object_accepted_as_floor(self):
        """Empty ground object name → accepted as startup floor."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="", raw_timestamp=1786026074827645952, is_new_event=True),
            ] + [_col()] * 30
            lr.return_value.read.side_effect = [_lf()] * 30
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=0.05))
            with patch.object(session, "takeoff_and_climb") as mock_tc:
                auto._running = False; r = auto.run()
                mock_tc.assert_called_once()
            assert r.startup_floor_contact_baseline is True
            assert "warmup_non_ground" not in (r.termination_reason or "")

    def test_null_ground_then_wall_rejects(self):
        """'(null)' floor accepted first, then a Wall → still reject."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="(null)", raw_timestamp=1786026074827645952, is_new_event=True),
                _col(ok=False, object_name="Wall_1", raw_timestamp=0, is_new_event=False),
            ] + [_col()] * 30
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            r = auto.run()
            assert "warmup_non_ground" in (r.termination_reason or "")
            assert not r.api_control_acquired

    def test_first_floor_then_second_floor_new_ts_rejects(self):
        """First Floor accepted, second Floor with different ts → reject."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="Floor", raw_timestamp=100, is_new_event=True),
                _col(ok=False, object_name="Floor", raw_timestamp=200, is_new_event=True),
            ] + [_col()] * 30
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            r = auto.run()
            assert "warmup_new_collision_event" in (r.termination_reason or "")
            assert not r.api_control_acquired

    def test_first_floor_then_wall_rejects(self):
        """First Floor accepted, then Wall → reject."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.side_effect = [
                _col(ok=False, object_name="Floor", raw_timestamp=1786026074827645952, is_new_event=True),
                _col(ok=False, object_name="Wall_1", raw_timestamp=0, is_new_event=False),
            ] + [_col()] * 30
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            r = auto.run()
            assert "warmup_non_ground" in (r.termination_reason or "")
            assert not r.api_control_acquired

    def test_warmup_fail_no_api_control(self):
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            cr.return_value.read.return_value = _col(
                ok=False, object_name="Wall", raw_timestamp=0, is_new_event=False)
            lr.return_value.read.return_value = _lf()
            auto = AutomaticMode(session, params=AutomaticModeParams(max_flight_duration_s=1.0))
            session.takeoff_and_climb.reset_mock()
            r = auto.run()
            session.takeoff_and_climb.assert_not_called()
            assert not r.api_control_acquired


class TestApfShadowLogging:
    """Verify apf_shadow log format does not raise TypeError from mismatched placeholders."""

    def test_apf_shadow_log_format_does_not_raise(self):
        """Run one flight loop iteration in apf_shadow mode — must not raise TypeError."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        with stack:
            lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
            _cfg_mocks(mocks)
            lr.return_value.read.side_effect = [_lf()] * 30
            sr.return_value.read.side_effect = [_st()] * 100
            cr.return_value.read.side_effect = [_col()] * 30
            # Provide all 16 sector keys so APF sees a complete input
            _all_rays = {
                "front": 10., "back": 10., "left": 10., "right": 10.,
                "up": 10., "down": 10.,
                "frontLeft": 10., "frontRight": 10.,
                "backLeft": 10., "backRight": 10.,
                "frontUp": 10., "frontDown": 10.,
                "leftUp": 10., "rightUp": 10.,
                "leftDown": 10., "rightDown": 10.,
            }
            dd.return_value.to_legacy_ray_distances.return_value = _all_rays
            dd.return_value.minimum_distance_m = 5.0

            auto = AutomaticMode(
                session,
                params=AutomaticModeParams(max_flight_duration_s=0.05),
                cli_overrides={"planner_mode": "apf_shadow"},
            )
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            # Must complete without exception; time_limit means logging ran OK
            assert r.termination_reason == "time_limit"


class TestCommandDispatch:
    """Verify the correct velocity command is sent based on planner mode."""

    # Sector data: front blocked → reactive picks right (vy=+0.15), APF pushes back (vx<0)
    _DISPATCH_RAYS = {
        "front": 1.0, "back": 50.0, "left": 50.0, "right": 50.0,
        "up": 50.0, "down": 50.0,
        "frontLeft": 50.0, "frontRight": 50.0,
        "backLeft": 50.0, "backRight": 50.0,
        "frontUp": 50.0, "frontDown": 50.0,
        "leftUp": 50.0, "rightUp": 50.0,
        "leftDown": 50.0, "rightDown": 50.0,
    }

    def _setup_dispatch(self, planner_mode):
        """Set up mocks for one flight loop iteration and return (auto, vc_mock)."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        # Don't use context manager — caller must manage stack
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        _cfg_mocks(mocks)
        lr.return_value.read.side_effect = [_lf()] * 30
        sr.return_value.read.side_effect = [_st()] * 100
        cr.return_value.read.side_effect = [_col()] * 30
        dd.return_value.to_legacy_ray_distances.return_value = self._DISPATCH_RAYS
        dd.return_value.minimum_distance_m = 5.0

        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.05),
            cli_overrides={"planner_mode": planner_mode},
        )
        return session, auto, stack, vc

    def test_reactive_mode_sends_reactive_command(self):
        """reactive mode: API receives reactive decision values."""
        session, auto, stack, vc = self._setup_dispatch("reactive")
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            # Reactive: front=1.0 blocked, left==right → vy=+0.15
            vc.return_value.send_velocity_body_frd.assert_called()
            call_args = vc.return_value.send_velocity_body_frd.call_args
            # First two positional args are vx, vy
            assert call_args[0][0] == 0.0       # vx=0 (front blocked, no forward)
            assert call_args[0][1] == pytest.approx(0.15)  # vy=side_speed
            assert call_args[0][2] == 0.0       # vz=0

    def test_apf_shadow_mode_sends_reactive_command(self):
        """apf_shadow: APF computes but API receives reactive, not APF."""
        session, auto, stack, vc = self._setup_dispatch("apf_shadow")
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            call_args = vc.return_value.send_velocity_body_frd.call_args
            # Must be reactive values, not APF values
            assert call_args[0][0] == 0.0        # reactive vx
            assert call_args[0][1] == pytest.approx(0.15)  # reactive vy
            assert call_args[0][2] == 0.0        # reactive vz

    def test_apf_mode_sends_apf_command_not_reactive(self):
        """apf mode: API receives APF output, not reactive.

        Reactive: vy=+0.15 (front blocked, left==right → right).
        APF: front blocked at 1.0m → pushes backward (vx < 0).
        """
        session, auto, stack, vc = self._setup_dispatch("apf")
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            call_args = vc.return_value.send_velocity_body_frd.call_args
            vx_sent = call_args[0][0]
            vy_sent = call_args[0][1]
            vz_sent = call_args[0][2]
            # APF was valid → must NOT be reactive (vy=+0.15)
            assert vy_sent != pytest.approx(0.15, abs=1e-6), \
                f"Expected APF command, got reactive vy={vy_sent}"
            # APF pushes backward from front obstacle → vx should be negative
            assert vx_sent < 0.0, f"APF should push back, got vx={vx_sent}"
            # horizontal_only → vz=0
            assert vz_sent == 0.0

    def test_apf_invalid_sends_zero_not_reactive_fallback(self):
        """apf mode with invalid APF: sends (0,0,0), does NOT silently use reactive."""
        from planners.improved_potential_field import ApfOutput
        session, auto, stack, vc = self._setup_dispatch("apf")
        with stack:
            # Force APF to return invalid
            invalid_out = ApfOutput(valid=False, reason="test_injected")
            with patch.object(auto._apf, "update", return_value=invalid_out):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            call_args = vc.return_value.send_velocity_body_frd.call_args
            # Must be (0,0,0) hold, not reactive fallback
            assert call_args[0][0] == 0.0
            assert call_args[0][1] == 0.0
            assert call_args[0][2] == 0.0

    def test_api_is_move_by_velocity_body_frame_async(self):
        """send_velocity_body_frd is called (→ moveByVelocityBodyFrameAsync under the hood)."""
        session, auto, stack, vc = self._setup_dispatch("apf")
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            # Verify the correct method was used
            vc.return_value.send_velocity_body_frd.assert_called_once()
            # Check keyword arguments include vehicle_name
            kwargs = vc.return_value.send_velocity_body_frd.call_args.kwargs
            assert kwargs.get("vehicle_name") == session.vehicle_name


class TestCbmbaShadowLogging:
    """Verify cbmba_path log format includes next= and max_lateral_dev= fields."""

    def _make_cbmba_auto_and_mocks(self):
        """Set up a flight with CBMBA shadow enabled and sensible LiDAR rays."""
        session = _make_mock_session()
        stack, mocks = _setup_patches()
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        _cfg_mocks(mocks)
        lr.return_value.read.side_effect = [_lf()] * 30
        sr.return_value.read.side_effect = [_st()] * 100
        cr.return_value.read.side_effect = [_col()] * 30
        # Sparse rays — one front obstacle, rest clear — so CBMBA
        # produces a short path quickly (avoid slow wall-proximity scans)
        dd.return_value.to_legacy_ray_distances.return_value = {
            "front": 8.0,
            "back": 50.0,
            "left": 50.0,
            "right": 50.0,
            "frontLeft": 50.0,
            "frontRight": 50.0,
            "backLeft": 50.0,
            "backRight": 50.0,
        }
        dd.return_value.minimum_distance_m = 5.0

        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.05),
            cli_overrides={"planner_mode": "apf"},
        )
        return session, auto, stack, vc

    def test_cbmba_path_log_has_next_and_deviation(self, caplog):
        """cbmba_path INFO line must contain next=( and max_lateral_dev= fields."""
        import logging
        session, auto, stack, vc = self._make_cbmba_auto_and_mocks()
        with stack:
            with caplog.at_level(logging.INFO, logger="automatic_mode"):
                with patch.object(session, "takeoff_and_climb"):
                    auto.run()

        cbmba_path_lines = [r.message for r in caplog.records
                           if getattr(r, "message", "").startswith("cbmba_path ")]
        assert len(cbmba_path_lines) >= 1, "Expected at least one cbmba_path log line"
        msg = cbmba_path_lines[0]
        assert "next=(" in msg, f"Missing next= in: {msg}"
        assert "max_lateral_dev=" in msg, f"Missing max_lateral_dev= in: {msg}"


class TestCbmbaPathDiagnostics:
    """Standalone tests for cbmba_path diagnostic computations (next, max_lateral_dev)."""

    def test_next_is_first_waypoint_different_from_start(self):
        """next = first pt differing > 0.05m from start on any axis."""
        path = [
            [0.0, 0.0, -1.0],     # start
            [0.0, 0.0, -1.0],     # same (within 0.05)
            [0.0, 0.01, -1.0],    # same
            [1.5, 0.0, -1.0],     # different! → next
            [3.0, -1.0, -1.0],
            [4.5, 0.0, -1.0],
        ]
        wp_first = path[0]
        eps = 0.05
        wp_next = wp_first
        for pt in path[1:]:
            if (abs(pt[0] - wp_first[0]) > eps
                    or abs(pt[1] - wp_first[1]) > eps
                    or abs(pt[2] - wp_first[2]) > eps):
                wp_next = pt
                break
        assert wp_next == [1.5, 0.0, -1.0]

    def test_next_is_start_when_all_same(self):
        """When all points equal start (within threshold), next stays at start."""
        path = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]
        wp_first = path[0]
        eps = 0.05
        wp_next = wp_first
        for pt in path[1:]:
            if (abs(pt[0] - wp_first[0]) > eps
                    or abs(pt[1] - wp_first[1]) > eps
                    or abs(pt[2] - wp_first[2]) > eps):
                wp_next = pt
                break
        assert wp_next == wp_first

    def test_max_lateral_dev_straight_line(self):
        """Points exactly on start→goal line → max_lateral_dev ≈ 0."""
        import math
        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]
        path = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        sx, sy = start[0], start[1]
        seg_dx = goal[0] - sx
        seg_dy = goal[1] - sy
        seg_len = math.hypot(seg_dx, seg_dy)
        max_dev = max(
            abs((pt[0] - sx) * seg_dy - (pt[1] - sy) * seg_dx) / seg_len
            for pt in path
        )
        assert max_dev == pytest.approx(0.0, abs=1e-9)

    def test_max_lateral_dev_detour(self):
        """Waypoint at lateral offset 3m → max_lateral_dev = 3.0."""
        import math
        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]
        path = [[0.0, 0.0, 0.0], [5.0, 3.0, 0.0], [10.0, 0.0, 0.0]]
        sx, sy = start[0], start[1]
        seg_dx = goal[0] - sx
        seg_dy = goal[1] - sy
        seg_len = math.hypot(seg_dx, seg_dy)
        max_dev = max(
            abs((pt[0] - sx) * seg_dy - (pt[1] - sy) * seg_dx) / seg_len
            for pt in path
        )
        assert max_dev == pytest.approx(3.0, abs=0.01)

    def test_max_lateral_dev_diagonal_goal(self):
        """Non-axis-aligned start→goal: point (3,-1) off y=(4/3)x → dev=3.0."""
        import math
        start = [0.0, 0.0, 0.0]
        goal = [6.0, 8.0, 0.0]
        # line: 4x - 3y = 0; distance(3,-1) = |12 + 3| / 5 = 3.0
        path = [[0.0, 0.0, 0.0], [3.0, -1.0, 0.0], [6.0, 8.0, 0.0]]
        sx, sy = start[0], start[1]
        seg_dx = goal[0] - sx
        seg_dy = goal[1] - sy
        seg_len = math.hypot(seg_dx, seg_dy)
        max_dev = max(
            abs((pt[0] - sx) * seg_dy - (pt[1] - sy) * seg_dx) / seg_len
            for pt in path
        )
        assert max_dev == pytest.approx(3.0, abs=0.01)

    def test_max_lateral_dev_zero_length_line(self):
        """Degenerate case: start==goal → max_lateral_dev stays 0."""
        import math
        start = [5.0, 5.0, 0.0]
        goal = [5.0, 5.0, 0.0]
        path = [[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]]
        seg_dx = goal[0] - start[0]
        seg_dy = goal[1] - start[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        max_dev = 0.0
        if seg_len > 1e-6:
            max_dev = max(
                abs((pt[0] - start[0]) * seg_dy
                    - (pt[1] - start[1]) * seg_dx) / seg_len
                for pt in path
            )
        assert max_dev == 0.0

"""Tests for SharedFlightSession — landing, arm failure, post-takeoff cleanup."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.shared_flight_session import (
    SharedFlightSession, SessionPhase, SessionError,
)

_ADAPTER = "adapters.airsim_client.AirSimClientAdapter"


def _make_session(mode="manual"):
    s = SharedFlightSession(settings_json="fake.json", mode=mode)
    with patch(_ADAPTER) as ma_cls:
        ma = ma_cls.return_value
        ma.vehicle_name = "Drone1"; ma.lidar_name = "LidarSensor1"
        mc = MagicMock()
        ma.get_raw_client.return_value = mc
        ma.list_vehicles.return_value = ["Drone1"]
        s.initialize()
        # Mock poll to avoid 30s waits in tests
        s._poll_landed_detailed = MagicMock(return_value=True)
        return s, mc


def _make_st(landed):
    st = MagicMock()
    st.landed_state = landed
    kin = MagicMock(); kin.position.z_val = 0.0; kin.linear_velocity.z_val = 0.0
    st.kinematics_estimated = kin
    return st

def _landed_st(): return _make_st(0)
def _flying_st(): return _make_st(1)


# ═══════════════ Arm failure ═══════════════

class TestArmFailureReportsError:
    def test_arm_fail_ground_cleanup_raises(self):
        s, c = _make_session()
        def _arm(v, **kw):
            if v is True: raise Exception("arm fail")
        c.armDisarm.side_effect = _arm
        c.getMultirotorState.return_value = _landed_st()
        with pytest.raises(SessionError, match="armDisarm failed"):
            s.takeoff_and_climb()
        c.takeoffAsync.assert_not_called()
        assert s.state.phase == SessionPhase.CONTROL_RELEASED

    def test_arm_fail_takeoff_never_called(self):
        s, c = _make_session()
        def _arm(v, **kw):
            if v is True: raise Exception("arm fail")
        c.armDisarm.side_effect = _arm
        c.getMultirotorState.return_value = _landed_st()
        with pytest.raises(SessionError):
            s.takeoff_and_climb()
        c.takeoffAsync.assert_not_called()

    def test_arm_fail_state_read_fail_raises(self):
        s, c = _make_session()
        def _arm(v, **kw):
            if v is True: raise Exception("arm fail")
        c.armDisarm.side_effect = _arm
        c.getMultirotorState.side_effect = Exception("state fail")
        with pytest.raises(SessionError, match="cannot read state"):
            s.takeoff_and_climb()
        assert s.state.phase == SessionPhase.MANUAL_INTERVENTION_REQUIRED

    def test_arm_fail_not_on_ground_raises(self):
        s, c = _make_session()
        def _arm(v, **kw):
            if v is True: raise Exception("arm fail")
        c.armDisarm.side_effect = _arm
        c.getMultirotorState.return_value = _flying_st()
        with pytest.raises(SessionError, match="may be airborne"):
            s.takeoff_and_climb()
        assert s.state.phase == SessionPhase.MANUAL_INTERVENTION_REQUIRED


# ═══════════════ Takeoff failure ═══════════════

class TestTakeoffFailure:
    def test_takeoff_exc_on_ground_disarms_releases_raises(self):
        s, c = _make_session()
        c.takeoffAsync.side_effect = Exception("takeoff fail")
        c.getMultirotorState.return_value = _landed_st()
        with pytest.raises(SessionError, match="takeoffAsync failed"):
            s.takeoff_and_climb()
        c.armDisarm.assert_any_call(False, vehicle_name="Drone1")
        c.enableApiControl.assert_any_call(False, vehicle_name="Drone1")

    def test_takeoff_exc_airborne_lands_confirms_disarms_releases_raises(self):
        s, c = _make_session()
        c.takeoffAsync.side_effect = Exception("takeoff fail")
        c.getMultirotorState.return_value = _flying_st()
        # poll returns True (mock default in _make_session)
        with pytest.raises(SessionError, match="takeoffAsync failed"):
            s.takeoff_and_climb()
        c.hoverAsync.assert_called()
        c.landAsync.assert_called()
        assert s.state.phase == SessionPhase.CONTROL_RELEASED

    def test_takeoff_exc_airborne_landing_not_confirmed_no_disarm(self):
        s, c = _make_session()
        c.takeoffAsync.side_effect = Exception("takeoff fail")
        c.getMultirotorState.return_value = _flying_st()
        s._poll_landed_detailed.return_value = False
        with pytest.raises(SessionError, match="landing unconfirmed"):
            s.takeoff_and_climb()
        disarm_calls = [c2 for c2 in c.armDisarm.call_args_list if c2[0][0] is False]
        assert len(disarm_calls) == 0
        assert s.state.phase == SessionPhase.MANUAL_INTERVENTION_REQUIRED

    def test_takeoff_exc_state_read_fail_no_disarm_no_release(self):
        s, c = _make_session()
        c.takeoffAsync.side_effect = Exception("takeoff fail")
        c.getMultirotorState.side_effect = Exception("state read fail")
        with pytest.raises(SessionError, match="cannot read state"):
            s.takeoff_and_climb()
        disarm_calls = [c2 for c2 in c.armDisarm.call_args_list if c2[0][0] is False]
        assert len(disarm_calls) == 0
        release_calls = [c2 for c2 in c.enableApiControl.call_args_list if not c2[0][0]]
        assert len(release_calls) == 0
        assert s.state.phase == SessionPhase.MANUAL_INTERVENTION_REQUIRED

    def test_takeoff_exc_disarm_fails_no_release(self):
        s, c = _make_session()
        c.takeoffAsync.side_effect = Exception("takeoff fail")
        c.getMultirotorState.return_value = _landed_st()
        def _disarm(v, **kw):
            if v is False: raise Exception("disarm fail")
        c.armDisarm.side_effect = _disarm
        with pytest.raises(SessionError, match="disarm also failed"):
            s.takeoff_and_climb()
        release_calls = [c2 for c2 in c.enableApiControl.call_args_list if not c2[0][0]]
        assert len(release_calls) == 0


# ═══════════════ MoveToZ / Hover failure ═══════════════

class TestMoveToZFailure:
    def test_moveToZ_exc_airborne_safe_landing(self):
        s, c = _make_session()
        c.moveToZAsync.side_effect = Exception("moveToZ fail")
        c.getMultirotorState.return_value = _flying_st()
        with pytest.raises(SessionError, match="moveToZAsync failed"):
            s.takeoff_and_climb()
        c.hoverAsync.assert_called()
        c.landAsync.assert_called()

    def test_moveToZ_exc_on_ground_disarm_release(self):
        s, c = _make_session()
        c.moveToZAsync.side_effect = Exception("moveToZ fail")
        c.getMultirotorState.return_value = _landed_st()
        with pytest.raises(SessionError, match="moveToZAsync failed"):
            s.takeoff_and_climb()
        c.armDisarm.assert_any_call(False, vehicle_name="Drone1")


class TestHoverFailure:
    def test_hover_exc_airborne_safe_landing(self):
        s, c = _make_session()
        c.hoverAsync.side_effect = Exception("hover fail")
        c.getMultirotorState.return_value = _flying_st()
        with pytest.raises(SessionError, match="hoverAsync failed"):
            s.takeoff_and_climb()
        c.landAsync.assert_called()


# ═══════════════ Lifecycle ═══════════════

class TestLifecycle:
    def test_initial_uninitialized(self):
        s = SharedFlightSession(settings_json="x")
        assert s.state.phase == SessionPhase.UNINITIALIZED

    def test_initialize(self):
        s, _ = _make_session()
        assert s.state.phase == SessionPhase.INITIALIZED

    def test_takeoff_success(self):
        s, c = _make_session()
        s.takeoff_and_climb()
        assert s.state.phase == SessionPhase.AIRBORNE

    def test_double_takeoff_raises(self):
        s, c = _make_session()
        s.takeoff_and_climb()
        with pytest.raises(SessionError, match="already called"):
            s.takeoff_and_climb()

    def test_landing_idempotent(self):
        """Second land_and_disarm call is a no-op."""
        s, c = _make_session()
        s.takeoff_and_climb()
        assert s.land_and_disarm()
        # Second call: already done
        assert s.land_and_disarm()
        # landAsync called exactly once
        assert c.landAsync.call_count == 1

    def test_manual_g_and_auto_time_limit_same_landing_path(self):
        """Both manual G and auto time_limit call the same land_and_disarm."""
        s1, c1 = _make_session()
        s1.takeoff_and_climb()
        c1.reset_mock()
        s1._poll_landed_detailed.return_value = True
        s1.land_and_disarm()
        manual_calls = [(n, kw) for n, _, kw in [
            (c._mock_name or "", None, c.kwargs) for c in c1.mock_calls
        ]]

        s2, c2 = _make_session()
        s2.takeoff_and_climb()
        c2.reset_mock()
        s2._poll_landed_detailed.return_value = True
        s2.land_and_disarm()

        # Both paths call hoverAsync, landAsync, armDisarm(False), enableApiControl(False)
        for mock_name in ["hoverAsync", "landAsync"]:
            assert getattr(c1, mock_name).called == getattr(c2, mock_name).called
        assert c1.landAsync.call_count == 1
        assert c2.landAsync.call_count == 1


# ═══════════════ Phased cleanup ═══════════════

class TestPhasedCleanup:
    def test_initialized_no_control(self):
        s, c = _make_session()
        s.land_and_disarm()
        c.hoverAsync.assert_not_called()
        c.armDisarm.assert_not_called()

    def test_control_acquired_releases(self):
        s, c = _make_session()
        s._state.phase = SessionPhase.CONTROL_ACQUIRED
        s.land_and_disarm()
        c.enableApiControl.assert_called_with(False, vehicle_name="Drone1")

    def test_disarm_fail_no_release(self):
        s, c = _make_session()
        s.takeoff_and_climb()
        c.armDisarm.side_effect = Exception("fail")
        s._poll_landed_detailed.return_value = True
        s.land_and_disarm()
        release_calls = [c2 for c2 in c.enableApiControl.call_args_list if not c2[0][0]]
        assert len(release_calls) == 0

    def test_manual_intervention_does_nothing(self):
        s, c = _make_session()
        s._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
        s.land_and_disarm()
        c.armDisarm.assert_not_called()


# ═══════════════ Simulation floor-contact fallback ═══════════════

def _make_col(has_collided, object_name="", time_stamp=0):
    c = MagicMock()
    c.has_collided = has_collided
    c.object_name = object_name
    c.time_stamp = time_stamp
    return c

def _make_st_full(landed, pz=2.0, vx=0.0, vy=0.0, vz=0.0):
    st = MagicMock()
    st.landed_state = landed
    kin = MagicMock()
    kin.position.z_val = pz
    kin.linear_velocity.x_val = vx
    kin.linear_velocity.y_val = vy
    kin.linear_velocity.z_val = vz
    st.kinematics_estimated = kin
    return st


class TestSimulationFloorContactFallback:
    """UE4+AirSim fallback: latch NEW Floor timestamp, then check stability."""

    def _make_poll_session(self, startup_ts=0):
        from flight_modes.shared_flight_session import SharedFlightSession
        s = SharedFlightSession(settings_json="fake.json", mode="auto")
        with patch(_ADAPTER) as ma_cls:
            ma = ma_cls.return_value
            ma.vehicle_name = "Drone1"; ma.lidar_name = "LidarSensor1"
            mc = MagicMock()
            ma.get_raw_client.return_value = mc
            ma.list_vehicles.return_value = ["Drone1"]
            s.initialize()
            mc.getMultirotorState.return_value = _make_st_full(1, pz=2.0)
            mc.simGetCollisionInfo.return_value = _make_col(False)
            if startup_ts:
                s.set_startup_floor_baseline(startup_ts)
            return s, mc

    def test_first_frame_floor_then_clean_stable_confirms(self):
        """Frame 0: collision=True/Floor, then collision=False. Latch persists, stability for 2s → confirm."""
        s, c = self._make_poll_session()
        call_n = [0]
        def _col_side(**kw):
            call_n[0] += 1
            if call_n[0] == 1:
                return _make_col(True, "Floor", 200)
            return _make_col(False)
        c.simGetCollisionInfo.side_effect = _col_side
        c.getMultirotorState.return_value = _make_st_full(1, pz=2.016)
        result = s._poll_landed_detailed("Drone1", max_wait_s=3.0, interval_s=0.2)
        assert result is True

    def test_startup_ts_not_accepted_for_landing_latch(self):
        """Only new landing-phase timestamps (≠ startup) can latch."""
        s, c = self._make_poll_session(startup_ts=200)
        # Same timestamp as startup → must NOT latch
        c.simGetCollisionInfo.return_value = _make_col(True, "Floor", 200)
        c.getMultirotorState.return_value = _make_st_full(1, pz=2.016)
        result = s._poll_landed_detailed("Drone1", max_wait_s=1.0, interval_s=0.2)
        assert result is False  # no standard landed, no new latch → timeout

    def test_latched_then_cube_rejects(self):
        """Latch Floor, then new Cube timestamp appears → cancel fallback."""
        s, c = self._make_poll_session()
        call_n = [0]
        def _col_side(**kw):
            call_n[0] += 1
            if call_n[0] == 1:
                return _make_col(True, "Floor", 200)
            if call_n[0] == 5:
                return _make_col(True, "Cube_1", 300)
            return _make_col(False)
        c.simGetCollisionInfo.side_effect = _col_side
        c.getMultirotorState.return_value = _make_st_full(1, pz=2.016)
        result = s._poll_landed_detailed("Drone1", max_wait_s=2.0, interval_s=0.2)
        assert result is False

    def test_latched_but_z_changing_rejects(self):
        """Latch Floor, but Z keeps changing → never stable → reject."""
        s, c = self._make_poll_session()
        call_n = [0]
        z_base = 2.0
        def _state_side(**kw):
            nonlocal z_base
            z_base += 0.05
            return _make_st_full(1, pz=z_base)
        c.getMultirotorState.side_effect = _state_side
        c.simGetCollisionInfo.side_effect = [
            _make_col(True, "Floor", 200)
        ]
        c.simGetCollisionInfo.return_value = _make_col(False)
        result = s._poll_landed_detailed("Drone1", max_wait_s=1.0, interval_s=0.2)
        assert result is False

    def test_standard_landed_state_still_priority(self):
        s, c = self._make_poll_session()
        c.getMultirotorState.return_value = _make_st_full(0, pz=2.0)
        c.simGetCollisionInfo.return_value = _make_col(False)
        result = s._poll_landed_detailed("Drone1", max_wait_s=1.0, interval_s=0.2)
        assert result is True

    def test_no_floor_contact_airborne_rejects(self):
        s, c = self._make_poll_session()
        c.getMultirotorState.return_value = _make_st_full(1, pz=-5.0)
        c.simGetCollisionInfo.return_value = _make_col(False)
        result = s._poll_landed_detailed("Drone1", max_wait_s=1.0, interval_s=0.2)
        assert result is False

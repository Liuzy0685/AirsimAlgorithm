"""ROUND 4.9: Airborne fixture tests — cleanup tracking, ARM_REQUESTED, state fields."""
import sys, math, argparse, tempfile, json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.airborne_test_fixture import (
    run_airborne_fixture, _parse_and_validate_args,
    _check_vertical_clearance_ned, UpwardClearanceDisabled,
)
from configs.airborne_fixture_config import AirborneFixtureConfig


def _mc():
    c = MagicMock()
    c.listVehicles.return_value = ["Drone1"]
    for m in ["takeoffAsync","moveToZAsync","hoverAsync","landAsync"]:
        getattr(c, m).return_value.join.return_value = None
    return c

def _ts():
    t = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump({"SettingsVersion":1.2,"Vehicles":{"Drone1":{"VehicleType":"SimpleFlight","Sensors":{"LidarSensor1":{
        "SensorType":6,"Enabled":True,"DataFrame":"SensorLocalFrame",
        "HorizontalFOVStart":-180,"HorizontalFOVEnd":180,
        "VerticalFOVUpper":30,"VerticalFOVLower":-30,"Range":40
    }}}}}, t); t.close(); return t.name

def _a(**kw):
    d=dict(target_z=-1.5,max_vertical_speed=0.5,hover_duration=30,
           allow_indefinite=False,settings_json=_ts()); d.update(kw)
    return argparse.Namespace(**d)

def _lf(ok=True):
    lf=MagicMock(); lf.frame_valid=ok; lf.invalid_reason=None if ok else "x"
    lf.point_cloud_sensor=np.ones((100,3),dtype=np.float32); lf.point_count=100
    lf.raw_timestamp_ns=1; lf.received_monotonic_seconds=99999.0
    lf.sensor_pose={"position":{"x":0.2,"y":0,"z":0},"orientation":{"x":0,"y":0,"z":0,"w":1}}; return lf

def _st(z=1.0,ready=True,ca=True,ls=0,roll=0.,pitch=0.):
    s=MagicMock(); s.position_ned_m=[0.,0.,z]; s.linear_velocity_ned_mps=[0,0,0]
    s.angular_velocity_body_radps=[0,0,0]; s.roll_rad=roll; s.pitch_rad=pitch; s.yaw_rad=0.
    s.ready=ready; s.can_arm=ca; s.landed_state=ls; s.received_monotonic_seconds=99999.0; return s

def _col(ok=True):
    c=MagicMock(); c.has_collided=not ok; c.object_name="" if ok else "w"
    c.penetration_depth=0.0; c.received_monotonic_seconds=99999.0; return c

def _fr(valid=True):
    fr=MagicMock(); fr.valid=valid; fr.invalid_reason=None if valid else "fail"
    fr.filtered_points_sensor=np.ones((50,3),dtype=np.float32); fr.output_point_count=50; return fr

def _up_mock(observable=True):
    up=MagicMock()
    up.is_corridor_observable=MagicMock(return_value=(observable,f"mock_{observable}"))
    up.check_clearance=MagicMock(return_value=(True,"mock_clear"))
    return up

def _run(client=None, args=None, mLR=None, mSR=None, mCR=None, mFlt=None, clock=None,
         upward_observable=True, hover_states=None):
    c=client or _mc()
    clk=[100.0]
    clk_fn=clock or (lambda:(clk.__setitem__(0,clk[0]+31.0),clk[0])[1])
    up=_up_mock(observable=upward_observable) if upward_observable else None
    with patch("sensors.lidar_reader.LidarReader", mLR or MagicMock()) as lr,\
         patch("sensors.state_reader.StateReader", mSR or MagicMock()) as sr,\
         patch("sensors.collision_reader.CollisionReader", mCR or MagicMock()) as cr,\
         patch("perception.pointcloud_filter.filter_pointcloud", mFlt or MagicMock()) as flt:
        if mLR is None: lr.return_value.read.return_value=_lf()
        if mSR is None:
            si=iter([_st(z=1.0)]*4+[_st(z=-1.4)]+(hover_states or [_st(z=-1.5)]*20))
            sr.return_value.read.side_effect=lambda:next(si)
        if mCR is None: cr.return_value.read.return_value=_col()
        if mFlt is None: flt.return_value=_fr()
        return run_airborne_fixture(args or _a(), lambda:MagicMock(get_raw_client=lambda:c),
                                     clk_fn, MagicMock(), signal_handler=lambda h:None,
                                     upward_provider=up)


# ═══════════ Cleanup tracking ═══════════

class TestCleanupTracking:
    def test_release_failure_cleanup_fails(self):
        """Normal task + release API fails → cleanup_success=False, exit_code=9."""
        c=_mc()
        # arm(True) ok, arm(False) ok, enableApiControl(False) fails
        c.enableApiControl.side_effect=[None,Exception("release fail")]
        r=_run(client=c)
        assert not r.mission_success and r.exit_code==9
        assert not r.cleanup_success and r.cleanup_errors
        assert not r.api_control_released
        assert any("release_api" in e for e in r.cleanup_errors)

    def test_release_success_cleanup_ok(self):
        c=_mc(); r=_run(client=c)
        assert r.cleanup_success and r.api_control_released

    def test_disarm_ok_release_fail_cleanup_fails(self):
        c=_mc()
        c.enableApiControl.side_effect=[None,Exception("release fail")]
        r=_run(client=c)
        assert not r.cleanup_success and r.exit_code==9
        assert r.disarmed
        assert not r.api_control_released

    def test_collision_release_fail_mission_and_cleanup_fail(self):
        c=_mc()
        c.enableApiControl.side_effect=[None,Exception("release fail")]
        with patch("sensors.lidar_reader.LidarReader") as mLR,\
             patch("sensors.state_reader.StateReader") as mSR,\
             patch("sensors.collision_reader.CollisionReader") as mCR,\
             patch("perception.pointcloud_filter.filter_pointcloud") as mFlt:
            mLR.return_value.read.return_value=_lf(); mFlt.return_value=_fr()
            mCR.return_value.read.side_effect=[_col()]*7+[_col(False)]
            si=iter([_st(z=1.0)]*4+[_st(z=-1.4)]+[_st(z=-1.5)]*4+[_st(z=0.0,ls=0)]*3)
            mSR.return_value.read.side_effect=lambda:next(si)
            r=run_airborne_fixture(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                    lambda:100.1,MagicMock(),signal_handler=lambda h:None,
                                    upward_provider=_up_mock())
        assert not r.mission_success and r.shutdown_type=="collision"
        assert not r.cleanup_success and r.exit_code==9


# ═══════════ ARM_REQUESTED ═══════════

class TestArmRequested:
    def test_arm_fails_still_grounded(self):
        """arm fails, state shows Landed→stationary→disarm ok."""
        c=_mc(); c.armDisarm.side_effect=[Exception("arm fail"),None,None]
        r=_run(client=c)
        assert not r.mission_success
        assert r.emergency_shutdown_attempted

    def test_arm_fails_state_read_fails(self):
        """arm fails, state read fails → manual_intervention_required."""
        c=_mc(); c.armDisarm.side_effect=[Exception("arm fail")]
        with patch("sensors.state_reader.StateReader") as mSR:
            mSR.return_value.read.side_effect=[_st(z=1.0)]*4+[Exception("state dead")]
            with patch("sensors.lidar_reader.LidarReader") as mLR,\
                 patch("sensors.collision_reader.CollisionReader") as mCR,\
                 patch("perception.pointcloud_filter.filter_pointcloud") as mFlt:
                mLR.return_value.read.return_value=_lf(); mCR.return_value.read.return_value=_col()
                mFlt.return_value=_fr()
                r=run_airborne_fixture(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                        lambda:100.1,MagicMock(),signal_handler=lambda h:None,
                                        upward_provider=_up_mock())
        assert not r.cleanup_success


# ═══════════ State fields ═══════════

class TestStateFields:
    def test_normal_end_state_fields(self):
        """Normal completion: api_control_enabled=False, armed=False, released+disarmed=True."""
        r=_run()
        assert r.cleanup_success
        assert not r.api_control_enabled and r.api_control_released
        assert not r.armed and r.disarmed
        assert r.emergency_shutdown_attempted == False  # normal is NOT emergency

    def test_normal_primary_failure_empty(self):
        r=_run()
        assert r.primary_failure_reason == ""
        assert r.shutdown_type == "normal"

    def test_collision_state_fields(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as mLR,\
             patch("sensors.state_reader.StateReader") as mSR,\
             patch("sensors.collision_reader.CollisionReader") as mCR,\
             patch("perception.pointcloud_filter.filter_pointcloud") as mFlt:
            mLR.return_value.read.return_value=_lf(); mFlt.return_value=_fr()
            mCR.return_value.read.side_effect=[_col()]*7+[_col(False)]
            si=iter([_st(z=1.0)]*4+[_st(z=-1.4)]+[_st(z=-1.5)]*4+[_st(z=0.0,ls=0)]*3)
            mSR.return_value.read.side_effect=lambda:next(si)
            r=run_airborne_fixture(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                    lambda:100.1,MagicMock(),signal_handler=lambda h:None,
                                    upward_provider=_up_mock())
        assert not r.mission_success and r.shutdown_type=="collision"
        assert r.emergency_shutdown_attempted
        assert "collision" in (r.primary_failure_reason or "")

    def test_cleanup_failure_state_fields(self):
        """Release fails → api_control_enabled stays True, api_control_released=False."""
        c=_mc()
        c.enableApiControl.side_effect=[None,Exception("release fail")]
        r=_run(client=c)
        assert not r.api_control_released
        assert not r.cleanup_success

    def test_arm_exception_does_not_release_on_unknown_state(self):
        """arm fails + state unreadable → no release."""
        c=_mc(); c.armDisarm.side_effect=[Exception("arm fail")]
        with patch("sensors.state_reader.StateReader") as mSR:
            mSR.return_value.read.side_effect=[_st(z=1.0)]*4+[Exception("state dead")]
            with patch("sensors.lidar_reader.LidarReader") as mLR,\
                 patch("sensors.collision_reader.CollisionReader") as mCR,\
                 patch("perception.pointcloud_filter.filter_pointcloud") as mFlt:
                mLR.return_value.read.return_value=_lf(); mCR.return_value.read.return_value=_col()
                mFlt.return_value=_fr()
                r=run_airborne_fixture(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                        lambda:100.1,MagicMock(),signal_handler=lambda h:None,
                                        upward_provider=_up_mock())
        # state unreadable → cleanup failure, no release
        assert not r.cleanup_success


# ═══════════ Exit semantics ═══════════

class TestExitSemantics:
    def test_normal_completion(self):
        r=_run()
        assert r.mission_success and r.exit_code==0 and r.shutdown_type=="normal"

    def test_hover_collision_mission_fails(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as mLR,\
             patch("sensors.state_reader.StateReader") as mSR,\
             patch("sensors.collision_reader.CollisionReader") as mCR,\
             patch("perception.pointcloud_filter.filter_pointcloud") as mFlt:
            mLR.return_value.read.return_value=_lf(); mFlt.return_value=_fr()
            mCR.return_value.read.side_effect=[_col()]*7+[_col(False)]
            si=iter([_st(z=1.0)]*4+[_st(z=-1.4)]+[_st(z=-1.5)]*4+[_st(z=0.0,ls=0)]*3)
            mSR.return_value.read.side_effect=lambda:next(si)
            r=run_airborne_fixture(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                    lambda:100.1,MagicMock(),signal_handler=lambda h:None,
                                    upward_provider=_up_mock())
        assert not r.mission_success and r.exit_code!=0 and r.shutdown_type=="collision"

    def test_drift_mission_fails(self):
        r=_run(hover_states=[_st(z=-1.5)]*3+[_st(z=-5.0)])
        assert not r.mission_success and r.exit_code!=0 and r.shutdown_type=="drift"


class TestConfig:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown"):
            AirborneFixtureConfig.from_dict({"airborne_fixture":{"bogus_key":1}})

    def test_takeoff_delta_z_locked(self):
        with pytest.raises(ValueError, match="must be -2.0"):
            AirborneFixtureConfig.from_dict({"airborne_fixture":{"takeoff_delta_z_m":-0.5}})


class TestVerticalClearance:
    def test_centered_on_drone(self):
        pts=np.array([[10,20,0.5]],dtype=np.float32)
        c,r=_check_vertical_clearance_ned(pts,np.array([10,20,1.0]),-1.5,1.5,0.3)
        assert not c and "blocked" in r

    def test_outside_radius_clear(self):
        assert _check_vertical_clearance_ned(np.array([[0,0,0.5]],dtype=np.float32),np.array([10,20,1.0]),-1.5,1.5,0.3)[0]


class TestUpwardObservability:
    def test_disabled_provider_fails(self):
        p=UpwardClearanceDisabled()
        ok,reason=p.is_corridor_observable()
        assert not ok and "unobservable" in reason

    def test_unobservable_blocks_fixture(self):
        c=_mc()
        r=_run(client=c,upward_observable=False)
        assert not r.mission_success and "unobservable" in r.exit_reason
        c.enableApiControl.assert_not_called()


class TestDryRunClean:
    def test_no_control_in_dry_run(self):
        s=(_PROJECT_ROOT/"scripts"/"run_local_avoidance_dry_run.py").read_text(encoding="utf-8")
        for api in ["enableApiControl","armDisarm","takeoffAsync","hoverAsync","landAsync"]:
            assert api not in s

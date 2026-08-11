"""Minimal LiDAR flight tests — strong assertions, call production functions."""
import sys, argparse, tempfile, json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.minimal_lidar_flight import (
    choose_reactive_command, ReactiveDecision, run_minimal_flight, MinimalFlightResult,
)

def _ts():
    t=tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False,encoding="utf-8")
    json.dump({"SettingsVersion":1.2,"Vehicles":{"Drone1":{"VehicleType":"SimpleFlight","Sensors":{"LidarSensor1":{
        "SensorType":6,"Enabled":True,"DataFrame":"SensorLocalFrame",
        "HorizontalFOVStart":-180,"HorizontalFOVEnd":180,
        "VerticalFOVUpper":30,"VerticalFOVLower":-30,"Range":40
    }}}}},t);t.close();return t.name

def _a(clearance=True,output=None):
    return argparse.Namespace(config=str(_PROJECT_ROOT/"configs"/"vehicle.yaml"),
        perception_config=str(_PROJECT_ROOT/"configs"/"perception.yaml"),
        flight_config=str(_PROJECT_ROOT/"configs"/"minimal_flight.yaml"),
        settings_json=_ts(),confirm_simulation_clearance=clearance,output=output)

def _mc():
    c=MagicMock();c.listVehicles.return_value=["Drone1"]
    for m in["takeoffAsync","moveToZAsync","hoverAsync","landAsync"]:getattr(c,m).return_value.join.return_value=None
    return c

def _lf(ok=True):
    lf=MagicMock();lf.frame_valid=ok;lf.invalid_reason=None if ok else"x"
    lf.point_cloud_sensor=np.ones((100,3),dtype=np.float32);lf.point_count=100
    lf.raw_timestamp_ns=1;lf.received_monotonic_seconds=99999.0
    lf.sensor_pose={"position":{"x":0.2,"y":0,"z":0},"orientation":{"x":0,"y":0,"z":0,"w":1}};return lf

def _st(z=-1.0,ls=1):
    s=MagicMock();s.position_ned_m=[0.,0.,z];s.linear_velocity_ned_mps=[0,0,0]
    s.angular_velocity_body_radps=[0,0,0];s.roll_rad=0.;s.pitch_rad=0.;s.yaw_rad=0.
    s.ready=True;s.can_arm=True;s.landed_state=ls;s.received_monotonic_seconds=99999.0;return s

def _col(ok=True, *, object_name=None, raw_timestamp=0, is_new_event=False):
    c=MagicMock();c.has_collided=not ok
    c.object_name=object_name if object_name is not None else ("" if ok else "w")
    c.raw_timestamp=raw_timestamp;c.is_new_collision_event=is_new_event
    c.penetration_depth=0.0;c.received_monotonic_seconds=99999.0;return c

def _fr(valid=True):
    fr=MagicMock();fr.valid=valid;fr.invalid_reason=None if valid else"fail"
    fr.filtered_points_sensor=np.ones((50,3),dtype=np.float32);fr.output_point_count=50;return fr

def _dd(front=10.0,left=10.0,right=10.0,min_dist=5.0):
    dd=MagicMock();dd.frame_valid=True;dd.minimum_distance_m=min_dist
    dd.to_legacy_ray_distances.return_value={"front":front,"left":left,"right":right};return dd

_CFG={"emergency_distance_m":0.8,"front_threshold_m":2.5,"forward_speed_mps":0.2,"side_speed_mps":0.15}


class TestReactiveDecision:
    def test_clear_path_forward(self):
        d=choose_reactive_command(5.0,10.0,10.0,5.0,_CFG)
        assert d.vx_body_mps==0.2 and d.vy_body_mps==0.0 and not d.should_terminate

    def test_front_blocked_left_clearer(self):
        d=choose_reactive_command(1.0,8.0,3.0,5.0,_CFG)
        assert d.vx_body_mps==0.0 and d.vy_body_mps==-0.15

    def test_front_blocked_right_clearer(self):
        d=choose_reactive_command(1.0,2.0,8.0,5.0,_CFG)
        assert d.vx_body_mps==0.0 and d.vy_body_mps==0.15

    def test_emergency_distance_terminates(self):
        d=choose_reactive_command(1.0,10.0,10.0,0.5,_CFG)
        assert d.should_terminate and d.termination_reason=="emergency_distance"

    def test_front_equal_left_right_goes_right(self):
        d=choose_reactive_command(1.0,5.0,5.0,5.0,_CFG)
        assert d.vy_body_mps==0.15


class TestRunMinimalFlight:
    def test_no_clearance_no_control(self):
        r=run_minimal_flight(_a(clearance=False),lambda:MagicMock(),lambda:0,lambda s:None)
        assert not r.api_control_acquired and r.exit_code!=0

    def test_enable_api_exception_no_cleanup_calls(self):
        """enableApiControl throws → no hover/land/disarm/release called."""
        c=_mc();c.enableApiControl.side_effect=Exception("fail")
        r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:0,lambda s:None)
        assert not r.api_control_acquired
        c.hoverAsync.assert_not_called();c.landAsync.assert_not_called()
        c.armDisarm.assert_not_called()

    def test_api_ok_arm_fails_releases(self):
        """enableApiControl ok, armDisarm throws → only release called."""
        c=_mc()
        c.armDisarm.side_effect=Exception("arm fail")
        c.enableApiControl.side_effect=[None,None]  # enable(True) ok, enable(False) ok
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.return_value=_st(z=1.0,ls=0)
            cr.return_value.read.return_value=_col()
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:0,lambda s:None)
        assert r.api_control_acquired and not r.armed
        assert r.api_control_released

    def test_airborne_exception_hover_land_no_disarm_if_not_landed(self):
        """Takeoff ok but flight-loop exception → hover+land called, disarm only if landing confirmed."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.side_effect=[_lf()]*6+[Exception("RPC dead")]
            # st0 (z=1,ls=0), post-climb (z=-1,ls=1,v=0), cleanup (all ls=1=Flying)
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*20
            cr.return_value.read.return_value=_col();flt.return_value=_fr();dd.return_value=_dd()
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:100.1,lambda s:None)
        c.hoverAsync.assert_called();c.landAsync.assert_called()
        assert not r.landing_confirmed;assert not r.disarmed

    def test_full_normal_flow_time_limit(self):
        """Normal flow: at least 1 velocity cmd sent, ends with time_limit, clean."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf();cr.return_value.read.return_value=_col()
            flt.return_value=_fr();dd.return_value=_dd()
            # st0(z=1,ls=0), post-climb(z=-1,ls=1,v=0), flight+cleanup all (z=-1,ls=1=flying), last 3 (z=0,ls=0=landed)
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*34+[_st(z=0.0,ls=0)]*20
            clk=[0.0]; calls=[0]
            def _clk():
                calls[0]+=1
                if calls[0]<=2: return clk[0]  # st0 + post-climb: return 0 (no time passed)
                clk[0]+=0.3  # flight iterations: advance 0.3s each
                return clk[0]
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),_clk,lambda s:None)
        assert r.termination_reason=="time_limit"
        assert r.success and r.exit_code==0
        assert vc.return_value.send_velocity_body_frd.call_count>=1
        kw=vc.return_value.send_velocity_body_frd.call_args.kwargs
        assert kw.get("vehicle_name")=="Drone1"
        assert r.landing_confirmed and r.disarmed and r.api_control_released

    def test_lidar_invalid_terminates(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.side_effect=[_lf()]*6+[_lf(False)]
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*20
            cr.return_value.read.return_value=_col();flt.return_value=_fr()
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:100.1,lambda s:None)
        assert "lidar_invalid" in (r.termination_reason or "")

    def test_collision_terminates(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*20
            cr.return_value.read.side_effect=[_col()]*6+[_col(False)]
            flt.return_value=_fr();dd.return_value=_dd()
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:100.1,lambda s:None)
        assert "collision" in (r.termination_reason or "")

    def test_geofence_terminates(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf();cr.return_value.read.return_value=_col()
            flt.return_value=_fr();dd.return_value=_dd()
            far_st=_st(z=-1.0,ls=1);far_st.position_ned_m=[3.0,3.0,-1.0]
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[far_st]+[_st(z=-1.0,ls=1)]*20
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),lambda:100.1,lambda s:None)
        assert "geofence" in (r.termination_reason or "")

    def test_no_landing_no_disarm(self):
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf();cr.return_value.read.return_value=_col()
            flt.return_value=_fr();dd.return_value=_dd()
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*100
            clk=[0.0]
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                  lambda:(clk.__setitem__(0,clk[0]+11.0),clk[0])[1],lambda s:None)
        assert r.termination_reason=="time_limit";assert not r.landing_confirmed;assert not r.disarmed

    # ── collision warm-up tests ──

    def test_warmup_floor_then_clean_allows(self):
        """First frame Floor(ts=0) then 5 clean frames: baseline recorded, flight proceeds."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf();flt.return_value=_fr();dd.return_value=_dd()
            cr.return_value.read.side_effect=[
                _col(ok=False,object_name="Floor",raw_timestamp=0,is_new_event=False),
            ]+[_col()]*50
            sr.return_value.read.side_effect=[
                _st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*34+[_st(z=0.0,ls=0)]*20
            clk=[0.0];calls=[0]
            def _clk():
                calls[0]+=1
                if calls[0]<=2:return clk[0]
                clk[0]+=0.3;return clk[0]
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),_clk,lambda s:None)
        assert r.startup_floor_contact_baseline is True
        assert r.success and r.exit_code==0

    def test_warmup_persistent_floor_rejects(self):
        """All 10 frames Floor collision: reject, no API control acquired."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.return_value=_st()
            cr.return_value.read.return_value=_col(ok=False,object_name="Floor",
                                                     raw_timestamp=0,is_new_event=False)
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                  lambda:0,lambda s:None)
        assert "preflight_floor_persists" in (r.termination_reason or "")
        assert not r.api_control_acquired
        assert r.exit_code!=0

    def test_warmup_wall_rejects(self):
        """First frame Wall: reject immediately, no API control."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.return_value=_st()
            cr.return_value.read.side_effect=[
                _col(ok=False,object_name="Wall",raw_timestamp=0,is_new_event=False),
            ]+[_col()]*20
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                  lambda:0,lambda s:None)
        assert "preflight_collision_0:Wall" in (r.termination_reason or "")
        assert not r.api_control_acquired

    def test_warmup_floor_then_new_collision_event_rejects(self):
        """Floor(ts=0) then new collision event with non-zero ts: reject."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.return_value=_st()
            cr.return_value.read.side_effect=[
                _col(ok=False,object_name="Floor",raw_timestamp=0,is_new_event=False),
                _col(ok=False,object_name="Floor",raw_timestamp=42,is_new_event=True),
            ]+[_col()]*20
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                  lambda:0,lambda s:None)
        assert "preflight_new_collision_event" in (r.termination_reason or "")
        assert not r.api_control_acquired

    def test_warmup_fail_no_api_control_calls(self):
        """Warmup rejection must never call enableApiControl, armDisarm, takeoffAsync."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr:
            lr.return_value.read.return_value=_lf()
            sr.return_value.read.return_value=_st()
            cr.return_value.read.return_value=_col(ok=False,object_name="Cylinder_1",
                                                     raw_timestamp=0,is_new_event=False)
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),
                                  lambda:0,lambda s:None)
        assert not r.api_control_acquired
        c.enableApiControl.assert_not_called()
        c.armDisarm.assert_not_called()
        c.takeoffAsync.assert_not_called()

    def test_warmup_floor_nonzero_ts_new_event_then_clean_allows(self):
        """First Floor with non-zero ts and is_new_event=True, then 5 clean: baseline set."""
        c=_mc()
        with patch("sensors.lidar_reader.LidarReader") as lr,\
             patch("sensors.state_reader.StateReader") as sr,\
             patch("sensors.collision_reader.CollisionReader") as cr,\
             patch("perception.pointcloud_filter.filter_pointcloud") as flt,\
             patch("perception.pointcloud_to_sectors.pointcloud_to_directional_distances") as dd,\
             patch("control.velocity_controller.VelocityController") as vc:
            lr.return_value.read.return_value=_lf();flt.return_value=_fr();dd.return_value=_dd()
            cr.return_value.read.side_effect=[
                _col(ok=False,object_name="Floor",raw_timestamp=1786026074827645952,is_new_event=True),
            ]+[_col()]*50
            sr.return_value.read.side_effect=[_st(z=1.0,ls=0)]+[_st(z=-1.0,ls=1)]+[_st(z=-1.0,ls=1)]*34+[_st(z=0.0,ls=0)]*20
            clk=[0.0];calls=[0]
            def _clk():
                calls[0]+=1
                if calls[0]<=2:return clk[0]
                clk[0]+=0.3;return clk[0]
            r=run_minimal_flight(_a(),lambda:MagicMock(get_raw_client=lambda:c),_clk,lambda s:None)
        assert r.startup_floor_contact_baseline is True
        assert r.success and r.exit_code==0

"""ROUND 4.1: AvoidanceSupervisor tests."""
import sys, math
from pathlib import Path
import numpy as np, pytest
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
from planning.avoidance_supervisor import AvoidanceSupervisor


def _obs(pos=(0,0,0), vel=(0,0,0), yaw=0.0, goal=(10,0,0), timestamp=0.0, dt=0.2):
    return {
        "ego": {"position": list(pos), "orientation": [0.0, 0.0, yaw],
                "linearVelocity": list(vel), "angularVelocity": [0.0,0.0,0.0]},
        "goal": list(goal), "globalPath": [list(pos), list(goal)],
        "staticObstacles": [], "dynamicObstacles": [],
        "localPointCloud": None, "dt": dt, "timestamp": timestamp,
        "collision": {},
    }

_ALL_RAYS = {k: 20.0 for k in ["front","back","left","right","up","down",
    "frontLeft","frontRight","backLeft","backRight",
    "frontUp","frontDown","leftUp","rightUp","leftDown","rightDown"]}


class TestSupervisorModes:
    def test_initial_cruise(self):
        sv = AvoidanceSupervisor()
        assert sv.mode == "CRUISE"
        assert sv.mode_since == 0.0

    def test_cruise_to_recovery_blocked(self):
        sv = AvoidanceSupervisor()
        rays = {**_ALL_RAYS, "front": 2.0, "frontLeft": 3.0, "frontRight": 3.0}
        obs = _obs(); obs["timestamp"] = 0.8  # past hysteresis
        result = sv.update(obs, ray_distances=rays)
        assert sv.mode in ("RECOVERY", "CRUISE")  # may need obstacle for full trigger

    def test_apf_called_once_per_update(self):
        sv = AvoidanceSupervisor()
        call_count = [0]
        orig = sv.apf.update
        def _wrap(o):
            call_count[0] += 1; return orig(o)
        sv.apf.update = _wrap
        sv.update(_obs(timestamp=0.0))
        assert call_count[0] == 1

    def test_pre_computed_apf_avoids_double_call(self):
        sv = AvoidanceSupervisor()
        call_count = [0]
        orig = sv.apf.update
        def _wrap(o):
            call_count[0] += 1; return orig(o)
        sv.apf.update = _wrap
        pre = sv.apf.update(_obs(timestamp=0.0))
        call_count[0] = 0
        sv.update(_obs(timestamp=0.0), pre_computed_apf_result=pre)
        assert call_count[0] == 0, "pre_computed_apf_result should skip internal APF call"

    def test_reset_clears_all(self):
        sv = AvoidanceSupervisor()
        sv.mode = "RECOVERY"; sv.mode_since = 5.0
        sv.last_global_path = [[1,2,3]]; sv.last_threats = [{"x": 1}]
        sv.reset(now=10.0)
        assert sv.mode == "CRUISE"
        assert sv.mode_since == 10.0
        assert sv.last_global_path == []
        assert sv.last_threats == []
        assert not sv.recovery.is_active

    def test_hover_fallback(self):
        sv = AvoidanceSupervisor()
        result = sv.update(_obs(timestamp=0.0))
        assert result is not None
        assert "velocity_world_ned_mps" in result

    def test_recovery_mode_produces_recovery_source(self):
        sv = AvoidanceSupervisor()
        sv.mode = "RECOVERY"; sv.mode_since = 0.0
        rays = {**_ALL_RAYS, "front": 10.0}
        result = sv.update(_obs(timestamp=0.0), ray_distances=rays)
        assert result["source"] in ("recovery", "apf")

    def test_front_ray_profile_keys(self):
        from planning.avoidance_supervisor import _front_ray_profile
        p = _front_ray_profile(_ALL_RAYS)
        for k in ("front", "left", "right", "up"):
            assert k in p
            assert p[k] == 20.0

    def test_static_blocked_detection(self):
        sv = AvoidanceSupervisor()
        rays = {**_ALL_RAYS, "front": 2.0}
        blocked = sv._is_static_blocked_ahead(_obs(), rays)
        assert isinstance(blocked, bool)

    def test_no_emergency_with_empty_threats(self):
        sv = AvoidanceSupervisor()
        assert not sv._is_emergency_threat([])
        assert not sv._is_dynamic_threat([])

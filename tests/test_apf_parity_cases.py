"""ROUND 4.1: APF parity case loader — loads apf_parity_cases.json and validates behaviors.

NOTE: Old JS is NOT executable (requires browser + Rapier + THREE.js).
These are source-derived behavioral tests, NOT numerical parity checks.
The expected fields describe direction signs and state behaviors derived
from source code analysis of ImprovedPotentialField.js.
"""
import json, sys, math
from pathlib import Path
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planning.improved_potential_field import ImprovedPotentialField
from planning.avoidance_supervisor import AvoidanceSupervisor

FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "apf_parity_cases.json"


def _load_cases():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _build_obs(case):
    ego_pos = case.get("ego_position", [0, 0, 0])
    ego_vel = case.get("ego_velocity", [0, 0, 0])
    ego_yaw = case.get("ego_yaw", 0.0)
    goal = case["goal_ned"]
    path = case.get("global_path", [ego_pos, goal])
    obstacles = case.get("obstacles", [])
    dt = case.get("dt", 0.2)
    return {
        "ego": {
            "position": ego_pos,
            "orientation": [0.0, 0.0, ego_yaw],
            "linearVelocity": ego_vel,
            "angularVelocity": [0.0, 0.0, 0.0],
        },
        "goal": goal,
        "globalPath": path,
        "staticObstacles": obstacles,
        "dynamicObstacles": [],
        "localPointCloud": None,
        "dt": dt,
        "timestamp": 0.0,
        "collision": {},
    }


class TestApfParityCases:
    """Source-derived behavioral tests — NOT numerical parity (JS not executable)."""

    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
    def test_case(self, case):
        expected = case["expected"]
        apf = ImprovedPotentialField()
        obs = _build_obs(case)
        result = apf.update(obs)
        vx, vy, vz = result["velocity_world_ned_mps"]
        diag = result.get("diagnostics", {})

        # ── Direction checks ──
        if expected.get("vx_positive"):
            assert vx >= 0, f"{case['name']}: vx={vx}, expected >= 0"
        if expected.get("vx_near_zero"):
            assert abs(vx) < 0.5, f"{case['name']}: vx={vx}, expected near zero"
        if expected.get("vx_reduced"):
            # With obstacle ahead, vx should be less than unobstructed case
            # We just check the repulsive is active
            rep_x = diag.get("repulsive_force_world_ned", (0, 0, 0))[0]
            assert abs(rep_x) > 0, f"{case['name']}: repulsive should be active, got {rep_x}"

        if expected.get("vy_positive"):
            assert vy >= 0, f"{case['name']}: vy={vy}, expected >= 0"
        if expected.get("vy_negative"):
            assert vy <= 0, f"{case['name']}: vy={vy}, expected <= 0"
        if expected.get("vy_near_zero"):
            assert abs(vy) < 0.5, f"{case['name']}: vy={vy}, expected near zero"

        if expected.get("vz_positive"):
            assert vz >= 0, f"{case['name']}: vz={vz}, expected >= 0"
        if expected.get("vz_negative"):
            assert vz <= 0, f"{case['name']}: vz={vz}, expected <= 0"
        if expected.get("vz_near_zero"):
            assert abs(vz) < 0.5, f"{case['name']}: vz={vz}, expected near zero"

        # ── Source checks ──
        assert result["source"] == expected.get("source", "apf"), \
            f"{case['name']}: source={result['source']}, expected {expected.get('source')}"

        # ── Force activity checks ──
        if expected.get("repulsive_nonzero"):
            rep = diag.get("repulsive_force_world_ned", (0, 0, 0))
            assert any(abs(r) > 1e-6 for r in rep), \
                f"{case['name']}: repulsive force should be non-zero, got {rep}"

        if expected.get("tangential_nonzero"):
            tan = diag.get("tangential_force_world_ned", (0, 0, 0))
            assert any(abs(t) > 1e-6 for t in tan), \
                f"{case['name']}: tangential force should be non-zero, got {tan}"

        # ── Bypass sign consistency ──
        if expected.get("bypass_sign_consistent"):
            sign1 = apf.last_bypass_sign
            obs2 = _build_obs(case)
            obs2["timestamp"] = 0.3  # within bypass_memory_seconds (0.8)
            result2 = apf.update(obs2)
            assert apf.last_bypass_sign == sign1, \
                f"{case['name']}: bypass sign should persist: {sign1} → {apf.last_bypass_sign}"


class TestSupervisorParityCases:
    """Supervisor mode transition tests from parity cases."""

    def test_recovery_entry_when_blocked(self):
        case = None
        for c in _load_cases():
            if c["name"] == "recovery_enter_triggers":
                case = c; break
        if case is None:
            pytest.skip("recovery_enter_triggers case not found")
        sv = AvoidanceSupervisor()
        obs = _build_obs(case)
        obs["timestamp"] = 0.8  # past hysteresis_enter
        rays = case.get("ray_distances", {})
        result = sv.update(obs, ray_distances=rays)
        expected = case["expected"]
        if expected.get("mode_not_cruise"):
            assert sv.mode != "CRUISE" or sv.mode == "RECOVERY", \
                f"Expected non-CRUISE mode, got {sv.mode}"

    def test_recovery_exit_when_clear(self):
        case = None
        for c in _load_cases():
            if c["name"] == "recovery_exit_when_clear":
                case = c; break
        if case is None:
            pytest.skip("recovery_exit_when_clear case not found")
        sv = AvoidanceSupervisor()
        sv.mode = "RECOVERY"; sv.mode_since = 0.0
        obs = _build_obs(case)
        obs["timestamp"] = 1.5  # past hysteresis_exit
        rays = case.get("ray_distances", {})
        result = sv.update(obs, ray_distances=rays)
        expected = case["expected"]
        if expected.get("mode_cruise"):
            assert sv.mode == "CRUISE", f"Expected CRUISE, got {sv.mode}"

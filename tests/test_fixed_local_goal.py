"""ROUND 4.1: fixed_local_goal tests."""
import sys, math
from pathlib import Path
import pytest
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
from planning.fixed_local_goal import compute_fixed_local_goal


class TestFixedLocalGoal:
    def test_body_relative_yaw_zero(self):
        goal, desc = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 5.0}},
                                               (0.0, 0.0, 0.0), 0.0)
        assert goal[0] == pytest.approx(5.0)
        assert goal[1] == pytest.approx(0.0)

    def test_body_relative_yaw_pi_over_2(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 5.0}},
                                            (0.0, 0.0, 0.0), math.pi / 2)
        assert goal[0] == pytest.approx(0.0, abs=1e-9)
        assert goal[1] == pytest.approx(5.0)

    def test_body_relative_yaw_pi(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 5.0}},
                                            (0.0, 0.0, 0.0), math.pi)
        assert goal[0] == pytest.approx(-5.0, abs=1e-9)
        assert goal[1] == pytest.approx(0.0, abs=1e-9)

    def test_right_offset(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 0.0, "right_m": 3.0}},
                                            (0.0, 0.0, 0.0), 0.0)
        assert goal[1] == pytest.approx(3.0)

    def test_down_offset(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 0.0, "down_m": 2.0}},
                                            (0.0, 0.0, 10.0), 0.0)
        assert goal[2] == pytest.approx(12.0)

    def test_absolute_local_ned(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "absolute_local_ned", "north_m": 100.0, "east_m": 50.0, "down_m": -5.0}},
                                            (10.0, 20.0, 30.0), 0.0)
        assert goal == (110.0, 70.0, -5.0)

    def test_position_offset_preserved(self):
        goal, _ = compute_fixed_local_goal({"goal": {"mode": "body_relative_at_start", "forward_m": 10.0}},
                                            (100.0, 200.0, -50.0), 0.0)
        assert goal[0] == pytest.approx(110.0)
        assert goal[1] == pytest.approx(200.0)
        assert goal[2] == pytest.approx(-50.0)

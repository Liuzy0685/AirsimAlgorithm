"""Phase B unit tests for the trajectory-centric navigation hardening.

Covers the new pure-computation units introduced in Phase B:

- planner_to_body_frame sign contract (LEFT → negative body Y)
- family_side
- REJOIN gating (lateral-error trigger) + smooth-curvature variants
- escape_hint (no-feasible → spatial hint for Recovery)
- geometric consistency (blocked previous trajectory → zero bonus)
- OccupancyGridMap.get_occupied_points_in_radius + self_filter_radius
- TrajectoryTracker pure-pursuit lateral sign

No AirSim RPC is exercised.
"""

import math

import numpy as np
import pytest

from mapping.occupancy_grid import (
    OccupancyGridMap,
    OccupancyGridParams,
)
from mapping.distance_field import DistanceField
from planners.local_trajectory_planner import (
    LocalTrajectoryPlanner,
    TrajectoryPlannerParams,
    TrajectoryMemory,
    STRAIGHT,
    LEFT,
    RIGHT,
    HARD_LEFT,
    HARD_RIGHT,
    REJOIN_SOFT,
    REJOIN_MEDIUM,
    family_side,
    planner_to_body_frame,
)
from planners.trajectory_tracker import TrajectoryTracker, TrackerResult


def _df(obstacles):
    df = DistanceField()
    df.set_obstacles(obstacles)
    return df


# ── sign contract ──


class TestSignContract:
    def test_left_curvature_is_negative_body_y(self):
        vx, vy = planner_to_body_frame(-0.45, False)
        assert vy < 0.0
        assert vx > 0.0  # forward

    def test_right_curvature_is_positive_body_y(self):
        vx, vy = planner_to_body_frame(0.45, False)
        assert vy > 0.0
        assert vx > 0.0

    def test_reverse_flips_forward_sign_not_lateral(self):
        vx, vy = planner_to_body_frame(0.45, True)
        assert vx < 0.0  # backing up
        assert vy > 0.0  # lateral sense unchanged

    def test_straight_is_pure_forward(self):
        vx, vy = planner_to_body_frame(0.0, False)
        assert vx == pytest.approx(1.0)
        assert vy == pytest.approx(0.0)

    def test_family_side(self):
        assert family_side(LEFT) == -1
        assert family_side(HARD_LEFT) == -1
        assert family_side(RIGHT) == 1
        assert family_side(HARD_RIGHT) == 1
        assert family_side(STRAIGHT) == 0


# ── rejoin gating ──


class TestRejoinGating:
    def _plan(self, drone, yaw, goal, path):
        planner = LocalTrajectoryPlanner(
            params=TrajectoryPlannerParams(), memory=TrajectoryMemory(),
        )
        return planner.plan(
            drone_position_ned=drone, yaw_rad=yaw, goal_xy=goal,
            global_path=path, distance_field=_df([]), global_path_version=0,
        )

    def test_rejoin_only_when_laterally_displaced(self):
        on_path = self._plan(
            (0.0, 0.0, 0.0), 0.0, (10.0, 0.0), [[0.0, 0.0], [10.0, 0.0]],
        )
        fams = {c.family for c in on_path.candidates}
        assert REJOIN_SOFT not in fams
        assert REJOIN_MEDIUM not in fams

        displaced = self._plan(
            (0.0, 0.0, 0.0), 0.0, (10.0, 2.0), [[0.0, 2.0], [10.0, 2.0]],
        )
        fams2 = {c.family for c in displaced.candidates}
        assert REJOIN_SOFT in fams2
        assert REJOIN_MEDIUM in fams2


# ── escape hint ──


class TestEscapeHint:
    def test_no_feasible_produces_escape_hint(self):
        planner = LocalTrajectoryPlanner(
            params=TrajectoryPlannerParams(), memory=TrajectoryMemory(),
        )
        r = planner.plan(
            drone_position_ned=(0.0, 0.0, 0.0), yaw_rad=0.0, goal_xy=(10.0, 0.0),
            global_path=None, distance_field=_df([(0.0, 0.0)]), global_path_version=0,
        )
        assert r.selected is None
        assert r.escape_hint is not None
        assert "side" in r.escape_hint
        assert "side_label" in r.escape_hint
        assert "clearance_m" in r.escape_hint


# ── geometric consistency ──


class TestGeometricConsistency:
    def test_overlap_consistency(self):
        planner = LocalTrajectoryPlanner(
            params=TrajectoryPlannerParams(), memory=TrajectoryMemory(),
        )
        pts = [(i * 0.25, 0.0) for i in range(17)]
        # Identical previous → consistency 1.0.
        assert planner._geometric_consistency(pts, pts, _df([])) == pytest.approx(1.0)

    def test_no_previous_points(self):
        planner = LocalTrajectoryPlanner(
            params=TrajectoryPlannerParams(), memory=TrajectoryMemory(),
        )
        pts = [(i * 0.25, 0.0) for i in range(17)]
        assert planner._geometric_consistency(pts, None, _df([])) == 0.0
        assert planner._geometric_consistency(pts, [], _df([])) == 0.0

    def test_blocked_previous_yields_zero(self):
        planner = LocalTrajectoryPlanner(
            params=TrajectoryPlannerParams(), memory=TrajectoryMemory(),
        )
        pts = [(i * 0.25, 0.0) for i in range(17)]
        # Previous trajectory now blocked at its start → zero bonus.
        assert planner._geometric_consistency(pts, pts, _df([(0.0, 0.0)])) == 0.0


# ── occupancy grid radius + self-filter ──


class TestOccupancyGridLocal:
    def test_radius_query_and_self_filter(self):
        m = OccupancyGridMap(OccupancyGridParams(
            resolution_m=0.5, self_filter_radius_m=0.5,
        ))
        # Self return at 0.3 m → filtered out (within self_filter_radius).
        m.update(np.array([[0.3, 0.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)
        # Real obstacle at ~5 m → occupied.
        m.update(np.array([[5.0, 0.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)

        near = m.get_occupied_points_in_radius(0.0, 0.0, 2.0)
        far = m.get_occupied_points_in_radius(0.0, 0.0, 10.0)

        # Self-filtered return never became an obstacle near the origin.
        assert near == []
        # The 5 m obstacle is captured by the wider window.
        assert len(far) > 0
        assert all(math.hypot(x, y) > 2.0 for x, y in far)


# ── tracker ──


class TestTrajectoryTracker:
    def _tracker(self):
        return TrajectoryTracker(
            lookahead_m=1.0, sample_spacing_m=0.25,
            forward_speed_mps=0.25, lateral_speed_mps=0.20,
            command_lookahead_m=1.0,
        )

    def test_lookahead_to_right_gives_positive_vy(self):
        # Lookahead point (index 4) is to the East of a North-facing drone.
        pts = [(i * 0.25, 0.0) for i in range(6)]
        pts[4] = (1.0, 1.0)
        r = self._tracker().compute_command(pts, (0.0, 0.0, 0.0), 0.0)
        assert isinstance(r, TrackerResult)
        assert r.vx == pytest.approx(0.25)
        assert r.vy > 0.0  # right
        assert r.curvature > 0.0

    def test_lookahead_to_left_gives_negative_vy(self):
        pts = [(i * 0.25, 0.0) for i in range(6)]
        pts[4] = (1.0, -1.0)
        r = self._tracker().compute_command(pts, (0.0, 0.0, 0.0), 0.0)
        assert r.vy < 0.0  # left
        assert r.curvature < 0.0

    def test_empty_trajectory_zero_command(self):
        r = self._tracker().compute_command([], (0.0, 0.0, 0.0), 0.0)
        assert r.vx == 0.0 and r.vy == 0.0 and r.vz == 0.0

    def test_lateral_velocity_clamped(self):
        # Large lateral error → curvature large, but vy clamps to lateral_speed.
        pts = [(i * 0.25, 0.0) for i in range(6)]
        pts[4] = (1.0, 20.0)
        r = self._tracker().compute_command(pts, (0.0, 0.0, 0.0), 0.0)
        assert abs(r.vy) <= 0.20 + 1e-9

    def test_3d_feedforward_and_vertical_error_enter_command(self):
        tracker = TrajectoryTracker(
            lookahead_m=0.25,
            sample_spacing_m=0.25,
            forward_speed_mps=0.25,
            lateral_speed_mps=0.20,
            command_lookahead_m=1.0,
            feedforward_gain=1.0,
            lateral_position_kp=0.10,
            vertical_position_kp=0.20,
            velocity_kd=0.0,
        )
        pts = [(0.0, 0.0, -0.5), (0.25, 0.10, -0.8), (0.50, 0.10, -1.0)]
        ff = [(0.25, 0.0, -0.10), (0.25, 0.0, -0.10), (0.0, 0.0, 0.0)]
        r = tracker.compute_command(
            pts,
            (0.0, 0.0, -0.5),
            0.0,
            trajectory_feedforward_body=ff,
            current_velocity_ned=(0.0, 0.0, 0.0),
        )
        assert r.vx == pytest.approx(0.25)
        assert r.vy > 0.0
        assert r.vz < 0.0

    def test_terminal_capture_uses_actual_goal_position(self):
        tracker = TrajectoryTracker(
            lookahead_m=1.0,
            sample_spacing_m=0.25,
            forward_speed_mps=0.25,
            lateral_speed_mps=0.20,
            terminal_slowdown_radius_m=2.0,
            terminal_goal_kp=0.5,
            terminal_goal_max_speed_mps=0.15,
            terminal_capture_radius_m=0.02,
        )
        # The cached trajectory points ahead, but the actual goal is behind
        # the lookahead point.  Terminal capture must command toward the goal.
        r = tracker.compute_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (1.0, 0.0, 0.0),
            0.0,
            goal_ned=(1.8, 0.5, 0.0),
        )
        assert r.vx > 0.0
        assert r.vy > 0.0
        assert r.vx > 0.10
        assert math.hypot(r.vx, r.vy) <= 0.15 + 1e-9

    def test_terminal_capture_holds_only_inside_tight_radius(self):
        tracker = TrajectoryTracker(
            lookahead_m=1.0,
            sample_spacing_m=0.25,
            forward_speed_mps=0.25,
            lateral_speed_mps=0.20,
            terminal_slowdown_radius_m=2.0,
            terminal_goal_kp=1.0,
            terminal_goal_max_speed_mps=0.05,
            terminal_capture_radius_m=0.02,
        )
        far = tracker.compute_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (0.030, 0.0, 0.0),
            0.0,
            goal_ned=(0.0, 0.0, 0.0),
        )
        near = tracker.compute_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (0.010, 0.0, 0.0),
            0.0,
            goal_ned=(0.0, 0.0, 0.0),
        )
        assert abs(far.vx) > 0.0
        assert near.vx == pytest.approx(0.0)
        assert near.vy == pytest.approx(0.0)

    def test_terminal_approach_radius_starts_goal_directed_control(self):
        tracker = TrajectoryTracker(
            lookahead_m=1.0,
            sample_spacing_m=0.25,
            forward_speed_mps=0.25,
            lateral_speed_mps=0.20,
            terminal_goal_approach_radius_m=2.0,
            terminal_slowdown_radius_m=0.5,
            terminal_goal_kp=0.5,
            terminal_goal_max_speed_mps=0.05,
            terminal_capture_radius_m=0.01,
        )
        r = tracker.compute_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (1.5, 0.2, 0.0),
            0.0,
            goal_ned=(0.0, 0.0, 0.0),
        )
        assert r.vx < 0.0
        assert r.vy < 0.0

"""Tests for the trajectory-centric navigation layer (Phase A).

Covers the four new pure-computation modules plus the sensor→world helper:

- mapping/occupancy_grid.py       (persistent 2D occupancy grid + ray casting)
- mapping/distance_field.py       (2D local ESDF)
- planners/local_trajectory_planner.py (deterministic candidates, scoring,
                                          memory, receding horizon)
- planners/goal_termination.py    (dwell-based mission_complete)
- flight_modes.automatic_mode._sensor_points_to_world_xy

No AirSim RPC is exercised here — everything is deterministic geometry.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mapping.occupancy_grid import (
    OccupancyGridMap,
    OccupancyGridParams,
    UNKNOWN,
    FREE,
    OCCUPIED,
)
from mapping.distance_field import DistanceField
from planners.local_trajectory_planner import (
    LocalTrajectoryPlanner,
    TrajectoryPlannerParams,
    TrajectoryMemory,
    STRAIGHT,
    LEFT,
    RIGHT,
    REJOIN,
    REJOIN_SOFT,
    REJOIN_MEDIUM,
)
from planners.goal_termination import GoalTerminationChecker, GoalTerminationParams
from flight_modes.automatic_mode import _sensor_points_to_world_xy


# ── helpers ──


def _df(obstacles):
    df = DistanceField()
    df.set_obstacles(obstacles)
    return df


def _plan(drone=(0.0, 0.0, 0.0), yaw=0.0, goal=(10.0, 0.0), global_path=None,
          obstacles=None, params=None, memory=None, unknown_query=None,
          global_path_version=0):
    planner = LocalTrajectoryPlanner(
        params=params, memory=memory or TrajectoryMemory(),
    )
    return planner.plan(
        drone_position_ned=drone, yaw_rad=yaw, goal_xy=goal,
        global_path=global_path, distance_field=_df(obstacles or []),
        unknown_query=unknown_query, global_path_version=global_path_version,
    )


def _candidate(result, family):
    for c in result.candidates:
        if c.family == family:
            return c
    return None


# ── occupancy grid ──


class TestOccupancyGrid:
    def _grid(self, **kw):
        return OccupancyGridMap(OccupancyGridParams(**kw))

    def test_ray_cast_marks_free_and_occupied(self):
        m = self._grid()
        # Single horizontal return 2 m ahead of a drone at (0,0,0), yaw=0.
        pts = np.array([[2.0, 0.0, 0.0]])
        m.update(pts, (0.0, 0.0, 0.0), 0.0)

        # Hit cell (world ~2 m) → OCCUPIED (+ inflation).
        assert m.state_at(2.0, 0.0) == OCCUPIED
        # Cell mid-ray (~1 m) → FREE.
        assert m.state_at(1.0, 0.0) == FREE
        # Never-observed cell → UNKNOWN.
        assert m.state_at(20.0, 20.0) == UNKNOWN
        assert len(m.get_occupied_points()) > 0

    def test_horizontal_band_filters_out_of_plane_points(self):
        m = self._grid()
        # z=2.0 exceeds the 1.0 m horizontal band → ignored entirely.
        pts = np.array([[2.0, 0.0, 2.0]])
        m.update(pts, (0.0, 0.0, 0.0), 0.0)
        assert m.get_occupied_points() == []

    def test_origin_fixed_on_first_update(self):
        m = self._grid()
        m.update(np.array([[2.0, 0.0, 0.0]]), (5.0, 5.0, 0.0), 0.0)
        # First update fixes the origin; world (7,5) is the 2 m hit.
        assert m.state_at(7.0, 5.0) == OCCUPIED

    def test_to_obstacles_cbmba_compatible(self):
        m = self._grid()
        m.update(np.array([[2.0, 0.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)
        obs = m.to_obstacles(-1.0)
        assert obs
        for o in obs:
            assert o["type"] == "map"
            assert o["dynamic"] is False
            assert len(o["position"]) == 3
            assert o["position"][2] == -1.0
            assert o["footprint_half_extents"]

    def test_exploration_interfaces(self):
        m = self._grid()
        assert m.get_explored_area() == 0.0
        assert m.get_unknown_ratio() == 1.0
        m.update(np.array([[2.0, 0.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)
        assert m.get_explored_area() > 0.0
        assert 0.0 < m.get_unknown_ratio() < 1.0
        assert isinstance(m.get_frontier_cells(), list)


# ── distance field ──


class TestDistanceField:
    def test_distance_and_clearance(self):
        df = _df([(3.0, 0.0), (0.0, 4.0)])
        assert df.distance_at(0.0, 0.0) == pytest.approx(3.0)
        assert df.distance_at(3.0, 0.0) == pytest.approx(0.0)
        assert df.trajectory_min_clearance([(0.0, 0.0), (3.0, 0.0)]) == pytest.approx(0.0)
        assert df.trajectory_min_clearance([(0.0, 0.0), (1.0, 0.0)]) == pytest.approx(2.0)

    def test_empty_field_infinite(self):
        df = _df([])
        assert math.isinf(df.distance_at(0.0, 0.0))
        assert math.isinf(df.trajectory_min_clearance([(0.0, 0.0)]))

    def test_gradient_points_away_from_obstacle(self):
        df = _df([(0.0, 0.0)])
        gx, gy = df.gradient_at(1.0, 0.0)
        # Gradient of distance from the origin at (1,0) points +x.
        assert gx == pytest.approx(1.0, abs=1e-3)
        assert abs(gy) < 1e-3


# ── trajectory planner ──


class TestTrajectoryPlanner:
    def test_narrow_passage_prefers_centered_straight(self):
        # Two nearby side surfaces leave the centerline open. The planner must
        # not alternate between side arcs just because both pillars are close.
        params = TrajectoryPlannerParams(
            hard_clearance_m=0.90,
            narrow_passage_enabled=True,
            narrow_passage_side_probe_m=1.0,
            narrow_passage_side_obstacle_max_distance_m=2.5,
            narrow_passage_max_center_clearance_m=1.6,
        )
        r = _plan(
            goal=(10.0, 0.0),
            obstacles=[(3.0, -1.0), (3.0, 1.0)],
            params=params,
        )
        assert r.selected is not None
        assert r.selected.family == STRAIGHT
        assert r.family_switch is None

    def test_narrow_passage_holds_straight_when_one_side_leaves_scan(self):
        # The opening can make one side disappear from the local field just
        # before the vehicle exits. Keep the already safe centerline briefly.
        memory = TrajectoryMemory()
        params = TrajectoryPlannerParams(
            hard_clearance_m=0.90,
            narrow_passage_enabled=True,
            narrow_passage_hold_enabled=True,
            narrow_passage_hold_max_duration_s=6.0,
            narrow_passage_hold_max_distance_m=4.5,
        )
        planner = LocalTrajectoryPlanner(params=params, memory=memory)
        r1 = planner.plan(
            (0.0, 0.0, 0.0), 0.0, (10.0, 0.0), None,
            _df([(3.0, -1.0), (3.0, 1.0)]), None, 0,
        )
        assert r1.selected.family == STRAIGHT
        assert r1.narrow_passage_active

        # One side is no longer represented, while the direct path remains
        # safe. A normal score comparison could select a side arc here.
        r2 = planner.plan(
            (1.0, 0.0, 0.0), 0.0, (10.0, 0.0), None,
            _df([(3.0, 1.0)]), None, 0,
        )
        assert r2.selected.family == STRAIGHT
        assert r2.narrow_passage_active

    def test_straight_wins_with_no_obstacles(self):
        r = _plan()
        assert r.selected is not None
        assert r.selected.family == STRAIGHT
        assert r.valid_count == r.generated
        # Full trajectory is planned; command is a finite scalar velocity.
        assert len(r.selected.points) > 4
        assert r.command_vx == pytest.approx(0.25)   # forward_speed_mps
        assert abs(r.command_vy) < 1e-9

    def test_cylinder_blocks_straight_but_not_curved(self):
        # Obstacle 3 m directly ahead; hard clearance is 1 m.
        r = _plan(obstacles=[(3.0, 0.0)])
        straight = _candidate(r, STRAIGHT)
        assert straight is not None
        assert not straight.valid
        assert straight.invalid_reason == "clearance"
        assert r.selected is not None
        assert r.selected.valid
        assert r.selected.family != STRAIGHT

    def test_wall_blocks_straight_whole_trajectory(self):
        # A dense vertical wall at x=3 spans the whole corridor.
        wall = [(3.0, y) for y in np.arange(-8.0, 8.01, 0.25)]
        r = _plan(obstacles=wall, goal=(10.0, 0.0))
        straight = _candidate(r, STRAIGHT)
        assert straight is not None and not straight.valid
        # Some lateral candidate that turns before the wall stays feasible.
        assert r.valid_count >= 1
        assert r.selected is not None and r.selected.valid

    def test_rejoin_generated_when_laterally_displaced(self):
        # Drone at (0,0) facing North, path offset 2 m to the East (right).
        # Lateral error 2 m > 0.75 m trigger → smooth rejoin variants appear.
        r = _plan(
            drone=(0.0, 0.0, 0.0), yaw=0.0, goal=(10.0, 2.0),
            global_path=[[0.0, 2.0], [10.0, 2.0]],
        )
        assert _candidate(r, REJOIN_SOFT) is not None
        assert _candidate(r, REJOIN_MEDIUM) is not None
        # Rejoin curves toward the path → better alignment than straight.
        straight = _candidate(r, STRAIGHT)
        rejoin_soft = _candidate(r, REJOIN_SOFT)
        assert straight is not None and rejoin_soft is not None
        assert rejoin_soft.global_path_alignment > straight.global_path_alignment

    def test_no_rejoin_candidate_when_on_path(self):
        # On-path (lateral error 0) → no rejoin even with a global path.
        r = _plan(
            drone=(0.0, 0.0, 0.0), yaw=0.0, goal=(10.0, 0.0),
            global_path=[[0.0, 0.0], [10.0, 0.0]],
        )
        assert _candidate(r, REJOIN_SOFT) is None
        assert _candidate(r, REJOIN_MEDIUM) is None

    def test_no_rejoin_candidate_without_global_path(self):
        r = _plan(goal=(10.0, 0.0), global_path=None)
        assert _candidate(r, REJOIN_SOFT) is None

    def test_no_feasible_trajectory_returns_zero_command(self):
        # Obstacle exactly at the drone position invalidates every candidate.
        r = _plan(obstacles=[(0.0, 0.0)])
        assert r.selected is None
        assert r.valid_count == 0
        assert r.command_vx == 0.0 and r.command_vy == 0.0

    def test_temporal_consistency_no_switch_on_repeat(self):
        memory = TrajectoryMemory()
        df = _df([])
        planner = LocalTrajectoryPlanner(params=None, memory=memory)
        r1 = planner.plan((0.0, 0.0, 0.0), 0.0, (10.0, 0.0), None, df, None, 0)
        r2 = planner.plan((0.0, 0.0, 0.0), 0.0, (10.0, 0.0), None, df, None, 0)
        assert r1.selected.family == STRAIGHT
        assert r2.selected.family == STRAIGHT
        assert r2.family_switch is None
        # The repeated family receives a consistency bonus.
        straight2 = _candidate(r2, STRAIGHT)
        assert straight2.consistency == pytest.approx(1.0)

    def test_family_switch_detected_on_change(self):
        memory = TrajectoryMemory()
        planner = LocalTrajectoryPlanner(params=None, memory=memory)
        empty = _df([])
        r1 = planner.plan((0.0, 0.0, 0.0), 0.0, (10.0, 0.0), None, empty, None, 0)
        assert r1.selected.family == STRAIGHT
        # Introduce an obstacle that blocks straight: family must change.
        blocked = _df([(3.0, 0.0)])
        r2 = planner.plan((0.0, 0.0, 0.0), 0.0, (10.0, 0.0), None, blocked, None, 0)
        assert r2.selected.family != STRAIGHT
        assert r2.family_switch is not None
        assert r2.family_switch[0] == STRAIGHT

    def test_unknown_penalty_applies_when_query_marks_unknown(self):
        r = _plan(
            obstacles=[],
            unknown_query=lambda x, y: x > 2.0,  # region beyond 2 m unknown
        )
        straight = _candidate(r, STRAIGHT)
        # Straight path samples beyond x=2 get penalised.
        assert straight.unknown_penalty > 0.0

    def test_reverse_penalised_but_available(self):
        # A full wall ahead makes forward infeasible; reverse is penalised but valid.
        wall = [(2.0, y) for y in np.arange(-10.0, 10.01, 0.25)]
        r = _plan(obstacles=wall, goal=(10.0, 0.0))
        reverse_families = {"REVERSE_LEFT", "REVERSE_RIGHT"}
        assert any(c.family in reverse_families and c.valid for c in r.candidates)


# ── memory ──


class TestTrajectoryMemory:
    def test_history_trimmed(self):
        m = TrajectoryMemory(history_length=3)
        for fam in ("A", "B", "C", "D"):
            m.record(fam, 1.0, 0)
        assert m.previous_family == "D"
        assert m.history == ["B", "C", "D"]

    def test_version_tracked(self):
        m = TrajectoryMemory()
        m.record("STRAIGHT", 1.0, 7)
        assert m.previous_global_path_version == 7


# ── goal termination ──


class TestGoalTermination:
    def test_dwell_before_reached(self):
        gt = GoalTerminationChecker(GoalTerminationParams(
            distance_tolerance_m=1.0, altitude_tolerance_m=0.4,
            max_speed_mps=0.25, dwell_time_s=1.0,
        ))
        goal = (10.0, 0.0, -1.0)
        r1 = gt.update((10.2, 0.1, -1.1), 0.1, goal, 0.0)
        assert r1.within_distance and r1.within_altitude and r1.speed_low
        assert not r1.reached
        assert not r1.dwelled

        r2 = gt.update((10.2, 0.1, -1.1), 0.1, goal, 1.1)
        assert r2.reached and r2.dwelled
        assert r2.dwell_elapsed_s == pytest.approx(1.1)

    def test_reset_when_outside_tolerance(self):
        gt = GoalTerminationChecker(GoalTerminationParams(dwell_time_s=0.5))
        goal = (10.0, 0.0, -1.0)
        gt.update((10.2, 0.1, -1.1), 0.1, goal, 0.0)
        gt.update((10.2, 0.1, -1.1), 0.1, goal, 0.3)
        # Move away before dwell completes → resets.
        r = gt.update((15.0, 0.0, -1.0), 0.1, goal, 0.4)
        assert not r.within_distance
        assert not r.reached

    def test_speed_too_high_blocks_reached(self):
        gt = GoalTerminationChecker(GoalTerminationParams(
            max_speed_mps=0.25, dwell_time_s=0.0,
        ))
        r = gt.update((10.0, 0.0, -1.0), 1.0, (10.0, 0.0, -1.0), 0.0)
        assert not r.speed_low
        assert not r.reached

    def test_3d_distance_and_vertical_speed_are_required(self):
        gt = GoalTerminationChecker(GoalTerminationParams(
            distance_tolerance_m=1.0,
            altitude_tolerance_m=0.4,
            max_speed_mps=0.25,
            max_vertical_speed_mps=0.20,
            position_std_tolerance_m=0.20,
            history_size_frames=1,
            dwell_time_s=0.0,
        ))
        goal = (10.0, 0.0, -1.0)
        r = gt.update(
            (10.0, 0.0, -1.0),
            speed_mps=0.0,
            goal_ned=goal,
            now=0.0,
            velocity_ned_mps=(0.0, 0.0, 0.3),
        )
        assert not r.speed_low
        assert not r.reached

    def test_position_history_must_be_stable(self):
        gt = GoalTerminationChecker(GoalTerminationParams(
            distance_tolerance_m=1.0,
            altitude_tolerance_m=0.4,
            max_speed_mps=0.25,
            position_std_tolerance_m=0.05,
            history_size_frames=3,
            dwell_time_s=0.0,
        ))
        goal = (10.0, 0.0, -1.0)
        for now, x in enumerate((9.8, 10.2, 10.0)):
            r = gt.update(
                (x, 0.0, -1.0), 0.0, goal, float(now),
                velocity_ned_mps=(0.0, 0.0, 0.0),
            )
        assert not r.position_stable
        assert not r.reached


# ── sensor → world helper ──


class TestSensorPointsToWorldXY:
    def test_conversion_and_band_filter(self):
        pts = np.array([
            [2.0, 0.0, 0.0],   # forward 2 m (in band)
            [0.0, 1.0, 0.0],   # right 1 m (in band)
            [2.0, 0.0, 2.0],   # out of band (ignored)
        ])
        out = _sensor_points_to_world_xy(pts, (0.0, 0.0, 0.0), 0.0)
        assert (2.0, 0.0) in out
        assert (0.0, 1.0) in out
        assert len(out) == 2

    def test_empty_and_none(self):
        assert _sensor_points_to_world_xy(None, (0, 0, 0), 0.0) == []
        assert _sensor_points_to_world_xy(np.empty((0, 3)), (0, 0, 0), 0.0) == []

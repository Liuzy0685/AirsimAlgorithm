"""Phase C1-R unit tests for the realtime-loop decoupling fixes.

Covers the pure-computation units introduced/refactored in Phase C1-R:

- GlobalPlannerWorker (background CBMBA A* — request/result/search_count/shutdown)
- _record_cbmba_search (per-frame duplicate-search guard)
- _downsample_xy / _prune_global_path (path/point helpers)
- trajectory_flight.yaml + double-inflation config resolution

No AirSim RPC is exercised.
"""

import math
import sys
import time
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from flight_modes.automatic_mode import (
    AutomaticMode,
    GlobalPlannerWorker,
    _replan_result_action,
    _resolve_mission_goal,
)
from planners.goal_termination import GoalTerminationChecker, GoalTerminationParams


def _fake_result(success=True, path=None):
    return types.SimpleNamespace(
        success=success,
        path_world=path or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        nodes_expanded=42,
        planning_time_ms=12.5,
        grid_size=100,
        max_lateral_deviation_m=0.3,
    )


class _FakePlanner:
    def __init__(self, result=None, delay_s=0.0):
        self._result = result if result is not None else _fake_result()
        self._delay_s = delay_s
        self.calls = []

    def plan_with_result(self, obstacles, start, goal):
        self.calls.append((obstacles, start, goal))
        if self._delay_s:
            time.sleep(self._delay_s)
        return self._result


def _wait_for_result(worker, timeout_s=2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        res = worker.get_latest_result()
        if res is not None:
            return res
        time.sleep(0.005)
    return None


# ── GlobalPlannerWorker ──


class TestGlobalPlannerWorker:
    def test_request_replan_returns_increasing_ids(self):
        w = GlobalPlannerWorker(_FakePlanner())
        ids = [
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="r1"),
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="r2"),
        ]
        assert ids[0] == 1 and ids[1] == 2

    def test_no_result_before_start(self):
        w = GlobalPlannerWorker(_FakePlanner())
        assert w.get_latest_result() is None

    def test_worker_publishes_result_on_background_thread(self):
        planner = _FakePlanner(delay_s=0.02)
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        try:
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="init")
            res = _wait_for_result(w)
            assert res is not None
            assert res["success"] is True
            assert res["request_id"] == 1
            assert res["planning_time_ms"] == 12.5
            assert len(planner.calls) == 1
        finally:
            w.shutdown()

    def test_search_count_increments(self):
        planner = _FakePlanner(delay_s=0.02)
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        try:
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0])
            _wait_for_result(w)
            assert w.search_count == 1
        finally:
            w.shutdown()

    def test_failed_search_still_published_with_success_false(self):
        planner = _FakePlanner(result=_fake_result(success=False, path=[]))
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        try:
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0])
            res = _wait_for_result(w)
            assert res is not None
            assert res["success"] is False
        finally:
            w.shutdown()

    def test_shutdown_stops_worker(self):
        planner = _FakePlanner(delay_s=0.02)
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        w.shutdown()
        assert w._thread is None


# ── per-frame CBMBA duplicate guard ──


class TestCbmbaSearchCounter:
    def _auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._cbmba_search_total = 0
        a._cbmba_searches_this_frame = 0
        return a

    def test_single_search_increments_total(self):
        a = self._auto()
        a._record_cbmba_search("cbmba_shadow")
        assert a._cbmba_search_total == 1
        assert a._cbmba_searches_this_frame == 1

    def test_second_search_in_same_frame_flags_duplicate(self):
        a = self._auto()
        a._record_cbmba_search("cbmba_shadow")
        a._record_cbmba_search("trajectory_global_replan")
        assert a._cbmba_search_total == 2
        assert a._cbmba_searches_this_frame == 2

    def test_frame_reset_clears_duplicate_flag(self):
        a = self._auto()
        a._record_cbmba_search("cbmba_shadow")
        a._cbmba_searches_this_frame = 0  # loop top-of-frame reset
        a._record_cbmba_search("cbmba_shadow")
        assert a._cbmba_searches_this_frame == 1  # no duplicate within new frame


# ── path/point helpers ──


class TestPathHelpers:
    def test_downsample_xy_dedups_within_cell(self):
        pts = [(0.0, 0.0), (0.1, 0.1), (0.49, 0.49), (1.0, 0.0)]
        out = AutomaticMode._downsample_xy(pts, 0.5)
        # (0,0),(0.1,0.1),(0.49,0.49) share the (0,0) voxel; (1,0) is separate.
        assert len(out) == 2

    def test_downsample_xy_empty(self):
        assert AutomaticMode._downsample_xy([], 0.5) == []

    def test_prune_global_path_drops_passed_waypoints(self):
        path = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        a = AutomaticMode.__new__(AutomaticMode)
        out = a._prune_global_path(path, (2.1, 0.0))
        # Nearest waypoint is index 2; prune keeps from index 2 onward.
        assert out[0] == [2.0, 0.0, 0.0]
        assert len(out) == 2


# ── config resolution ──


class TestTrajectoryFlightConfig:
    def test_trajectory_flight_yaml_sized_for_42m_goal(self):
        cfg_path = _PROJECT_ROOT / "configs" / "trajectory_flight.yaml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        fc = raw["minimal_flight"]
        assert fc["geofence_radius_m"] >= 42.6
        assert fc["max_flight_duration_s"] >= 120.0

    def test_trajectory_planner_yaml_inflation_single_source(self):
        cfg_path = _PROJECT_ROOT / "configs" / "trajectory_planner.yaml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        # Phase C1-R: grid no longer pre-inflates; CBMBA inflation_radius is
        # the single source of global-path inflation.
        assert raw["occupancy_grid"]["inflation_cells"] == 0


# ── mission-goal resolution (Goal Z semantics) ──


class TestResolveMissionGoal:
    def test_cli_fixed_goal_overrides_missing_or_present_actor(self):
        goal, source, actor = _resolve_mission_goal(
            actor_xyz=(42.0, 4.0, 0.4),
            target_z_ned=-1.0,
            fallback_start_ned=(0.0, 0.0, 0.0),
            fallback_heading_rad=0.0,
            fallback_dist_m=15.0,
            goal_xy_override=(15.0, 0.0),
        )
        assert goal == (15.0, 0.0, -1.0)
        assert source == "cli_fixed"
        assert actor is None

    def test_actor_goal_z_is_cruise_altitude_not_ground(self):
        # MissionEnd actor sits on the ground at Z ≈ +0.4 NED; the drone cruises
        # at target_z_ned = -1.0.  The navigation goal Z must be the cruise
        # altitude, not the actor's ground-level Z.
        goal, source, actor = _resolve_mission_goal(
            actor_xyz=(42.0, 4.0, 0.4),
            target_z_ned=-1.0,
            fallback_start_ned=(0.0, 0.0, 0.0),
            fallback_heading_rad=0.0,
            fallback_dist_m=15.0,
        )
        assert source == "actor"
        assert actor == (42.0, 4.0, 0.4)  # scene metadata preserved verbatim
        assert goal == (42.0, 4.0, -1.0)  # Z = cruise, XY = actor

    def test_config_fixed_fallback_goal_z_is_start_z(self):
        # Legacy config_fixed goal preserves the initial airborne Z (start Z),
        # not the cruise target — matching test_guided_apf_shadow's
        # test_fixed_goal_z_preserved.
        goal, source, actor = _resolve_mission_goal(
            actor_xyz=None,
            target_z_ned=-1.0,
            fallback_start_ned=(0.0, 0.0, -0.54),
            fallback_heading_rad=0.0,
            fallback_dist_m=15.0,
        )
        assert source == "config_fixed"
        assert actor is None
        # heading 0 → +X by fallback_dist; Z = initial airborne Z.
        assert goal == (15.0, 0.0, -0.54)

    def test_config_fixed_fallback_respects_heading(self):
        goal, source, _ = _resolve_mission_goal(
            actor_xyz=None,
            target_z_ned=-1.0,
            fallback_start_ned=(1.0, 2.0, -0.5),
            fallback_heading_rad=math.pi / 2,
            fallback_dist_m=10.0,
        )
        assert source == "config_fixed"
        # heading 90° (East, +Y in NED) → goal Y = 2 + 10.
        assert abs(goal[0] - 1.0) < 1e-9
        assert abs(goal[1] - 12.0) < 1e-9
        assert goal[2] == -0.5  # start Z preserved


# ── goal termination with ground actor (Z semantics) ──


class TestGroundActorGoalTermination:
    """The ground-actor Z bug: feeding actor Z (+0.4) into GoalTerminationChecker
    while the drone cruises at Z=-1.0 made |dz| = 1.4 > 0.4 forever.  With the
    fix, the checker is fed goal Z = cruise altitude, so arrival is detected."""

    def _checker(self, dwell_time_s=1.0):
        return GoalTerminationChecker(
            GoalTerminationParams(
                enabled=True,
                distance_tolerance_m=1.0,
                altitude_tolerance_m=0.4,
                max_speed_mps=0.25,
                dwell_time_s=dwell_time_s,
            )
        )

    def test_ground_actor_completes_at_cruise_altitude(self):
        # Navigation goal: (42, 4, -1.0); drone arrives at (42, 4, -1.0).
        checker = self._checker()
        goal = (42.0, 4.0, -1.0)
        pos = (42.0, 4.0, -1.0)
        r1 = checker.update(pos, speed_mps=0.1, goal_ned=goal, now=0.0)
        assert r1.within_distance and r1.within_altitude and r1.speed_low
        assert not r1.reached  # not dwelled yet
        r2 = checker.update(pos, speed_mps=0.1, goal_ned=goal, now=1.0)
        assert r2.dwelled and r2.reached

    def test_xy_not_reached_fails(self):
        checker = self._checker()
        goal = (42.0, 4.0, -1.0)
        pos = (40.0, 4.0, -1.0)  # 2 m short of goal XY
        r = checker.update(pos, speed_mps=0.1, goal_ned=goal, now=0.0)
        assert not r.within_distance
        assert not r.reached

    def test_altitude_hold_fail_fails(self):
        # Drone is at cruise target_z=-1.0 but its actual Z drifted to -0.2:
        # 0.8 m of altitude error > 0.4 tolerance → cannot terminate.
        checker = self._checker()
        goal = (42.0, 4.0, -1.0)
        pos = (42.0, 4.0, -0.2)
        r1 = checker.update(pos, speed_mps=0.1, goal_ned=goal, now=0.0)
        assert r1.within_distance
        assert not r1.within_altitude
        assert not r1.reached
        r2 = checker.update(pos, speed_mps=0.1, goal_ned=goal, now=2.0)
        assert not r2.reached  # altitude never satisfied → never dwells

    def test_actor_ground_z_would_fail_without_z_fix(self):
        # Regression guard: if the checker were fed the raw actor Z (+0.4) while
        # the drone is at cruise Z=-1.0, arrival must NOT be detected.  This
        # documents why the goal Z must be the cruise altitude, not actor Z.
        checker = self._checker()
        actor_goal = (42.0, 4.0, 0.4)  # raw actor pose (ground marker)
        pos = (42.0, 4.0, -1.0)
        r = checker.update(pos, speed_mps=0.1, goal_ned=actor_goal, now=0.0)
        assert not r.within_altitude
        assert not r.reached


# ── stale CBMBA worker result (monotonic request id) ──


class TestReplanResultAction:
    def test_stale_result_ignored(self):
        # An older request (id 3) finishes after id 4 was already applied.
        assert _replan_result_action(request_id=3, applied_id=4) == "ignore_stale"

    def test_newer_result_applied(self):
        assert _replan_result_action(request_id=5, applied_id=4) == "apply"

    def test_equal_result_is_noop(self):
        assert _replan_result_action(request_id=4, applied_id=4) == "noop"

    def test_applied_id_never_decreases(self):
        # Replaying the same guard over a late-finish sequence must never let a
        # stale id overwrite the applied id: applied only moves upward.
        applied = -1
        for request_id in (1, 2, 2, 3, 1, 4, 2):  # includes a stale re-delivery
            action = _replan_result_action(request_id, applied)
            if action == "apply":
                assert request_id > applied  # monotonic precondition
                applied = request_id
        assert applied == 4

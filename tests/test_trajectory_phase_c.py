"""Phase C0 unit tests for trajectory-centric flight validation.

Covers the new pure-computation units introduced in Phase C0 (tests A–J):

- A. adaptive horizon (sec 8)
- B. family-switch hysteresis hold-time / improvement gates (sec 4)
- C. direct LEFT↔RIGHT switch penalty (sec 5)
- D. REJOIN clear-front gate (sec 6)
- E. REJOIN dynamic alignment bonus weight (sec 6)
- F. invalid-reason histogram (sec 7)
- G. family transition log breakdown (sec 3)
- H. mission progress monitor: stuck vs mission-stalled (sec 23/24)
- I. obstacle avoidance episode tracker (sec 2)
- J. single-obstacle behaviour monitor (sec 25)
- plus: end-of-flight metrics summary + CSV trace round-trip (sec 1/21/31)

No AirSim RPC is exercised.
"""

import math

import pytest

from mapping.distance_field import DistanceField
from planners.local_trajectory_planner import (
    HARD_LEFT,
    LEFT,
    REJOIN_SOFT,
    RIGHT,
    STRAIGHT,
    LocalTrajectoryPlanner,
    TrajectoryPlannerParams,
    TrajectoryMemory,
    _rejoin_variants,
)
from flight_modes.trajectory_flight_metrics import (
    FamilyTransitionLog,
    FlightTraceWriter,
    MissionProgressMonitor,
    ObstacleAvoidanceEpisodeTracker,
    SingleObstacleBehaviorMonitor,
    TrajectoryFlightMetrics,
)


def _df(obstacles):
    df = DistanceField()
    df.set_obstacles(obstacles)
    return df


def _planner(params=None, memory=None, clock=None):
    return LocalTrajectoryPlanner(
        params=params or TrajectoryPlannerParams(),
        memory=memory or TrajectoryMemory(),
        clock=clock,
    )


# ── A. adaptive horizon (sec 8) ──


class TestAdaptiveHorizon:
    def test_horizon_shrinks_near_obstacle(self):
        pl = _planner()
        assert pl._adaptive_horizon(1.0) == pytest.approx(2.0)      # near → min
        assert pl._adaptive_horizon(3.5) == pytest.approx(3.0)      # mid
        assert pl._adaptive_horizon(10.0) == pytest.approx(4.0)     # far → max
        assert pl._adaptive_horizon(float("inf")) == pytest.approx(4.0)

    def test_horizon_fixed_when_disabled(self):
        p = TrajectoryPlannerParams(adaptive_horizon_enabled=False, horizon_m=4.0)
        pl = _planner(params=p)
        assert pl._adaptive_horizon(1.0) == pytest.approx(4.0)


# ── B. family-switch hysteresis (sec 4) ──


class TestFamilySwitchHysteresis:
    def test_hold_time_gate_suppresses_immediate_switch(self):
        clock = [0.0]
        mem = TrajectoryMemory()
        mem.previous_family = LEFT
        mem.previous_score = 10.0
        mem.current_family_held_since = 0.0
        pl = _planner(memory=mem, clock=lambda: clock[0])
        clock[0] = 0.1  # held only 0.1s < 0.5s hold time
        r = pl.plan((0, 0, 0), 0.0, (10.0, 0.0), [[0, 0], [10, 0]], _df([]))
        assert r.selected is not None
        assert r.selected.family == LEFT       # incumbent survives
        assert r.family_switch is None         # no switch recorded

    def test_insufficient_improvement_gate(self):
        clock = [1.0]
        mem = TrajectoryMemory()
        mem.previous_family = LEFT
        mem.previous_score = 10.0
        mem.current_family_held_since = 0.0
        p = TrajectoryPlannerParams(family_switch_min_score_improvement=100.0)
        pl = _planner(params=p, memory=mem, clock=lambda: clock[0])
        # held_for = 1.0s ≥ hold time, but improvement can never reach 100.
        r = pl.plan((0, 0, 0), 0.0, (10.0, 0.0), [[0, 0], [10, 0]], _df([]))
        assert r.selected.family == LEFT
        assert r.family_switch is None

    def test_switch_allowed_when_gates_pass(self):
        clock = [1.0]
        mem = TrajectoryMemory()
        mem.previous_family = LEFT
        mem.previous_score = -float("inf")
        mem.current_family_held_since = 0.0
        p = TrajectoryPlannerParams(
            family_switch_min_hold_time_s=0.0,
            family_switch_min_score_improvement=0.0,
        )
        pl = _planner(params=p, memory=mem, clock=lambda: clock[0])
        r = pl.plan((0, 0, 0), 0.0, (10.0, 0.0), [[0, 0], [10, 0]], _df([]))
        assert r.selected.family == STRAIGHT
        assert r.family_switch == (LEFT, STRAIGHT, "score_exceeded")


# ── C. direct opposite-switch penalty (sec 5) ──


class TestDirectOppositePenalty:
    def test_opposite_side_flagged_only(self):
        mem = TrajectoryMemory()
        mem.previous_family = RIGHT
        pl = _planner(memory=mem)
        r = pl.plan((0, 0, 0), 0.0, (10.0, 0.0), [[0, 0], [10, 0]], _df([]))
        by_fam = {c.family: c for c in r.candidates}
        assert by_fam[LEFT].direct_opposite_switch is True
        assert by_fam[HARD_LEFT].direct_opposite_switch is True
        assert by_fam[RIGHT].direct_opposite_switch is False
        assert by_fam[STRAIGHT].direct_opposite_switch is False


# ── D. REJOIN clear-front gate (sec 6) ──


class TestRejoinClearFrontGate:
    def test_rejoin_suppressed_when_front_blocked(self):
        path = [[0.0, 2.0], [10.0, 2.0]]  # drone displaced 2 m to the left
        p = TrajectoryPlannerParams()
        open_ = _rejoin_variants(0.0, 0.0, 0.0, path, p, front_clear_m=10.0)
        blocked = _rejoin_variants(0.0, 0.0, 0.0, path, p, front_clear_m=1.0)
        assert len(open_) > 0
        assert blocked == []


# ── E. REJOIN dynamic alignment bonus (sec 6) ──


class TestRejoinDynamicBonus:
    def test_bonus_scales_with_weight(self):
        p0 = TrajectoryPlannerParams(rejoin_alignment_bonus_weight=0.0)
        p2 = TrajectoryPlannerParams(rejoin_alignment_bonus_weight=2.0)
        pl0 = _planner(params=p0)
        pl2 = _planner(params=p2)
        pts = [(i * 0.25, 0.0) for i in range(17)]
        gpath = [(0.0, 0.0), (10.0, 0.0)]
        goal_unit = (1.0, 0.0)
        c0 = pl0.evaluate_candidate(
            REJOIN_SOFT, pts, 0.15, False, (0, 0, 0), 0.0,
            goal_unit, gpath, _df([]),
        )
        c2 = pl2.evaluate_candidate(
            REJOIN_SOFT, pts, 0.15, False, (0, 0, 0), 0.0,
            goal_unit, gpath, _df([]),
        )
        # Only the rejoin bonus differs → Δscore == 2.0 × alignment.
        assert c2.total_score - c0.total_score == pytest.approx(
            2.0 * c0.global_path_alignment
        )


# ── F. invalid-reason histogram (sec 7) ──


class TestInvalidReasonHistogram:
    def test_clearance_reason_counted(self):
        pl = _planner()
        r = pl.plan((0, 0, 0), 0.0, (10.0, 0.0), [[0, 0], [10, 0]],
                    _df([(1.0, 0.0)]))
        assert r.invalid_reason_counts.get("clearance", 0) >= 1


# ── G. family transition log (sec 3) ──


class TestFamilyTransitionLog:
    def test_breakdown_and_direct_opposite_count(self):
        log = FamilyTransitionLog()
        log.record("STRAIGHT", "LEFT", "score_exceeded", 10, 1.0)
        log.record("LEFT", "RIGHT", "score_exceeded", 20, 2.0, direct_opposite=True)
        assert log.breakdown[("STRAIGHT", "LEFT")] == 1
        assert log.breakdown[("LEFT", "RIGHT")] == 1
        assert log.direct_opposite_count == 1
        assert len(log) == 2


# ── H. mission progress monitor (sec 23/24) ──


class TestMissionProgressMonitor:
    def test_stuck_vs_mission_stalled(self):
        mon = MissionProgressMonitor(
            window_s=1.0, min_progress_m=1.0,
            check_interval_s=0.0, stuck_epsilon_m=0.2,
        )
        mon.set_goal((10.0, 0.0))

        # No motion over the window → stuck.
        mon.reset(0.0, (0.0, 0.0))
        st = mon.update(1.0, (0.0, 0.0))
        assert st is not None and st.is_stuck and not st.is_mission_stalled

        # Moving, but not toward the goal → mission-stalled.
        mon.reset(0.0, (0.0, 0.0))
        st2 = mon.update(1.0, (0.0, 2.0))
        assert st2 is not None and st2.is_mission_stalled and not st2.is_stuck


# ── I. obstacle avoidance episode (sec 2) ──


class TestObstacleEpisode:
    def test_episode_open_and_close(self):
        tr = ObstacleAvoidanceEpisodeTracker(
            start_distance_m=3.0, end_distance_m=3.5, hold_frames=3,
        )
        # Approach below start_distance → opens.
        assert tr.update(1, 1.0, 2.5, -1, (0, 0), (5, 0)) is None
        assert tr.active
        # Rise above end_distance for 3 frames → closes (success).
        assert tr.update(2, 1.1, 4.0, -1, (0, 0), (5, 0)) is None
        assert tr.update(3, 1.2, 4.0, -1, (0, 0), (5, 0)) is None
        ep = tr.update(4, 1.3, 4.0, -1, (0, 0), (5, 0))
        assert ep is not None
        assert ep.side == "LEFT"
        assert ep.success
        assert not tr.active


# ── J. single-obstacle behaviour monitor (sec 25) ──


class TestSingleObstacleMonitor:
    def test_fixed_obstacle_is_persistent(self):
        mon = SingleObstacleBehaviorMonitor(min_samples=2)
        for i in range(1, 6):
            mon.update((i * 0.5, 0.0), (5.0, 0.0))
        v = mon.finalize()
        assert v.persistent
        assert v.note == "fixed"

    def test_self_return_tracks_uav(self):
        mon = SingleObstacleBehaviorMonitor(min_samples=2)
        for i in range(1, 6):
            mon.update((i * 0.5, 0.0), (i * 0.5, 0.0))  # centroid follows the UAV
        v = mon.finalize()
        assert not v.persistent
        assert v.note == "tracks_uav"


# ── H. current-LiDAR block → immediate global replan (sec 14) ──


class TestCurrentLidarBlock:
    def test_blocked_global_path_below_hard_clearance(self):
        # The sec-14 replan predicate: a global path whose minimum clearance
        # against the freshly-built distance field (which includes this frame's
        # LiDAR) drops below hard clearance must be treated as blocked.
        df = _df([(1.0, 0.0)])                      # LiDAR obstacle dead ahead
        blocked_path = [(0.25 * i, 0.0) for i in range(9)]   # 0..2 m, through it
        clear_path = [(0.25 * i, 2.0) for i in range(9)]     # 2 m off to the side
        assert df.trajectory_min_clearance(blocked_path) < 1.0
        assert df.trajectory_min_clearance(clear_path) >= 1.0


# ── end-of-flight metrics + CSV trace (sec 1/21/31) ──


class TestFlightMetricsAndTrace:
    def test_metrics_summary_and_log_string(self):
        m = TrajectoryFlightMetrics(goal_xy=(10.0, 0.0))
        m.record_frame((0.0, 0.0), 5.0, 0.25, 0.0)       # first frame → start
        m.record_frame((1.0, 0.0), 4.0, 0.25, 0.0)       # 1 m forward
        m.record_frame((2.0, 0.0), 3.0, 0.25, 0.0)       # 2 m forward
        m.flight_duration_s = 10.0
        m.frames_completed = 3
        m.control_loop_overruns = 1
        m.control_loop_max_overrun_ms = 90.0
        m.control_loop_avg_dt_ms = 55.0
        m.lidar_stale_frames = 1
        m.finalize("mission_complete", True, (2.0, 0.0))

        s = m.summary()
        assert s["success"] is True
        assert s["termination_reason"] == "mission_complete"
        assert s["initial_distance_to_goal_m"] == pytest.approx(10.0)
        assert s["final_distance_to_goal_m"] == pytest.approx(8.0)
        assert s["total_path_length_m"] == pytest.approx(2.0)
        assert s["min_obstacle_clearance_m"] == pytest.approx(3.0)
        assert s["control_loop_overruns"] == 1
        assert "trajectory_flight_summary" in m.to_log_string()

    def test_trace_writer_round_trip(self, tmp_path):
        csv_path = tmp_path / "trace.csv"
        w = FlightTraceWriter(str(csv_path), flush_interval=1)
        w.write_row([1, 0.1, 0.0, 0.0, -3.0, 0.0, 0.25, 0.0, 0.0,
                     0.25, 0.0, "trajectory", "STRAIGHT", 5.0, 0.0, 0.0])
        w.close()

        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2                      # header + one row
        assert lines[0].startswith("frame")
        assert "STRAIGHT" in lines[1]

"""Safety, bounded search, and live tracker integration for adaptive curves."""
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from flight_modes.automatic_mode import _build_trajectory_params
from mapping.distance_field import DistanceField
from mapping.occupancy_grid import OccupancyGridParams
from planners.local_trajectory_planner import (
    LocalTrajectoryPlanner, TrajectoryMemory, TrajectoryPlannerParams,
)
from planners.trajectory_tracker import TrajectoryTracker
from planners.process_workers import LocalTrajectoryPlannerWorker


def params(**overrides):
    return replace(TrajectoryPlannerParams(
        adaptive_candidates_enabled=True, max_compute_ms=200.,
        narrow_passage_enabled=False, narrow_passage_hold_enabled=False,
    ), **overrides)


def plan(planner, obstacles=(), position=(0., 0., -2.), goal=(8., 2.4), yaw=0.):
    return planner.plan(position, yaw, goal, None, DistanceField(list(obstacles)))


def test_generates_intermediate_curvature_and_improves_goal_alignment():
    p = params(adaptive_refinements=0)
    result = plan(LocalTrajectoryPlanner(p))
    assert result.selected is not None
    assert result.selected.generation_source != "fixed"
    assert result.selected.curvature not in (0., -.2, -.45, -.9, .2, .45, .9)
    fixed = [c for c in result.candidates if c.generation_source == "fixed" and c.valid]
    assert result.selected.goal_heading_alignment > max(c.goal_heading_alignment for c in fixed)
    assert result.selected.total_score >= max(c.total_score for c in fixed)


def test_obstacle_gradient_changes_proposals_without_map_access():
    planner = LocalTrajectoryPlanner(params())
    left = plan(planner, [(1.5, -.6)])
    right = plan(LocalTrajectoryPlanner(params()), [(1.5, .6)])
    left_proposals = [c.curvature for c in left.candidates if c.generation_source == "adaptive_obstacle"]
    right_proposals = [c.curvature for c in right.candidates if c.generation_source == "adaptive_obstacle"]
    assert left_proposals and right_proposals
    assert left_proposals[0] > right_proposals[0]


def test_total_candidates_and_fine_search_are_capped():
    p = params(adaptive_max_candidates=11, adaptive_proposals=8, adaptive_refinements=4)
    result = plan(LocalTrajectoryPlanner(p))
    assert result.generated <= 11
    assert result.evaluated_count == len(result.candidates) <= 11
    assert result.refinement_count <= 4


def test_fine_candidates_remain_constant_curvature_arcs():
    result = plan(LocalTrajectoryPlanner(params()))
    fine = [c for c in result.candidates if c.generation_source == "refined"]
    assert fine
    for cand in fine:
        assert abs(cand.curvature) <= .9
        if abs(cand.curvature) > 1e-8:
            radius = abs(1 / cand.curvature)
            assert all(math.hypot(x, y - 1 / cand.curvature) == pytest.approx(radius)
                       for x, y in cand.points)


def test_new_frame_regenerates_curves_and_rechecks_new_obstacles():
    memory = TrajectoryMemory()
    planner = LocalTrajectoryPlanner(params(), memory=memory)
    old = plan(planner)
    assert old.selected is not None
    # No cached curve is accepted on its previous clearance certificate.
    obstacle = old.selected.points[len(old.selected.points) // 2]
    result = plan(planner, [obstacle], position=(.2, 0., -2.))
    df = DistanceField([obstacle])
    for cand in result.candidates:
        assert cand.points[0] == pytest.approx((.2, 0.))
        if cand.valid:
            assert df.trajectory_clearances(cand.points)[0] >= planner.params.hard_clearance_m


class BudgetClock:
    now = 0.

    def __call__(self):
        return self.now


class CostlyField(DistanceField):
    def __init__(self, clock, seconds):
        super().__init__()
        self.clock, self.seconds = clock, seconds

    def trajectory_clearances(self, trajectory, *, check_budget=None):
        self.clock.now += self.seconds
        if check_budget:
            check_budget()
        return math.inf, math.inf


def test_deadline_keeps_only_fully_checked_fixed_fallback():
    clock = BudgetClock()
    planner = LocalTrajectoryPlanner(params(max_compute_ms=10.), compute_clock=clock)
    result = planner.plan((0., 0., -2.), 0., (8., 2.), None, CostlyField(clock, .006))
    assert result.budget_exhausted
    assert result.evaluated_count == 1
    assert result.selected is result.candidates[0]
    assert result.selected.generation_source == "fixed"
    assert result.fallback_reason == "budget_best_checked"
    # Deliberately slow indivisible work may overrun the cooperative deadline.
    assert result.compute_ms == pytest.approx(12.)


def test_no_checked_candidate_at_deadline_never_replays_old_plan():
    clock = BudgetClock()
    memory = TrajectoryMemory(previous_points=[(0., 0.), (4., 0.)], previous_family="STRAIGHT")
    planner = LocalTrajectoryPlanner(params(max_compute_ms=10.), memory=memory, compute_clock=clock)
    result = planner.plan((0., 0., -2.), 0., (8., 0.), None, CostlyField(clock, .02))
    assert result.selected is None
    assert result.evaluated_count == 0
    assert result.command_vx == result.command_vy == 0.
    assert result.fallback_reason == "budget_no_feasible_candidate"


def test_fixed_candidates_remain_available_when_adaptive_work_is_disabled():
    result = plan(LocalTrajectoryPlanner(params(adaptive_proposals=0, adaptive_refinements=0)))
    assert result.selected is not None
    assert result.generated == 9
    assert result.fallback_reason == "fixed_candidates"
    assert all(c.generation_source == "fixed" for c in result.candidates)


def test_optional_seed_failure_preserves_current_fixed_fallback(monkeypatch):
    planner = LocalTrajectoryPlanner(params())
    def failed(*args):
        raise ValueError("failed optional seed")
    monkeypatch.setattr(planner, "_adaptive_curvatures", failed)
    result = plan(planner)
    assert result.selected is not None
    assert result.selected.generation_source == "fixed"
    assert result.fallback_reason == "adaptive_seed_error_fixed"


def test_generator_failure_falls_back_to_checked_fixed_arcs():
    class FailedGenerator:
        def generate(self, *args, **kwargs):
            raise RuntimeError("generator unavailable")
    result = plan(LocalTrajectoryPlanner(params(), generator=FailedGenerator()))
    assert result.selected is not None
    assert result.fallback_reason == "generator_error_fixed"


def test_no_feasible_candidates_does_not_claim_a_fixed_fallback():
    result = plan(LocalTrajectoryPlanner(params()), [(0., 0.)])
    assert result.selected is None
    assert result.fallback_reason == "no_feasible_candidate"


def test_segment_check_catches_thin_obstacle_between_samples():
    planner = LocalTrajectoryPlanner(params(hard_clearance_m=.2))
    df = DistanceField([(.5, 0.)])
    assert df.trajectory_min_clearance([(0., 0.), (1., 0.)]) == .5
    candidate = planner.evaluate_candidate("STRAIGHT", [(0., 0.), (1., 0.)], 0., False,
        (0., 0., -2.), 0., (1., 0.), [], df)
    assert not candidate.valid
    assert candidate.invalid_reason == "clearance"


def test_exact_segment_distance_matches_analytic_geometry():
    df = DistanceField([(.5, .3), (3., 4.)])
    minimum, mean = df.trajectory_clearances([(0., 0.), (1., 0.), (1., 0.)])
    assert minimum == pytest.approx(.3)
    assert mean == pytest.approx(math.hypot(.5, .3))


def test_large_distance_field_matches_scalar_reference_and_updates_cache():
    obstacles = np.random.default_rng(4).normal(size=(100, 2))
    df = DistanceField(obstacles.tolist())
    assert df.distance_at(.2, .1) == pytest.approx(min(math.hypot(x-.2, y-.1) for x, y in obstacles))
    df.set_obstacles([(20., 20.)])
    assert df.distance_at(.2, .1) == pytest.approx(math.hypot(19.8, 19.9))


@pytest.mark.parametrize("reverse,goal", [(False, (10., 0., -2.)), (True, None),
                                        (False, (.5, .4, -2.))])
def test_tracker_respects_planner_cap_including_terminal_blending(reverse, goal):
    tracker = TrajectoryTracker(forward_speed_mps=1., lateral_speed_mps=1.)
    result = tracker.compute_command([(0., 0.), (1., .3), (2., .6)], (0., 0., -2.), 0.,
        is_reverse=reverse, goal_ned=goal, speed_limit_mps=.06)
    assert math.hypot(result.vx, result.vy) <= .06 + 1e-12


@pytest.mark.parametrize("cap", [0., -1., math.nan, math.inf])
def test_invalid_or_zero_speed_cap_stops_translation(cap):
    result = TrajectoryTracker().compute_command([(0., 0.), (2., 0.)], (0., 0., -2.), 0.,
                                                speed_limit_mps=cap)
    assert result.vx == result.vy == 0.


def test_planner_command_respects_clearance_and_turn_speed_caps():
    p = params(adaptive_lateral_accel_mps2=.01, forward_speed_mps=1.)
    result = plan(LocalTrajectoryPlanner(p))
    assert result.speed_limit_mps < 1.
    assert math.hypot(result.command_vx, result.command_vy) <= result.speed_limit_mps + 1e-12


def test_live_yaml_wires_feature_but_old_configs_keep_legacy_behavior():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/trajectory_planner.yaml").read_text(encoding="utf-8"))
    live = _build_trajectory_params(cfg)
    assert live.adaptive_candidates_enabled
    assert live.adaptive_max_candidates == 18
    assert not _build_trajectory_params({}).adaptive_candidates_enabled


def test_real_background_worker_transports_adaptive_result_and_speed_cap():
    worker = LocalTrajectoryPlannerWorker(
        traj_config=asdict(params()), occ_config=asdict(OccupancyGridParams()),
        memory_history_length=10, dfield_radius_m=10., downsample_m=.25)
    try:
        worker.start()
        request_id = worker.request_plan(dict(drone_position_ned=[0., 0., -2.], yaw_rad=0.,
            goal_xy=[8., 2.4], global_path=[], global_path_version=0, lidar_points=[]))
        envelope = None
        deadline = time.monotonic() + 8.
        while time.monotonic() < deadline:
            envelope = worker.poll_latest_result()
            if envelope is not None:
                break
            time.sleep(.01)
        assert envelope is not None
        assert envelope["request_id"] == request_id
        assert "error" not in envelope
        result = envelope["result"]
        assert result.selected is not None
        assert result.selected.generation_source != "fixed"
        assert result.speed_limit_mps is not None
        assert result.evaluated_count <= 18
    finally:
        worker.shutdown()


@pytest.mark.parametrize("overrides", [
    {"adaptive_max_candidates": 100}, {"adaptive_max_candidates": True},
    {"adaptive_proposals": 1.5}, {"adaptive_refinements": -1},
    {"max_compute_ms": math.nan}, {"sample_spacing_m": 0.},
    {"adaptive_curvature_step": math.inf}, {"horizon_m": 0.},
    {"adaptive_candidates_enabled": "false"},
])
def test_invalid_search_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        params(**overrides)

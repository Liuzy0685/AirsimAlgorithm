"""
Local trajectory-centric planner — deterministic receding-horizon baseline.

This module realises the "trajectory-centric" idea from Beyond Waypoints /
DreamFly **without** a learned diffusion generator:

    global reference path
        ↓
    generate K short executable trajectories (deterministic arcs)
        ↓
    evaluate each (whole-trajectory clearance, alignment, smoothness, …)
        ↓
    select one trajectory
        ↓
    execute ONLY the first segment (tracked at high rate by a separate Tracker)
        ↓
    replan on a slower cadence (closed-loop)

CBMBA stays the **global** planner.  This module only decides *how to move
the next few metres*.  It is a pure-computation module: no AirSim RPC, no
velocity dispatch — it returns a selected trajectory and the caller (or a
``TrajectoryTracker``) turns that into a body-frame velocity command.

Trajectory representation
-------------------------
Every candidate is a **continuous path** (a list of world-NED XY samples), not
an endpoint.  Collision checking therefore validates the whole path, not just
the destination — the key problem this layer solves.

Frame & sign contract
---------------------
Body frame is FRD:  +X forward, +Y right, +Z down.

    LEFT   → negative body Y  (curvature < 0)
    RIGHT  → positive body Y  (curvature > 0)

This sign is preserved through every stage (generator → world transform →
command extraction → AirSim ``moveByVelocityBodyFrameAsync``).  The single
conversion is ``planner_to_body_frame`` below; do **not** scatter ad-hoc
``-y`` flips anywhere else.

Candidate families
------------------
STRAIGHT, SOFT_LEFT, LEFT, HARD_LEFT, SOFT_RIGHT, RIGHT, HARD_RIGHT,
REVERSE_LEFT, REVERSE_RIGHT, and (when a global path exists and the lateral
error exceeds a trigger) REJOIN / REJOIN_SOFT / REJOIN_MEDIUM.
Deterministic generation uses constant-curvature arcs (circles), the
simplest dependency-free smooth primitive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ── candidate families ──

STRAIGHT = "STRAIGHT"
SOFT_LEFT = "SOFT_LEFT"
LEFT = "LEFT"
HARD_LEFT = "HARD_LEFT"
SOFT_RIGHT = "SOFT_RIGHT"
RIGHT = "RIGHT"
HARD_RIGHT = "HARD_RIGHT"
REVERSE_LEFT = "REVERSE_LEFT"
REVERSE_RIGHT = "REVERSE_RIGHT"
REJOIN = "REJOIN"
REJOIN_SOFT = "REJOIN_SOFT"
REJOIN_MEDIUM = "REJOIN_MEDIUM"
GOAL_DIRECT = "GOAL_DIRECT"

# Forward families: (family, curvature 1/m).  curvature > 0 → right, < 0 → left.
_FORWARD_FAMILIES: Tuple[Tuple[str, float], ...] = (
    (STRAIGHT, 0.0),
    (SOFT_LEFT, -0.20),
    (LEFT, -0.45),
    (HARD_LEFT, -0.90),
    (SOFT_RIGHT, 0.20),
    (RIGHT, 0.45),
    (HARD_RIGHT, 0.90),
)

# Reverse families: (family, base curvature) — the arc shape mirrors the
# corresponding forward family across the body Y axis (backs up).
_REVERSE_FAMILIES: Tuple[Tuple[str, float], ...] = (
    (REVERSE_LEFT, -0.45),
    (REVERSE_RIGHT, 0.45),
)

# Rejoin variants: (family, curvature magnitude) — curvature sign is chosen
# toward the global path.  These are constant-curvature arcs (smooth, not a
# straight cut), so a rejoin cannot skim along an obstacle face.
_REJOIN_VARIANTS: Tuple[Tuple[str, float], ...] = (
    (REJOIN_SOFT, 0.15),
    (REJOIN_MEDIUM, 0.35),
)

_CURVATURE_MAX = 0.90  # matches the hardest family; used for normalisation

# Lateral sign per family: -1 = left, +1 = right, 0 = neutral.
_FAMILY_SIDE: Dict[str, int] = {
    STRAIGHT: 0,
    SOFT_LEFT: -1, LEFT: -1, HARD_LEFT: -1,
    SOFT_RIGHT: +1, RIGHT: +1, HARD_RIGHT: +1,
    REVERSE_LEFT: -1, REVERSE_RIGHT: +1,
    REJOIN: 0, REJOIN_SOFT: 0, REJOIN_MEDIUM: 0,
    GOAL_DIRECT: 0,
}


def family_side(family: str) -> int:
    """Lateral sign of a family: -1 left, +1 right, 0 neutral."""
    return _FAMILY_SIDE.get(family, 0)


def planner_to_body_frame(curvature: float, is_reverse: bool) -> Tuple[float, float]:
    """Convert a planner curvature to a body-frame (vx, vy) *direction*.

    Body FRD: +X forward, +Y right.  ``curvature`` is signed 1/m (>0 right).
    This is the ONE canonical conversion — the sign contract is:  LEFT →
    negative body Y, RIGHT → positive body Y.
    """
    vx = -1.0 if is_reverse else 1.0
    vy = curvature  # sign directly: >0 → +Y (right), <0 → -Y (left)
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        return (vx, 0.0)
    return (vx / mag, vy / mag)


# ── parameters ──


@dataclass
class TrajectoryPlannerParams:
    """Configuration for the local trajectory planner."""

    enabled: bool = True
    num_candidates: int = 9            # base geometric candidates (7 fwd + 2 rev)
    horizon_m: float = 4.0
    sample_spacing_m: float = 0.25
    planning_hz: float = 8.0
    max_compute_ms: float = 20.0

    hard_clearance_m: float = 1.0
    preferred_clearance_m: float = 1.8

    enable_distance_field_refinement: bool = False
    refinement_gain: float = 0.2

    forward_speed_mps: float = 0.25
    lateral_speed_mps: float = 0.20
    command_lookahead_m: float = 1.0

    alignment_scale_m: float = 3.0          # exp decay scale for whole-path alignment
    consistency_scale_m: float = 0.5        # exp decay scale for geometric consistency
    side_consistency_bonus: float = 0.3     # small bonus for keeping the last side

    # REJOIN gating
    rejoin_trigger_lateral_error_m: float = 0.75
    rejoin_completion_lateral_error_m: float = 0.30
    # Dynamic rejoin bonus (sec 6): reward alignment, but only offer a rejoin
    # when the front is clear for this many metres (never into a wall).
    rejoin_alignment_bonus_weight: float = 1.0
    rejoin_clear_front_required_m: float = 3.0

    # Family-switch hysteresis (sec 4): a *soft* gate, never a hard side lock.
    # A switch away from the current family is suppressed unless (a) the family
    # has been held for ``family_switch_min_hold_time_s`` and (b) the challenger
    # beats the incumbent score by ``family_switch_min_score_improvement``.
    family_switch_min_score_improvement: float = 0.15
    family_switch_min_hold_time_s: float = 0.5
    # Direct LEFT↔RIGHT switch penalty (sec 5) — a direct opposite flip is
    # discouraged (oscillation) but not forbidden.
    direct_opposite_switch_penalty: float = 1.0

    # Adaptive horizon (sec 8): shorten the planning horizon near obstacles.
    adaptive_horizon_enabled: bool = True
    min_horizon_m: float = 2.0
    mid_horizon_m: float = 3.0
    max_horizon_m: float = 4.0
    adaptive_near_threshold_m: float = 3.0
    adaptive_mid_threshold_m: float = 4.0

    # APF safety-filter limits (consumed by automatic_mode, kept here so the
    # trajectory layer owns its own safety budget).
    apf_max_lateral_correction_mps: float = 0.25
    apf_max_speed_reduction_ratio: float = 0.8

    # Tracker (high-rate pure-pursuit) lookahead.
    tracker_lookahead_m: float = 1.0
    trajectory_window_steps: int = 12

    weights: Dict[str, float] = field(default_factory=lambda: {
        "goal_progress": 2.0,
        "global_path_alignment": 3.0,
        "clearance": 3.0,
        "smoothness": 1.0,
        "trajectory_consistency": 1.5,
        "curvature_penalty": 0.5,
        "reverse_penalty": 2.0,
        "unknown_penalty": 0.5,
    })


# ── candidate ──


@dataclass
class TrajectoryCandidate:
    """One evaluated trajectory candidate."""

    family: str = STRAIGHT
    points: List[Tuple[float, float]] = field(default_factory=list)  # world NED XY
    curvature: float = 0.0            # signed curvature (1/m)
    is_reverse: bool = False
    valid: bool = True
    invalid_reason: str = ""

    min_clearance_m: float = float("inf")
    mean_clearance_m: float = float("inf")

    # scoring breakdown (for detailed logs)
    goal_progress: float = 0.0
    global_path_alignment: float = 0.0
    clearance: float = 0.0
    smoothness: float = 0.0
    consistency: float = 0.0
    curvature_penalty: float = 0.0
    reverse_penalty: float = 0.0
    unknown_penalty: float = 0.0
    path_deviation_m: float = float("inf")
    total_score: float = -float("inf")
    direct_opposite_switch: bool = False  # penalised LEFT↔RIGHT flip (sec 5)

    # body-frame command direction of the first segment
    command_vx_body: float = 0.0
    command_vy_body: float = 0.0
    points_ned: List[Tuple[float, float, float]] = field(default_factory=list)
    feedforward_body: List[Tuple[float, float, float]] = field(default_factory=list)


# ── memory ──


@dataclass
class TrajectoryMemory:
    """Limited history so trajectory choice has temporal continuity.

    Continuity is now *geometric* (overlap of the previous trajectory with
    the new one) rather than a coarse same-family flag.  ``previous_points``
    holds the last selected trajectory's world-XY samples.
    """

    previous_family: Optional[str] = None
    previous_score: float = -float("inf")
    previous_global_path_version: int = 0
    previous_points: List[Tuple[float, float]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    history_length: int = 10
    # Monotonic time the current family was first selected (hysteresis, sec 4).
    current_family_held_since: float = -float("inf")

    def record(self, family: str, score: float, path_version: int,
               points: Optional[List[Tuple[float, float]]] = None) -> None:
        self.previous_family = family
        self.previous_score = score
        self.previous_global_path_version = path_version
        self.previous_points = list(points) if points else []
        self.history.append(family)
        if len(self.history) > self.history_length:
            self.history = self.history[-self.history_length:]

    def reset(self) -> None:
        """Forget the last trajectory (used on Recovery exit)."""
        self.previous_family = None
        self.previous_score = -float("inf")
        self.previous_points = []
        self.history = []
        self.current_family_held_since = -float("inf")


# ── plan result ──


@dataclass
class TrajectoryPlanResult:
    """Output of one ``LocalTrajectoryPlanner.plan()`` call."""

    selected: Optional[TrajectoryCandidate] = None
    candidates: List[TrajectoryCandidate] = field(default_factory=list)
    generated: int = 0
    valid_count: int = 0
    valid_ratio: float = 0.0
    best_clearance_m: float = float("inf")
    family_switch: Optional[Tuple[str, str, str]] = None  # (from, to, reason)
    compute_ms: float = 0.0
    command_vx: float = 0.0
    command_vy: float = 0.0
    command_vz: float = 0.0
    # Spatial hint for Recovery when every candidate is infeasible.
    escape_hint: Optional[Dict[str, object]] = None
    # Invalid-reason histogram (sec 7): why candidates were rejected.
    invalid_reason_counts: Dict[str, int] = field(default_factory=dict)
    # The (possibly adaptive) horizon actually used this tick (sec 8).
    horizon_m: float = 0.0
    # Nearest-obstacle distance driving the adaptive horizon.
    min_distance_m: float = float("inf")


# ── generator backend (deterministic now; diffusion later) ──


class TrajectoryGeneratorBackend:
    """Interface for trajectory generation.

    Future learned backends (e.g. a diffusion generator) implement the same
    ``generate`` contract: local map + goal direction + global path + history
    → K candidate trajectories.  The deterministic backend ignores most inputs.
    """

    def generate(self, params: TrajectoryPlannerParams, **ctx) -> List[Tuple[str, List[Tuple[float, float]], float, bool]]:
        raise NotImplementedError


class DeterministicTrajectoryGenerator(TrajectoryGeneratorBackend):
    """Constant-curvature arc generator (world-NED XY samples)."""

    def generate(self, params: TrajectoryPlannerParams, **ctx) -> List[Tuple[str, List[Tuple[float, float]], float, bool]]:
        yaw_rad = ctx["yaw_rad"]
        px = ctx["drone_position_ned"][0]
        py = ctx["drone_position_ned"][1]
        global_path = ctx.get("global_path") or []
        horizon_m = ctx.get("horizon_m", params.horizon_m)
        front_clear_m = ctx.get("front_clear_m", float("inf"))

        out: List[Tuple[str, List[Tuple[float, float]], float, bool]] = []

        # ── forward families ──
        for family, curv in _FORWARD_FAMILIES:
            body_pts = _arc_points(curv, horizon_m, params.sample_spacing_m)
            out.append((family, _body_to_world(body_pts, px, py, yaw_rad), curv, False))

        # ── reverse families (mirror of the corresponding forward arc) ──
        for family, curv in _REVERSE_FAMILIES:
            body_pts = _arc_points(curv, horizon_m, params.sample_spacing_m)
            mirrored = [(-bx, by) for (bx, by) in body_pts]  # back up, same lateral sense
            out.append((family, _body_to_world(mirrored, px, py, yaw_rad), curv, True))

        # ── rejoin variants (only when displaced AND the front is clear) ──
        for family, curv in _rejoin_variants(
            px, py, yaw_rad, global_path, params, front_clear_m,
        ):
            body_pts = _arc_points(curv, horizon_m, params.sample_spacing_m)
            out.append((family, _body_to_world(body_pts, px, py, yaw_rad), curv, False))

        goal_xy = ctx.get("goal_xy")
        if goal_xy is not None:
            goal_pts = _goal_direct_points(
                (px, py),
                (float(goal_xy[0]), float(goal_xy[1])),
                horizon_m,
                params.sample_spacing_m,
            )
            if len(goal_pts) >= 2:
                out.append((GOAL_DIRECT, goal_pts, 0.0, False))

        return out


# ── planner ──


class LocalTrajectoryPlanner:
    """Generate, evaluate, and select a local trajectory.

    Pure computation.  Returns a ``TrajectoryPlanResult``; the selected
    trajectory's ``command_*`` fields give a body-FRD velocity for the
    current tick, and the full ``points`` list is the plan that a separate
    ``TrajectoryTracker`` can follow at higher rate.
    """

    def __init__(
        self,
        params: Optional[TrajectoryPlannerParams] = None,
        memory: Optional[TrajectoryMemory] = None,
        generator: Optional[TrajectoryGeneratorBackend] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.params = params or TrajectoryPlannerParams()
        self._memory = memory or TrajectoryMemory()
        self._generator = generator or DeterministicTrajectoryGenerator()
        self._clock = clock if clock is not None else time.monotonic

    # ── public API ──

    def plan(
        self,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        goal_xy: Tuple[float, float],
        global_path: Optional[List[List[float]]],
        distance_field,
        unknown_query: Optional[Callable[[float, float], bool]] = None,
        global_path_version: int = 0,
        side_hint: int = 0,
        goal_z_ned: Optional[float] = None,
    ) -> TrajectoryPlanResult:
        """Plan one receding-horizon step.

        Args:
            drone_position_ned: ``(x, y, z)`` world NED.
            yaw_rad: drone yaw (0 = North, π/2 = East).
            goal_xy: world-NED XY of the mission goal.
            global_path: CBMBA reference path (list of ``[x, y, z]`` or ``(x, y)``).
            distance_field: ``DistanceField`` over merged map + LiDAR obstacles.
            unknown_query: optional ``(x, y) -> bool`` returning True if UNKNOWN.
            global_path_version: integer that increments when the global path
                changes, used for memory / hysteresis bookkeeping.
            side_hint: optional preferred lateral side (-1/0/+1) — a *soft*
                bonus only, never a hard constraint.
        """
        import time as _time_module
        t0 = _time_module.perf_counter()
        p = self.params
        result = TrajectoryPlanResult()

        # ── adaptive horizon + front clearance (sec 6/8) ──
        min_distance = distance_field.distance_at(
            drone_position_ned[0], drone_position_ned[1],
        )
        horizon_m = self._adaptive_horizon(min_distance)
        front_clear_m = self._front_clearance(drone_position_ned, yaw_rad, distance_field)
        result.horizon_m = horizon_m
        result.min_distance_m = min_distance

        ctx = {
            "drone_position_ned": drone_position_ned,
            "yaw_rad": yaw_rad,
            "goal_xy": goal_xy,
            "goal_z_ned": goal_z_ned,
            "global_path": global_path or [],
            "horizon_m": horizon_m,
            "front_clear_m": front_clear_m,
        }
        raw = self._generator.generate(p, **ctx)
        result.generated = len(raw)

        path_xy = _path_to_xy(global_path)
        goal_unit = _unit_xy((goal_xy[0] - drone_position_ned[0], goal_xy[1] - drone_position_ned[1]))
        prev_points = self._memory.previous_points

        # Resolve the side hint: explicit hint wins, else the last family's side.
        if side_hint == 0 and self._memory.previous_family:
            side_hint = family_side(self._memory.previous_family)

        candidates: List[TrajectoryCandidate] = []
        for family, points, curv, is_reverse in raw:
            cand = self.evaluate_candidate(
                family, points, curv, is_reverse,
                drone_position_ned, yaw_rad, goal_unit, path_xy, distance_field,
                unknown_query, prev_points, side_hint, goal_z_ned,
            )
            candidates.append(cand)

        # ── distance-field refinement (optional, default off) ──
        if p.enable_distance_field_refinement:
            for cand in candidates:
                if cand.valid:
                    self._refine_candidate(cand, distance_field)

        # ── direct opposite-switch penalty (sec 5): discourage LEFT↔RIGHT flips ──
        prev = self._memory.previous_family
        prev_side = family_side(prev) if prev else 0
        if prev_side != 0:
            for cand in candidates:
                cs = family_side(cand.family)
                if cs != 0 and cs == -prev_side:
                    cand.total_score -= p.direct_opposite_switch_penalty
                    cand.direct_opposite_switch = True

        # ── validity stats + invalid-reason histogram (sec 7) ──
        valid_cands = [c for c in candidates if c.valid]
        result.candidates = candidates
        result.valid_count = len(valid_cands)
        result.valid_ratio = len(valid_cands) / len(candidates) if candidates else 0.0
        result.best_clearance_m = max((c.min_clearance_m for c in valid_cands), default=float("inf"))
        result.invalid_reason_counts = _histogram(
            c.invalid_reason for c in candidates if not c.valid
        )

        if not valid_cands:
            result.escape_hint = self._build_escape_hint(candidates)
            result.compute_ms = (_time_module.perf_counter() - t0) * 1000.0
            return result  # no feasible trajectory → dispatcher falls back

        best = max(valid_cands, key=lambda c: c.total_score)

        # ── family-switch hysteresis (sec 4) — a *soft* gate, never a hard lock ──
        now = self._clock()
        switch_reason = "score_exceeded"
        if prev is not None and prev != best.family:
            held_for = now - self._memory.current_family_held_since
            if held_for < p.family_switch_min_hold_time_s:
                switch_reason = "hold_time"
            elif best.total_score - self._memory.previous_score < p.family_switch_min_score_improvement:
                switch_reason = "insufficient_improvement"
            if switch_reason != "score_exceeded":
                incumbent = _candidate_by_family(valid_cands, prev)
                if incumbent is not None:
                    best = incumbent

        # ── command derivation (execute first segment) ──
        cmd_vx, cmd_vy = self._command_from_candidate(best)
        result.selected = best
        result.command_vx = cmd_vx
        result.command_vy = cmd_vy
        result.command_vz = 0.0

        # ── family switch detection (for logs) ──
        if prev is not None and prev != best.family:
            result.family_switch = (prev, best.family, switch_reason)
            self._memory.current_family_held_since = now
        elif self._memory.current_family_held_since == -float("inf"):
            self._memory.current_family_held_since = now
        self._memory.record(best.family, best.total_score, global_path_version, best.points)

        result.compute_ms = (_time_module.perf_counter() - t0) * 1000.0
        return result

    # ── evaluation ──

    def evaluate_candidate(
        self,
        family: str,
        points: List[Tuple[float, float]],
        curvature: float,
        is_reverse: bool,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        goal_unit: Tuple[float, float],
        global_path_xy: List[Tuple[float, float]],
        distance_field,
        unknown_query: Optional[Callable[[float, float], bool]] = None,
        previous_points: Optional[List[Tuple[float, float]]] = None,
        side_hint: int = 0,
        goal_z_ned: Optional[float] = None,
    ) -> TrajectoryCandidate:
        """Score one candidate.  Returns a ``TrajectoryCandidate`` (possibly invalid)."""
        p = self.params
        cand = TrajectoryCandidate(
            family=family,
            points=points,
            curvature=curvature,
            is_reverse=is_reverse,
        )
        if not points or len(points) < 2:
            cand.valid = False
            cand.invalid_reason = "empty_trajectory"
            return cand

        # ── whole-trajectory clearance ──
        cand.min_clearance_m = distance_field.trajectory_min_clearance(points)
        cand.mean_clearance_m = distance_field.trajectory_mean_clearance(points)

        if cand.min_clearance_m < p.hard_clearance_m:
            cand.valid = False
            cand.invalid_reason = "clearance"
            return cand

        px0, py0 = drone_position_ned[0], drone_position_ned[1]

        # ── goal progress: endpoint + mean forward-arc (not endpoint only) ──
        endpoint = points[-1]
        endpoint_progress = (endpoint[0] - px0) * goal_unit[0] + (endpoint[1] - py0) * goal_unit[1]
        arc_progress = sum((pt[0] - px0) * goal_unit[0] + (pt[1] - py0) * goal_unit[1] for pt in points) / len(points)
        cand.goal_progress = _clamp01(0.5 * (endpoint_progress + arc_progress) / p.horizon_m)

        # ── whole-path alignment (mean error over ALL points, exp decay) ──
        if global_path_xy:
            mean_err = sum(_point_to_path_distance_xy(pt, global_path_xy) for pt in points) / len(points)
            cand.path_deviation_m = mean_err
            cand.global_path_alignment = math.exp(-mean_err / p.alignment_scale_m)
        else:
            cand.path_deviation_m = float("inf")
            cand.global_path_alignment = 0.0

        cand.clearance = _clamp01(cand.min_clearance_m / p.preferred_clearance_m)
        cand.smoothness = _clamp01(1.0 - abs(curvature) / _CURVATURE_MAX)
        cand.consistency = self._geometric_consistency(points, previous_points, distance_field)

        cand.curvature_penalty = _clamp01(abs(curvature) / _CURVATURE_MAX)
        cand.reverse_penalty = 1.0 if is_reverse else 0.0

        if unknown_query is not None:
            unknown_count = sum(1 for (qx, qy) in points if unknown_query(qx, qy))
            cand.unknown_penalty = unknown_count / len(points)
        else:
            cand.unknown_penalty = 0.0

        w = p.weights
        cand.total_score = (
            w.get("goal_progress", 0.0) * cand.goal_progress
            + w.get("global_path_alignment", 0.0) * cand.global_path_alignment
            + w.get("clearance", 0.0) * cand.clearance
            + w.get("smoothness", 0.0) * cand.smoothness
            + w.get("trajectory_consistency", 0.0) * cand.consistency
            - w.get("curvature_penalty", 0.0) * cand.curvature_penalty
            - w.get("reverse_penalty", 0.0) * cand.reverse_penalty
            - w.get("unknown_penalty", 0.0) * cand.unknown_penalty
        )

        # ── soft side-consistency bonus (never a hard sign override) ──
        if side_hint != 0 and family_side(family) == side_hint:
            cand.total_score += p.side_consistency_bonus

        # ── dynamic rejoin bonus (sec 6): reward alignment when rejoining ──
        if family in (REJOIN, REJOIN_SOFT, REJOIN_MEDIUM):
            cand.total_score += p.rejoin_alignment_bonus_weight * cand.global_path_alignment

        # ── command direction (body frame, first segment) ──
        cand.command_vx_body, cand.command_vy_body = self._command_direction(
            points, curvature, is_reverse, drone_position_ned, yaw_rad, family,
        )
        cand.points_ned = _build_3d_window(
            points,
            drone_position_ned[2],
            drone_position_ned[2] if goal_z_ned is None else goal_z_ned,
            p.trajectory_window_steps,
        )
        cand.feedforward_body = _build_feedforward_body(
            cand.points_ned,
            yaw_rad,
            max(0.05, p.sample_spacing_m / max(0.05, p.forward_speed_mps)),
            p.forward_speed_mps,
            p.lateral_speed_mps,
        )
        return cand

    def _geometric_consistency(
        self,
        points: List[Tuple[float, float]],
        previous_points: Optional[List[Tuple[float, float]]],
        distance_field,
    ) -> float:
        """Overlap consistency: how similar the first ~2 m are to the previous plan.

        Returns 0 if there is no previous trajectory, or if the previous
        trajectory is now blocked (its clearance fell below hard clearance) —
        a stale plan must not bias the drone back into a wall.
        """
        p = self.params
        if not previous_points or len(previous_points) < 2:
            return 0.0
        # Previous plan blocked by a new obstacle → no consistency bonus.
        if distance_field.trajectory_min_clearance(previous_points) < p.hard_clearance_m:
            return 0.0
        n = min(len(points), len(previous_points))
        mean_dist = sum(
            math.hypot(points[i][0] - previous_points[i][0], points[i][1] - previous_points[i][1])
            for i in range(n)
        ) / n
        return math.exp(-mean_dist / p.consistency_scale_m)

    def _build_escape_hint(self, candidates: List[TrajectoryCandidate]) -> Dict[str, object]:
        """Best infeasible direction (for Recovery)."""
        best = max(
            candidates,
            key=lambda c: c.min_clearance_m if math.isfinite(c.min_clearance_m) else -1e9,
            default=None,
        )
        if best is None:
            return {}
        side = family_side(best.family)
        return {
            "side": side,
            "side_label": { -1: "LEFT", 1: "RIGHT", 0: "NONE" }[side],
            "clearance_m": best.min_clearance_m,
            "family": best.family,
        }

    # ── helpers ──

    def _command_direction(
        self,
        points: List[Tuple[float, float]],
        curvature: float,
        is_reverse: bool,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        family: str = "",
    ) -> Tuple[float, float]:
        """Body-frame unit direction of the first segment (vx, vy).

        Uses the canonical ``planner_to_body_frame`` conversion so the sign
        contract (LEFT → negative body Y) is enforced in one place.
        """
        if family == GOAL_DIRECT and len(points) >= 2:
            dx = points[1][0] - drone_position_ned[0]
            dy = points[1][1] - drone_position_ned[1]
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)
            vx = dx * cos_y + dy * sin_y
            vy = -dx * sin_y + dy * cos_y
            mag = math.hypot(vx, vy)
            if mag > 1e-9:
                return (vx / mag, vy / mag)
        return planner_to_body_frame(curvature * self.params.command_lookahead_m, is_reverse)

    def _command_from_candidate(self, cand: TrajectoryCandidate) -> Tuple[float, float]:
        p = self.params
        vx = cand.command_vx_body * p.forward_speed_mps
        vy = cand.command_vy_body * p.forward_speed_mps
        # Clamp lateral velocity to the configured lateral speed.
        vy = max(-p.lateral_speed_mps, min(p.lateral_speed_mps, vy))
        return vx, vy

    def _refine_candidate(self, cand: TrajectoryCandidate, distance_field) -> None:
        """Optional distance-gradient refinement (pushes points away from obstacles).

        Only nudge points that are below preferred clearance, then re-check
        clearance.  If the refined path still violates hard clearance, mark invalid.
        """
        p = self.params
        refined: List[Tuple[float, float]] = []
        for (x, y) in cand.points:
            d = distance_field.distance_at(x, y)
            if d < p.preferred_clearance_m:
                gx, gy = distance_field.gradient_at(x, y)
                push = (p.preferred_clearance_m - d) * p.refinement_gain
                x = x + gx * push
                y = y + gy * push
            refined.append((x, y))
        cand.points = refined
        cand.min_clearance_m = distance_field.trajectory_min_clearance(refined)
        cand.mean_clearance_m = distance_field.trajectory_mean_clearance(refined)
        if cand.min_clearance_m < p.hard_clearance_m:
            cand.valid = False
            cand.invalid_reason = "clearance_after_refinement"

    def _adaptive_horizon(self, min_distance_m: float) -> float:
        """Shrink the planning horizon near obstacles (sec 8).

        Hard clearance is unchanged — only the look-ahead length shortens, so a
        close obstacle yields tighter, more conservative trajectories.
        """
        p = self.params
        if not p.adaptive_horizon_enabled:
            return p.horizon_m
        if math.isinf(min_distance_m):
            return p.max_horizon_m
        if min_distance_m < p.adaptive_near_threshold_m:
            return p.min_horizon_m
        if min_distance_m < p.adaptive_mid_threshold_m:
            return p.mid_horizon_m
        return p.max_horizon_m

    def _front_clearance(
        self,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        distance_field,
    ) -> float:
        """Min clearance along a forward ray of ``rejoin_clear_front_required_m``.

        Used to gate rejoin candidates (sec 6) so a rejoin is never offered
        into the face of an obstacle.
        """
        p = self.params
        px = drone_position_ned[0]
        py = drone_position_ned[1]
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        best = float("inf")
        d = 0.0
        step = max(0.25, p.sample_spacing_m)
        while d <= p.rejoin_clear_front_required_m + 1e-9:
            best = min(best, distance_field.distance_at(
                px + cos_y * d, py + sin_y * d,
            ))
            d += step
        return best


# ── geometry helpers ──


def _histogram(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def _candidate_by_family(
    candidates: List[TrajectoryCandidate], family: str,
) -> Optional[TrajectoryCandidate]:
    for c in candidates:
        if c.family == family:
            return c
    return None


def _arc_points(curvature: float, horizon_m: float, spacing_m: float) -> List[Tuple[float, float]]:
    """Constant-curvature arc in body frame (forward +X, right +Y).

    curvature > 0 turns right (+Y), curvature < 0 turns left (-Y).
    Returns points from s=0 to horizon inclusive at ``spacing_m`` steps.
    """
    points: List[Tuple[float, float]] = []
    s = 0.0
    while s <= horizon_m + 1e-9:
        if abs(curvature) < 1e-9:
            x, y = s, 0.0
        else:
            x = math.sin(curvature * s) / curvature
            y = (1.0 - math.cos(curvature * s)) / curvature
        points.append((x, y))
        s += spacing_m
    return points


def _goal_direct_points(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    horizon_m: float,
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """Short straight trajectory window from the current position to the goal."""
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return [start_xy]
    travel = min(dist, max(0.05, horizon_m))
    step = max(0.05, spacing_m)
    n = max(1, int(math.ceil(travel / step)))
    ux = dx / dist
    uy = dy / dist
    return [
        (start_xy[0] + ux * travel * i / n, start_xy[1] + uy * travel * i / n)
        for i in range(n + 1)
    ]


def _body_to_world(
    body_pts: Iterable[Tuple[float, float]],
    px: float, py: float, yaw_rad: float,
) -> List[Tuple[float, float]]:
    """Body (forward, right) → world NED XY (forward=(cos,sin), right=(-sin,cos))."""
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    return [
        (px + bx * cos_y - by * sin_y, py + bx * sin_y + by * cos_y)
        for (bx, by) in body_pts
    ]


def _rejoin_variants(
    px: float, py: float, yaw_rad: float,
    global_path: List[List[float]], params: TrajectoryPlannerParams,
    front_clear_m: float = float("inf"),
) -> List[Tuple[str, float]]:
    """Return (family, curvature) rejoin variants, or [] when not needed.

    Rejoin candidates are generated only when the drone is laterally displaced
    from the global path by more than ``rejoin_trigger_lateral_error_m`` AND
    the front is clear for at least ``rejoin_clear_front_required_m`` (sec 6) —
    a rejoin must never steer the drone into the face of an obstacle.
    Each variant is a constant-curvature arc curving toward the path (never a
    straight cut that could skim an obstacle face).
    """
    if front_clear_m < params.rejoin_clear_front_required_m:
        return []
    path_xy = _path_to_xy(global_path)
    if len(path_xy) < 2:
        return []

    _, nearest_pt, _ = _nearest_on_path((px, py), path_xy)
    ox = nearest_pt[0] - px
    oy = nearest_pt[1] - py
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    # Body-frame lateral error: + = path to the right, - = path to the left.
    lateral_error = -ox * sin_y + oy * cos_y

    if abs(lateral_error) < params.rejoin_trigger_lateral_error_m:
        return []

    curv_sign = 1.0 if lateral_error >= 0 else -1.0
    return [(family, curv_sign * mag) for (family, mag) in _REJOIN_VARIANTS]


def _path_to_xy(path: Optional[List[List[float]]]) -> List[Tuple[float, float]]:
    if not path:
        return []
    out: List[Tuple[float, float]] = []
    for wp in path:
        if wp is None or len(wp) < 2:
            continue
        out.append((float(wp[0]), float(wp[1])))
    return out


def _point_to_path_distance_xy(point: Tuple[float, float], path_xy: List[Tuple[float, float]]) -> float:
    if not path_xy:
        return float("inf")
    best = float("inf")
    for i in range(len(path_xy) - 1):
        d = _point_segment_dist(point, path_xy[i], path_xy[i + 1])
        if d < best:
            best = d
    if len(path_xy) == 1:
        best = math.hypot(point[0] - path_xy[0][0], point[1] - path_xy[0][1])
    return best


def _point_segment_dist(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_len_sq))
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


def _nearest_on_path(
    point: Tuple[float, float], path_xy: List[Tuple[float, float]],
) -> Tuple[int, Tuple[float, float], float]:
    best_idx = 0
    best_pt = path_xy[0]
    best_d = math.hypot(point[0] - path_xy[0][0], point[1] - path_xy[0][1])
    for i in range(len(path_xy) - 1):
        d = _point_segment_dist(point, path_xy[i], path_xy[i + 1])
        a, b = path_xy[i], path_xy[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            proj = a
        else:
            t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / seg_len_sq))
            proj = (a[0] + t * dx, a[1] + t * dy)
        if d < best_d:
            best_d = d
            best_pt = proj
            best_idx = i
    return best_idx, best_pt, best_d


def _path_lookahead(
    path_xy: List[Tuple[float, float]], start_idx: int, distance_m: float,
) -> Optional[Tuple[float, float]]:
    """Walk forward along path from start_idx until distance_m is accumulated."""
    if distance_m <= 0:
        return path_xy[start_idx]
    acc = 0.0
    for i in range(start_idx, len(path_xy) - 1):
        seg_len = math.hypot(
            path_xy[i + 1][0] - path_xy[i][0],
            path_xy[i + 1][1] - path_xy[i][1],
        )
        if acc + seg_len >= distance_m:
            remain = distance_m - acc
            t = remain / seg_len if seg_len > 1e-9 else 0.0
            return (
                path_xy[i][0] + t * (path_xy[i + 1][0] - path_xy[i][0]),
                path_xy[i][1] + t * (path_xy[i + 1][1] - path_xy[i][1]),
            )
        acc += seg_len
    return path_xy[-1]


def _unit_xy(v: Tuple[float, float]) -> Tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-9:
        return (1.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _build_3d_window(
    points_xy: List[Tuple[float, float]],
    current_z: float,
    target_z: float,
    max_steps: int,
) -> List[Tuple[float, float, float]]:
    """Attach a smooth Z profile to the first N XY trajectory samples."""
    if not points_xy:
        return []
    n = min(len(points_xy), max(1, int(max_steps)))
    out: List[Tuple[float, float, float]] = []
    denom = max(1, n - 1)
    for i, (x, y) in enumerate(points_xy[:n]):
        u = i / denom
        s = u * u * (3.0 - 2.0 * u)
        z = current_z + (target_z - current_z) * s
        out.append((x, y, z))
    return out


def _build_feedforward_body(
    points_ned: List[Tuple[float, float, float]],
    yaw_rad: float,
    dt_s: float,
    max_forward_mps: float,
    max_lateral_mps: float,
) -> List[Tuple[float, float, float]]:
    """Finite-difference velocity feed-forward for the 3D trajectory window."""
    if not points_ned:
        return []
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    out: List[Tuple[float, float, float]] = []
    for i, pt in enumerate(points_ned):
        nxt = points_ned[min(i + 1, len(points_ned) - 1)]
        vx_w = (nxt[0] - pt[0]) / dt_s
        vy_w = (nxt[1] - pt[1]) / dt_s
        vz = (nxt[2] - pt[2]) / dt_s
        vx_b = vx_w * cos_y + vy_w * sin_y
        vy_b = -vx_w * sin_y + vy_w * cos_y
        vx_b = max(-max_forward_mps, min(max_forward_mps, vx_b))
        vy_b = max(-max_lateral_mps, min(max_lateral_mps, vy_b))
        out.append((vx_b, vy_b, vz))
    return out

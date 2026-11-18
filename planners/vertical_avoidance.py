"""
Vertical avoidance — climb-over vs. go-around selector for trajectory mode.

Pure calculation module.  Does NOT call any AirSim API.

The 2D local trajectory planner (``local_trajectory_planner.py``) can only
steer laterally (STRAIGHT / LEFT / RIGHT / …), so a low building directly ahead
is always bypassed sideways even when the space above it is wide open.  This
module adds the missing vertical axis: given the 16-sector LiDAR distances and
the current / target altitude it decides, each tick, whether to keep the
default lateral avoidance or to climb over the obstacle.

Policy (product decision):
    * default = lateral (go around);
    * climb only when the obstacle ahead is **low** (``frontUp`` is open), there
      is headroom above, the climb ceiling is not reached, AND both sides are
      narrow enough that going around is clearly more expensive.

Coordinate convention: NED Z is positive downward, so climbing means Z
decreases.  ``target_z`` is the cruise altitude (e.g. -1.0 m); a climb target
is always ``target_z - max_climb_m`` (further negative = higher).

Missing / NaN vertical sectors are treated as zero clearance so a climb can
never be authorised by incomplete sensing (same fail-closed rule as
``recovery_commander._upward_clearance``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


# Sectors used for upward clearance (same set as recovery_commander).
_UPWARD_SECTORS = ("up", "frontUp", "leftUp", "rightUp")


@dataclass
class VerticalAvoidanceParams:
    """Configurable vertical-avoidance thresholds (all metres unless noted)."""

    enabled: bool = True
    front_blocked_m: float = 2.5      # front < this → obstacle directly ahead
    front_clear_m: float = 4.0        # front >= this → obstacle passed (exit)
    low_obstacle_front_up_m: float = 3.0  # frontUp >= this → obstacle is low enough to climb over
    up_clearance_m: float = 2.5       # min(up, frontUp, leftUp, rightUp) required to climb
    max_climb_m: float = 2.0          # max altitude gain above the cruise target
    climb_step_m: float = 0.6         # per-tick incremental climb-target step
    narrow_side_m: float = 2.0        # min(left,right) < this → lateral is "expensive"
    ceiling_margin_m: float = 0.3     # stop this far short of the max-climb ceiling
    hysteresis_s: float = 0.5         # min active time before an "obstacle passed" exit

    # Persistent escape (state machine) parameters.  A climb is NOT allowed to
    # exit on a single LiDAR-clear frame: the front must stay clear for
    # ``clear_frames_required`` consecutive frames AND the drone must have
    # climbed at least ``minimum_escape_height_m``.  After that it holds its
    # altitude and flies forward ``forward_clearance_distance_m`` before
    # descending back to the route (bounded by ``forward_clearance_max_duration_s``).
    minimum_escape_height_m: float = 0.8
    clear_frames_required: int = 5
    forward_clearance_distance_m: float = 1.5
    forward_clearance_speed_mps: float = 0.25
    forward_clearance_max_duration_s: float = 8.0


@dataclass
class VerticalAvoidanceDecision:
    """Output of one ``select()`` call (read-only diagnostics + action)."""

    action: str = "lateral"            # "lateral" | "climb"
    target_z: Optional[float] = None   # temporary climb target (NED), only when climb
    reason: str = ""

    front_m: float = float("inf")
    front_up_m: float = float("inf")
    up_clearance_m: float = 0.0
    min_side_m: float = float("inf")
    climb_cap_m: float = float("inf")  # most-negative allowed Z (climb ceiling)


def safe_ray(value) -> float:
    """Coerce None / non-numeric / non-finite LiDAR values to 0.0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def upward_clearance(rays: Dict[str, float]) -> float:
    """Most restrictive distance in the upward-facing sectors (metres).

    Missing sectors are treated as zero clearance (see module docstring).
    """
    return min(safe_ray(rays.get(name)) for name in _UPWARD_SECTORS)


class VerticalAvoidanceSelector:
    """Decide whether to climb over a low obstacle or keep lateral avoidance."""

    def __init__(self, params: Optional[VerticalAvoidanceParams] = None):
        self._params = params or VerticalAvoidanceParams()

    def select(
        self,
        rays: Dict[str, float],
        current_z: float,
        target_z: float,
    ) -> VerticalAvoidanceDecision:
        """Return the vertical-avoidance decision for one frame.

        Args:
            rays: 16-sector legacy distances (keys include front, frontUp,
                up, leftUp, rightUp, left, right).
            current_z: current drone altitude (NED, positive down).
            target_z: cruise altitude (NED).

        Returns:
            VerticalAvoidanceDecision.  ``target_z`` is set only when
            ``action == "climb"``.
        """
        p = self._params

        front = safe_ray(rays.get("front"))
        front_up = safe_ray(rays.get("frontUp"))
        up_clear = upward_clearance(rays)
        left = safe_ray(rays.get("left"))
        right = safe_ray(rays.get("right"))
        min_side = min(left, right)

        climb_cap = float(target_z) - p.max_climb_m

        def _decision(action: str, reason: str) -> VerticalAvoidanceDecision:
            return VerticalAvoidanceDecision(
                action=action,
                reason=reason,
                front_m=front,
                front_up_m=front_up,
                up_clearance_m=up_clear,
                min_side_m=min_side,
                climb_cap_m=climb_cap,
            )

        if not p.enabled:
            return _decision("lateral", "disabled")

        # Guard: altitude inputs must be finite before any Z comparison.
        if not math.isfinite(current_z) or not math.isfinite(target_z):
            return _decision("lateral", "invalid_altitude")

        # 1. No obstacle directly ahead.
        if front >= p.front_blocked_m:
            return _decision("lateral", "front_clear")

        # 2. Obstacle is too tall to climb over (no open space above its face).
        if front_up < p.low_obstacle_front_up_m:
            return _decision("lateral", "obstacle_tall")

        # 3. Insufficient headroom above.
        if up_clear < p.up_clearance_m:
            return _decision("lateral", "up_clearance_insufficient")

        # 4. Already at/above the climb ceiling.
        if current_z <= climb_cap + p.ceiling_margin_m:
            return _decision("lateral", "ceiling_reached")

        # 5. Sides are wide enough → lateral is the convenient default.
        if min_side >= p.narrow_side_m:
            return _decision("lateral", "sides_open")

        # 6. Climb over: incremental step toward the bounded ceiling.
        target = max(climb_cap, float(current_z) - p.climb_step_m)
        d = _decision("climb", "climb_over")
        d.target_z = target
        return d

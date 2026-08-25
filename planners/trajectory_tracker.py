"""
High-rate trajectory tracker (pure pursuit).

The planner produces a full ``horizon_m`` trajectory on a slow cadence
(5–10 Hz).  The tracker follows that cached trajectory on every control tick
(20–30 Hz) by re-deriving a body-frame velocity from the current pose — it
does **not** simply replay a cached velocity command.

Body frame is FRD: +X forward, +Y right, +Z down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TrackerResult:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    lookahead_point: Optional[Tuple[float, float]] = None
    curvature: float = 0.0
    lateral_error_m: float = 0.0


class TrajectoryTracker:
    """Pure-pursuit tracker over a cached world-NED XY trajectory."""

    def __init__(self, lookahead_m: float = 1.0, sample_spacing_m: float = 0.25,
                 forward_speed_mps: float = 0.25, lateral_speed_mps: float = 0.20,
                 command_lookahead_m: float = 1.0) -> None:
        self.lookahead_m = lookahead_m
        self.sample_spacing_m = sample_spacing_m
        self.forward_speed_mps = forward_speed_mps
        self.lateral_speed_mps = lateral_speed_mps
        self.command_lookahead_m = command_lookahead_m

    def compute_command(
        self,
        trajectory_points: List[Tuple[float, float]],
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
    ) -> TrackerResult:
        """Compute a body-frame velocity to track the cached trajectory.

        Pure pursuit: pick the trajectory sample ~``lookahead_m`` ahead,
        measure its body-frame lateral offset, and turn toward it.
        """
        res = TrackerResult()
        if not trajectory_points:
            return res

        lp = self._lookahead_point(trajectory_points)
        res.lookahead_point = lp

        dx = lp[0] - drone_position_ned[0]
        dy = lp[1] - drone_position_ned[1]
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # body-frame offset of the lookahead point
        fwd = dx * cos_y + dy * sin_y
        lat = -dx * sin_y + dy * cos_y
        res.lateral_error_m = lat

        L = max(0.1, self.lookahead_m)
        # Standard pure-pursuit curvature: kappa = 2 * lateral / L^2.
        curvature = 2.0 * lat / (L * L)
        res.curvature = curvature

        res.vx = self.forward_speed_mps
        res.vy = curvature * self.command_lookahead_m * self.forward_speed_mps
        res.vy = max(-self.lateral_speed_mps, min(self.lateral_speed_mps, res.vy))
        res.vz = 0.0
        return res

    def _lookahead_point(self, trajectory_points: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not trajectory_points:
            return (0.0, 0.0)
        idx = min(
            len(trajectory_points) - 1,
            int(round(self.lookahead_m / max(0.05, self.sample_spacing_m))),
        )
        return trajectory_points[idx]

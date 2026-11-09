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
    yaw_rate_radps: float = 0.0


class TrajectoryTracker:
    """Pure-pursuit tracker over a cached world-NED XY trajectory."""

    def __init__(self, lookahead_m: float = 1.0, sample_spacing_m: float = 0.25,
                 forward_speed_mps: float = 0.25, lateral_speed_mps: float = 0.20,
                 command_lookahead_m: float = 1.0,
                 yaw_gain: float = 1.4,
                 max_yaw_rate_radps: float = 0.5,
                 goal_blend_distance_m: float = 4.0,
                 goal_direct_distance_m: float = 2.0,
                 goal_slowdown_distance_m: float = 4.0,
                 terminal_goal_approach_radius_m: float = 0.0,
                 terminal_slowdown_radius_m: float = 0.0,
                 terminal_goal_kp: float = 0.5,
                 terminal_goal_max_speed_mps: Optional[float] = None,
                 terminal_braking_accel_mps2: float = 0.35,
                 terminal_capture_radius_m: float = 0.02) -> None:
        self.lookahead_m = lookahead_m
        self.sample_spacing_m = sample_spacing_m
        self.forward_speed_mps = forward_speed_mps
        self.lateral_speed_mps = lateral_speed_mps
        self.command_lookahead_m = command_lookahead_m
        self.yaw_gain = max(0.0, float(yaw_gain))
        self.max_yaw_rate_radps = max(0.0, float(max_yaw_rate_radps))
        self.goal_blend_distance_m = max(0.0, float(goal_blend_distance_m))
        self.goal_direct_distance_m = max(0.0, float(goal_direct_distance_m))
        self.goal_slowdown_distance_m = max(0.1, float(goal_slowdown_distance_m))
        self.terminal_goal_approach_radius_m = max(
            0.0, float(terminal_goal_approach_radius_m),
        )
        self.terminal_slowdown_radius_m = max(
            0.0, float(terminal_slowdown_radius_m),
        )
        self.terminal_goal_kp = max(0.0, float(terminal_goal_kp))
        self.terminal_goal_max_speed_mps = (
            max(0.0, float(terminal_goal_max_speed_mps))
            if terminal_goal_max_speed_mps is not None
            else abs(self.forward_speed_mps)
        )
        self.terminal_braking_accel_mps2 = max(
            0.01, float(terminal_braking_accel_mps2),
        )
        self.terminal_capture_radius_m = max(
            0.0, float(terminal_capture_radius_m),
        )

    def compute_command(
        self,
        trajectory_points: List[Tuple[float, float]],
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        is_reverse: bool = False,
        goal_xy: Optional[Tuple[float, float]] = None,
        goal_ned: Optional[Tuple[float, float, float]] = None,
    ) -> TrackerResult:
        """Compute a body-frame velocity to track the cached trajectory.

        Pure pursuit: pick the trajectory sample ~``lookahead_m`` ahead,
        measure its body-frame lateral offset, and turn toward it.
        """
        res = TrackerResult()
        if not trajectory_points:
            return res

        # A cached trajectory is generated from an older pose.  Find the
        # nearest point first, then look ahead from there; a fixed list index
        # eventually falls behind the drone and causes the old "fly past the
        # goal, then turn back" behavior.
        nearest_idx = self._nearest_index(trajectory_points, drone_position_ned)
        lp = self._lookahead_point(
            trajectory_points, drone_position_ned, nearest_idx=nearest_idx,
        )

        # Near MissionEnd, progressively bias the lookahead toward the live
        # goal.  This makes arrival independent of the remaining length of a
        # stale 4 m arc and avoids following a blue trajectory beyond the goal.
        # ``goal_xy`` remains supported for existing callers; goal_ned is the
        # richer endpoint interface used by automatic mode.
        if goal_xy is None and goal_ned is not None:
            goal_xy = (float(goal_ned[0]), float(goal_ned[1]))

        goal_distance = float("inf")
        if goal_xy is not None:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            goal_distance = math.hypot(
                gx - drone_position_ned[0], gy - drone_position_ned[1],
            )
            if goal_distance <= self.goal_direct_distance_m:
                lp = (gx, gy)
            elif goal_distance < self.goal_blend_distance_m:
                span = max(1e-6, self.goal_blend_distance_m - self.goal_direct_distance_m)
                blend = (self.goal_blend_distance_m - goal_distance) / span
                blend = max(0.0, min(1.0, blend))
                lp = (
                    lp[0] * (1.0 - blend) + gx * blend,
                    lp[1] * (1.0 - blend) + gy * blend,
                )
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

        # The planner may deliberately select a reverse family.  The previous
        # tracker always sent positive vx, so REVERSE_* plans were executed as
        # forward motion and the vehicle remained against the obstacle.  Keep
        # the lateral sign from the path, while flipping only the forward axis.
        speed = abs(self.forward_speed_mps)
        if math.isfinite(goal_distance) and goal_distance < self.goal_slowdown_distance_m:
            # Keep the command below the goal termination speed inside the
            # final metre, while retaining a small nonzero command to settle.
            speed *= max(0.20, goal_distance / self.goal_slowdown_distance_m)

        # Turn the vehicle toward the local path tangent, not the far lookahead
        # point.  The latter can stay lateral after the drone has passed the
        # bend and was the source of prolonged turns/large yaw overshoot.
        tangent = self._path_tangent(trajectory_points, nearest_idx, is_reverse)
        if tangent is None or (goal_xy is not None
                               and goal_distance < self.goal_blend_distance_m):
            tangent_fwd, tangent_lat = fwd, lat
        else:
            tx, ty = tangent
            tangent_fwd = tx * cos_y + ty * sin_y
            tangent_lat = -tx * sin_y + ty * cos_y
        bearing = math.atan2(tangent_lat, tangent_fwd)
        if not is_reverse and fwd < 0.0:
            res.vx = 0.0
            res.vy = 0.0
        else:
            res.vx = -speed if is_reverse else speed
        res.vy = curvature * self.command_lookahead_m * speed
        res.vy = max(-self.lateral_speed_mps, min(self.lateral_speed_mps, res.vy))
        res.yaw_rate_radps = max(
            -self.max_yaw_rate_radps,
            min(self.max_yaw_rate_radps, self.yaw_gain * bearing),
        )
        res.vz = 0.0

        # Inside the terminal approach area, command toward the live goal and
        # apply a braking cap. This prevents a cached blue arc from carrying
        # the vehicle past MissionEnd before the dwell checker can stop it.
        terminal_radius = max(
            self.terminal_goal_approach_radius_m,
            self.terminal_slowdown_radius_m,
        )
        if (
            goal_ned is not None
            and not is_reverse
            and terminal_radius > 0.0
            and goal_distance <= terminal_radius
        ):
            goal_dx = float(goal_ned[0]) - drone_position_ned[0]
            goal_dy = float(goal_ned[1]) - drone_position_ned[1]
            goal_fwd = goal_dx * cos_y + goal_dy * sin_y
            goal_lat = -goal_dx * sin_y + goal_dy * cos_y
            goal_vx = self.terminal_goal_kp * goal_fwd
            goal_vy = self.terminal_goal_kp * goal_lat

            braking_cap = math.sqrt(
                2.0 * self.terminal_braking_accel_mps2 * goal_distance
            )
            terminal_cap = min(
                self.terminal_goal_max_speed_mps,
                braking_cap,
            )
            goal_mag = math.hypot(goal_vx, goal_vy)
            if goal_mag > terminal_cap > 0.0:
                scale = terminal_cap / goal_mag
                goal_vx *= scale
                goal_vy *= scale

            if goal_distance <= self.terminal_capture_radius_m:
                goal_vx = goal_vy = 0.0
            res.vx = goal_vx
            res.vy = goal_vy

        return res

    @staticmethod
    def _nearest_index(
        trajectory_points: List[Tuple[float, float]],
        drone_position_ned: Tuple[float, float, float],
    ) -> int:
        px, py = drone_position_ned[0], drone_position_ned[1]
        return min(
            range(len(trajectory_points)),
            key=lambda i: (trajectory_points[i][0] - px) ** 2
                           + (trajectory_points[i][1] - py) ** 2,
        )

    @staticmethod
    def _path_tangent(
        trajectory_points: List[Tuple[float, float]],
        nearest_idx: int,
        is_reverse: bool,
    ) -> Optional[Tuple[float, float]]:
        """Return the unit world tangent at the nearest path sample."""
        if len(trajectory_points) < 2:
            return None
        if is_reverse:
            a_idx = nearest_idx
            b_idx = max(0, nearest_idx - 1)
        else:
            a_idx = nearest_idx
            b_idx = min(len(trajectory_points) - 1, nearest_idx + 1)
            if b_idx == a_idx:
                a_idx = max(0, nearest_idx - 1)
        dx = trajectory_points[b_idx][0] - trajectory_points[a_idx][0]
        dy = trajectory_points[b_idx][1] - trajectory_points[a_idx][1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        return dx / norm, dy / norm

    def _lookahead_point(
        self,
        trajectory_points: List[Tuple[float, float]],
        drone_position_ned: Optional[Tuple[float, float, float]] = None,
        nearest_idx: Optional[int] = None,
    ) -> Tuple[float, float]:
        if not trajectory_points:
            return (0.0, 0.0)

        if drone_position_ned is None:
            idx = min(
                len(trajectory_points) - 1,
                int(round(self.lookahead_m / max(0.05, self.sample_spacing_m))),
            )
            return trajectory_points[idx]

        nearest = (
            nearest_idx if nearest_idx is not None
            else self._nearest_index(trajectory_points, drone_position_ned)
        )
        travelled = 0.0
        for i in range(nearest + 1, len(trajectory_points)):
            ax, ay = trajectory_points[i - 1]
            bx, by = trajectory_points[i]
            travelled += math.hypot(bx - ax, by - ay)
            if travelled >= self.lookahead_m:
                return trajectory_points[i]
        return trajectory_points[-1]

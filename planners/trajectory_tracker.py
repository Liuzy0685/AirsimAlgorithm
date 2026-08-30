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
from typing import List, Optional, Sequence, Tuple


@dataclass
class TrackerResult:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    lookahead_point: Optional[Tuple[float, ...]] = None
    curvature: float = 0.0
    lateral_error_m: float = 0.0
    vertical_error_m: float = 0.0
    feedforward_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class TrajectoryTracker:
    """Pure-pursuit tracker over a cached world-NED XY trajectory."""

    def __init__(self, lookahead_m: float = 1.0, sample_spacing_m: float = 0.25,
                 forward_speed_mps: float = 0.25, lateral_speed_mps: float = 0.20,
                 command_lookahead_m: float = 1.0,
                 feedforward_gain: float = 1.0,
                 lateral_position_kp: float = 0.0,
                 vertical_position_kp: float = 0.0,
                 velocity_kd: float = 0.0,
                 terminal_goal_approach_radius_m: float = 0.0,
                 terminal_slowdown_radius_m: float = 0.0,
                 recovery_suppress_radius_m: float = 0.0,
                 terminal_goal_kp: float = 0.5,
                 terminal_goal_max_speed_mps: Optional[float] = None,
                 terminal_braking_accel_mps2: float = 0.35,
                 terminal_capture_radius_m: float = 0.02) -> None:
        self.lookahead_m = lookahead_m
        self.sample_spacing_m = sample_spacing_m
        self.forward_speed_mps = forward_speed_mps
        self.lateral_speed_mps = lateral_speed_mps
        self.command_lookahead_m = command_lookahead_m
        self.feedforward_gain = feedforward_gain
        self.lateral_position_kp = lateral_position_kp
        self.vertical_position_kp = vertical_position_kp
        self.velocity_kd = velocity_kd
        self.terminal_goal_approach_radius_m = max(
            0.0, float(terminal_goal_approach_radius_m),
        )
        self.terminal_slowdown_radius_m = terminal_slowdown_radius_m
        self.recovery_suppress_radius_m = max(0.0, float(recovery_suppress_radius_m))
        self.terminal_goal_kp = max(0.0, float(terminal_goal_kp))
        self.terminal_goal_max_speed_mps = (
            max(0.0, float(terminal_goal_max_speed_mps))
            if terminal_goal_max_speed_mps is not None else forward_speed_mps
        )
        self.terminal_braking_accel_mps2 = max(
            0.01, float(terminal_braking_accel_mps2),
        )
        self.terminal_capture_radius_m = max(
            0.0, float(terminal_capture_radius_m),
        )

    def compute_command(
        self,
        trajectory_points: Sequence[Sequence[float]],
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        trajectory_feedforward_body: Optional[Sequence[Sequence[float]]] = None,
        current_velocity_ned: Optional[Tuple[float, float, float]] = None,
        goal_ned: Optional[Tuple[float, float, float]] = None,
    ) -> TrackerResult:
        """Compute a body-frame velocity to track the cached trajectory.

        Pure pursuit: pick the trajectory sample ~``lookahead_m`` ahead,
        measure its body-frame lateral offset, and turn toward it.  When a
        trajectory window supplies feed-forward velocities, the tracker uses
        them as the base command and adds bounded P/D corrections.
        """
        res = TrackerResult()
        if not trajectory_points:
            return res

        idx = self._lookahead_index(trajectory_points)
        lp = tuple(float(v) for v in trajectory_points[idx])
        res.lookahead_point = lp

        dx = lp[0] - drone_position_ned[0]
        dy = lp[1] - drone_position_ned[1]
        dz = (lp[2] if len(lp) > 2 else drone_position_ned[2]) - drone_position_ned[2]
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # body-frame offset of the lookahead point
        fwd = dx * cos_y + dy * sin_y
        lat = -dx * sin_y + dy * cos_y
        res.lateral_error_m = lat
        res.vertical_error_m = dz

        L = max(0.1, self.lookahead_m)
        # Standard pure-pursuit curvature: kappa = 2 * lateral / L^2.
        curvature = 2.0 * lat / (L * L)
        res.curvature = curvature

        pure_vx = self.forward_speed_mps
        pure_vy = curvature * self.command_lookahead_m * self.forward_speed_mps
        pure_vy = max(-self.lateral_speed_mps, min(self.lateral_speed_mps, pure_vy))

        ff = None
        if trajectory_feedforward_body:
            ff_idx = min(idx, len(trajectory_feedforward_body) - 1)
            raw = trajectory_feedforward_body[ff_idx]
            if len(raw) >= 3:
                ff = (float(raw[0]), float(raw[1]), float(raw[2]))
        if ff is None:
            ff = (pure_vx, pure_vy, 0.0)
        res.feedforward_body = ff

        cvx_b = cvy_b = cvz_b = 0.0
        if current_velocity_ned is not None:
            cvx_b = current_velocity_ned[0] * cos_y + current_velocity_ned[1] * sin_y
            cvy_b = -current_velocity_ned[0] * sin_y + current_velocity_ned[1] * cos_y
            cvz_b = current_velocity_ned[2]

        res.vx = ff[0] * self.feedforward_gain - self.velocity_kd * cvx_b
        res.vy = (
            ff[1] * self.feedforward_gain
            + self.lateral_position_kp * lat
            - self.velocity_kd * cvy_b
        )
        res.vz = (
            ff[2] * self.feedforward_gain
            + self.vertical_position_kp * dz
            - self.velocity_kd * cvz_b
        )

        if goal_ned is not None and self.terminal_slowdown_radius_m > 0.0:
            dist_xy = math.hypot(
                drone_position_ned[0] - goal_ned[0],
                drone_position_ned[1] - goal_ned[1],
            )
            if dist_xy < self.terminal_slowdown_radius_m:
                # Once inside the terminal capture radius, use the actual goal
                # position as the target.  The cached trajectory is still used
                # outside this radius, but a fixed lookahead point can leave a
                # small residual error near the endpoint.
                goal_dx = goal_ned[0] - drone_position_ned[0]
                goal_dy = goal_ned[1] - drone_position_ned[1]
                goal_fwd = goal_dx * cos_y + goal_dy * sin_y
                goal_lat = -goal_dx * sin_y + goal_dy * cos_y
                goal_vx = self.terminal_goal_kp * goal_fwd
                goal_vy = self.terminal_goal_kp * goal_lat
                goal_mag = math.hypot(goal_vx, goal_vy)
                braking_cap = math.sqrt(
                    2.0 * self.terminal_braking_accel_mps2 * dist_xy
                )
                terminal_cap = min(
                    self.terminal_goal_max_speed_mps,
                    braking_cap,
                )
                if goal_mag > terminal_cap > 0.0:
                    scale = terminal_cap / goal_mag
                    goal_vx *= scale
                    goal_vy *= scale
                res.vx = goal_vx
                res.vy = goal_vy

                # The braking cap above already provides a smooth
                # distance-dependent slowdown.  Do not multiply by another
                # smoothstep here: doing so makes the vehicle crawl while it
                # is still nearly a metre from the goal.
                if dist_xy <= self.terminal_capture_radius_m:
                    res.vx = 0.0
                    res.vy = 0.0
            elif (
                self.terminal_goal_approach_radius_m > 0.0
                and dist_xy <= self.terminal_goal_approach_radius_m
            ):
                goal_dx = goal_ned[0] - drone_position_ned[0]
                goal_dy = goal_ned[1] - drone_position_ned[1]
                goal_fwd = goal_dx * cos_y + goal_dy * sin_y
                goal_lat = -goal_dx * sin_y + goal_dy * cos_y
                goal_vx = self.terminal_goal_kp * goal_fwd
                goal_vy = self.terminal_goal_kp * goal_lat
                goal_mag = math.hypot(goal_vx, goal_vy)
                braking_cap = math.sqrt(
                    2.0 * self.terminal_braking_accel_mps2 * dist_xy
                )
                terminal_cap = min(
                    self.terminal_goal_max_speed_mps,
                    braking_cap,
                )
                if goal_mag > terminal_cap > 0.0:
                    scale = terminal_cap / goal_mag
                    goal_vx *= scale
                    goal_vy *= scale
                res.vx = goal_vx
                res.vy = goal_vy

        res.vx = max(-self.forward_speed_mps, min(self.forward_speed_mps, res.vx))
        res.vy = max(-self.lateral_speed_mps, min(self.lateral_speed_mps, res.vy))
        return res

    def _lookahead_index(self, trajectory_points: Sequence[Sequence[float]]) -> int:
        if not trajectory_points:
            return 0
        return min(
            len(trajectory_points) - 1,
            int(round(self.lookahead_m / max(0.05, self.sample_spacing_m))),
        )

    def _lookahead_point(self, trajectory_points: Sequence[Sequence[float]]) -> Tuple[float, float]:
        if not trajectory_points:
            return (0.0, 0.0)
        pt = trajectory_points[self._lookahead_index(trajectory_points)]
        return (float(pt[0]), float(pt[1]))

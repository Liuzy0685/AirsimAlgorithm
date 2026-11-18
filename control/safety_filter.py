"""SafetyFilter — formation-target safety gate before dispatch (step 4, §4/§5).

A formation target may only reach the flight controller after passing this
filter.  Checks (fail-closed):

1. the command is present and valid,
2. the neighbor set is complete (missing/stale peers ⇒ reject, never guess),
3. the desired velocity is finite and within horizontal/vertical limits
   (out-of-limit magnitudes are clamped, not silently forwarded),
4. the predicted closest approach to every neighbor over the prediction
   horizon respects the minimum inter-vehicle distance.

Hover is **not** a universal default here: the filter only *rejects* an unsafe
target; the arbitrator/gateway decides the actual fallback (decelerate, hold,
recover, or emergency) based on the wider state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from common.models import FormationCommand, VehicleState

Vec3 = tuple


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    desired_velocity_ned_mps: tuple


def _norm(v) -> float:
    return math.sqrt(sum(float(c) ** 2 for c in v))


def _clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _predicted_min_distance(own_pos, own_vel, nbr_pos, nbr_vel, horizon_s: float) -> float:
    """Minimum 3D center distance predicted over ``horizon_s`` (linear CPA)."""
    rel = [float(n) - float(o) for n, o in zip(nbr_pos, own_pos)]
    rel_vel = [float(nv) - float(ov) for nv, ov in zip(nbr_vel, own_vel)]
    speed_sq = sum(c * c for c in rel_vel)
    if speed_sq < 1e-9:
        return _norm(rel)
    t = -sum(r * v for r, v in zip(rel, rel_vel)) / speed_sq
    t = max(0.0, min(float(horizon_s), t))
    return _norm([r + v * t for r, v in zip(rel, rel_vel)])


class SafetyFilter:
    """Reject formation targets that violate speed or inter-vehicle limits."""

    def __init__(
        self,
        *,
        max_horizontal_speed_mps: float = 4.0,
        max_vertical_speed_mps: float = 1.0,
        safety_distance_m: float = 2.0,
        prediction_horizon_s: float = 3.0,
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (
            max_horizontal_speed_mps, max_vertical_speed_mps,
            safety_distance_m, prediction_horizon_s,
        )):
            raise ValueError("safety limits must be finite and positive")
        self.max_horizontal_speed_mps = float(max_horizontal_speed_mps)
        self.max_vertical_speed_mps = float(max_vertical_speed_mps)
        self.safety_distance_m = float(safety_distance_m)
        self.prediction_horizon_s = float(prediction_horizon_s)

    def check(
        self,
        command: Optional[FormationCommand],
        *,
        own_state: VehicleState,
        neighbors: Sequence[VehicleState],
        neighbors_complete: bool = True,
    ) -> SafetyDecision:
        """Gate a formation command; returns the decision and clamped velocity."""
        if command is None or not getattr(command, "valid", False):
            return SafetyDecision(False, "invalid_formation_command", (0.0, 0.0, 0.0))
        if not neighbors_complete:
            return SafetyDecision(False, "neighbor_state_incomplete", (0.0, 0.0, 0.0))

        velocity = self._velocity_from_command(command, own_state)
        return self.check_velocity(
            velocity, own_state=own_state, neighbors=neighbors,
            neighbors_complete=neighbors_complete,
        )

    def check_velocity(
        self,
        velocity,
        *,
        own_state: VehicleState,
        neighbors: Sequence[VehicleState],
        neighbors_complete: bool = True,
    ) -> SafetyDecision:
        """Gate an explicit desired velocity (finite, limited, separation)."""
        if not neighbors_complete:
            return SafetyDecision(False, "neighbor_state_incomplete", (0.0, 0.0, 0.0))
        try:
            vx, vy, vz = (float(c) for c in velocity)
        except (TypeError, ValueError):
            return SafetyDecision(False, "non_finite", (0.0, 0.0, 0.0))
        if not all(math.isfinite(c) for c in (vx, vy, vz)):
            return SafetyDecision(False, "non_finite", (0.0, 0.0, 0.0))

        # Speed limits — clamp (documented, never silently exceed).
        h = math.hypot(vx, vy)
        if h > self.max_horizontal_speed_mps:
            scale = self.max_horizontal_speed_mps / h
            vx, vy = vx * scale, vy * scale
        vz = _clamp(vz, self.max_vertical_speed_mps)
        desired = (vx, vy, vz)

        # Inter-vehicle minimum separation.
        own_pos = own_state.position_ned_m
        for nbr in neighbors:
            dmin = _predicted_min_distance(
                own_pos, desired,
                nbr.position_ned_m, nbr.linear_velocity_ned_mps,
                self.prediction_horizon_s,
            )
            if dmin < self.safety_distance_m:
                return SafetyDecision(
                    False, f"inter_vehicle_conflict: dmin={dmin:.2f}m", (0.0, 0.0, 0.0)
                )
        return SafetyDecision(True, "", desired)

    def _velocity_from_command(self, command: FormationCommand,
                               own_state: VehicleState) -> tuple:
        if command.mode == "velocity":
            return tuple(command.desired_velocity_ned_mps)
        target = command.target_position_ned_m
        own = own_state.position_ned_m
        d = [float(t) - float(o) for t, o in zip(target, own)]
        dist = _norm(d)
        if dist < 1e-6:
            return (0.0, 0.0, 0.0)
        unit = [c / dist for c in d]
        # Horizontal magnitude caps at the horizontal limit; vertical is clamped.
        speed = min(dist, self.max_horizontal_speed_mps)
        vx = unit[0] * speed
        vy = unit[1] * speed
        vz = _clamp(unit[2] * speed, self.max_vertical_speed_mps)
        return (vx, vy, vz)

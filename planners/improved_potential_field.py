"""
Improved Artificial Potential Field — sector-based reactive planner.

Pure calculation module.  Does NOT call any AirSim API.

Inputs:
- 16-sector distances + point counts (from DirectionalDistances)
- Goal direction in body-frame
- Current body-frame velocity
- Configurable parameters

Outputs (ApfOutput):
- desired_vx_body, desired_vy_body, desired_vz_body (m/s, body FRD)
- attractive_force, repulsive_force (diagnostic only)
- nearest_distance, valid, reason
- NaN/Inf guard, speed clamping, epsilon on all divisions

Coordinate convention (body FRD):
    +X = forward
    +Y = right
    +Z = down
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── 16 legacy sector names in order ──
_LEGACY_NAMES = [
    "front", "back", "left", "right", "up", "down",
    "frontLeft", "frontRight", "backLeft", "backRight",
    "frontUp", "frontDown", "leftUp", "rightUp", "leftDown", "rightDown",
]

# Sectors excluded from horizontal force in horizontal_only mode.
# These face downward (+Z body FRD) and their X/Y repulsive components
# are ground artefacts, not true lateral obstacles.
_HORIZONTAL_EXCLUDED_SECTORS = frozenset({
    "down", "frontDown", "backDown", "leftDown", "rightDown",
})

# Sector → body-frame direction unit vector [vx, vy, vz]
_SECTOR_DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "front":      ( 1.0,  0.0,  0.0),
    "back":       (-1.0,  0.0,  0.0),
    "left":       ( 0.0, -1.0,  0.0),
    "right":      ( 0.0,  1.0,  0.0),
    "up":         ( 0.0,  0.0, -1.0),
    "down":       ( 0.0,  0.0,  1.0),
    "frontLeft":  ( 0.707, -0.707,  0.0),
    "frontRight": ( 0.707,  0.707,  0.0),
    "backLeft":   (-0.707, -0.707,  0.0),
    "backRight":  (-0.707,  0.707,  0.0),
    "frontUp":    ( 0.707,  0.0, -0.707),
    "frontDown":  ( 0.707,  0.0,  0.707),
    "leftUp":     ( 0.0, -0.707, -0.707),
    "rightUp":    ( 0.0,  0.707, -0.707),
    "leftDown":   ( 0.0, -0.707,  0.707),
    "rightDown":  ( 0.0,  0.707,  0.707),
}


@dataclass
class ApfOutput:
    """Structured APF output with diagnostic separation."""
    desired_vx_body: float = 0.0
    desired_vy_body: float = 0.0
    desired_vz_body: float = 0.0
    attractive_force: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    repulsive_force: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_magnitude: float = 0.0
    command_magnitude: float = 0.0
    nearest_distance: float = float("inf")
    valid: bool = True
    reason: str = ""
    nan_detected: bool = False
    inf_detected: bool = False
    saturated: bool = False
    per_sector_contributions: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class ApfParams:
    """Configurable APF parameters with conservative defaults."""
    attractive_gain: float = 0.8
    repulsive_gain: float = 4.0
    damping_gain: float = 0.5
    safe_distance_m: float = 8.0
    emergency_distance_m: float = 0.8
    max_horizontal_speed_mps: float = 0.20
    max_vertical_speed_mps: float = 0.10
    epsilon: float = 1e-6
    horizontal_only: bool = True
    enable_per_sector_diagnostics: bool = False


class ImprovedPotentialField:
    """Sector-based APF planner.

    Converts 16-sector LiDAR distances into virtual obstacles, computes
    attractive + repulsive + damping forces, then normalizes and clamps
    to produce a safe body-frame velocity command.
    """

    def __init__(self, params: Optional[ApfParams] = None):
        self._params = params or ApfParams()

    # ── public API ──

    def update(
        self,
        sector_distances: Dict[str, float],
        sector_point_counts: Optional[Dict[str, int]] = None,
        goal_body: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        current_velocity_body: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        minimum_distance_m: float = float("inf"),
        lateral_guidance_bias: float = 0.0,
    ) -> ApfOutput:
        """Compute one APF planning step.

        Args:
            sector_distances: dict of legacy_name → distance_m for each sector.
            sector_point_counts: optional dict of legacy_name → point_count.
            goal_body: goal direction in body-frame (vx, vy, vz).  Will be normalized.
            current_velocity_body: current body-frame velocity (vx, vy, vz).
            minimum_distance_m: global minimum distance across all points.
            lateral_guidance_bias: optional lateral bias added to attractive_y
                **after** the normal goal_body computation.  Default 0.0
                preserves byte-for-byte identical behaviour.  Intended for
                shadow-only CBMBA lateral guidance experiments.

        Returns:
            ApfOutput with desired body-frame velocity and diagnostics.
        """
        p = self._params
        eps = p.epsilon

        # ── Validate goal ──
        gx, gy, gz = float(goal_body[0]), float(goal_body[1]), float(goal_body[2])
        goal_len = math.sqrt(gx * gx + gy * gy + gz * gz)
        if goal_len < eps:
            goal_len = 1.0; gx = 1.0
        gx /= goal_len; gy /= goal_len; gz /= goal_len

        # ── Current velocity ──
        cvx, cvy, cvz = float(current_velocity_body[0]), float(current_velocity_body[1]), float(current_velocity_body[2])

        # ── NaN/Inf checks ──
        nan_inf_reason = ""
        for label, val in [("gx", gx), ("gy", gy), ("gz", gz),
                            ("cvx", cvx), ("cvy", cvy), ("cvz", cvz),
                            ("lateral_guidance_bias", lateral_guidance_bias)]:
            if math.isnan(val):
                nan_inf_reason = f"NaN in {label}"
                break
            if math.isinf(val):
                nan_inf_reason = f"Inf in {label}"
                break
        if nan_inf_reason:
            return ApfOutput(
                valid=False, reason=nan_inf_reason,
                nan_detected="NaN" in nan_inf_reason, inf_detected="Inf" in nan_inf_reason,
                nearest_distance=minimum_distance_m,
            )

        # ── Attractive force (toward goal) ──
        att_x = gx * p.attractive_gain
        att_y = gy * p.attractive_gain + lateral_guidance_bias
        att_z = gz * p.attractive_gain

        # ── Repulsive force (from virtual sector obstacles) ──
        rep_x, rep_y, rep_z = 0.0, 0.0, 0.0
        nearest = float(minimum_distance_m)
        per_sector: List[Dict[str, object]] = []

        for name, distance in sector_distances.items():
            if name not in _SECTOR_DIRECTIONS:
                continue
            d = float(distance)
            if not (d > 0 and d < p.safe_distance_m):
                continue

            # Weight by point count if available
            weight = 1.0
            if sector_point_counts and name in sector_point_counts:
                pc = sector_point_counts[name]
                weight = max(float(pc) / 5.0, 0.2) if pc > 0 else 1.0

            # Direction AWAY from obstacle (negate sector direction)
            dx, dy, dz = _SECTOR_DIRECTIONS[name]
            ax, ay, az = -dx, -dy, -dz  # repulsive pushes away
            # Repulsive magnitude: ~ 1/d²
            dist_scale = (1.0 / max(d, eps)) - (1.0 / p.safe_distance_m)
            # Clamp distance scale to prevent explosion at very close range
            dist_scale = min(dist_scale, 100.0)
            mag = p.repulsive_gain * weight * dist_scale / max(d * d, eps)

            # NaN/Inf guard on magnitude
            if math.isnan(mag) or math.isinf(mag):
                return ApfOutput(
                    valid=False, reason=f"repulsive_magnitude_diverged:{name}",
                    nan_detected=math.isnan(mag), inf_detected=math.isinf(mag),
                    nearest_distance=nearest,
                )

            rx = ax * mag
            ry = ay * mag
            rz = az * mag

            # In horizontal_only mode, Down-series sectors contribute only
            # to Z (diagnostic); their X/Y components are ground artefacts.
            excluded_from_horizontal = (
                p.horizontal_only and name in _HORIZONTAL_EXCLUDED_SECTORS
            )
            if excluded_from_horizontal:
                rep_z += rz  # Z for diagnostics; excluded from norm anyway
            else:
                rep_x += rx; rep_y += ry; rep_z += rz
            nearest = min(nearest, d)

            if p.enable_per_sector_diagnostics:
                per_sector.append({
                    "name": name, "distance": d,
                    "dir_x": dx, "dir_y": dy, "dir_z": dz,
                    "rep_x": rx, "rep_y": ry, "rep_z": rz,
                    "used_for_control": not excluded_from_horizontal,
                })

        # ── Damping ──
        damp_x = -cvx * p.damping_gain
        damp_y = -cvy * p.damping_gain
        damp_z = -cvz * p.damping_gain

        # ── Raw forces (diagnostic only) ──
        raw_fx = att_x + rep_x + damp_x
        raw_fy = att_y + rep_y + damp_y
        raw_fz = att_z + rep_z + damp_z
        raw_mag = math.sqrt(raw_fx * raw_fx + raw_fy * raw_fy + raw_fz * raw_fz)

        # ── Normalize raw force → velocity command ──
        # When horizontal_only, exclude Z from normalization magnitude so
        # ground/ceiling repulsion does not dilute horizontal steering.
        if p.horizontal_only:
            h_raw_mag = math.sqrt(raw_fx * raw_fx + raw_fy * raw_fy)
            if h_raw_mag > eps:
                norm_x = raw_fx / h_raw_mag
                norm_y = raw_fy / h_raw_mag
            else:
                norm_x = norm_y = 0.0
            norm_z = 0.0
        else:
            if raw_mag > eps:
                norm_x = raw_fx / raw_mag
                norm_y = raw_fy / raw_mag
                norm_z = raw_fz / raw_mag
            else:
                norm_x = norm_y = norm_z = 0.0

        # Scale by speed limits
        cmd_x = norm_x * p.max_horizontal_speed_mps
        cmd_y = norm_y * p.max_horizontal_speed_mps
        cmd_z = norm_z * p.max_vertical_speed_mps
        cmd_mag = math.sqrt(cmd_x * cmd_x + cmd_y * cmd_y + cmd_z * cmd_z)

        # ── NaN/Inf on output ──
        for label, val in [("cmd_x", cmd_x), ("cmd_y", cmd_y), ("cmd_z", cmd_z)]:
            if math.isnan(val) or math.isinf(val):
                return ApfOutput(
                    valid=False, reason=f"output_{label}_invalid",
                    nan_detected=math.isnan(val), inf_detected=math.isinf(val),
                    attractive_force=(att_x, att_y, att_z),
                    repulsive_force=(rep_x, rep_y, rep_z),
                    force_magnitude=raw_mag, nearest_distance=nearest,
                )

        # ── Clamp horizontal speed ──
        h_speed = math.sqrt(cmd_x * cmd_x + cmd_y * cmd_y)
        saturated = False
        if h_speed > p.max_horizontal_speed_mps:
            scale = p.max_horizontal_speed_mps / h_speed
            cmd_x *= scale; cmd_y *= scale
            saturated = True
        if abs(cmd_z) > p.max_vertical_speed_mps:
            cmd_z = math.copysign(p.max_vertical_speed_mps, cmd_z)
            saturated = True

        # ── Horizontal-only mode: suppress vertical APF output ──
        if p.horizontal_only:
            cmd_z = 0.0

        return ApfOutput(
            desired_vx_body=cmd_x,
            desired_vy_body=cmd_y,
            desired_vz_body=cmd_z,
            attractive_force=(att_x, att_y, att_z),
            repulsive_force=(rep_x, rep_y, rep_z),
            force_magnitude=raw_mag,
            command_magnitude=cmd_mag,
            nearest_distance=nearest,
            valid=True,
            reason="",
            saturated=saturated,
            per_sector_contributions=per_sector if p.enable_per_sector_diagnostics else [],
        )

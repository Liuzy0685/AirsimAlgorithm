"""Fixed local goal computation — ROUND 4.

Supports two modes:
- body_relative_at_start: goal = drone_pos + R(yaw) * [forward, right, down]
- absolute_local_ned: goal = config value directly

Goal is computed once at startup (snapshot of drone yaw), then FIXED.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def compute_fixed_local_goal(
    config: Dict,
    drone_position_ned: Tuple[float, float, float],
    drone_yaw_rad: float,
) -> Tuple[Tuple[float, float, float], str]:
    """Compute a fixed local NED goal.

    Args:
        config: Full config dict (containing 'goal' section).
        drone_position_ned: (x_north, y_east, z_down) at startup.
        drone_yaw_rad: Drone yaw angle at startup (radians).

    Returns:
        ((goal_x, goal_y, goal_z), mode_description)
    """
    goal_cfg = config.get("goal", {}) or {}
    mode = goal_cfg.get("mode", "body_relative_at_start")

    if mode == "absolute_local_ned":
        x = float(goal_cfg.get("north_m", 5.0))
        y = float(goal_cfg.get("east_m", 0.0))
        z = float(goal_cfg.get("down_m", drone_position_ned[2]))
        return ((drone_position_ned[0] + x, drone_position_ned[1] + y, z),
                f"absolute_local_ned: offset=({x},{y},{z})")

    # body_relative_at_start (default)
    fwd_m = float(goal_cfg.get("forward_m", 5.0))
    right_m = float(goal_cfg.get("right_m", 0.0))
    down_m = float(goal_cfg.get("down_m", 0.0))

    # Rotate body-relative offset into local NED using drone's actual yaw
    cos_y = math.cos(drone_yaw_rad)
    sin_y = math.sin(drone_yaw_rad)

    dx_ned = fwd_m * cos_y - right_m * sin_y
    dy_ned = fwd_m * sin_y + right_m * cos_y
    dz_ned = down_m  # Z unaffected by yaw

    goal = (
        drone_position_ned[0] + dx_ned,
        drone_position_ned[1] + dy_ned,
        drone_position_ned[2] + dz_ned,
    )
    return (goal, f"body_relative_at_start: body({fwd_m},{right_m},{down_m}) → ned({dx_ned:.2f},{dy_ned:.2f},{dz_ned:.2f})")

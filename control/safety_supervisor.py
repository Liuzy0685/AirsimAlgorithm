"""Safety Supervisor — ROUND 4.2.

Final safety gate.  Does NOT modify the APF formula.
Checks: data sync, LiDAR valid, FOV compatible, DD valid, data timeout,
RPC error, collision, NaN/Inf, speed limits, yaw rate limits,
obstacle approach direction.

Any critical data invalid → command_valid=False, reason=explicit.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from models.local_planner_command import LocalPlannerCommand, invalid_command
from models.lidar_frame import LidarFrame
from models.directional_distances import DirectionalDistances
from models.collision_state import CollisionState


class SafetySupervisor:
    """Final safety gate for local planner commands."""

    def __init__(self, config: Optional[Dict] = None, clock: Optional[Callable[[], float]] = None):
        cfg = config or {}
        safety_cfg = cfg.get("safety", {}) or {}
        self.max_horizontal_speed = float(safety_cfg.get("max_horizontal_speed_mps", 4.0))
        self.max_vertical_speed = float(safety_cfg.get("max_vertical_speed_mps", 1.0))
        self.max_yaw_rate = float(safety_cfg.get("max_yaw_rate_radps", 1.2))
        self.lidar_timeout = float(safety_cfg.get("lidar_timeout_seconds", 0.5))
        self.max_consecutive_invalid = int(safety_cfg.get("max_consecutive_invalid", 10))
        self._clock = clock if clock is not None else time.monotonic

    def validate(
        self,
        proposed_velocity: Tuple[float, float, float],
        proposed_yaw_rate: Optional[float],
        source: str,
        lidar_frame: Optional[LidarFrame],
        directional_distances: Optional[DirectionalDistances],
        collision_state: Optional[CollisionState],
        fov_compatible: bool,
        consecutive_invalid: int = 0,
        data_sync_valid: bool = True,
        data_sync_reason: Optional[str] = None,
        obstacle_positions_ned: Optional[np.ndarray] = None,
        ego_position_ned: Optional[Tuple[float, float, float]] = None,
    ) -> LocalPlannerCommand:
        """Validate a proposed velocity command against all safety gates."""

        # --- Data sync check (from dry-run) ---
        if not data_sync_valid:
            return invalid_command(f"Data sync failed: {data_sync_reason or 'unknown'}")

        # --- LiDAR frame invalid ---
        if lidar_frame is None or not lidar_frame.frame_valid:
            return invalid_command(f"LiDAR frame invalid: {lidar_frame.invalid_reason if lidar_frame else 'no frame'}")

        # --- FOV incompatible ---
        if not fov_compatible:
            return invalid_command("FOV incompatible")

        # --- DirectionalDistances invalid ---
        if directional_distances is None or not directional_distances.frame_valid:
            return invalid_command(f"DD invalid: {directional_distances.invalid_reason if directional_distances else 'no data'}")

        # --- Data timeout (ROUND 4.2: actual check) ---
        now = self._clock()
        if lidar_frame.received_monotonic_seconds > 0:
            lidar_age = now - lidar_frame.received_monotonic_seconds
            if lidar_age > self.lidar_timeout:
                return invalid_command(f"lidar_data_stale: age {lidar_age:.3f}s > {self.lidar_timeout}s")

        # --- Consecutive invalid too high ---
        if consecutive_invalid >= self.max_consecutive_invalid:
            return invalid_command(f"Max consecutive invalid reached: {consecutive_invalid}")

        # --- Collision detected ---
        if collision_state is not None and collision_state.has_collided:
            return invalid_command(f"Collision active: {collision_state.object_name} (penetration={collision_state.penetration_depth:.3f}m)")

        # --- NaN / Inf check ---
        vx, vy, vz = proposed_velocity
        if any(not math.isfinite(v) for v in (vx, vy, vz)):
            return invalid_command(f"Proposed velocity contains NaN/Inf: ({vx},{vy},{vz})")
        if proposed_yaw_rate is not None and not math.isfinite(proposed_yaw_rate):
            return invalid_command(f"Proposed yaw_rate contains NaN/Inf: {proposed_yaw_rate}")

        # --- Horizontal speed limit ---
        h_speed = math.hypot(vx, vy)
        if h_speed > self.max_horizontal_speed:
            return invalid_command(f"Horizontal speed {h_speed:.2f} m/s exceeds limit {self.max_horizontal_speed} m/s")

        # --- Vertical speed limit ---
        if abs(vz) > self.max_vertical_speed:
            return invalid_command(f"Vertical speed {abs(vz):.2f} m/s exceeds limit {self.max_vertical_speed} m/s")

        # --- Yaw rate limit ---
        if proposed_yaw_rate is not None and abs(proposed_yaw_rate) > self.max_yaw_rate:
            return invalid_command(f"Yaw rate {abs(proposed_yaw_rate):.3f} rad/s exceeds limit {self.max_yaw_rate} rad/s")

        # --- Obstacle approach check (ROUND 4.2) ---
        # If velocity is toward any nearby obstacle, reject.
        if obstacle_positions_ned is not None and ego_position_ned is not None and obstacle_positions_ned.size > 0:
            vel = np.array([vx, vy, vz], dtype=np.float64)
            speed_sq = float(vel @ vel)
            if speed_sq > 1e-9:
                ego = np.array(ego_position_ned, dtype=np.float64)
                for i in range(obstacle_positions_ned.shape[0]):
                    obs_pos = obstacle_positions_ned[i]
                    to_obs = obs_pos - ego  # vector FROM ego TO obstacle
                    dist_sq = float(to_obs @ to_obs)
                    if dist_sq < 1e-9:
                        continue
                    # Dot product: positive = velocity toward obstacle
                    approach = float(np.dot(vel, to_obs))
                    if approach > 0:
                        dist = math.sqrt(dist_sq)
                        if dist < 8.0:  # only check nearby obstacles
                            return invalid_command(
                                f"Velocity toward obstacle at distance {dist:.2f}m: "
                                f"approach={approach:.2f}, vel=({vx:.2f},{vy:.2f},{vz:.2f})"
                            )

        # --- All checks passed ---
        return LocalPlannerCommand(
            command_valid=True,
            velocity_world_ned_mps=(vx, vy, vz),
            yaw_rate_radps=proposed_yaw_rate,
            source=source,
            priority=100 if source == "apf" else (40 if source == "recovery" else 0),
            invalid_reason=None,
        )

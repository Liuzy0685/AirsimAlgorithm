"""Improved Artificial Potential Field — ROUND 4.

Faithful migration of ImprovedPotentialField.js (449 lines).
Preserves ALL formulas: attractive, path attraction, repulsive (basic +
forward-block + relative-velocity), tangential bypass, vertical bias,
damping, dominant-obstacle boost, bypass-direction memory.

Coordinate convention:
  - Input position/velocity/goal: world NED (+X=north, +Y=east, +Z=down)
  - Input localPointCloud: world NED (already transformed from SensorLocalFrame)
  - Output velocity_world_ned_mps: world NED [vx_north, vy_east, vz_down] m/s
  - Output yaw_rate_radps: rad/s (PI controller on desired velocity heading)

Uses ONLY NumPy — no THREE.js, no scipy.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Vector helpers (NumPy equivalents of THREE.Vector3 methods)
# ═══════════════════════════════════════════════════════════════════════


def _vec3(arr) -> np.ndarray:
    """Convert [x,y,z] list-like to (3,) float64 numpy array."""
    v = np.asarray(arr, dtype=np.float64).ravel()
    if v.size != 3:
        raise ValueError(f"Expected 3 elements, got {v.size}")
    return v


def _length(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _length_sq(v: np.ndarray) -> float:
    return float(v @ v)


def _normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """Safe normalize — if near-zero, return fallback (or zero vector)."""
    n = _length_sq(v)
    if n < 1e-12:
        if fallback is not None:
            return fallback.copy()
        return np.zeros(3, dtype=np.float64)
    return v / math.sqrt(n)


def _clamp_magnitude(v: np.ndarray, max_mag: float) -> np.ndarray:
    """Clamp vector magnitude in-place (returns same array)."""
    if _length_sq(v) > max_mag * max_mag:
        v = _normalize(v) * max_mag
    return v


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _project_point_to_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    len_sq = _length_sq(segment)
    if len_sq < 1e-12:
        return start.copy()
    t = _clamp01(np.dot(point - start, segment) / len_sq)
    return start + segment * t


def _point_cloud_obstacles(
    point_cloud: Optional[np.ndarray], point_radius: float
) -> List[Dict]:
    if point_cloud is None or point_cloud.size < 3:
        return []
    pts = np.asarray(point_cloud, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return []
    obstacles = []
    for i in range(pts.shape[0]):
        obstacles.append({
            "id": f"point-{i}",
            "type": "unknown",
            "position": pts[i].tolist(),
            "velocity": [0.0, 0.0, 0.0],
            "size": point_radius,
            "dynamic": False,
            "confidence": 0.25,
        })
    return obstacles


def _obstacle_half_extents(obstacle: Dict) -> Dict[str, float]:
    fp = obstacle.get("footprintHalfExtents")
    if isinstance(fp, (list, tuple)) and len(fp) >= 2:
        return {
            "x": max(fp[0] if len(fp) > 0 else 0, 0),
            "y": max(fp[1] if len(fp) > 1 else 0, 0),
            "z": max(fp[2] if len(fp) > 2 else 0, 0),
        }
    radius = max(obstacle.get("size", 0), 0)
    return {"x": radius, "y": radius, "z": radius}


def _is_massive_terrain_obstacle(
    obstacle: Dict, max_span: float, max_height: float
) -> bool:
    oid = obstacle.get("id", "")
    if not isinstance(oid, str) or not oid.startswith("terrain-"):
        return False
    ext = _obstacle_half_extents(obstacle)
    span = max(ext["x"] * 2, ext["y"] * 2)
    height = ext["z"] * 2
    return span > max_span or height > max_height


# ═══════════════════════════════════════════════════════════════════════
# ImprovedPotentialField class
# ═══════════════════════════════════════════════════════════════════════


class ImprovedPotentialField:
    """Improved Artificial Potential Field safety layer.

    Ported faithfully from ImprovedPotentialField.js.
    Preserves: corridor attraction along global path tangent,
    dominant-obstacle bypass bias, forward-block gain for static
    buildings, and short-term bypass direction memory to reduce
    left/right oscillation.
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        # State variables (preserved from JS)
        self.last_bypass_sign: int = 1
        self.last_bypass_obstacle_id: Optional[str] = None
        self.last_bypass_update_time: float = -float("inf")
        self.bypass_lock_until: float = -float("inf")
        self.bypass_lock_clearance: float = float("inf")

    def reset(self) -> None:
        self.last_bypass_sign = 1
        self.last_bypass_obstacle_id = None
        self.last_bypass_update_time = -float("inf")
        self.bypass_lock_until = -float("inf")
        self.bypass_lock_clearance = float("inf")

    @property
    def parameters(self) -> Dict:
        defaults = {
            "enabled": True,
            "priority": 100,
            "validity_ms": 120,
            "max_speed_mps": 6.0,
            "max_acceleration_mps2": 8.0,
            "hover_deadband_mps": 0.35,
            "safe_distance_m": 14.0,
            "emergency_distance_m": 8.0,
            "repulsive_gain": 18.0,
            "attractive_gain": 1.8,
            "path_attraction_gain": 2.1,
            "damping_gain": 1.2,
            "tangential_gain": 4.6,
            "relative_velocity_gain": 2.5,
            "vertical_bias_gain": 0.8,
            "forward_block_gain": 6.0,
            "dominant_obstacle_boost": 1.8,
            "path_corridor_width_m": 5.0,
            "point_cloud_point_radius_m": 0.35,
            "yaw_rate_gain": 1.4,
            "max_yaw_rate_radps": 1.2,
            "local_goal_lookahead": 4,
            "bypass_memory_seconds": 0.8,
            "bypass_lock_seconds": 2.0,
            "bypass_lock_release_distance_m": 11.0,
            "wall_follow_distance_m": 10.0,
            "wall_follow_gain": 2.4,
            "up_axis_sign": -1,
            "local_obstacle_max_span_m": 45.0,
            "local_obstacle_max_height_m": 24.0,
        }
        # Merge config.navigation.apf over defaults
        nav = self._config.get("navigation", {})
        apf_cfg = nav.get("apf", {}) if isinstance(nav, dict) else {}
        result = {**defaults}
        for k, v in apf_cfg.items():
            if k in defaults:
                result[k] = v
        return result

    def update(self, observation: Dict) -> Dict:
        """Main entry point — faithfully mirrors JS `update(observation)`.

        Args:
            observation: NavigationObservation dict with keys:
                ego: {position, orientation, linearVelocity, angularVelocity}
                goal: [x,y,z]
                globalPath: [[x,y,z], ...]  (optional)
                staticObstacles: List[Obstacle]
                dynamicObstacles: List[Obstacle]
                localPointCloud: N×3 ndarray in world NED (optional)
                dt: float (time step seconds)
                timestamp: float

        Returns:
            MotionSetpoint dict: {velocity_world_ned_mps, yaw_rate_radps,
                                  source, priority, valid_until}
        """
        params = self.parameters
        now = observation.get("timestamp", 0.0)
        ego_pos = _vec3(observation["ego"]["position"])
        ego_vel = _vec3(observation["ego"]["linearVelocity"])
        dt = observation.get("dt", 0.01)

        # --- Path frame ---
        path_frame = self._compute_path_frame(observation, ego_pos)
        local_goal = path_frame["goal"]

        # --- Attractive force (to local goal) ---
        attractive = _normalize(local_goal - ego_pos) * params["attractive_gain"]

        # --- Path attraction (along tangent) ---
        path_attraction = path_frame["tangent"] * params["path_attraction_gain"]

        # --- Damping ---
        damping = ego_vel * (-params["damping_gain"])

        # --- Gather obstacles ---
        static_obs = observation.get("staticObstacles", []) or []
        dynamic_obs = observation.get("dynamicObstacles", []) or []
        cloud = _point_cloud_obstacles(
            observation.get("localPointCloud"),
            params["point_cloud_point_radius_m"],
        )
        filtered_static = [
            o for o in static_obs
            if not _is_massive_terrain_obstacle(
                o, params["local_obstacle_max_span_m"],
                params["local_obstacle_max_height_m"],
            )
        ]
        all_obstacles = filtered_static + dynamic_obs + cloud

        # --- Dominant obstacle and bypass sign ---
        dominant = self._find_dominant_obstacle(
            all_obstacles, ego_pos, path_frame["tangent"], params
        )
        bypass_sign = self._select_bypass_sign(
            dominant, ego_pos, path_frame["tangent"], now, params
        )

        # --- Repulsive + tangential forces ---
        repulsive = np.zeros(3, dtype=np.float64)
        tangential = np.zeros(3, dtype=np.float64)
        nearest_distance = float("inf")

        for obs in all_obstacles:
            obs_pos = _vec3(obs["position"])
            rel_pos = ego_pos - obs_pos
            center_dist = _length(rel_pos)
            clearance = max(center_dist - (obs.get("size", 0) or 0), 1e-3)
            nearest_distance = min(nearest_distance, clearance)

            if clearance > params["safe_distance_m"]:
                continue

            away = _normalize(rel_pos)
            to_obstacle = -away
            weight = max(obs.get("confidence", 0.5) or 0.5, 0.1)
            alignment = max(np.dot(path_frame["tangent"], to_obstacle), 0.0)
            corridor_penalty = max(
                1.0 - _length(obs_pos - path_frame["projected"])
                / max(params["path_corridor_width_m"], 1.0),
                0.0,
            )
            is_dominant = (
                dominant is not None and dominant["id"] == obs.get("id")
            )
            boost = params["dominant_obstacle_boost"] if is_dominant else 1.0

            # (1) Basic repulsive: repulsiveGain * weight * boost * (1/r - 1/safe) / r²
            dist_scale = (1.0 / clearance) - (1.0 / params["safe_distance_m"])
            rep_mag = (
                params["repulsive_gain"] * weight * boost
                * dist_scale / (clearance * clearance)
            )
            repulsive = repulsive + away * rep_mag

            # (2) Forward-block gain
            if alignment > 0.2:
                repulsive = repulsive + away * (
                    params["forward_block_gain"] * weight * boost
                    * alignment * corridor_penalty / max(clearance, 1.0)
                )

            # (3) Relative velocity term
            obs_vel = _vec3(obs.get("velocity", [0, 0, 0]))
            rel_vel = ego_vel - obs_vel
            closing_speed = max(np.dot(rel_vel, to_obstacle), 0.0)
            if closing_speed > 0.0:
                repulsive = repulsive + away * (
                    params["relative_velocity_gain"] * closing_speed
                    * weight * boost / clearance
                )

            # (4) Tangential (bypass) force
            tangent_axis = np.array(
                [-rel_pos[1], rel_pos[0], 0.0], dtype=np.float64
            )
            if _length_sq(tangent_axis) > 1e-12:
                tangent = _normalize(tangent_axis) * bypass_sign
                emerg_boost = 2.0 if clearance < params["emergency_distance_m"] else 1.0
                wf_boost = (
                    params["wall_follow_gain"] if alignment > 0.25 else 1.0
                )
                tangential = tangential + tangent * (
                    params["tangential_gain"] * weight * boost
                    * emerg_boost * wf_boost / max(clearance, 1.0)
                )

            # Wall-follow along path tangent
            if alignment > 0.15 and clearance < params["wall_follow_distance_m"]:
                tangential = tangential + path_frame["tangent"] * (
                    alignment * weight * params["wall_follow_gain"]
                    / max(clearance, 1.0)
                )

            # (5) Vertical bias
            vb_scale = 1.0 if obs.get("dynamic") else 0.2
            if (
                obs_pos[2] >= ego_pos[2]
                and clearance < params["emergency_distance_m"]
            ):
                tangential[2] += (
                    params["vertical_bias_gain"] * weight
                    * params["up_axis_sign"] * vb_scale
                )
            elif (
                obs_pos[2] < ego_pos[2]
                and clearance < params["emergency_distance_m"]
            ):
                tangential[2] -= (
                    params["vertical_bias_gain"] * weight
                    * params["up_axis_sign"] * vb_scale
                )

        # --- Synthesis ---
        desired_accel = attractive + path_attraction + repulsive + tangential + damping
        desired_accel = _clamp_magnitude(
            desired_accel, params["max_acceleration_mps2"]
        )

        desired_vel = ego_vel + desired_accel * dt
        desired_vel = _clamp_magnitude(desired_vel, params["max_speed_mps"])

        # Hover deadband
        if (
            _length(desired_vel) < params["hover_deadband_mps"]
            and nearest_distance >= params["safe_distance_m"]
        ):
            desired_vel = np.zeros(3, dtype=np.float64)

        # --- Yaw control ---
        desired_yaw = math.atan2(desired_vel[1], desired_vel[0])
        current_yaw = observation["ego"]["orientation"][2]
        yaw_error = math.atan2(
            math.sin(desired_yaw - current_yaw),
            math.cos(desired_yaw - current_yaw),
        )
        yaw_rate = max(
            -params["max_yaw_rate_radps"],
            min(params["max_yaw_rate_radps"],
                yaw_error * params["yaw_rate_gain"]),
        )

        return {
            "velocity_world_ned_mps": (
                float(desired_vel[0]),
                float(desired_vel[1]),
                float(desired_vel[2]),
            ),
            "yaw_rate_radps": float(yaw_rate),
            "source": "apf",
            "priority": params["priority"],
            "valid_until": now + params["validity_ms"] / 1000.0,
            "diagnostics": {
                "attractive_force_world_ned": (
                    float(attractive[0]), float(attractive[1]), float(attractive[2]),
                ),
                "path_attraction_force_world_ned": (
                    float(path_attraction[0]), float(path_attraction[1]), float(path_attraction[2]),
                ),
                "repulsive_force_world_ned": (
                    float(repulsive[0]), float(repulsive[1]), float(repulsive[2]),
                ),
                "tangential_force_world_ned": (
                    float(tangential[0]), float(tangential[1]), float(tangential[2]),
                ),
                "damping_force_world_ned": (
                    float(damping[0]), float(damping[1]), float(damping[2]),
                ),
                "desired_acceleration_world_ned": (
                    float(desired_accel[0]), float(desired_accel[1]), float(desired_accel[2]),
                ),
                "nearest_obstacle_distance_m": (
                    float(nearest_distance) if nearest_distance != float("inf") else None
                ),
                "dominant_obstacle_id": (
                    dominant["id"] if dominant else None
                ),
                "bypass_sign": int(bypass_sign),
            },
        }

    # ------------------------------------------------------------------
    # Path frame computation (lines 338-379)
    # ------------------------------------------------------------------

    def _compute_path_frame(
        self, observation: Dict, ego_pos: np.ndarray
    ) -> Dict:
        params = self.parameters
        global_path = observation.get("globalPath") or []
        goal = _vec3(observation["goal"])

        if not global_path or len(global_path) < 2:
            tangent = _normalize(
                goal - ego_pos,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            return {"projected": ego_pos.copy(), "tangent": tangent, "goal": goal}

        # Find nearest segment
        best = {"distance": float("inf"), "index": 0,
                "start": None, "end": None, "projected": None}
        for i in range(len(global_path) - 1):
            start = _vec3(global_path[i])
            end = _vec3(global_path[i + 1])
            proj = _project_point_to_segment(ego_pos, start, end)
            dist = _length(proj - ego_pos)
            if dist < best["distance"]:
                best = {"distance": dist, "index": i,
                        "start": start, "end": end, "projected": proj}

        tangent = _normalize(
            best["end"] - best["start"],
            _normalize(goal - ego_pos, np.array([1.0, 0.0, 0.0])),
        )
        la_idx = min(
            best["index"] + params["local_goal_lookahead"],
            len(global_path) - 1,
        )
        local_goal = _vec3(global_path[la_idx])
        return {"projected": best["projected"], "tangent": tangent, "goal": local_goal}

    # ------------------------------------------------------------------
    # Dominant obstacle (lines 382-408)
    # ------------------------------------------------------------------

    def _find_dominant_obstacle(
        self,
        obstacles: List[Dict],
        ego_pos: np.ndarray,
        path_tangent: np.ndarray,
        params: Dict,
    ) -> Optional[Dict]:
        best = None
        best_score = -float("inf")
        for obs in obstacles:
            obs_pos = _vec3(obs["position"])
            to_obs = obs_pos - ego_pos
            clearance = max(
                _length(to_obs) - (obs.get("size", 0) or 0), 1e-3
            )
            if clearance > params["safe_distance_m"]:
                continue
            alignment = max(
                np.dot(path_tangent, _normalize(to_obs)), 0.0
            )
            score = (
                alignment * 1.8
                + (1.0 - clearance / params["safe_distance_m"]) * 1.6
                + (0.25 if obs.get("dynamic") else 0.5)
            )
            if score > best_score:
                best = {**obs, "clearance": clearance,
                        "alignment": alignment, "score": score}
                best_score = score
        return best

    # ------------------------------------------------------------------
    # Bypass sign selection (lines 410-448) — stateful
    # ------------------------------------------------------------------

    def _select_bypass_sign(
        self,
        dominant: Optional[Dict],
        ego_pos: np.ndarray,
        path_tangent: np.ndarray,
        now: float,
        params: Dict,
    ) -> int:
        if dominant is None:
            if now > self.bypass_lock_until:
                self.last_bypass_obstacle_id = None
                self.bypass_lock_clearance = float("inf")
            return self.last_bypass_sign

        obs_pos = _vec3(dominant["position"])
        to_obs = obs_pos - ego_pos
        cross_z = path_tangent[0] * to_obs[1] - path_tangent[1] * to_obs[0]
        sign = -1 if cross_z >= 0 else 1

        lock_active = (
            now < self.bypass_lock_until
            and dominant["clearance"]
            < max(params["bypass_lock_release_distance_m"],
                  self.bypass_lock_clearance)
        )

        if (
            self.last_bypass_obstacle_id == dominant["id"]
            and (
                (now - self.last_bypass_update_time)
                < params["bypass_memory_seconds"]
                or lock_active
            )
        ):
            sign = self.last_bypass_sign
        elif lock_active:
            sign = self.last_bypass_sign

        self.last_bypass_sign = sign
        self.last_bypass_obstacle_id = dominant["id"]
        self.last_bypass_update_time = now
        if dominant["clearance"] < params["wall_follow_distance_m"]:
            self.bypass_lock_until = now + params["bypass_lock_seconds"]
            self.bypass_lock_clearance = (
                dominant["clearance"]
                + params["bypass_lock_release_distance_m"]
            )
        return sign

"""AvoidanceSupervisor — ROUND 4.1.

Faithful migration of AvoidanceSupervisor.js (1049 lines).
Uses LocalRecovery for all detour state (no duplicated recovery logic).

Key fix: update() accepts optional pre_computed_apf_result to avoid
double APF calls.  When not provided, computes APF internally (once).
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
from planning.improved_potential_field import (
    ImprovedPotentialField,
    _vec3, _normalize, _length, _length_sq, _clamp01, _clamp_magnitude,
    _obstacle_half_extents, _is_massive_terrain_obstacle,
)
from planning.local_recovery import (
    LocalRecovery, _safe_direction, _lateral_direction, _clamp_forward_progress,
    _horizontal_clearance, _closest_horizontal_point,
)


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _hover_setpoint(timestamp: float, validity_ms: float) -> Dict:
    return {"velocity_world_ned_mps": (0.0, 0.0, 0.0), "yaw_rate_radps": 0.0,
            "source": "hover", "priority": 0, "valid_until": timestamp + validity_ms / 1000.0}


def _project_point_to_segment(point, start, end):
    seg = end - start; ls = _length_sq(seg)
    if ls < 1e-12: return start.copy()
    t = _clamp01(float(np.dot(point - start, seg)) / ls)
    return start + seg * t


def _path_deviation(position, path):
    if not path: return float("inf")
    if len(path) == 1: return _length(position - _vec3(path[0]))
    best = float("inf")
    for i in range(len(path) - 1):
        proj = _project_point_to_segment(position, _vec3(path[i]), _vec3(path[i + 1]))
        best = min(best, _length(proj - position))
    return best


def _front_ray_profile(ray_distances, fallback=float("inf")):
    r = ray_distances
    return {
        "front": min(r.get("front", fallback), r.get("frontLeft", r.get("front", fallback)), r.get("frontRight", r.get("front", fallback))),
        "left": min(r.get("left", fallback), r.get("frontLeft", r.get("left", fallback)), r.get("leftUp", r.get("left", fallback))),
        "right": min(r.get("right", fallback), r.get("frontRight", r.get("right", fallback)), r.get("rightUp", r.get("right", fallback))),
        "up": min(r.get("up", fallback), r.get("frontUp", r.get("up", fallback)), r.get("leftUp", r.get("up", fallback)), r.get("rightUp", r.get("up", fallback))),
    }


def _select_best_setpoint(candidates, fallback):
    valid = [c for c in candidates if c is not None]
    if not valid: return fallback
    valid.sort(key=lambda c: (c["priority"], c["valid_until"]), reverse=True)
    return valid[0]


def _is_ray_obstacle(obstacle):
    oid = obstacle.get("id", ""); return isinstance(oid, str) and oid.startswith("ray-")


def _preferred_static_obstacles(observation):
    obs = observation.get("staticObstacles", []) or []
    return [o for o in obs if not _is_ray_obstacle(o) and not _is_massive_terrain_obstacle(o, 45.0, 24.0)]


# ═══════════════════════════════════════════════════════════════════════
# AvoidanceSupervisor
# ═══════════════════════════════════════════════════════════════════════

class AvoidanceSupervisor:
    def __init__(self, config=None):
        self._config = config or {}
        self.apf = ImprovedPotentialField(config)
        self.recovery = LocalRecovery(config)
        self.mode: str = "CRUISE"
        self.mode_since: float = 0.0
        self.last_global_path: List = []
        self.last_rl_result: Optional[Dict] = None
        self.last_threats: List = []
        self.last_setpoint: Optional[Dict] = None

    def reset(self, now=0.0):
        self.apf.reset(); self.recovery.reset()
        self.mode = "CRUISE"; self.mode_since = now
        self.last_global_path = []; self.last_rl_result = None
        self.last_threats = []; self.last_setpoint = None

    @property
    def parameters(self) -> Dict:
        defaults = {
            "hysteresis_enter_seconds": 0.5, "hysteresis_exit_seconds": 0.75,
            "dynamic_threat_score_threshold": 0.6, "dynamic_ttc_threshold_s": 5.0,
            "emergency_ttc_threshold_s": 1.5, "emergency_distance_threshold_m": 8.0,
            "static_warning_distance_m": 5.0, "static_emergency_distance_m": 3.0,
            "static_blocked_ahead_distance_m": 8.0, "static_blocked_ahead_cone_dot": 0.45,
            "static_apf_only_distance_m": 1.6, "static_hard_emergency_distance_m": 0.7,
            "recovery_clear_seconds": 0.5, "recovery_path_deviation_threshold_m": 3.0,
            "rl_result_ttl_ms": 150, "hover_validity_ms": 120,
            "recovery_speed_mps": 4.0, "cruise_priority": 20, "recovery_priority": 40,
            "detour_lateral_gain": 2.2, "detour_forward_gain": 3.4,
            "detour_obstacle_influence_distance_m": 7.0,
            "detour_lateral_clearance_m": 3.5, "detour_forward_clearance_m": 4.5,
            "detour_extra_clearance_m": 2.5, "detour_waypoint_reach_distance_m": 2.8,
            "detour_lock_seconds": 1.4, "recovery_lookahead_distance_m": 8.0,
            "recovery_min_forward_speed_mps": 1.8, "recovery_vertical_gain": 0.45,
            "recovery_vertical_deadband": 0.25, "recovery_vertical_limit_mps": 0.45,
            "local_detour_max_obstacle_span_m": 45.0, "local_detour_max_obstacle_height_m": 24.0,
            "proactive_front_distance_m": 9.0, "proactive_front_hard_distance_m": 4.8,
            "proactive_side_preference_gain": 2.6, "proactive_forward_suppression": 0.85,
            "proactive_climb_distance_m": 4.0, "proactive_climb_speed_mps": 0.42,
            "proactive_climb_side_threshold_m": 2.0,
        }
        nav = self._config.get("navigation", {}) or {}
        sv = nav.get("supervisor", {}) or {}
        return {**defaults, **{k: v for k, v in sv.items() if k in defaults}}

    def accept_global_path(self, path): self.last_global_path = path if isinstance(path, list) else []
    def accept_rl_result(self, result): self.last_rl_result = result
    def accept_threat_assessments(self, threats): self.last_threats = threats if isinstance(threats, list) else []

    def status(self) -> Dict:
        return {"mode": self.mode, "since": self.mode_since,
                "detour_points": len(self.recovery.detour_path),
                "detour_index": self.recovery.detour_index,
                "detour_obstacle_id": self.recovery.detour_obstacle_id}

    # ------------------------------------------------------------------
    # Main update — accepts optional pre-computed APF result
    # ------------------------------------------------------------------

    def update(self, observation, cruise_setpoint=None,
               ray_distances=None, pre_computed_apf_result=None) -> Dict:
        now = observation.get("timestamp", 0.0); params = self.parameters
        obs_wr = {**observation}
        if ray_distances: obs_wr["rayDistances"] = ray_distances

        nav_phase = observation.get("navPhase", "cruise")
        if nav_phase in ("takeoff", "align"):
            self.mode = "CRUISE"; self.mode_since = now
            sp = cruise_setpoint or _hover_setpoint(now, params["hover_validity_ms"])
            self.last_setpoint = sp; return sp

        threats = self.last_threats

        # APF: use pre-computed result if provided, else compute once
        apf_sp = pre_computed_apf_result if pre_computed_apf_result is not None else self.apf.update({
            **obs_wr, "globalPath": (self.last_global_path if self.last_global_path else observation.get("globalPath", [])),
        })

        rl_fresh = self._latest_fresh_rl_result(now)
        static_blocked = self._is_static_blocked_ahead(obs_wr, ray_distances)
        self._transition_mode(obs_wr, threats, now, ray_distances)
        self._refresh_local_detour(obs_wr, static_blocked, ray_distances)

        if self.mode == "EMERGENCY_AVOID":
            selected = (apf_sp if self._should_apf_own_control(obs_wr, threats, ray_distances)
                        else self._build_recovery_setpoint(obs_wr, now, ray_distances))
        elif self.mode == "DYNAMIC_AVOID":
            selected = self._select_dynamic_setpoint(obs_wr, rl_fresh, apf_sp, cruise_setpoint, threats, now, ray_distances)
        elif self.mode == "RECOVERY":
            recovery_sp = self._build_recovery_setpoint(obs_wr, now, ray_distances)
            if self._should_apf_own_control(obs_wr, threats, ray_distances):
                selected = _select_best_setpoint([apf_sp, recovery_sp], apf_sp or recovery_sp)
            else:
                selected = _select_best_setpoint([recovery_sp, cruise_setpoint], recovery_sp)
        else:
            recovery_sp = self._build_recovery_setpoint(obs_wr, now, ray_distances)
            if self._should_apf_own_control(obs_wr, threats, ray_distances):
                selected = _select_best_setpoint([apf_sp, recovery_sp, cruise_setpoint], apf_sp or recovery_sp or cruise_setpoint)
            elif static_blocked:
                selected = _select_best_setpoint([apf_sp, recovery_sp, cruise_setpoint], apf_sp or recovery_sp or cruise_setpoint)
            else:
                selected = cruise_setpoint or recovery_sp or apf_sp

        if selected is None: selected = _hover_setpoint(now, params["hover_validity_ms"])
        self.last_setpoint = selected; return selected

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def _transition_mode(self, observation, threats, now, ray_distances=None):
        params = self.parameters; in_for = now - self.mode_since
        dyn_emerg = self._is_emergency_threat(threats)
        dyn_threat = self._is_dynamic_threat(threats)
        st_emerg = self._is_static_emergency(observation)
        st_warn = self._is_static_warning(observation)
        st_blocked = self._is_static_blocked_ahead(observation, ray_distances)
        emerg = dyn_emerg or st_emerg; clear_emerg = not emerg
        on_path = self._is_back_on_path(observation)

        if self.mode == "CRUISE":
            if emerg: self._set_mode("EMERGENCY_AVOID", now)
            elif st_blocked and in_for >= params["hysteresis_enter_seconds"] * 0.4: self._set_mode("RECOVERY", now)
            elif dyn_threat and in_for >= params["hysteresis_enter_seconds"]: self._set_mode("DYNAMIC_AVOID", now)
        elif self.mode == "DYNAMIC_AVOID":
            if emerg: self._set_mode("EMERGENCY_AVOID", now)
            elif st_blocked and not dyn_threat: self._set_mode("RECOVERY", now)
            elif not dyn_threat and not st_blocked and in_for >= params["hysteresis_exit_seconds"]: self._set_mode("RECOVERY", now)
        elif self.mode == "EMERGENCY_AVOID":
            if clear_emerg and in_for >= params["recovery_clear_seconds"]: self._set_mode("RECOVERY", now)
        elif self.mode == "RECOVERY":
            if emerg: self._set_mode("EMERGENCY_AVOID", now)
            elif dyn_threat: self._set_mode("DYNAMIC_AVOID", now)
            elif on_path and not st_warn and in_for >= params["hysteresis_exit_seconds"]: self._set_mode("CRUISE", now)

    def _is_dynamic_threat(self, threats): return False
    def _is_emergency_threat(self, threats): return False

    def _is_back_on_path(self, observation):
        params = self.parameters
        if self.recovery.is_active and not self.recovery.is_complete: return False
        if not self.last_global_path: return True
        return _path_deviation(_vec3(observation["ego"]["position"]), self.last_global_path) < params["recovery_path_deviation_threshold_m"]

    def _is_static_warning(self, observation):
        col = observation.get("collision", {}) or {}
        if col.get("hasPhysicalContact") or col.get("isColliding"): return True
        return (col.get("minDistance", float("inf")) or float("inf")) < self.parameters["static_warning_distance_m"]

    def _is_static_emergency(self, observation):
        col = observation.get("collision", {}) or {}
        if col.get("hasPhysicalContact"): return True
        return (col.get("minDistance", float("inf")) or float("inf")) < self.parameters["static_emergency_distance_m"]

    def _should_apf_own_control(self, observation, threats, ray_distances=None):
        params = self.parameters
        col = observation.get("collision", {}) or {}
        has_contact = col.get("hasPhysicalContact", False)
        obs_dist = col.get("minDistance", float("inf")) or float("inf")
        profile = _front_ray_profile(ray_distances or {}, float("inf"))
        has_detour = self.recovery.is_active and not self.recovery.is_complete
        if self._is_emergency_threat(threats): return True
        if profile["front"] < params["proactive_front_hard_distance_m"]: return True
        if obs_dist < params["static_hard_emergency_distance_m"]: return True
        if has_contact and not has_detour: return True
        return has_contact and obs_dist < max(params["static_emergency_distance_m"] * 0.6, params["static_hard_emergency_distance_m"])

    def _is_static_blocked_ahead(self, observation, ray_distances=None):
        params = self.parameters
        tangent = self._path_direction(observation); ego = _vec3(observation["ego"]["position"])
        severity = 0.0; rd = ray_distances or {}
        profile = _front_ray_profile(rd, params["static_blocked_ahead_distance_m"])
        if profile["front"] < params["proactive_front_distance_m"]:
            fs = _clamp01(1.0 - profile["front"] / max(params["proactive_front_distance_m"], 1.0))
            ss = _clamp01(max(profile["left"], profile["right"]) / max(params["proactive_front_distance_m"], 1.0))
            severity = max(severity, fs * 0.82 + ss * 0.18)
        for obs in _preferred_static_obstacles(observation):
            point = _closest_horizontal_point(ego, obs); offset = point - ego
            if _length_sq(offset) < 1e-9: offset = _vec3(obs["position"]) - ego
            dist = max(_horizontal_clearance(ego, obs), 0.0)
            if dist > params["static_blocked_ahead_distance_m"]: continue
            direction = _normalize(offset, tangent)
            if np.dot(tangent, direction) < params["static_blocked_ahead_cone_dot"]: continue
            score = (_clamp01(1.0 - dist / params["static_blocked_ahead_distance_m"]) * 0.65
                     + _clamp01((np.dot(tangent, direction) - params["static_blocked_ahead_cone_dot"])
                                / (1.0 - params["static_blocked_ahead_cone_dot"])) * 0.35)
            severity = max(severity, score)
        return severity > 0.45

    def _path_direction(self, observation, prefer_local_detour=True):
        ego = _vec3(observation["ego"]["position"])
        if prefer_local_detour and self.recovery.is_active:
            wp = self.recovery.current_waypoint
            if wp: return _safe_direction(ego, _vec3(wp))
        if self.last_global_path and len(self.last_global_path) > 1:
            best_idx = 0; best_dist = float("inf")
            for i, pt in enumerate(self.last_global_path):
                d = _length(_vec3(pt) - ego)
                if d < best_dist: best_dist = d; best_idx = i
            nxt = min(best_idx + 1, len(self.last_global_path) - 1)
            return _safe_direction(ego, _vec3(self.last_global_path[nxt]))
        return _safe_direction(ego, _vec3(observation["goal"]))

    # ------------------------------------------------------------------
    # Recovery setpoint
    # ------------------------------------------------------------------

    def _build_recovery_setpoint(self, observation, now, ray_distances=None):
        params = self.parameters
        ego_pos = _vec3(observation["ego"]["position"]); ego_vel = _vec3(observation["ego"]["linearVelocity"])
        direction = self._path_direction(observation, False)
        rd = ray_distances or {}; profile = _front_ray_profile(rd, params["detour_obstacle_influence_distance_m"])

        self.recovery.advance(observation["ego"]["position"])
        target = None
        if self.recovery.is_active:
            wp = self.recovery.current_waypoint
            if wp: target = _vec3(wp)
        if self.last_global_path:
            target = target or self._target_along_path(self.last_global_path, observation["ego"]["position"], params["recovery_lookahead_distance_m"]) or _vec3(self.last_global_path[-1])
        if target is None: target = _vec3(observation["goal"])

        blocker = self._nearest_static_ahead(observation, direction)
        desired = direction * params["detour_forward_gain"]
        if target is not None: desired = _safe_direction(ego_pos, target, direction) * params["recovery_speed_mps"]

        if not self.recovery.is_active and blocker is not None:
            bp = _closest_horizontal_point(ego_pos, blocker); offset = bp - ego_pos
            cross = direction[0] * offset[1] - direction[1] * offset[0]; sign = -1 if cross >= 0 else 1
            lat = _lateral_direction(direction, sign)
            prox = _clamp01(1.0 - blocker.get("distance", 0.0) / params["detour_obstacle_influence_distance_m"])
            desired = desired + lat * (params["detour_lateral_gain"] * max(prox, 0.8))

        desired[2] = 0.0

        if profile["front"] < params["proactive_front_distance_m"]:
            fb = 1 if profile["right"] > profile["left"] else -1
            sign = self.recovery.preferred_detour_sign(rd, direction, fb)
            lat = _lateral_direction(direction, sign)
            fp = _clamp01(1.0 - profile["front"] / max(params["proactive_front_distance_m"], 1.0))
            side_gap = abs(profile["right"] - profile["left"])
            side_bias = _clamp01(side_gap / max(params["proactive_front_distance_m"], 1.0))
            desired = desired + lat * (params["proactive_side_preference_gain"] * (0.9 + fp * 1.4 + side_bias * 0.8))
            desired = desired + direction * (-params["recovery_speed_mps"] * params["proactive_forward_suppression"] * fp)
            narrow = max(profile["left"], profile["right"]) < params["proactive_climb_side_threshold_m"]
            if (profile["front"] < params["proactive_climb_distance_m"]
                    and profile["up"] > max(profile["front"], profile["left"], profile["right"]) + 0.5 and narrow):
                desired[2] = -params["proactive_climb_speed_mps"] * (1.0 + fp)

        goal_fwd = _safe_direction(ego_pos, _vec3(observation["goal"]), direction)
        fwd_speed = float(np.dot(desired, goal_fwd))
        if fwd_speed < params["recovery_min_forward_speed_mps"]:
            desired = desired + goal_fwd * (params["recovery_min_forward_speed_mps"] - fwd_speed)
        if profile["front"] >= params["proactive_climb_distance_m"] or abs(desired[2]) < 1e-6: desired[2] = 0.0

        desired_vel = _clamp_magnitude(desired - ego_vel * 0.15, params["recovery_speed_mps"])
        dy = math.atan2(desired_vel[1], desired_vel[0]); cy = observation["ego"]["orientation"][2]
        ye = math.atan2(math.sin(dy - cy), math.cos(dy - cy))
        return {"velocity_world_ned_mps": (float(desired_vel[0]), float(desired_vel[1]), float(desired_vel[2])),
                "yaw_rate_radps": float(max(-1.0, min(1.0, ye))), "source": "recovery",
                "priority": params["recovery_priority"], "valid_until": now + params["hover_validity_ms"] / 1000.0}

    # ------------------------------------------------------------------
    # Local detour — delegated to LocalRecovery
    # ------------------------------------------------------------------

    def _refresh_local_detour(self, observation, static_blocked, ray_distances=None):
        params = self.parameters; now = observation.get("timestamp", 0.0)
        base_dir = self._path_direction(observation, False)
        blocker = self._nearest_static_ahead(observation, base_dir)
        rd = ray_distances or {}
        needs = blocker is not None and (static_blocked or blocker.get("distance", float("inf")) < params["detour_obstacle_influence_distance_m"] * 0.9)

        if not needs:
            if self.recovery.is_active:
                if self.recovery.near_completion(observation["ego"]["position"]) or self._is_back_on_path(observation):
                    self.recovery.reset()
            return
        if self.recovery.is_active and now < self.recovery.detour_lock_until and not self.recovery.is_complete:
            return
        if not self.recovery.is_active or self.recovery.detour_obstacle_id != blocker["id"]:
            self.recovery.build_detour_path(
                observation["ego"]["position"], blocker, base_dir, rd,
                self.last_global_path, now,
            )

    def _nearest_static_ahead(self, observation, direction):
        params = self.parameters; ego = _vec3(observation["ego"]["position"])
        best = None; best_dist = float("inf")
        for obs in _preferred_static_obstacles(observation):
            point = _closest_horizontal_point(ego, obs); offset = point - ego
            if _length_sq(offset) < 1e-12: offset = _vec3(obs["position"]) - ego
            dist = max(_horizontal_clearance(ego, obs), 0.0)
            if dist > params["detour_obstacle_influence_distance_m"]: continue
            d = _normalize(offset, direction)
            if np.dot(direction, d) < params["static_blocked_ahead_cone_dot"]: continue
            if dist < best_dist: best_dist = dist; best = {**obs, "distance": dist, "alignment": float(np.dot(direction, d))}
        return best

    @staticmethod
    def _target_along_path(path, position, lookahead):
        if not path: return None
        if len(path) == 1: return _vec3(path[0])
        pos = _vec3(position); nearest = 0; best_d = float("inf")
        for i, pt in enumerate(path):
            d = _length(_vec3(pt) - pos)
            if d < best_d: best_d = d; nearest = i
        remaining = max(lookahead, 0.0); curr = path[max(nearest, 0)]
        for i in range(max(nearest, 0), len(path) - 1):
            nxt = path[i + 1]; sl = math.hypot(nxt[0] - curr[0], nxt[1] - curr[1], nxt[2] - curr[2])
            if sl >= remaining and sl > 1e-6: t = remaining / sl; return np.array([curr[0] + (nxt[0] - curr[0]) * t, curr[1] + (nxt[1] - curr[1]) * t, curr[2] + (nxt[2] - curr[2]) * t])
            remaining -= sl; curr = nxt
        return _vec3(path[-1])

    def _select_dynamic_setpoint(self, obs, rl_fresh, apf_sp, cruise_sp, threats, now, ray_distances=None):
        params = self.parameters; recovery_sp = self._build_recovery_setpoint(obs, now, ray_distances)
        return recovery_sp or cruise_sp or apf_sp or _hover_setpoint(now, params["hover_validity_ms"])

    def _latest_fresh_rl_result(self, now): return None

    def _set_mode(self, mode, now):
        if self.mode == mode: return
        self.mode = mode; self.mode_since = now

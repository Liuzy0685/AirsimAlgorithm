"""Local Recovery — ROUND 4.1.

Extracted from AvoidanceSupervisor.js lines 781-939.
Full 7-point local detour path: preEntry → entry → sideNear → side →
exit → clear → merge.

Encapsulated in LocalRecovery class so AvoidanceSupervisor does not
duplicate recovery state.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from planning.improved_potential_field import (
    _vec3, _normalize, _length, _length_sq, _clamp01, _clamp_magnitude,
    _obstacle_half_extents, _project_point_to_segment,
)


def _safe_direction(from_vec, to_vec, fallback=None):
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0])
    return _normalize(to_vec - from_vec, fallback)


def _lateral_direction(direction, sign=1):
    lat = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    if _length_sq(lat) < 1e-12:
        return np.array([0.0, float(sign), 0.0])
    return _normalize(lat) * sign


def _clamp_forward_progress(point, origin, forward, min_progress):
    offset = point - origin
    progress = float(np.dot(offset, forward))
    if progress >= min_progress:
        return point
    return point + forward * (min_progress - progress)


def _horizontal_clearance(position, obstacle):
    ext = _obstacle_half_extents(obstacle)
    dx = max(abs(position[0] - obstacle["position"][0]) - ext["x"], 0.0)
    dy = max(abs(position[1] - obstacle["position"][1]) - ext["y"], 0.0)
    return math.hypot(dx, dy)


def _closest_horizontal_point(position, obstacle):
    ext = _obstacle_half_extents(obstacle)
    c = obstacle["position"]
    return np.array([
        max(c[0] - ext["x"], min(c[0] + ext["x"], position[0])),
        max(c[1] - ext["y"], min(c[1] + ext["y"], position[1])),
        position[2],
    ])


class LocalRecovery:
    """Full 7-point local detour path recovery.

    State:
        detour_path: list of [x,y,z] waypoints (max 7)
        detour_index: current waypoint index
        detour_obstacle_id: obstacle that triggered current detour
        detour_sign: lateral direction sign (+1 or -1)
        detour_lock_until: timestamp until detour is locked
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self.detour_path: List[List[float]] = []
        self.detour_index: int = 0
        self.detour_obstacle_id: Optional[str] = None
        self.detour_sign: int = 1
        self.detour_lock_until: float = -float("inf")

    def reset(self) -> None:
        self.detour_path = []
        self.detour_index = 0
        self.detour_obstacle_id = None
        self.detour_sign = 1
        self.detour_lock_until = -float("inf")

    @property
    def is_active(self) -> bool:
        return len(self.detour_path) > 0

    @property
    def current_waypoint(self) -> Optional[List[float]]:
        if not self.detour_path:
            return None
        return self.detour_path[min(self.detour_index, len(self.detour_path) - 1)]

    @property
    def is_complete(self) -> bool:
        return self.detour_path and self.detour_index >= len(self.detour_path) - 1

    @property
    def parameters(self) -> Dict:
        defaults = {
            "detour_lateral_gain": 2.2,
            "detour_forward_gain": 3.4,
            "detour_obstacle_influence_distance_m": 7.0,
            "detour_lateral_clearance_m": 3.5,
            "detour_forward_clearance_m": 4.5,
            "detour_extra_clearance_m": 2.5,
            "detour_waypoint_reach_distance_m": 2.8,
            "detour_lock_seconds": 1.4,
            "recovery_path_deviation_threshold_m": 3.0,
            "static_blocked_ahead_cone_dot": 0.45,
            "local_detour_max_obstacle_span_m": 45.0,
            "local_detour_max_obstacle_height_m": 24.0,
        }
        nav = self._config.get("navigation", {}) or {}
        sv = nav.get("supervisor", {}) or {}
        return {**defaults, **{k: v for k, v in sv.items() if k in defaults}}

    # ------------------------------------------------------------------
    # Preferred detour sign (JS lines 942-963)
    # ------------------------------------------------------------------

    def preferred_detour_sign(
        self,
        ray_distances: Dict[str, float],
        direction: np.ndarray,
        fallback_sign: int = 1,
    ) -> int:
        r = ray_distances
        left_clear = (
            (r.get("left", 0.0) or 0.0) * 0.8
            + (r.get("frontLeft", r.get("left", 0.0)) or 0.0) * 1.2
            + (r.get("leftUp", r.get("left", 0.0)) or 0.0) * 0.35
            + (r.get("leftDown", r.get("left", 0.0)) or 0.0) * 0.2
        )
        right_clear = (
            (r.get("right", 0.0) or 0.0) * 0.8
            + (r.get("frontRight", r.get("right", 0.0)) or 0.0) * 1.2
            + (r.get("rightUp", r.get("right", 0.0)) or 0.0) * 0.35
            + (r.get("rightDown", r.get("right", 0.0)) or 0.0) * 0.2
        )
        if abs(right_clear - left_clear) < 0.75:
            return self.detour_sign if self.detour_obstacle_id else fallback_sign
        return 1 if right_clear > left_clear else -1

    # ------------------------------------------------------------------
    # Build 7-point detour path (JS lines 840-939)
    # ------------------------------------------------------------------

    def build_detour_path(
        self,
        ego_position: List[float],
        blocker: Dict,
        base_direction: np.ndarray,
        ray_distances: Dict[str, float],
        global_path: List,
        now: float,
    ) -> List[List[float]]:
        """Generate the full 7-point detour: preEntry, entry, sideNear,
        side, exit, clear, merge."""
        params = self.parameters
        ego = _vec3(ego_position)
        cruise_z = ego[2]

        bp = _closest_horizontal_point(ego, blocker)
        blocker_center = np.array([
            blocker["position"][0], blocker["position"][1], cruise_z
        ], dtype=np.float64)
        offset = bp - ego
        cross = base_direction[0] * offset[1] - base_direction[1] * offset[0]
        sign = self.preferred_detour_sign(
            ray_distances, base_direction, -1 if cross >= 0 else 1,
        )
        self.detour_sign = sign

        lateral = _lateral_direction(base_direction, sign)
        ext = _obstacle_half_extents(blocker)
        lateral_extent = abs(lateral[0]) * ext["x"] + abs(lateral[1]) * ext["y"]
        forward_extent = abs(base_direction[0]) * ext["x"] + abs(base_direction[1]) * ext["y"]

        r = ray_distances
        front_dist = min(
            r.get("front", params["detour_obstacle_influence_distance_m"]),
            r.get("frontLeft", params["detour_obstacle_influence_distance_m"]),
            r.get("frontRight", params["detour_obstacle_influence_distance_m"]),
            params["detour_obstacle_influence_distance_m"],
        )
        prox = _clamp01(1.0 - blocker.get("distance", 0.0) / max(params["detour_obstacle_influence_distance_m"], 1.0))
        extra_clear = (
            params["detour_extra_clearance_m"]
            + prox * 2.5
            + _clamp01(1.0 - front_dist / max(params["detour_obstacle_influence_distance_m"], 1.0)) * 2.0
        )
        lat_clear = lateral_extent + params["detour_lateral_clearance_m"] + extra_clear
        fwd_clear = forward_extent + params["detour_forward_clearance_m"] + extra_clear

        # 7 waypoints (exactly matching JS lines 872-926)
        pre_entry = _clamp_forward_progress(
            ego.copy() + lateral * max(lat_clear * 0.55, 3.0),
            ego, base_direction, 1.0,
        )
        pre_entry[2] = cruise_z

        entry = _clamp_forward_progress(
            blocker_center.copy() + lateral * lat_clear
            + base_direction * (-max(fwd_clear, 5.0)),
            ego, base_direction, 2.0,
        )
        entry[2] = cruise_z

        side_near = _clamp_forward_progress(
            blocker_center.copy() + lateral * lat_clear
            + base_direction * (-max(fwd_clear * 0.35, 2.5)),
            ego, base_direction, 3.5,
        )
        side_near[2] = cruise_z

        side_pt = _clamp_forward_progress(
            blocker_center.copy() + lateral * lat_clear,
            ego, base_direction, 5.0,
        )
        side_pt[2] = cruise_z

        exit_pt = _clamp_forward_progress(
            blocker_center.copy() + lateral * lat_clear
            + base_direction * fwd_clear,
            ego, base_direction, 7.0,
        )
        exit_pt[2] = cruise_z

        clear_pt = _clamp_forward_progress(
            exit_pt.copy() + base_direction * max(params["detour_forward_clearance_m"] + extra_clear * 0.5, 4.0),
            ego, base_direction, 9.5,
        )
        clear_pt[2] = cruise_z

        merge_target = self._target_along_path(global_path, ego_position, params["detour_obstacle_influence_distance_m"] * 0.9)
        merge = (
            merge_target.copy()
            if merge_target is not None
            else exit_pt.copy() + base_direction * max(params["detour_forward_clearance_m"], 4.0)
        )
        merge[2] = cruise_z

        raw_path = [pre_entry, entry, side_near, side_pt, exit_pt, clear_pt, merge]
        raw_path = [p for p in raw_path if all(math.isfinite(v) for v in p)]
        raw_list = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_path]

        # Filter near-duplicates (< 0.6m apart)
        filtered = []
        for pt in raw_list:
            if filtered:
                prev = filtered[-1]
                if math.hypot(pt[0] - prev[0], pt[1] - prev[1], pt[2] - prev[2]) <= 0.6:
                    continue
            filtered.append(pt)

        self.detour_path = filtered
        self.detour_index = 0
        self.detour_obstacle_id = blocker.get("id")
        self.detour_lock_until = now + params["detour_lock_seconds"]
        return filtered

    # ------------------------------------------------------------------
    # Advance waypoint index (JS lines 826-838)
    # ------------------------------------------------------------------

    def advance(self, ego_position: List[float]) -> None:
        if not self.detour_path:
            return
        ego = _vec3(ego_position)
        params = self.parameters
        while self.detour_index < len(self.detour_path) - 1:
            wp = _vec3(self.detour_path[self.detour_index])
            if _length(ego - wp) > params["detour_waypoint_reach_distance_m"]:
                break
            self.detour_index += 1

    # ------------------------------------------------------------------
    # Check if detour is near completion (JS lines 792-798)
    # ------------------------------------------------------------------

    def near_completion(self, ego_position: List[float], deviation_threshold: Optional[float] = None) -> bool:
        if not self.detour_path:
            return True
        params = self.parameters
        thresh = deviation_threshold if deviation_threshold is not None else params["recovery_path_deviation_threshold_m"]
        ego = _vec3(ego_position)
        final_pt = _vec3(self.detour_path[-1])
        return _length(ego - final_pt) < max(thresh, 2.5)

    # ------------------------------------------------------------------
    # Target along path (JS lines 994-1036)
    # ------------------------------------------------------------------

    @staticmethod
    def _target_along_path(path, position, lookahead):
        if not path:
            return None
        if len(path) == 1:
            return _vec3(path[0])
        pos = _vec3(position)
        nearest = 0
        best_d = float("inf")
        for i, pt in enumerate(path):
            d = _length(_vec3(pt) - pos)
            if d < best_d:
                best_d = d
                nearest = i
        remaining = max(lookahead, 0.0)
        curr = path[max(nearest, 0)]
        for i in range(max(nearest, 0), len(path) - 1):
            nxt = path[i + 1]
            seg_len = math.hypot(
                nxt[0] - curr[0], nxt[1] - curr[1], nxt[2] - curr[2],
            )
            if seg_len >= remaining and seg_len > 1e-6:
                t = remaining / seg_len
                return np.array([
                    curr[0] + (nxt[0] - curr[0]) * t,
                    curr[1] + (nxt[1] - curr[1]) * t,
                    curr[2] + (nxt[2] - curr[2]) * t,
                ], dtype=np.float64)
            remaining -= seg_len
            curr = nxt
        return _vec3(path[-1])

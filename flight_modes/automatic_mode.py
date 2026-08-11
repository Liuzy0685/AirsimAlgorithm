"""
Automatic flight mode — LiDAR-based autonomous obstacle avoidance.

All preflight checks execute BEFORE enableApiControl:
    1. Drone1 present
    2. LidarSensor1 consecutive valid frames
    3. No active collision
    4. FOV compatible (NOT hardcoded — validates sector coverage)
    5. Perception config valid
    6. minimal_flight.yaml loaded and parameters strictly validated
    7. target_z, speeds, time, geofence validated

CLI overrides take priority over YAML, but all values are strictly checked.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger("automatic_mode")

# ── reactive decision ──


@dataclass(frozen=True)
class ReactiveDecision:
    vx_body_mps: float = 0.0
    vy_body_mps: float = 0.0
    should_terminate: bool = False
    termination_reason: str = ""


def choose_reactive_command(
    front_m: float, left_m: float, right_m: float,
    minimum_distance_m: float, config: Dict[str, float],
) -> ReactiveDecision:
    emerg = config["emergency_distance_m"]
    ft = config["front_threshold_m"]
    fwd = config["forward_speed_mps"]
    side = config["side_speed_mps"]
    if minimum_distance_m < emerg:
        return ReactiveDecision(should_terminate=True, termination_reason="emergency_distance")
    if front_m > ft:
        return ReactiveDecision(vx_body_mps=fwd)
    if left_m > right_m:
        return ReactiveDecision(vy_body_mps=-side)
    return ReactiveDecision(vy_body_mps=side)


# ── flight result ──


@dataclass
class AutomaticFlightResult:
    success: bool = False
    termination_reason: str = "unknown"
    frames_completed: int = 0
    flight_duration_s: float = 0.0
    api_control_acquired: bool = False
    armed: bool = False
    takeoff_completed: bool = False
    airborne: bool = False
    landing_confirmed: bool = False
    disarmed: bool = False
    api_control_released: bool = False
    startup_floor_contact_baseline: bool = False


# ── flight config loading ──


def _load_flight_config(path: str) -> Dict[str, Any]:
    """Load and validate minimal_flight.yaml. Returns validated dict."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    fc = raw.get("minimal_flight", {})
    if not fc:
        raise ValueError("minimal_flight section missing from flight config")

    params = {}
    validations = [
        ("target_z_ned", -10.0, -0.1),
        ("max_vertical_speed_mps", 0.1, 2.0),
        ("max_flight_duration_s", 0.5, 120.0),
        ("command_duration_s", 0.05, 1.0),
        ("forward_speed_mps", 0.05, 2.0),
        ("side_speed_mps", 0.05, 1.0),
        ("front_threshold_m", 0.5, 20.0),
        ("emergency_distance_m", 0.3, 5.0),
        ("geofence_radius_m", 0.5, 50.0),
        ("preflight_lidar_frames", 1, 20),
        ("takeoff_timeout_s", 5.0, 60.0),
    ]
    for key, lo, hi in validations:
        val = fc.get(key)
        if val is None:
            raise ValueError(f"minimal_flight.{key} is missing")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"minimal_flight.{key} must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"minimal_flight.{key} must be finite")
        if not (lo <= val <= hi):
            raise ValueError(f"minimal_flight.{key}={val} must be in [{lo}, {hi}]")
        params[key] = float(val)
    return params


def _merge_params(yaml_params: Dict[str, Any], cli_overrides: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """CLI overrides win over YAML. Both sources already validated."""
    merged = dict(yaml_params)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if k in merged:
                merged[k] = float(v)
    return merged


@dataclass
class AutomaticModeParams:
    """Resolved flight parameters (after YAML + CLI merge)."""
    target_z_ned: float = -1.0
    max_vertical_speed_mps: float = 0.5
    max_flight_duration_s: float = 10.0
    command_duration_s: float = 0.2
    forward_speed_mps: float = 0.2
    side_speed_mps: float = 0.15
    front_threshold_m: float = 2.5
    emergency_distance_m: float = 0.8
    geofence_radius_m: float = 2.0
    preflight_lidar_frames: int = 3
    takeoff_timeout_s: float = 20.0

    @classmethod
    def from_yaml(cls, path: str, cli_overrides: Optional[Dict[str, float]] = None) -> AutomaticModeParams:
        yp = _load_flight_config(path)
        merged = _merge_params(yp, cli_overrides)
        return cls(
            target_z_ned=merged["target_z_ned"],
            max_vertical_speed_mps=merged["max_vertical_speed_mps"],
            max_flight_duration_s=merged["max_flight_duration_s"],
            command_duration_s=merged["command_duration_s"],
            forward_speed_mps=merged["forward_speed_mps"],
            side_speed_mps=merged["side_speed_mps"],
            front_threshold_m=merged["front_threshold_m"],
            emergency_distance_m=merged["emergency_distance_m"],
            geofence_radius_m=merged["geofence_radius_m"],
            preflight_lidar_frames=int(merged["preflight_lidar_frames"]),
            takeoff_timeout_s=merged["takeoff_timeout_s"],
        )


# ── CBMBA obstacle conversion (pure function, no AirSim calls) ──


def _sector_distances_to_obstacles(
    rays: dict,
    drone_position_ned: Tuple[float, float, float],
    yaw_rad: float,
    max_range: float = 15.0,
    obstacle_radius: float = 0.8,
) -> list:
    """Convert LiDAR sector distances into CBMBA-compatible obstacle dicts.

    This is a compatibility layer — it does NOT modify the A* core.
    Each LiDAR ray that hits an obstacle (< max_range) produces one
    obstacle dict at the estimated world position.

    Sector directions (body-frame angles relative to forward/X):
        front  = 0°,  frontLeft  = -22.5°,  frontRight = +22.5°
        left   = -90°, right      = +90°
        backLeft = -157.5°,        backRight  = +157.5°
        back   = 180°

    Args:
        rays: Dict of sector_name → distance_m (from dd.to_legacy_ray_distances()).
        drone_position_ned: Drone NED position (x, y, z).
        yaw_rad: Drone yaw in radians (NED: 0=North, π/2=East).
        max_range: Max distance to consider (farther = no obstacle detected).
        obstacle_radius: Assigned obstacle radius (half-size).

    Returns:
        List of obstacle dicts suitable for CbmbaAStarPlanner.plan().
    """
    # Body-frame sector angles (radians): forward=0, right=π/2
    SECTOR_ANGLES = {
        "front": 0.0,
        "frontLeft": -math.pi / 8,        # -22.5°
        "frontRight": math.pi / 8,         # +22.5°
        "left": -math.pi / 2,              # -90°
        "right": math.pi / 2,              # +90°
        "backLeft": -math.pi * 7 / 8,      # -157.5°
        "backRight": math.pi * 7 / 8,      # +157.5°
        "back": math.pi,                    # 180°
    }

    obstacles = []
    px, py, pz = drone_position_ned

    for sector_name, distance in rays.items():
        if sector_name not in SECTOR_ANGLES:
            continue
        if not isinstance(distance, (int, float)):
            continue
        if not math.isfinite(distance):
            continue
        if distance >= max_range or distance <= 0:
            continue

        # Body-frame direction
        body_angle = SECTOR_ANGLES[sector_name]
        # Rotate by yaw to get world-frame (NED) direction
        world_angle = yaw_rad + body_angle
        dir_x = math.cos(world_angle)
        dir_y = math.sin(world_angle)

        # Estimated obstacle world position
        obs_x = px + dir_x * distance
        obs_y = py + dir_y * distance
        obs_z = pz  # LiDAR sectors are horizontal — assume at same altitude

        # Body-frame XY hit point (diagnostic — CBMBA ignores unknown keys)
        body_hit_x = math.cos(body_angle) * distance
        body_hit_y = math.sin(body_angle) * distance

        obstacles.append({
            "position": [obs_x, obs_y, obs_z],
            "footprint_half_extents": [0.0, 0.0, 0.0],
            "type": "lidar",
            "velocity": [0.0, 0.0, 0.0],
            "dynamic": False,
            "confidence": 0.7,
            # ── diagnostic metadata (not consumed by CBMBA) ──
            "_diag_sector": sector_name,
            "_diag_distance": distance,
            "_diag_body_xy": (body_hit_x, body_hit_y),
        })

    return obstacles


# ── automatic mode ──


class AutomaticMode:
    """LiDAR-based autonomous obstacle avoidance flight.

    All preflight checks execute BEFORE enableApiControl (in session.takeoff_and_climb).
    """

    def __init__(
        self,
        session: Any,
        perception_config_path: Optional[str] = None,
        flight_config_path: Optional[str] = None,
        params: Optional[AutomaticModeParams] = None,
        cli_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._session = session
        self._client = session.client
        self._adapter = session.adapter
        self._vn = session.vehicle_name

        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self._perception_config_path = (
            perception_config_path or str(_PROJECT_ROOT / "configs" / "perception.yaml")
        )
        self._flight_config_path = (
            flight_config_path or str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        )

        # Load and validate flight config
        if params is not None:
            self._params = params
        else:
            self._params = AutomaticModeParams.from_yaml(
                self._flight_config_path, cli_overrides
            )
        logger.info("Flight config loaded from %s", self._flight_config_path)

        self._running = False
        self._last_velocity_future = None

        # Planner mode: "reactive" (default) | "apf_shadow" | "apf"
        self._planner_mode = cli_overrides.get("planner_mode", "reactive") if cli_overrides else "reactive"
        logger.info("planner_mode=%s", self._planner_mode)
        self._guided_apf_control = bool(cli_overrides.get("guided_apf_control", False)) if cli_overrides else False
        logger.info("guided_apf_control=%s", self._guided_apf_control)
        from planners.improved_potential_field import ImprovedPotentialField, ApfParams
        self._apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True,
            enable_per_sector_diagnostics=False,
        ))
        from planners.local_recovery import LocalRecovery, RecoveryParams, RecoveryDecision
        self._recovery = LocalRecovery(RecoveryParams(
            history_window_s=4.0,
            stuck_time_window_s=2.5,
            stuck_position_epsilon_m=0.15,
            stuck_min_frames=10,
            oscillation_time_window_s=2.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        from planners.recovery_commander import RecoveryStateMachine
        self._recovery_sm = RecoveryStateMachine()

        # ── CBMBA A* shadow planner (compute + log only; never dispatches) ──
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams as _CbmbaParams

        # ── runtime resolution override (Phase 2G) ──
        _cbmba_resolution: float = 0.75
        if cli_overrides:
            _override_res = cli_overrides.get("cbmba_resolution")
            if _override_res is not None:
                if not math.isfinite(_override_res):
                    raise ValueError(
                        f"--cbmba-resolution must be finite, got {_override_res}"
                    )
                if _override_res <= 0:
                    raise ValueError(
                        f"--cbmba-resolution must be > 0, got {_override_res}"
                    )
                _cbmba_resolution = float(_override_res)

        self._cbmba = CbmbaAStarPlanner(_CbmbaParams(
            resolution=_cbmba_resolution,
            inflation_radius=1.5,
            max_search_nodes=2000,
            wall_penalty_radius=0,        # skip expensive proximity scan in shadow
            adaptive_long_step_cells=1,    # 26 neighbors instead of 52
        ))
        self._cbmba_enabled = True
        logger.info("cbmba_shadow  enabled=true  resolution=%.2f  max_search_nodes=%d",
                     self._cbmba.params.resolution, self._cbmba.params.max_search_nodes)

        # ── CBMBA guidance adapter (shadow only; never dispatches) ──
        from planners.cbmba_guidance import CbmbaGuidance, CbmbaGuidanceParams as _CbmbaGuidanceParams
        self._cbmba_guidance = CbmbaGuidance(_CbmbaGuidanceParams(
            min_forward_progress=0.25,
            min_waypoint_distance=0.5,
        ))
        self._cbmba_guidance_enabled = True
        logger.info("cbmba_guidance_shadow  enabled=true  min_forward_progress=%.2f  min_waypoint_distance=%.2f",
                     self._cbmba_guidance.params.min_forward_progress,
                     self._cbmba_guidance.params.min_waypoint_distance)

        # ── diagnostic rate-limit state ──
        self._diag_last_obstacle_count: int = -1
        self._diag_last_obstacle_log_time: float = -float("inf")
        self._diag_last_path_points: Optional[tuple] = None
        self._diag_last_path_log_time: float = -float("inf")

        # ── recovery test trigger (CLI only; one-shot) ──
        self._recovery_test_trigger = (
            cli_overrides.get("recovery_test_trigger") if cli_overrides else None
        )
        self._recovery_test_trigger_fired = False
        # Delay: ~1s after first airborne APF frame (≈10-15 frames at 10 Hz)
        self._recovery_test_trigger_delay_frames = 15
        if self._recovery_test_trigger is not None:
            logger.info(
                "recovery_test_trigger_enabled  type=%s  delay_frames=%d  once_per_process=true",
                self._recovery_test_trigger,
                self._recovery_test_trigger_delay_frames,
            )

    # ── public API ──

    def run(self) -> AutomaticFlightResult:
        rk: Dict[str, Any] = dict(
            success=False, termination_reason="unknown",
            frames_completed=0, flight_duration_s=0.0,
            api_control_acquired=False, armed=False,
            takeoff_completed=False, airborne=False,
            landing_confirmed=False, disarmed=False,
            api_control_released=False,
            startup_floor_contact_baseline=False,
        )

        try:
            # ────────────────────────────────────────────────
            # PREFLIGHT (BEFORE enableApiControl — all checks here)
            # ────────────────────────────────────────────────

            from sensors.lidar_reader import LidarReader
            from sensors.state_reader import StateReader
            from sensors.collision_reader import CollisionReader

            lidar = LidarReader(self._adapter)
            sr = StateReader(self._adapter, vehicle_name=self._vn)
            cr = CollisionReader(self._adapter, vehicle_name=self._vn)

            # 1. Drone1 existence (via state read)
            try:
                st0 = sr.read()
            except Exception as e:
                rk["termination_reason"] = f"preflight_state:{e}"
                return AutomaticFlightResult(**rk)

            # 2. LiDAR consecutive valid frames
            pf = self._params.preflight_lidar_frames
            for i in range(pf):
                try:
                    lf = lidar.read()
                except Exception as e:
                    rk["termination_reason"] = f"preflight_lidar_read_{i}:{e}"
                    return AutomaticFlightResult(**rk)
                if not lf.frame_valid:
                    rk["termination_reason"] = f"preflight_lidar_{i}:{lf.invalid_reason}"
                    return AutomaticFlightResult(**rk)

            # 3. Collision warm-up (before enableApiControl)
            #    First Floor/Floor_3 contact → accepted as startup candidate
            #    (even with non-zero ts or is_new_collision_event=True).
            #    Subsequent new collision events → reject.
            _WARMUP_MAX = 10
            _WARMUP_INTERVAL = 0.15
            _WARMUP_CLEAR = 5
            _FLOOR_OK = frozenset({"Floor", "Floor_3"})
            saw_floor = False
            cons_clean = 0
            saw_new_event_after_first = False
            initial_floor_ts = 0
            for i in range(_WARMUP_MAX):
                try:
                    col = cr.read()
                except Exception as e:
                    rk["termination_reason"] = f"warmup_read_error_{i}:{e}"
                    return AutomaticFlightResult(**rk)
                if col.has_collided:
                    if col.object_name not in _FLOOR_OK:
                        rk["termination_reason"] = f"warmup_non_ground_{i}:{col.object_name}"
                        return AutomaticFlightResult(**rk)
                    # Floor/Floor_3 contact
                    if not saw_floor:
                        # First floor contact — always accept as candidate
                        saw_floor = True
                        initial_floor_ts = col.raw_timestamp
                    elif col.is_new_collision_event and col.raw_timestamp != initial_floor_ts:
                        # Subsequent new collision event (different ts) → reject
                        saw_new_event_after_first = True
                    cons_clean = 0
                else:
                    cons_clean += 1
                if cons_clean >= _WARMUP_CLEAR:
                    break
                time.sleep(_WARMUP_INTERVAL)
            if saw_new_event_after_first:
                rk["termination_reason"] = "warmup_new_collision_event"
                return AutomaticFlightResult(**rk)
            if saw_floor and cons_clean < _WARMUP_CLEAR:
                rk["termination_reason"] = "warmup_floor_persists"
                return AutomaticFlightResult(**rk)
            if saw_floor:
                rk["startup_floor_contact_baseline"] = True
                logger.info("Startup floor contact baseline established (ts=%d).", initial_floor_ts)
                # Pass startup timestamp to session for landing-phase floor latch detection
                self._session.set_startup_floor_baseline(initial_floor_ts)

            # 4. FOV compatibility (NOT hardcoded)
            from perception.perception_config import load_perception_config
            from perception.sensor_fov import load_lidar_fov, validate_sector_fov_coverage

            try:
                pcfg = load_perception_config(self._perception_config_path)
            except Exception as e:
                rk["termination_reason"] = f"preflight_perception_config:{e}"
                return AutomaticFlightResult(**rk)

            try:
                fov = load_lidar_fov(self._session.settings_json, self._vn, self._adapter.lidar_name)
            except Exception as e:
                rk["termination_reason"] = f"preflight_fov_load:{e}"
                return AutomaticFlightResult(**rk)

            fov_results = validate_sector_fov_coverage(pcfg, fov)
            incompatible = [s for s in fov_results if not s.fully_observable]
            if incompatible:
                names = [s.legacy_name for s in incompatible]
                rk["termination_reason"] = f"preflight_fov_incompatible:{names}"
                return AutomaticFlightResult(**rk)

            # 5. Perception config valid (already loaded above)
            sz_cfg = pcfg.sectorization
            pc_cfg = pcfg.pointcloud
            sdefs = list(sz_cfg.sectors)

            # Build FOV observability map from real validation results
            fov_obs = {}
            for sts in fov_results:
                for sd in sdefs:
                    if sd.legacy_name == sts.legacy_name:
                        fov_obs[sd.name] = (sts.fully_observable, 1.0)

            # 6. minimal_flight.yaml already loaded via AutomaticModeParams
            logger.info("Preflight passed — all checks OK.")

            # ────────────────────────────────────────────────
            # TAKEOFF (enableApiControl happens HERE, inside session)
            # ────────────────────────────────────────────────
            self._session.takeoff_and_climb(target_z=self._params.target_z_ned)
            rk["api_control_acquired"] = True
            rk["armed"] = True
            rk["takeoff_completed"] = True
            rk["airborne"] = True
            logger.info("Airborne — starting LiDAR control loop.")

            # ── perception pipeline (post-takeoff) ──
            from perception.pointcloud_filter import filter_pointcloud
            from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
            from control.velocity_controller import VelocityController

            vc = VelocityController(
                self._adapter,
                max_horizontal_speed_mps=self._params.forward_speed_mps,
                max_vertical_speed_mps=0.0,
                command_duration_seconds=self._params.command_duration_s,
            )

            # Re-read state for spawn reference
            st0 = sr.read()
            spawn = (st0.position_ned_m[0], st0.position_ned_m[1])

            # ── Fixed mission goal (computed once; never rolls with drone) ──
            _mission_heading = st0.yaw_rad
            _mission_goal_dist = 15.0
            _mission_goal = (
                st0.position_ned_m[0] + math.cos(_mission_heading) * _mission_goal_dist,
                st0.position_ned_m[1] + math.sin(_mission_heading) * _mission_goal_dist,
                st0.position_ned_m[2],
            )
            logger.info(
                "cbmba_mission_goal  "
                "start=(%.2f,%.2f,%.2f)  "
                "goal=(%.2f,%.2f,%.2f)  "
                "distance=%.1f  "
                "heading=%.4f  "
                "fixed=true",
                st0.position_ned_m[0], st0.position_ned_m[1], st0.position_ned_m[2],
                _mission_goal[0], _mission_goal[1], _mission_goal[2],
                _mission_goal_dist,
                _mission_heading,
            )

            reactive_config = {
                "emergency_distance_m": self._params.emergency_distance_m,
                "front_threshold_m": self._params.front_threshold_m,
                "forward_speed_mps": self._params.forward_speed_mps,
                "side_speed_mps": self._params.side_speed_mps,
            }

            # ── flight loop ──
            from planners.local_recovery import RecoveryDecision as _RecoveryDecision
            t0 = time.monotonic()
            fn = 0
            term = "time_limit"

            self._running = True
            while self._running:
                fn += 1
                if time.monotonic() - t0 >= self._params.max_flight_duration_s:
                    term = "time_limit"
                    break

                try:
                    lf = lidar.read()
                    st = sr.read()
                    col = cr.read()
                except Exception:
                    term = "rpc_error"
                    break

                if not lf.frame_valid:
                    term = f"lidar_invalid:{lf.invalid_reason}"
                    break
                if col.has_collided:
                    term = f"collision:{col.object_name}"
                    break
                if math.hypot(st.position_ned_m[0] - spawn[0], st.position_ned_m[1] - spawn[1]) > self._params.geofence_radius_m:
                    term = "geofence"
                    break

                fr = filter_pointcloud(
                    lf.point_cloud_sensor,
                    min_range_m=pc_cfg.min_range_m, max_range_m=pc_cfg.max_range_m,
                    self_exclusion={
                        "enabled": pc_cfg.self_exclusion.enabled,
                        "x_min_m": pc_cfg.self_exclusion.x_min_m,
                        "x_max_m": pc_cfg.self_exclusion.x_max_m,
                        "y_min_m": pc_cfg.self_exclusion.y_min_m,
                        "y_max_m": pc_cfg.self_exclusion.y_max_m,
                        "z_min_m": pc_cfg.self_exclusion.z_min_m,
                        "z_max_m": pc_cfg.self_exclusion.z_max_m,
                    },
                    voxel_downsample=pc_cfg.voxel_downsample.enabled,
                    voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m,
                )
                if not fr.valid:
                    term = f"filter:{fr.invalid_reason}"
                    break

                try:
                    dd = pointcloud_to_directional_distances(
                        fr.filtered_points_sensor, sector_defs=sdefs,
                        default_max_range_m=sz_cfg.default_max_range_m,
                        default_min_points=sz_cfg.default_min_points,
                        distance_strategy=sz_cfg.default_distance_strategy,
                        nearest_k=sz_cfg.nearest_k, percentile=sz_cfg.percentile,
                        frame_valid=True, fov_compatible=len(incompatible) == 0,
                        fov_observability=fov_obs,
                    )
                except Exception:
                    term = "sector_error"
                    break
                if not dd.frame_valid:
                    term = f"dd:{dd.invalid_reason}"
                    break

                try:
                    rays = dd.to_legacy_ray_distances()
                except Exception:
                    term = "legacy_error"
                    break

                dec = choose_reactive_command(
                    rays.get("front", float("inf")),
                    rays.get("left", float("inf")),
                    rays.get("right", float("inf")),
                    dd.minimum_distance_m, reactive_config,
                )
                if dec.should_terminate:
                    term = dec.termination_reason
                    break

                # ── LocalRecovery shadow detection (compute + log only; no control) ──
                recovery_decision = _RecoveryDecision()    # safe default if try fails
                try:
                    # NED → body-frame velocity conversion
                    _yaw = st.yaw_rad
                    _vn, _ve, _vd = st.linear_velocity_ned_mps
                    _vx_body = _vn * math.cos(_yaw) + _ve * math.sin(_yaw)
                    _vy_body = -_vn * math.sin(_yaw) + _ve * math.cos(_yaw)
                    _vz_body = _vd

                    recovery_decision = self._recovery.update(
                        timestamp=time.monotonic(),
                        position=(st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
                        velocity_body=(_vx_body, _vy_body, _vz_body),
                        yaw_rad=_yaw,
                    )
                    logger.info(
                        "recovery_shadow  stuck=%s  oscillating=%s  needs=%s  "
                        "stuck_dur=%.2f  stuck_delta=%.3f  "
                        "position=(%.3f,%.3f,%.3f)  "
                        "oldest_position=(%.3f,%.3f,%.3f)  "
                        "osc_flips=%d  osc_lateral=%.3f  "
                        "candidates=%s  reason=%s  window=%d",
                        recovery_decision.is_stuck,
                        recovery_decision.is_oscillating,
                        recovery_decision.needs_recovery,
                        recovery_decision.stuck_duration_s,
                        recovery_decision.stuck_position_delta_m,
                        recovery_decision.stuck_latest_position[0],
                        recovery_decision.stuck_latest_position[1],
                        recovery_decision.stuck_latest_position[2],
                        recovery_decision.stuck_oldest_position[0],
                        recovery_decision.stuck_oldest_position[1],
                        recovery_decision.stuck_oldest_position[2],
                        recovery_decision.oscillation_vy_sign_flips,
                        recovery_decision.oscillation_lateral_progress_m,
                        recovery_decision.candidate_actions,
                        recovery_decision.reason,
                        recovery_decision.window_size_frames,
                    )
                except Exception as _recovery_exc:
                    logger.warning("recovery_compute_error: %s", _recovery_exc)

                # ── Recovery test trigger: one-shot synthetic injection ──
                if (self._recovery_test_trigger is not None
                        and not self._recovery_test_trigger_fired
                        and rk.get("airborne")
                        and fn >= self._recovery_test_trigger_delay_frames):
                    self._recovery_test_trigger_fired = True
                    trigger_type = self._recovery_test_trigger
                    synthetic = _RecoveryDecision(
                        is_stuck=(trigger_type == "stuck"),
                        is_oscillating=(trigger_type == "oscillation"),
                        needs_recovery=True,
                        reason=f"test_trigger:{trigger_type}",
                    )
                    logger.info(
                        "recovery_test_trigger  type=%s  "
                        "stuck=%s  oscillating=%s  needs=%s",
                        trigger_type,
                        synthetic.is_stuck,
                        synthetic.is_oscillating,
                        synthetic.needs_recovery,
                    )
                    # Inject: replace the real recovery_decision for the state machine.
                    # The shadow log above still reflects real AirSim state.
                    recovery_decision = synthetic

                # ── Recovery takeover state machine ──
                recovery_result = self._recovery_sm.tick(
                    time.monotonic(), recovery_decision, rays,
                )
                if recovery_result.event == "enter":
                    logger.info(
                        "recovery_enter  reason=%s  action=(%.3f,%.3f,%.3f)  "
                        "cmd=(%.3f,%.3f,%.3f)",
                        recovery_decision.reason,
                        recovery_result.vx_body,
                        recovery_result.vy_body,
                        recovery_result.vz_body,
                        recovery_result.vx_body,
                        recovery_result.vy_body,
                        recovery_result.vz_body,
                    )
                elif recovery_result.event == "active":
                    logger.info(
                        "recovery_active  elapsed=%.2f",
                        recovery_result.elapsed_s,
                    )
                elif recovery_result.event == "exit_timeout":
                    logger.info(
                        "recovery_exit  reason=timeout  elapsed=%.2f",
                        recovery_result.elapsed_s,
                    )
                    logger.info("handoff_to_apf")
                    logger.info(
                        "recovery_cooldown  remaining=%.2f",
                        recovery_result.cooldown_remaining_s,
                    )
                elif recovery_result.event == "cooldown_expired":
                    logger.info("recovery_cooldown  expired")
                elif recovery_result.event and recovery_result.event.startswith("exit_safety"):
                    logger.info(
                        "recovery_exit  reason=%s  elapsed=%.2f",
                        recovery_result.event, recovery_result.elapsed_s,
                    )
                    logger.info("handoff_to_apf")
                    logger.info(
                        "recovery_cooldown  remaining=%.2f",
                        recovery_result.cooldown_remaining_s,
                    )

                # ── APF computation (apf_shadow: log only; apf: control) ──
                apf_output = None
                apf_label = "apf_shadow" if self._planner_mode == "apf_shadow" else "apf_control"
                if self._planner_mode in ("apf_shadow", "apf"):
                    try:
                        from planners.improved_potential_field import ImprovedPotentialField
                        apf_output = self._apf.update(
                            sector_distances=rays,
                            sector_point_counts=None,
                            goal_body=(1.0, 0.0, 0.0),
                            current_velocity_body=(dec.vx_body_mps, dec.vy_body_mps, 0.0),
                            minimum_distance_m=dd.minimum_distance_m,
                        )
                        logger.info(
                            "%s  front=%.2f left=%.2f right=%.2f minD=%.2f  "
                            "reactive=(%.3f,%.3f,%.3f)  "
                            "attractive=(%.3f,%.3f,%.3f)  "
                            "repulsive=(%.3f,%.3f,%.3f)  "
                            "force_mag=%.3f  "
                            "apf_cmd=(%.3f,%.3f,%.3f)  cmd_mag=%.3f  "
                            "valid=%s  reason=%s  nan=%s  inf=%s  sat=%s",
                            apf_label,
                            rays.get("front", float("inf")),
                            rays.get("left", float("inf")),
                            rays.get("right", float("inf")),
                            dd.minimum_distance_m,
                            dec.vx_body_mps, dec.vy_body_mps, 0.0,
                            apf_output.attractive_force[0],
                            apf_output.attractive_force[1],
                            apf_output.attractive_force[2],
                            apf_output.repulsive_force[0],
                            apf_output.repulsive_force[1],
                            apf_output.repulsive_force[2],
                            apf_output.force_magnitude,
                            apf_output.desired_vx_body,
                            apf_output.desired_vy_body,
                            apf_output.desired_vz_body,
                            apf_output.command_magnitude,
                            apf_output.valid, apf_output.reason,
                            apf_output.nan_detected, apf_output.inf_detected,
                            apf_output.saturated,
                        )
                        # ── per-sector repulsive contributions (diagnostic) ──
                        if apf_output.per_sector_contributions:
                            for sc in apf_output.per_sector_contributions:
                                logger.info(
                                    "apf_sector  name=%-10s  dist=%.2f  "
                                    "dir=(%+.3f,%+.3f,%+.3f)  "
                                    "rep=(%+.4f,%+.4f,%+.4f)  "
                                    "used_for_control=%s",
                                    sc["name"], sc["distance"],
                                    sc["dir_x"], sc["dir_y"], sc["dir_z"],
                                    sc["rep_x"], sc["rep_y"], sc["rep_z"],
                                    sc.get("used_for_control", True),
                                )
                    except Exception as e:
                        logger.warning("apf_compute_error: %s", e)

                # ── CBMBA A* shadow (compute + log only; never dispatches) ──
                if self._cbmba_enabled:
                    try:
                        cbmba_start = [
                            st.position_ned_m[0],
                            st.position_ned_m[1],
                            st.position_ned_m[2],
                        ]
                        # Fixed mission goal (computed once at airborne; never rolls)
                        cbmba_goal = [
                            _mission_goal[0],
                            _mission_goal[1],
                            _mission_goal[2],
                        ]
                        # Build synthetic obstacles from LiDAR sector distances
                        cbmba_obstacles = _sector_distances_to_obstacles(
                            rays, st.position_ned_m, _yaw,
                        )
                        # ── diagnostic: obstacle samples (count change or ≤1 Hz) ──
                        _now_abs = time.monotonic()
                        _obs_count = len(cbmba_obstacles)
                        _obs_changed = _obs_count != self._diag_last_obstacle_count
                        _obs_stale = (_now_abs - self._diag_last_obstacle_log_time) >= 1.0
                        if cbmba_obstacles and (_obs_changed or _obs_stale or fn == 1):
                            _sample_parts = []
                            for _obs in cbmba_obstacles[:12]:  # cap per log line
                                _sec = _obs.get("_diag_sector", "?")
                                _dist = _obs.get("_diag_distance", -1.0)
                                _bxy = _obs.get("_diag_body_xy", (0.0, 0.0))
                                _wpos = _obs["position"]
                                _sample_parts.append(
                                    f"{{sector={_sec} dist={_dist:.2f} "
                                    f"body=({_bxy[0]:.2f},{_bxy[1]:.2f}) "
                                    f"world=({_wpos[0]:.2f},{_wpos[1]:.2f}) "
                                    f"footprint={_obs['footprint_half_extents']}}}"
                                )
                            logger.info(
                                "cbmba_obstacles  "
                                "count=%d  "
                                "samples=[%s]  "
                                "planner_inflation=%.1f  "
                                "effective_extent=%.1f",
                                _obs_count,
                                "  ".join(_sample_parts) if _sample_parts else "none",
                                self._cbmba.params.inflation_radius,
                                self._cbmba.params.inflation_radius,
                            )
                            self._diag_last_obstacle_count = _obs_count
                            self._diag_last_obstacle_log_time = _now_abs
                        elif not cbmba_obstacles and _obs_changed:
                            logger.info(
                                "cbmba_obstacles  count=0  samples=[]  "
                                "planner_inflation=%.1f  effective_extent=%.1f",
                                self._cbmba.params.inflation_radius,
                                self._cbmba.params.inflation_radius,
                            )
                            self._diag_last_obstacle_count = 0
                            self._diag_last_obstacle_log_time = _now_abs

                        cbmba_result = self._cbmba.plan_with_result(
                            cbmba_obstacles, cbmba_start, cbmba_goal,
                        )

                        # ── diagnostic: CBMBA path XY (material change or ≤1 Hz) ──
                        if cbmba_result.success and len(cbmba_result.path_world) >= 2:
                            _path_xy = tuple(
                                (round(p[0], 2), round(p[1], 2))
                                for p in cbmba_result.path_world
                            )
                            _path_changed = _path_xy != self._diag_last_path_points
                            _path_stale = (_now_abs - self._diag_last_path_log_time) >= 1.0
                            if _path_changed or _path_stale:
                                _xs = [p[0] for p in cbmba_result.path_world]
                                _ys = [p[1] for p in cbmba_result.path_world]
                                _pt_strs = [f"({p[0]:.2f},{p[1]:.2f})" for p in cbmba_result.path_world]
                                logger.info(
                                    "cbmba_path_xy  "
                                    "points=[%s]  "
                                    "min_x=%.2f  max_x=%.2f  "
                                    "min_y=%.2f  max_y=%.2f",
                                    " ".join(_pt_strs),
                                    min(_xs), max(_xs),
                                    min(_ys), max(_ys),
                                )
                                self._diag_last_path_points = _path_xy
                                self._diag_last_path_log_time = _now_abs
                        logger.info(
                            "cbmba_shadow  success=%s  nodes=%d  path_len=%d  "
                            "grid_size=%d  time_ms=%.2f  "
                            "start=(%.2f,%.2f,%.2f)  goal=(%.2f,%.2f,%.2f)  "
                            "num_obstacles=%d  fixed=true",
                            cbmba_result.success,
                            cbmba_result.nodes_expanded,
                            len(cbmba_result.path_world),
                            cbmba_result.grid_size,
                            cbmba_result.planning_time_ms,
                            cbmba_start[0], cbmba_start[1], cbmba_start[2],
                            cbmba_goal[0], cbmba_goal[1], cbmba_goal[2],
                            len(cbmba_obstacles),
                        )
                        # Log path shape for diagnostics
                        if cbmba_result.success and len(cbmba_result.path_world) >= 2:
                            wp_first = cbmba_result.path_world[0]
                            wp_last = cbmba_result.path_world[-1]
                            # ── next = first waypoint meaningfully different from start ──
                            _eps = 0.05
                            wp_next = wp_first
                            for _pt in cbmba_result.path_world[1:]:
                                if (abs(_pt[0] - wp_first[0]) > _eps
                                        or abs(_pt[1] - wp_first[1]) > _eps
                                        or abs(_pt[2] - wp_first[2]) > _eps):
                                    wp_next = _pt
                                    break
                            # ── max_lateral_dev = max perpendicular distance from start→goal XY line ──
                            _sx, _sy = cbmba_start[0], cbmba_start[1]
                            _gx, _gy = cbmba_goal[0], cbmba_goal[1]
                            _seg_dx = _gx - _sx
                            _seg_dy = _gy - _sy
                            _seg_len = math.hypot(_seg_dx, _seg_dy)
                            _max_dev = 0.0
                            if _seg_len > 1e-6:
                                _max_dev = max(
                                    abs((_pt[0] - _sx) * _seg_dy - (_pt[1] - _sy) * _seg_dx) / _seg_len
                                    for _pt in cbmba_result.path_world
                                )
                            logger.info(
                                "cbmba_path  waypoints=%d  "
                                "first=(%.2f,%.2f,%.2f)  "
                                "next=(%.2f,%.2f,%.2f)  "
                                "last=(%.2f,%.2f,%.2f)  "
                                "max_lateral_dev=%.3f",
                                len(cbmba_result.path_world),
                                wp_first[0], wp_first[1], wp_first[2],
                                wp_next[0], wp_next[1], wp_next[2],
                                wp_last[0], wp_last[1], wp_last[2],
                                _max_dev,
                            )
                    except Exception as _cbmba_exc:
                        logger.warning("cbmba_compute_error: %s", _cbmba_exc)

                # ── CBMBA guidance shadow (segment-crossing; never dispatches) ──
                _guidance_result = None  # saved for guided APF shadow below
                if self._cbmba_enabled and self._cbmba_guidance_enabled:
                    try:
                        _cbmba_path = getattr(self._cbmba, "last_path", None)
                        if _cbmba_path and len(_cbmba_path) >= 2:
                            _guidance_result = self._cbmba_guidance.select_waypoint(
                                (st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
                                _yaw,
                                _cbmba_path,
                            )
                            if _guidance_result.valid:
                                _seg = _guidance_result.source_segment
                                _seg_str = f"({_seg[0]},{_seg[1]})" if _seg else "none"
                                logger.info(
                                    "cbmba_guidance_shadow  valid=true  "
                                    "source_segment=%s  "
                                    "interpolated=%s  "
                                    "body_target=(%.2f,%.2f)  "
                                    "forward_progress=%.2f  "
                                    "lateral_offset=%.2f  "
                                    "direction=(%.3f,%.3f)  "
                                    "reason=%s",
                                    _seg_str,
                                    "true" if _guidance_result.interpolated else "false",
                                    _guidance_result.target_body_xy[0],
                                    _guidance_result.target_body_xy[1],
                                    _guidance_result.forward_progress_m,
                                    _guidance_result.lateral_offset_m,
                                    _guidance_result.direction_body_xy[0],
                                    _guidance_result.direction_body_xy[1],
                                    _guidance_result.reason,
                                )
                            else:
                                logger.info(
                                    "cbmba_guidance_shadow  valid=false  reason=%s",
                                    _guidance_result.reason,
                                )
                    except Exception as _guid_exc:
                        logger.warning("cbmba_guidance_error: %s", _guid_exc)

                # ── Guided APF lateral shadow (CBMBA → lateral attractive bias) ──
                _guided_output = None
                _g_cmd = (0.0, 0.0)
                _n_cmd = (0.0, 0.0)
                _guide_valid_flag = False
                _fallback_reason = ""
                if apf_output is not None and apf_output.valid:
                    try:
                        _lateral_bias = 0.0
                        _guide_valid_flag = False
                        _guide_dir_x, _guide_dir_y = 1.0, 0.0
                        if _guidance_result is not None and _guidance_result.valid:
                            _gdx, _gdy = _guidance_result.direction_body_xy
                            if (math.isfinite(_gdx) and math.isfinite(_gdy)
                                    and (_gdx != 0.0 or _gdy != 0.0)):
                                _guide_dir_x = _gdx
                                _guide_dir_y = _gdy
                                # Lateral bias bounded by ±attractive_gain
                                # because |guidance_direction_y| ≤ 1 (unit vector)
                                _lateral_bias = self._apf._params.attractive_gain * _gdy
                                _guide_valid_flag = True

                        _guided_output = self._apf.update(
                            sector_distances=rays,
                            sector_point_counts=None,
                            goal_body=(1.0, 0.0, 0.0),
                            current_velocity_body=(dec.vx_body_mps, dec.vy_body_mps, 0.0),
                            minimum_distance_m=dd.minimum_distance_m,
                            lateral_guidance_bias=_lateral_bias,
                        )

                        _n_att = apf_output.attractive_force
                        _g_att = _guided_output.attractive_force
                        _rep = apf_output.repulsive_force
                        _n_cmd = (apf_output.desired_vx_body, apf_output.desired_vy_body)
                        _g_cmd = (_guided_output.desired_vx_body, _guided_output.desired_vy_body)
                        _delta_x = _g_cmd[0] - _n_cmd[0]
                        _delta_y = _g_cmd[1] - _n_cmd[1]
                        _forward_preserved = abs(_g_att[0] - _n_att[0]) < 1e-9

                        logger.info(
                            "guided_apf_lateral_shadow  "
                            "guidance_valid=%s  "
                            "guidance_direction=(%.3f,%.3f)  "
                            "normal_attractive=(%.3f,%.3f)  "
                            "guided_attractive=(%.3f,%.3f)  "
                            "repulsive=(%.3f,%.3f)  "
                            "normal_cmd=(%.3f,%.3f)  "
                            "guided_cmd=(%.3f,%.3f)  "
                            "cmd_delta=(%.3f,%.3f)  "
                            "forward_preserved=%s  "
                            "guided_cmd_mag=%.3f  "
                            "valid=%s  reason=%s",
                            "true" if _guide_valid_flag else "false",
                            _guide_dir_x, _guide_dir_y,
                            _n_att[0], _n_att[1],
                            _g_att[0], _g_att[1],
                            _rep[0], _rep[1],
                            _n_cmd[0], _n_cmd[1],
                            _g_cmd[0], _g_cmd[1],
                            _delta_x, _delta_y,
                            "true" if _forward_preserved else "false",
                            _guided_output.command_magnitude,
                            _guided_output.valid,
                            _guided_output.reason,
                        )
                    except Exception as _gapf_exc:
                        logger.warning("guided_apf_lateral_shadow_error: %s", _gapf_exc)

                # ── Command dispatch ──
                # Priority (highest first):
                #   1. Safety: collision / geofence / emergency → break (already handled above)
                #   2. Recovery: if state machine says should_override
                #   3. Guided APF (opt-in + conditions met)
                #   4. Normal APF fallback
                #   5. Reactive
                if recovery_result.should_override:
                    selected_vx = recovery_result.vx_body
                    selected_vy = recovery_result.vy_body
                    selected_vz = recovery_result.vz_body
                    command_source = "recovery"
                elif self._planner_mode == "apf":
                    # ── Guided APF takeover evaluation ──
                    _takeover = False
                    if self._guided_apf_control:
                        if (_guide_valid_flag
                                and _guided_output is not None
                                and _guided_output.valid
                                and math.isfinite(_g_cmd[0])
                                and math.isfinite(_g_cmd[1])):
                            # forward_sign_guard: if normal pushes forward but
                            # guided pushes backward, refuse takeover
                            if _n_cmd[0] > 0.0 and _g_cmd[0] < 0.0:
                                _fallback_reason = "forward_sign_guard"
                            else:
                                _takeover = True
                        else:
                            if not _guide_valid_flag:
                                _fallback_reason = "guidance_invalid"
                            elif _guided_output is None:
                                _fallback_reason = "guided_unavailable"
                            elif not _guided_output.valid:
                                _fallback_reason = "guided_invalid"
                            else:
                                _fallback_reason = "guided_nan_inf"

                    if _takeover:
                        selected_vx = _g_cmd[0]
                        selected_vy = _g_cmd[1]
                        selected_vz = 0.0
                        command_source = "guided_apf"
                    elif apf_output is not None and apf_output.valid:
                        selected_vx = apf_output.desired_vx_body
                        selected_vy = apf_output.desired_vy_body
                        selected_vz = apf_output.desired_vz_body
                        command_source = "apf"
                    else:
                        selected_vx = selected_vy = selected_vz = 0.0
                        command_source = "apf_invalid_hold"

                    # ── Guided APF takeover log (only when feature enabled) ──
                    if self._guided_apf_control:
                        logger.info(
                            "guided_apf_takeover  "
                            "enabled=true  "
                            "guidance_valid=%s  "
                            "guided_valid=%s  "
                            "normal_cmd=(%.4f,%.4f)  "
                            "guided_cmd=(%.4f,%.4f)  "
                            "source=%s  "
                            "fallback_reason=%s",
                            "true" if _guide_valid_flag else "false",
                            "true" if (_guided_output is not None and _guided_output.valid) else "false",
                            _n_cmd[0], _n_cmd[1],
                            _g_cmd[0], _g_cmd[1],
                            command_source,
                            _fallback_reason if not _takeover else "",
                        )
                else:
                    # reactive or apf_shadow: reactive commands the drone
                    selected_vx = dec.vx_body_mps
                    selected_vy = dec.vy_body_mps
                    selected_vz = 0.0
                    command_source = "reactive"

                logger.info(
                    "control_dispatch  planner_mode=%s  source=%s  "
                    "cmd=(%.4f,%.4f,%.4f)  api=moveByVelocityBodyFrameAsync",
                    self._planner_mode, command_source,
                    selected_vx, selected_vy, selected_vz,
                )

                try:
                    self._last_velocity_future = vc.send_velocity_body_frd(
                        selected_vx, selected_vy, selected_vz,
                        duration=self._params.command_duration_s,
                        vehicle_name=self._vn,
                    )
                except Exception:
                    term = "velocity_send_error"
                    break

                time.sleep(self._params.command_duration_s)

            # ── loop exited — stop producing commands, join last future ──
            logger.info("last_velocity_future_wait_started")
            if self._last_velocity_future is not None:
                try:
                    self._last_velocity_future.join()
                    logger.info("last_velocity_future_wait_completed")
                except Exception as e:
                    logger.warning("last_velocity_future_join_error: %s", e)
                self._last_velocity_future = None

            rk.update(termination_reason=term, frames_completed=fn,
                       flight_duration_s=time.monotonic() - t0)

        except KeyboardInterrupt:
            rk["termination_reason"] = "ctrl_c"
            logger.info("Ctrl+C.")
        except Exception as e:
            rk["termination_reason"] = f"exception:{e}"
            logger.exception("Unhandled exception.")
        finally:
            self._running = False
            # Cleanup is owned by CLI's finally → session.cleanup() → land_and_disarm()
            # automatic_mode only reports termination_reason, no independent landing.
            rk["success"] = (rk["termination_reason"] == "time_limit")

        return AutomaticFlightResult(**rk)

    def stop(self) -> None:
        self._running = False

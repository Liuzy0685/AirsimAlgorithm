#!/usr/bin/env python
"""ROUND 4.1 — Read-only local avoidance dry-run.  SENDS NO FLIGHT COMMANDS.

Pipeline:
  Connect → FOV check → Warm-up (5 frames, no commands) →
  Collision clear check (5 frames) → Main loop:
    Read state + LiDAR + collision → Filter → Sectorize → Transform to NED →
    Compute goal → APF (once) → Supervisor → SafetySupervisor →
    Print HOLD / proposed command.

Fixes from ROUND 4:
  - APF returns diagnostics dict; dry-run reads it directly (no _vec3 hack).
  - Transform fail-closed: missing/bad sensor_pose → HOLD.
  - Data sync checks: monotonic receive times, skew + age limits.
  - Consecutive invalid reaches threshold → terminate dry-run.
  - Full per-frame output: attract, repulse, tangential, APF speed, mode, recovery, safety.
"""
from __future__ import annotations
import argparse, logging, math, signal, sys, time, yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from configs.runtime_config import load_lidar_runtime_config
from perception.perception_config import load_perception_config
from perception.pointcloud_filter import filter_pointcloud
from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
from perception.sensor_fov import (
    load_lidar_fov, validate_sector_fov_coverage, check_max_range_against_fov,
)
from sensors.lidar_reader import LidarReader
from sensors.state_reader import StateReader
from sensors.collision_reader import CollisionReader
from transforms.lidar_to_local_ned import sensor_to_local_ned
from planning.fixed_local_goal import compute_fixed_local_goal
from planning.avoidance_supervisor import AvoidanceSupervisor
from control.safety_supervisor import SafetySupervisor
from models.local_planner_command import LocalPlannerCommand, invalid_command
from models.directional_distances import DirectionalDistances

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dry_run")


def _fail(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def _validate_sensor_pose(sensor_pose: Optional[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Validate sensor_pose from LidarFrame. Returns (position, orientation_xyzw, error).

    Contract: LidarReader produces sensor_pose with keys x/y/z/w (NOT x_val/y_val/etc).
    Fail-closed with specific error codes.
    """
    if sensor_pose is None or not isinstance(sensor_pose, dict):
        return None, None, "sensor_pose_missing"

    pos = sensor_pose.get("position")
    if not isinstance(pos, dict):
        return None, None, "sensor_pose_position_missing"

    orient = sensor_pose.get("orientation")
    if not isinstance(orient, dict):
        return None, None, "sensor_pose_orientation_missing"

    # Read each field with explicit error
    for field, src in [("x", pos), ("y", pos), ("z", pos),
                        ("x", orient), ("y", orient), ("z", orient), ("w", orient)]:
        if field not in src:
            return None, None, f"sensor_pose_field_missing:{field}"
        v = src[field]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, None, f"sensor_pose_field_not_numeric:{field}"
        if np.isnan(v) or np.isinf(v):
            return None, None, "sensor_pose_nonfinite"

    px, py, pz = float(pos["x"]), float(pos["y"]), float(pos["z"])
    qx, qy, qz, qw = float(orient["x"]), float(orient["y"]), float(orient["z"]), float(orient["w"])

    pos_arr = np.array([px, py, pz], dtype=np.float64)
    orient_arr = np.array([qx, qy, qz, qw], dtype=np.float64)

    q_norm = np.linalg.norm(orient_arr)
    if q_norm < 1e-15:
        return None, None, "sensor_pose_zero_quaternion"
    orient_arr = orient_arr / q_norm
    return pos_arr, orient_arr, None


def _record_frame_times(lf, st, col):
    """Record monotonic receive times for all sensors from their data model fields."""
    return {
        "lidar_mono": lf.received_monotonic_seconds if lf else 0.0,
        "state_mono": st.received_monotonic_seconds if st else 0.0,
        "collision_mono": col.received_monotonic_seconds if col else 0.0,
    }


def _check_data_sync(times: Dict, max_skew: float, max_age: float, now: Optional[float] = None) -> Optional[str]:
    """Check sensor data sync. Returns error string or None.

    - skew: max(times) - min(times) must be <= max_skew
    - age: now - min(times) must be <= max_age (check OLDEST data)
    - any time missing, non-finite, or <= 0 → data_timestamp_missing
    """
    if now is None:
        now = time.monotonic()
    values = list(times.values())
    if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        return "data_timestamp_missing"
    t_min = min(values)
    t_max = max(values)
    if t_max - t_min > max_skew:
        return f"data_sync_error: sensor skew {t_max - t_min:.3f}s > {max_skew}s"
    if now - t_min > max_age:
        return f"data_stale: oldest sensor age {now - t_min:.3f}s > {max_age}s"
    return None


def main():
    p = argparse.ArgumentParser(description="ROUND 4.1 — Read-only local avoidance dry-run")
    p.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "vehicle.yaml"))
    p.add_argument("--perception-config", default=str(_PROJECT_ROOT / "configs" / "perception.yaml"))
    p.add_argument("--planner-config", default=str(_PROJECT_ROOT / "configs" / "local_planner.yaml"))
    p.add_argument("--settings-json", required=True, help="Path to AirSim settings.json")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--interval", type=float, default=0.2)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    with open(args.planner_config, "r", encoding="utf-8") as fh:
        planner_cfg = yaml.safe_load(fh) or {}
    sync_cfg = planner_cfg.get("synchronization", {}) or {}
    max_skew = float(sync_cfg.get("max_sensor_state_skew_seconds", 0.1))
    max_age = float(sync_cfg.get("max_data_age_seconds", 0.5))
    max_invalid = int(planner_cfg.get("safety", {}).get("max_consecutive_invalid", 10))
    max_safety_holds = int(planner_cfg.get("safety", {}).get("max_consecutive_safety_holds", 0))
    # 0 means no limit on safety holds in dry-run (only data errors terminate)

    # ── Steps 1-4: Config, FOV, Connect (unchanged) ──
    try:
        rt = load_lidar_runtime_config(args.config)
    except Exception as e:
        _fail(f"Vehicle config: {e}")
    v_name, l_name = rt.airsim.vehicle_name, rt.airsim.lidar_name

    try:
        pcfg = load_perception_config(args.perception_config)
    except Exception as e:
        _fail(f"Perception config: {e}")

    sp = Path(args.settings_json)
    if not sp.is_file():
        _fail(f"Settings JSON not found: {sp}")
    try:
        fov = load_lidar_fov(str(sp), v_name, l_name)
    except Exception as e:
        _fail(f"FOV load: {e}")
    print(f"LiDAR FOV: V=[{fov.vertical_lower_deg},{fov.vertical_upper_deg}], R={fov.range_m}m")
    range_errs = check_max_range_against_fov(pcfg, fov)
    if range_errs:
        for e in range_errs: print(f"  RANGE ERROR: {e}", file=sys.stderr)
        _fail("max_range exceeds LiDAR Range")
    fov_stats = validate_sector_fov_coverage(pcfg, fov)
    required = {s.legacy_name for s in pcfg.sectorization.sectors}
    unobs = [s.legacy_name for s in fov_stats if s.legacy_name in required and not s.fully_observable]
    if unobs:
        print(f"FOV INCOMPATIBLE: {unobs}", file=sys.stderr)
        sys.exit(1)
    print("FOV FULLY COMPATIBLE\n")

    fov_obs = {}
    for s in fov_stats:
        for sd in pcfg.sectorization.sectors:
            if sd.legacy_name == s.legacy_name:
                fov_obs[sd.name] = (s.fully_observable, min(s.horizontal_coverage_fraction, s.vertical_coverage_fraction))
                break

    adapter = AirSimClientAdapter(config_path=args.config, readonly=True)
    try: adapter.connect()
    except Exception as e: _fail(f"Connect: {e}")
    try:
        vehicles = adapter.list_vehicles()
        if v_name not in vehicles: _fail(f"{v_name!r} not in {vehicles}")
    except Exception as e: _fail(f"listVehicles: {e}")

    lidar = LidarReader(adapter, frame_timeout_seconds=rt.frame_timeout_seconds)
    state_reader = StateReader(adapter, vehicle_name=v_name)
    collision_reader = CollisionReader(adapter, vehicle_name=v_name)

    # ── Warm-up ──
    warmup = planner_cfg.get("warmup", {}).get("frames", 5)
    print(f"Warm-up: {warmup} frames...")
    for i in range(warmup):
        try:
            lf = lidar.read(); st = state_reader.read(); col = collision_reader.read()
        except Exception as e:
            print(f"  Warm-up {i+1}: RPC error: {e}")
            time.sleep(args.interval)
            continue
        valid_str = "OK" if lf.frame_valid else f"INVALID: {lf.invalid_reason}"
        print(f"  Warm-up {i+1}: {lf.point_count} pts, {valid_str}")
        time.sleep(args.interval)

    # ── Collision baseline ──
    clear_needed = planner_cfg.get("warmup", {}).get("collision_clear_frames", 5)
    print(f"\nCollision baseline: need {clear_needed} clear frames...")
    col0 = collision_reader.read()
    if col0.has_collided:
        print(f"  INITIAL COLLISION: obj={col0.object_name} "
              f"pos=({col0.position_ned_m[0]:.2f},{col0.position_ned_m[1]:.2f},{col0.position_ned_m[2]:.2f}) "
              f"pen={col0.penetration_depth:.3f}m")
    clear_count = 0
    while clear_count < clear_needed:
        try: col = collision_reader.read()
        except Exception as e: print(f"  Collision read error: {e}"); time.sleep(args.interval); continue
        if col.has_collided:
            print(f"  COLLISION during baseline: {col.object_name} — ABORT")
            _fail("Collision during baseline")
        clear_count += 1
        print(f"  Clear {clear_count}/{clear_needed}")
        time.sleep(args.interval)
    print("Collision baseline established.\n")

    # ── Goal (one-time) ──
    st0 = state_reader.read()
    goal_ned, goal_desc = compute_fixed_local_goal(
        planner_cfg,
        (st0.position_ned_m[0], st0.position_ned_m[1], st0.position_ned_m[2]),
        st0.yaw_rad,
    )
    print(f"Fixed goal: {goal_desc} → NED({goal_ned[0]:.2f},{goal_ned[1]:.2f},{goal_ned[2]:.2f})\n")

    supervisor = AvoidanceSupervisor(planner_cfg)
    safety = SafetySupervisor(planner_cfg)

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None
    def _w(line):
        print(line)
        if out_fh: out_fh.write(line + "\n"); out_fh.flush()

    _w(f"{'Frm':>4s} | {'pos_x':>7s} | {'pos_y':>7s} | {'pos_z':>7s} | "
       f"{'goal_x':>7s} | {'goal_y':>7s} | {'goal_z':>7s} | "
       f"{'raw':>5s} | {'flt':>5s} | {'minD':>6s} | "
       f"{'front':>7s} | {'fL':>7s} | {'fR':>7s} | {'left':>7s} | "
       f"{'right':>7s} | {'up':>7s} | {'down':>7s} | {'back':>7s} | "
       f"{'vx':>7s} | {'vy':>7s} | {'vz':>7s} | {'valid':>5s}")
    _w("-" * 200)

    running = True
    def _sig(_, __):
        nonlocal running; _w("\nCtrl+C — stopping."); running = False
    signal.signal(signal.SIGINT, _sig)

    consecutive_invalid_data = 0   # sensor/data errors only → termination
    safety_hold_count = 0           # safety rejections → logged, not terminating in dry-run
    valid_command_count = 0         # commands that passed all checks
    frame_num = 0
    pc_cfg = pcfg.pointcloud
    sz_cfg = pcfg.sectorization
    sector_defs = list(sz_cfg.sectors)

    while running and frame_num < args.frames:
        t0 = time.monotonic()
        frame_num += 1

        # ── Read sensors ──
        try:
            lf = lidar.read()
            st = state_reader.read()
            col = collision_reader.read()
        except Exception as e:
            _w(f"{frame_num:4d} | RPC error: {e} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid:
                _w(f"MAX INVALID ({consecutive_invalid_data}/{max_invalid}) — terminating dry-run.")
                break
            _sleep(t0, args.interval)
            continue

        # ── Data sync check ──
        times = _record_frame_times(lf, st, col)
        sync_err = _check_data_sync(times, max_skew, max_age)
        if sync_err:
            _w(f"{frame_num:4d} | {sync_err} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid:
                _w(f"MAX INVALID ({consecutive_invalid_data}/{max_invalid}) — terminating.")
                break
            _sleep(t0, args.interval)
            continue

        # ── Collision ──
        if col.has_collided:
            _w(f"{frame_num:4d} | COLLISION: {col.object_name} pen={col.penetration_depth:.3f}m → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid:
                _w(f"MAX INVALID ({consecutive_invalid_data}/{max_invalid}) — terminating.")
                break
            _sleep(t0, args.interval)
            continue

        # ── LiDAR ──
        if not lf.frame_valid:
            _w(f"{frame_num:4d} | LiDAR INVALID: {lf.invalid_reason} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid:
                _w(f"MAX INVALID ({consecutive_invalid_data}/{max_invalid}) — terminating.")
                break
            _sleep(t0, args.interval)
            continue

        # ── Filter ──
        fr = filter_pointcloud(
            lf.point_cloud_sensor,
            min_range_m=pc_cfg.min_range_m, max_range_m=pc_cfg.max_range_m,
            self_exclusion={
                "enabled": pc_cfg.self_exclusion.enabled,
                "x_min_m": pc_cfg.self_exclusion.x_min_m, "x_max_m": pc_cfg.self_exclusion.x_max_m,
                "y_min_m": pc_cfg.self_exclusion.y_min_m, "y_max_m": pc_cfg.self_exclusion.y_max_m,
                "z_min_m": pc_cfg.self_exclusion.z_min_m, "z_max_m": pc_cfg.self_exclusion.z_max_m,
            },
            voxel_downsample=pc_cfg.voxel_downsample.enabled,
            voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m,
        )
        if not fr.valid:
            _w(f"{frame_num:4d} | FILTER INVALID: {fr.invalid_reason} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        # ── Sectorize ──
        try:
            dd = pointcloud_to_directional_distances(
                fr.filtered_points_sensor, sector_defs=sector_defs,
                default_max_range_m=sz_cfg.default_max_range_m,
                default_min_points=sz_cfg.default_min_points,
                distance_strategy=sz_cfg.default_distance_strategy,
                nearest_k=sz_cfg.nearest_k, percentile=sz_cfg.percentile,
                frame_valid=lf.frame_valid,
                raw_timestamp_ns=lf.raw_timestamp_ns,
                received_monotonic_seconds=lf.received_monotonic_seconds,
                fov_compatible=True, fov_invalid_sectors=(),
                fov_observability=fov_obs,
            )
        except Exception as e:
            _w(f"{frame_num:4d} | SECTOR ERROR: {e} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        if not dd.frame_valid:
            _w(f"{frame_num:4d} | DD INVALID: {dd.invalid_reason} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        # ── Legacy ray distances ──
        try:
            ray_dists = dd.to_legacy_ray_distances()
        except Exception as e:
            _w(f"{frame_num:4d} | LEGACY ERROR: {e} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        # ── Transform: fail-closed ──
        pos_arr, orient_arr, pose_err = _validate_sensor_pose(lf.sensor_pose)
        if pose_err is not None:
            _w(f"{frame_num:4d} | POSE ERROR: {pose_err} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        try:
            pc_ned = sensor_to_local_ned(fr.filtered_points_sensor, pos_arr, orient_arr)
        except Exception as e:
            _w(f"{frame_num:4d} | TRANSFORM ERROR: {e} → HOLD")
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid: _w(f"MAX INVALID — terminating."); break
            _sleep(t0, args.interval)
            continue

        # ── Build observation ──
        obs = {
            "ego": {
                "position": [st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]],
                "orientation": [st.roll_rad, st.pitch_rad, st.yaw_rad],
                "linearVelocity": st.linear_velocity_ned_mps,
                "angularVelocity": st.angular_velocity_body_radps,
            },
            "goal": list(goal_ned),
            "globalPath": [list(st.position_ned_m), list(goal_ned)],
            "staticObstacles": [],
            "dynamicObstacles": [],
            "localPointCloud": pc_ned.astype(np.float64),
            "dt": args.interval,
            "timestamp": time.monotonic(),
            "collision": {
                "hasPhysicalContact": col.has_collided,
                "isColliding": col.has_collided,
                "minDistance": col.penetration_depth if col.has_collided else float("inf"),
            },
        }

        # ── APF (called ONCE) ──
        apf_result = supervisor.apf.update(obs)
        diag = apf_result.get("diagnostics", {})

        # ── Supervisor (uses same frame, does NOT call APF again) ──
        sv_result = supervisor.update(obs, cruise_setpoint=None, ray_distances=ray_dists, pre_computed_apf_result=apf_result)

        # ── Safety ──
        proposed_v = sv_result["velocity_world_ned_mps"]
        proposed_yr = sv_result.get("yaw_rate_radps")
        cmd = safety.validate(
            proposed_v, proposed_yr, sv_result["source"],
            lf, dd, col, fov_compatible=True,
            consecutive_invalid=consecutive_invalid_data,
            data_sync_valid=(sync_err is None),
            data_sync_reason=sync_err,
            obstacle_positions_ned=pc_ned if pc_ned.size > 0 else None,
            ego_position_ned=(st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
        )

        if cmd.command_valid:
            consecutive_invalid_data = 0
            valid_command_count += 1
        else:
            # Distinguish sensor/data errors from safety holds
            reason = cmd.invalid_reason or ""
            if "toward obstacle" in reason:
                # Safety hold — log but don't count as data error
                safety_hold_count += 1
            else:
                consecutive_invalid_data += 1

        # ── Output ──
        def _dk(k):
            return f"{ray_dists.get(k, float('inf')):6.2f}" if k in ray_dists else "    NA"
        vx, vy, vz = proposed_v
        _w(f"{frame_num:4d} | {st.position_ned_m[0]:7.2f} | {st.position_ned_m[1]:7.2f} | {st.position_ned_m[2]:7.2f} | "
           f"{goal_ned[0]:7.2f} | {goal_ned[1]:7.2f} | {goal_ned[2]:7.2f} | "
           f"{lf.point_count:5d} | {fr.output_point_count:5d} | {dd.minimum_distance_m:6.2f} | "
           f"{_dk('front')} | {_dk('frontLeft')} | {_dk('frontRight')} | {_dk('left')} | "
           f"{_dk('right')} | {_dk('up')} | {_dk('down')} | {_dk('back')} | "
           f"{vx:7.2f} | {vy:7.2f} | {vz:7.2f} | "
           f"{'OK' if cmd.command_valid else ('SAFETY_HOLD' if ('toward obstacle' in (cmd.invalid_reason or '')) else 'DATA_ERR')}")
        if not cmd.command_valid:
            _w(f"  → HOLD: {cmd.invalid_reason}")

        # Diagnostics
        att = diag.get("attractive_force_world_ned", (0.0, 0.0, 0.0))
        rep = diag.get("repulsive_force_world_ned", (0.0, 0.0, 0.0))
        tan = diag.get("tangential_force_world_ned", (0.0, 0.0, 0.0))
        apf_speed = math.hypot(vx, vy, vz)
        _w(f"  APF: attract=({att[0]:.2f},{att[1]:.2f},{att[2]:.2f}) "
           f"repulse=({rep[0]:.2f},{rep[1]:.2f},{rep[2]:.2f}) "
           f"tangent=({tan[0]:.2f},{tan[1]:.2f},{tan[2]:.2f}) "
           f"speed={apf_speed:.2f}")
        _w(f"  mode={supervisor.mode} recovery={sv_result['source']} "
           f"bypass_sign={diag.get('bypass_sign','?')} "
           f"nearest_obs={diag.get('nearest_obstacle_distance_m','?')} "
           f"dominant={diag.get('dominant_obstacle_id','?')} "
           f"loop={((time.monotonic()-t0)*1000):.1f}ms")

        if consecutive_invalid_data >= max_invalid:
            _w(f"MAX DATA INVALID ({consecutive_invalid_data}/{max_invalid}) — terminating dry-run.")
            break

        if max_safety_holds > 0 and safety_hold_count >= max_safety_holds:
            _w(f"MAX SAFETY HOLDS ({safety_hold_count}/{max_safety_holds}) — terminating dry-run.")
            break

        _sleep(t0, args.interval)

    _w(f"\n{frame_num} frames. Done. No flight commands sent.")
    _w(f"Summary: valid_command={valid_command_count}, safety_hold={safety_hold_count}, invalid_data={consecutive_invalid_data}")

    if out_fh: out_fh.close()
    sys.exit(0)


def _sleep(t0, interval):
    elapsed = time.monotonic() - t0
    if elapsed < interval: time.sleep(interval - elapsed)


if __name__ == "__main__":
    main()

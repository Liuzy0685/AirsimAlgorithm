#!/usr/bin/env python
"""Airborne Test Fixture — ROUND 4.9.

Cleanup tracking: landing_ok, disarm_ok, release_ok all required.
ARM_REQUESTED phase. Normal completion != emergency.
"""
from __future__ import annotations
import argparse, logging, math, signal, sys, time, enum, json, yaml
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
from models.fixture_result import FixtureResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("airborne_fixture")

# ═══════════════════════ CLI ═══════════════════════
def _vf(v, label, lo=None, hi=None):
    if isinstance(v, bool): raise ValueError(f"{label} must be a number, got bool")
    if not isinstance(v, (int, float)): raise ValueError(f"{label} must be a number")
    fv = float(v)
    if not math.isfinite(fv): raise ValueError(f"{label} must be finite")
    if lo is not None and fv < lo: raise ValueError(f"{label} must be >= {lo}")
    if hi is not None and fv > hi: raise ValueError(f"{label} must be <= {hi}")
    return fv

def _parse_and_validate_args(argv=None, cfg_dict=None):
    cfg_dict = cfg_dict or {}
    af = cfg_dict.get("airborne_fixture", {}) or {}
    zr = af.get("target_z_range_m", [-3.0, -0.5])
    p = argparse.ArgumentParser(description="Airborne Test Fixture — ROUND 4.9")
    p.add_argument("--settings-json", required=True)
    p.add_argument("--target-z", type=float, default=-1.5)
    p.add_argument("--max-vertical-speed", type=float, default=0.5)
    p.add_argument("--hover-duration", type=float, default=float(af.get("default_hover_duration_s", 30)))
    p.add_argument("--allow-indefinite", action="store_true")
    args = p.parse_args(argv)
    _vf(args.target_z, "target_z", lo=zr[0], hi=zr[1])
    if args.target_z >= 0: raise ValueError("target_z must be negative (UP in NED)")
    _vf(args.max_vertical_speed, "max_vertical_speed", lo=0.01, hi=float(af.get("max_vertical_speed_mps", 0.5)))
    _vf(args.hover_duration, "hover_duration", lo=0.0)
    if args.hover_duration == 0 and not args.allow_indefinite:
        raise ValueError("--hover-duration 0 requires --allow-indefinite")
    return args

# ═══════════════════════ Upward provider ═══════════════════════
class UpwardClearanceProvider:
    def is_corridor_observable(self) -> Tuple[bool, str]: raise NotImplementedError
    def check_clearance(self, *a) -> Tuple[bool, str]: raise NotImplementedError

class UpwardClearanceDisabled(UpwardClearanceProvider):
    def is_corridor_observable(self): return False, "vertical_corridor_unobservable:no_upward_sensor_configured"
    def check_clearance(self, *a): return False, "upward_sensor_disabled"

class UpwardClearanceNotImplemented(UpwardClearanceProvider):
    def is_corridor_observable(self): return False, "upward_provider_not_implemented:enabled_but_no_runtime_provider"
    def check_clearance(self, *a): return False, "upward_provider_not_implemented"

def _create_upward_provider(af_cfg):
    if not af_cfg.upward_sensor_enabled: return UpwardClearanceDisabled()
    return UpwardClearanceNotImplemented()

# ═══════════════════════ Vertical NED clearance ═══════════════════════
def _check_vertical_clearance_ned(pc_ned, current_pos_ned, highest_z_ned, safety_radius_m, vertical_margin_m):
    if pc_ned.size == 0: return True, "no_points"
    pts = np.asarray(pc_ned, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3: return False, "invalid_shape"
    if not math.isfinite(highest_z_ned): return False, "non_finite_altitude"
    if any(not math.isfinite(v) for v in current_pos_ned): return False, "non_finite_position"
    if any(not math.isfinite(v) for v in pts.flat): return False, "non_finite_points"
    cz, cx, cy = float(current_pos_ned[2]), float(current_pos_ned[0]), float(current_pos_ned[1])
    z_upper = max(cz, highest_z_ned) + vertical_margin_m
    z_lower = min(cz, highest_z_ned) - vertical_margin_m
    r_xy = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    in_cyl = pts[r_xy <= safety_radius_m]
    if in_cyl.size == 0: return True, "no_points_in_cylinder"
    in_ch = in_cyl[(in_cyl[:,2] <= z_upper) & (in_cyl[:,2] >= z_lower)]
    if in_ch.size == 0: return True, "channel_clear"
    return False, f"vertical_clearance_blocked:{in_ch.shape[0]}pts,closest={float(np.min(np.abs(in_ch[:,2]-cz))):.2f}m"

# ═══════════════════════ Phases ═══════════════════════
class _Phase(enum.Enum):
    PREFLIGHT = 0; CONTROL_ACQUIRED = 1; ARM_REQUESTED = 2; ARMED = 3; TAKEOFF_STARTED = 4; AIRBORNE = 5

# ═══════════════════════ Safe shutdown ═══════════════════════
_CLEANUP_FAILURE_CODE = 9

def _safe_shutdown(client, vehicle_name, state_reader, phase, reason, af_cfg, clock, errs, rk):
    """Returns (cleanup_ok, note). ROUND 4.9: ALL required steps must succeed."""
    disarm_ok = False; release_ok = False

    if phase == _Phase.PREFLIGHT:
        return True, "preflight"

    if phase == _Phase.CONTROL_ACQUIRED:
        try: client.enableApiControl(False, vehicle_name=vehicle_name); rk.update(api_control_released=True, api_control_enabled=False); release_ok = True
        except Exception as e: errs.append(f"release_api:{e}")
        return release_ok, "CONTROL_ACQUIRED"

    # Helper for landing + disarm + release sequence
    def _land_disarm_release(airborne, note_prefix):
        nonlocal disarm_ok, release_ok
        st_ok = True
        if airborne:
            try: client.hoverAsync(vehicle_name=vehicle_name).join()
            except Exception as e: errs.append(f"hover:{e}")
            try: client.landAsync(timeout_sec=15, vehicle_name=vehicle_name).join()
            except Exception as e: errs.append(f"land:{e}")
            lf = int(af_cfg.get("landing_confirmation_frames", 3))
            for _ in range(lf):
                try:
                    if state_reader.read().landed_state == 0: rk["landing_confirmed"] = True; break
                except Exception: pass
                time.sleep(0.5)
            if not rk.get("landing_confirmed"):
                errs.append("landing_not_confirmed:manual_intervention_required")
                return False, f"landing_not_confirmed:{reason}"
        try: client.armDisarm(False, vehicle_name=vehicle_name); rk.update(disarmed=True, armed=False); disarm_ok = True
        except Exception as e: errs.append(f"disarm:{e}")
        if disarm_ok:
            try: client.enableApiControl(False, vehicle_name=vehicle_name); rk.update(api_control_released=True, api_control_enabled=False); release_ok = True
            except Exception as e: errs.append(f"release_api:{e}")
        if not disarm_ok: errs.append("manual_intervention_required:disarm_failed")
        if not st_ok: errs.append("manual_intervention_required:state_read_failed")
        return (disarm_ok and release_ok and st_ok), f"{note_prefix}:{reason}"

    if phase == _Phase.ARM_REQUESTED:
        st_ok = True; airborne = True
        try:
            st = state_reader.read()
            pos_ok = all(math.isfinite(v) for v in st.position_ned_m + st.linear_velocity_ned_mps)
            stationary = math.hypot(*st.linear_velocity_ned_mps) <= float(af_cfg.get("stationary_speed_threshold_mps", 0.2))
            airborne = (st.landed_state != 0 or not pos_ok or not stationary)
        except Exception: st_ok = False
        return _land_disarm_release(airborne, "ARM_REQUESTED")

    if phase in (_Phase.ARMED, _Phase.TAKEOFF_STARTED):
        st_ok = True; airborne = True
        try:
            st = state_reader.read()
            pos_ok = all(math.isfinite(v) for v in st.position_ned_m + st.linear_velocity_ned_mps)
            stationary = math.hypot(*st.linear_velocity_ned_mps) <= float(af_cfg.get("stationary_speed_threshold_mps", 0.2))
            airborne = (st.landed_state != 0 or phase == _Phase.TAKEOFF_STARTED or not pos_ok or not stationary)
        except Exception: st_ok = False
        return _land_disarm_release(airborne, phase.name)

    # AIRBORNE
    return _land_disarm_release(True, "AIRBORNE")

# ═══════════════════════ Mission exit codes ═══════════════════════
_MC = {"normal":0,"ctrl_c":1,"collision":5,"lidar_failure":6,"rpc_lost":3,"drift":7,"tilt":7,
       "stabilization_failed":4,"altitude_not_reached":4,"post_climb_read":4,"moveToZ_error":4,
       "takeoff_error":4,"arm_error":4,"enable_error":4}

# ═══════════════════════ Main fixture ═══════════════════════
def run_airborne_fixture(args, adapter_factory, clock, sleeper, signal_handler=None,
                        upward_provider=None) -> FixtureResult:
    from adapters.airsim_client import AirSimClientAdapter
    from configs.runtime_config import load_lidar_runtime_config
    from configs.airborne_fixture_config import AirborneFixtureConfig
    from perception.sensor_fov import load_lidar_fov, validate_sector_fov_coverage, check_max_range_against_fov
    from perception.perception_config import load_perception_config
    from perception.pointcloud_filter import filter_pointcloud
    from sensors.lidar_reader import LidarReader
    from sensors.state_reader import StateReader
    from sensors.collision_reader import CollisionReader
    from transforms.lidar_to_local_ned import sensor_to_local_ned

    errs = []
    rk = dict(mission_success=False, cleanup_success=False, exit_code=1, exit_reason="unknown",
              primary_failure_reason="", cleanup_failure_reason=None,
              api_control_enabled=False, api_control_released=False, armed=False, disarmed=False,
              takeoff_completed=False, target_altitude_reached=False, hover_seconds=0.0,
              emergency_shutdown_attempted=False, shutdown_type="", preflight_checks_passed=False,
              takeoff_allowed=False, collision_detected_during_flight=False, landing_confirmed=False,
              actual_altitude_achieved=None)
    phase = _Phase.PREFLIGHT

    def _done(reason, code=1, **kw):
        ms = kw.pop("mission_success", (code == 0))
        pf = kw.pop("primary_failure_reason", "" if code == 0 else reason)
        rk.update(exit_reason=reason, exit_code=code, primary_failure_reason=pf, **kw)
        for k in ("mission_success","cleanup_errors"): rk.pop(k, None)
        return FixtureResult(mission_success=ms, cleanup_errors=list(errs), **rk)

    def _fail(reason, shutdown_type, mission_code):
        rk.update(shutdown_type=shutdown_type, emergency_shutdown_attempted=True)
        cleanup_ok, note = _safe_shutdown(client, vehicle_name, state_reader, phase, reason, af_cfg.__dict__, clock, errs, rk)
        rk["cleanup_success"] = cleanup_ok
        if not cleanup_ok: rk["cleanup_failure_reason"] = note
        return _done(f"{reason}:{note}", code=(_CLEANUP_FAILURE_CODE if not cleanup_ok else mission_code),
                     primary_failure_reason=reason)

    # ── Config ──
    vcp = str(_PROJECT_ROOT / "configs" / "vehicle.yaml")
    pcp = str(_PROJECT_ROOT / "configs" / "perception.yaml")
    plp = str(_PROJECT_ROOT / "configs" / "local_planner.yaml")
    try: rt = load_lidar_runtime_config(vcp)
    except Exception as e: return _done(f"vehicle_config:{e}", code=2)
    vehicle_name, lidar_name = rt.airsim.vehicle_name, rt.airsim.lidar_name
    try: pcfg = load_perception_config(pcp)
    except Exception as e: return _done(f"perception_config:{e}", code=2)
    with open(plp, "r", encoding="utf-8") as fh: planner_cfg = yaml.safe_load(fh) or {}
    try: af_cfg = AirborneFixtureConfig.from_dict(planner_cfg)
    except Exception as e: return _done(f"config:{e}", code=2)
    up_provider = upward_provider if upward_provider is not None else _create_upward_provider(af_cfg)

    # ── FOV + VehicleType ──
    sp = Path(args.settings_json)
    if not sp.is_file(): return _done(f"settings_not_found:{sp}", code=2)
    try: fov = load_lidar_fov(str(sp), vehicle_name, lidar_name)
    except Exception as e: return _done(f"fov:{e}", code=2)
    with open(str(sp), "r", encoding="utf-8") as fh: raw_settings = json.load(fh)
    vehicles = raw_settings.get("Vehicles", {})
    if not isinstance(vehicles, dict) or vehicle_name not in vehicles:
        return _done(f"vehicle_not_in_settings:{vehicle_name}", code=2)
    veh = vehicles[vehicle_name]
    if not isinstance(veh, dict): return _done("vehicle_entry_not_dict", code=2)
    if veh.get("VehicleType", "") != "SimpleFlight":
        return _done(f"vehicle_type_not_simpleflight:{veh.get('VehicleType','')}", code=2)
    if check_max_range_against_fov(pcfg, fov): return _done("max_range_exceeds_lidar", code=2)
    if [s.legacy_name for s in validate_sector_fov_coverage(pcfg, fov) if not s.fully_observable]:
        return _done("fov_incompatible", code=2)

    # ── Connect ──
    try: adapter = adapter_factory()
    except Exception as e: return _done(f"connect:{e}", code=3)
    try: client = adapter.get_raw_client()
    except Exception as e: return _done(f"get_raw_client:{e}", code=3)
    try:
        if vehicle_name not in [str(v) for v in client.listVehicles()]:
            return _done(f"vehicle_not_found:{vehicle_name}", code=2)
    except Exception as e: return _done(f"listVehicles:{e}", code=3)

    lidar = LidarReader(adapter, frame_timeout_seconds=rt.frame_timeout_seconds)
    state_reader = StateReader(adapter, vehicle_name=vehicle_name)
    collision_reader = CollisionReader(adapter, vehicle_name=vehicle_name)

    # ── Preflight ──
    for i in range(af_cfg.preflight_lidar_frames):
        try: lf = lidar.read(); st = state_reader.read(); col = collision_reader.read()
        except Exception as e: return _done(f"preflight_rpc_{i}:{e}", code=3)
        if not lf.frame_valid: return _done(f"preflight_lidar_invalid_{i}", code=2)
        if col.has_collided: return _done(f"preflight_collision_{i}", code=2)
    st0 = state_reader.read()
    if not st0.ready: return _done("not_ready", code=2)
    if not st0.can_arm: return _done("cannot_arm", code=2)
    if st0.landed_state != 0: return _done(f"not_landed:{st0.landed_state}", code=2)
    pos0, vel0 = st0.position_ned_m, st0.linear_velocity_ned_mps
    if any(not math.isfinite(v) for v in pos0 + vel0): return _done("nonfinite", code=2)
    if math.hypot(*vel0) > af_cfg.stationary_speed_threshold_mps:
        return _done(f"not_stationary:{math.hypot(*vel0):.2f}", code=2)
    rk["preflight_checks_passed"] = True
    current_pos = np.array(pos0, dtype=np.float64)

    # ── Vertical corridor + clearance ──
    obs, obs_reason = up_provider.is_corridor_observable()
    if not obs: return _done(obs_reason, code=2, takeoff_allowed=False)
    lf0 = lidar.read()
    if not lf0.frame_valid: return _done(f"vc_lidar_invalid:{lf0.invalid_reason}", code=2, takeoff_allowed=False)
    pc_cfg = pcfg.pointcloud
    fr = filter_pointcloud(lf0.point_cloud_sensor, min_range_m=pc_cfg.min_range_m, max_range_m=pc_cfg.max_range_m,
        self_exclusion={"enabled": pc_cfg.self_exclusion.enabled, "x_min_m": pc_cfg.self_exclusion.x_min_m,
                        "x_max_m": pc_cfg.self_exclusion.x_max_m, "y_min_m": pc_cfg.self_exclusion.y_min_m,
                        "y_max_m": pc_cfg.self_exclusion.y_max_m, "z_min_m": pc_cfg.self_exclusion.z_min_m,
                        "z_max_m": pc_cfg.self_exclusion.z_max_m},
        voxel_downsample=pc_cfg.voxel_downsample.enabled, voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m)
    if not fr.valid: return _done(f"filter_invalid:{fr.invalid_reason}", code=2, takeoff_allowed=False)
    if fr.output_point_count < af_cfg.min_preflight_filtered_points:
        return _done(f"insufficient_points:{fr.output_point_count}", code=2, takeoff_allowed=False)
    pos_arr = orient_arr = None
    if lf0.sensor_pose and isinstance(lf0.sensor_pose, dict):
        pd = lf0.sensor_pose.get("position", {}); od = lf0.sensor_pose.get("orientation", {})
        if isinstance(pd, dict) and isinstance(od, dict):
            try:
                pos_arr = np.array([float(pd["x"]), float(pd["y"]), float(pd["z"])])
                orient_arr = np.array([float(od["x"]), float(od["y"]), float(od["z"]), float(od["w"])])
            except (KeyError, TypeError, ValueError): pass
    if pos_arr is None: return _done("pose_invalid", code=2, takeoff_allowed=False)
    try: pc_ned = sensor_to_local_ned(fr.filtered_points_sensor, pos_arr, orient_arr)
    except Exception as e: return _done(f"transform:{e}", code=2, takeoff_allowed=False)
    if af_cfg.takeoff_delta_z_m != -2.0:
        return _done(f"takeoff_delta_z_m must be -2.0 (SimpleFlight), got {af_cfg.takeoff_delta_z_m}", code=2)
    highest_z = min(args.target_z, current_pos[2] + (-2.0))
    clear, vc_reason = up_provider.check_clearance(pc_ned, current_pos, highest_z,
                                                    af_cfg.vc_safety_radius_m, af_cfg.vc_vertical_margin_m)
    if not clear: return _done(vc_reason, code=2, takeoff_allowed=False)
    rk["takeoff_allowed"] = True

    # ── Takeoff ──
    emergency = [False]
    if signal_handler: signal_handler(lambda s, f: emergency.__setitem__(0, True))
    try: client.enableApiControl(True, vehicle_name=vehicle_name); rk.update(api_control_enabled=True)
    except Exception as e: return _done(f"enable:{e}", code=4)
    phase = _Phase.CONTROL_ACQUIRED

    phase = _Phase.ARM_REQUESTED
    try: client.armDisarm(True, vehicle_name=vehicle_name); rk.update(armed=True)
    except Exception as e: return _fail(f"arm:{e}", "cleanup", _MC["arm_error"])
    phase = _Phase.ARMED

    phase = _Phase.TAKEOFF_STARTED
    try:
        client.takeoffAsync(timeout_sec=af_cfg.takeoff_timeout_s, vehicle_name=vehicle_name).join()
        rk["takeoff_completed"] = True
    except Exception as e: return _fail(f"takeoff:{e}", "cleanup", _MC["takeoff_error"])
    phase = _Phase.AIRBORNE

    try:
        client.moveToZAsync(z=args.target_z, velocity=args.max_vertical_speed,
                            timeout_sec=max(30, abs(args.target_z)/args.max_vertical_speed+10),
                            vehicle_name=vehicle_name).join()
    except Exception as e: return _fail(f"moveToZ:{e}", "cleanup", _MC["moveToZ_error"])

    # ── Post-climb ──
    try: st_post = state_reader.read()
    except Exception: return _fail("post_climb_read", "cleanup", _MC["post_climb_read"])
    az = st_post.position_ned_m[2]
    if not math.isfinite(az) or any(not math.isfinite(v) for v in st_post.position_ned_m + st_post.linear_velocity_ned_mps):
        return _fail("post_climb_nonfinite", "cleanup", _MC["post_climb_read"])
    if abs(az - args.target_z) > af_cfg.altitude_tolerance_m:
        return _fail(f"altitude_not_reached:{az:.2f}", "cleanup", _MC["altitude_not_reached"])
    rk.update(target_altitude_reached=True, actual_altitude_achieved=az)

    # ── Hover stabilization ──
    try: client.hoverAsync(vehicle_name=vehicle_name).join()
    except Exception as e: return _fail(f"hover_stab:{e}", "cleanup", _MC["stabilization_failed"])
    stable = 0
    for _ in range(af_cfg.hover_stabilization_frames * 6):
        try: st_s = state_reader.read(); col_s = collision_reader.read(); lf_s = lidar.read()
        except Exception: stable = 0; break
        if not lf_s.frame_valid: stable = 0; continue
        if col_s.has_collided: stable = 0; continue
        cz = st_s.position_ned_m[2]; vs = abs(st_s.linear_velocity_ned_mps[2])
        hs = math.hypot(st_s.linear_velocity_ned_mps[0], st_s.linear_velocity_ned_mps[1])
        tilt = max(abs(st_s.roll_rad), abs(st_s.pitch_rad))
        if (abs(cz - args.target_z) <= af_cfg.altitude_tolerance_m and vs <= af_cfg.hs_max_vertical_speed_mps
                and hs <= af_cfg.max_horizontal_speed_mps and tilt <= af_cfg.max_tilt_rad):
            stable += 1
        else: stable = 0
        if stable >= af_cfg.hover_stabilization_frames: break
        sleeper(0.2)
    if stable < af_cfg.hover_stabilization_frames:
        return _fail("stabilization_failed", "cleanup", _MC["stabilization_failed"])

    # ── Hover monitoring ──
    hs_t0 = clock(); hp = (current_pos[0], current_pos[1], az); sht = "normal"
    try:
        while True:
            if emergency[0]: sht = "ctrl_c"; break
            try: col = collision_reader.read()
            except Exception: sht = "rpc_lost"; break
            if col.has_collided: sht = "collision"; rk["collision_detected_during_flight"] = True; break
            try: lf = lidar.read()
            except Exception: sht = "rpc_lost"; break
            if not lf.frame_valid: sht = "lidar_failure"; break
            try: st = state_reader.read()
            except Exception: sht = "rpc_lost"; break
            cz = st.position_ned_m[2]; dr = math.hypot(st.position_ned_m[0]-hp[0], st.position_ned_m[1]-hp[1])
            hs2 = math.hypot(st.linear_velocity_ned_mps[0], st.linear_velocity_ned_mps[1])
            vs2 = abs(st.linear_velocity_ned_mps[2]); tilt2 = max(abs(st.roll_rad), abs(st.pitch_rad))
            if abs(cz - args.target_z) > af_cfg.max_altitude_error_m: sht = "drift"; break
            if dr > af_cfg.max_horizontal_drift_m: sht = "drift"; break
            if hs2 > af_cfg.max_horizontal_speed_mps: sht = "drift"; break
            if vs2 > af_cfg.hs_max_vertical_speed_mps: sht = "drift"; break
            if tilt2 > af_cfg.max_tilt_rad: sht = "tilt"; break
            if args.hover_duration > 0 and clock() - hs_t0 >= args.hover_duration: sht = "normal"; break
            sleeper(0.5)
    except Exception: sht = "rpc_lost"
    rk["hover_seconds"] = clock() - hs_t0; rk["shutdown_type"] = sht

    if sht == "normal":
        # Normal completion → clean shutdown, NOT emergency
        rk["primary_failure_reason"] = ""
        cleanup_ok, note = _safe_shutdown(client, vehicle_name, state_reader, phase, "normal_completion", af_cfg.__dict__, clock, errs, rk)
        rk["cleanup_success"] = cleanup_ok
        if not cleanup_ok: rk["cleanup_failure_reason"] = note
        return _done(f"normal_completion:{note}", code=(0 if cleanup_ok else _CLEANUP_FAILURE_CODE),
                     mission_success=cleanup_ok, primary_failure_reason="")

    # Abnormal → cleanup, mission always fails
    return _fail(sht, sht, _MC.get(sht, 4))


# ═══════════════════════ main ═══════════════════════
def main():
    with open(str(_PROJECT_ROOT / "configs" / "local_planner.yaml"), "r", encoding="utf-8") as fh:
        planner_cfg = yaml.safe_load(fh) or {}
    args = _parse_and_validate_args(cfg_dict=planner_cfg)
    def _af():
        from adapters.airsim_client import AirSimClientAdapter
        a = AirSimClientAdapter(config_path=str(_PROJECT_ROOT / "configs" / "vehicle.yaml"), readonly=False)
        a.connect(); return a
    result = run_airborne_fixture(args, _af, time.monotonic, time.sleep, lambda h: signal.signal(signal.SIGINT, h))
    print(f"\nFixture result: {result}")
    sys.exit(result.exit_code)

if __name__ == "__main__": main()

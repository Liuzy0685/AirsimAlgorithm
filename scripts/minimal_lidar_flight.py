#!/usr/bin/env python
"""Minimal LiDAR reactive flight — UE4+AirSim simulation ONLY.

Fixes: phased cleanup, takeoff altitude verification, UTF-8 log output.
"""
from __future__ import annotations
import argparse, logging, math, signal, sys, time, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("minimal_flight")


@dataclass(frozen=True)
class ReactiveDecision:
    vx_body_mps: float = 0.0; vy_body_mps: float = 0.0; vz_body_mps: float = 0.0
    should_terminate: bool = False; termination_reason: str = ""

def choose_reactive_command(front_m, left_m, right_m, minimum_distance_m, config):
    emerg=float(config.get("emergency_distance_m",0.8)); ft=float(config.get("front_threshold_m",2.5))
    fwd=float(config.get("forward_speed_mps",0.2)); side=float(config.get("side_speed_mps",0.15))
    if minimum_distance_m<emerg: return ReactiveDecision(should_terminate=True,termination_reason="emergency_distance")
    if front_m>ft: return ReactiveDecision(vx_body_mps=fwd)
    if left_m>right_m: return ReactiveDecision(vy_body_mps=-side)
    return ReactiveDecision(vy_body_mps=side)

@dataclass(frozen=True)
class MinimalFlightResult:
    success: bool=False; exit_code: int=1; termination_reason: str=""; frames_completed: int=0
    flight_duration_s: float=0.0; api_control_acquired: bool=False; armed: bool=False
    takeoff_completed: bool=False; airborne: bool=False; landing_confirmed: bool=False
    disarmed: bool=False; api_control_released: bool=False; cleanup_errors: Tuple[str,...]=()
    startup_floor_contact_baseline: bool=False

def _load_flight_config(path):
    with open(path,"r",encoding="utf-8") as fh:
        fc=(yaml.safe_load(fh) or {}).get("minimal_flight",{})
    for key,lo,hi in [("target_z_ned",-5.0,-0.3),("max_vertical_speed_mps",0.1,0.5),
        ("takeoff_timeout_s",5.0,60.0),("max_flight_duration_s",1.0,60.0),
        ("command_duration_s",0.05,0.5),("forward_speed_mps",0.05,1.0),
        ("side_speed_mps",0.05,0.5),("front_threshold_m",0.5,10.0),
        ("emergency_distance_m",0.3,3.0),("geofence_radius_m",0.5,20.0),
        ("preflight_lidar_frames",1,10),("landing_confirmation_frames",1,10)]:
        v=fc.get(key)
        if isinstance(v,bool): raise ValueError(f"{key} must be number")
        if not isinstance(v,(int,float)): raise ValueError(f"{key} must be number")
        if not math.isfinite(v): raise ValueError(f"{key} must be finite")
        if not (lo<=v<=hi): raise ValueError(f"{key}={v} must be in [{lo},{hi}]")
    return fc


def run_minimal_flight(args, adapter_factory, clock, sleeper):
    from adapters.airsim_client import AirSimClientAdapter
    from configs.runtime_config import load_lidar_runtime_config
    from perception.perception_config import load_perception_config
    from perception.pointcloud_filter import filter_pointcloud
    from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
    from perception.sensor_fov import load_lidar_fov, validate_sector_fov_coverage, check_max_range_against_fov
    from sensors.lidar_reader import LidarReader
    from sensors.state_reader import StateReader
    from sensors.collision_reader import CollisionReader
    from control.velocity_controller import VelocityController

    rk=dict(success=False,exit_code=1,termination_reason="",frames_completed=0,
            flight_duration_s=0.0,api_control_acquired=False,armed=False,
            takeoff_completed=False,airborne=False,landing_confirmed=False,
            disarmed=False,api_control_released=False,cleanup_errors=(),
            startup_floor_contact_baseline=False)
    def _done(**kw): rk.update(**kw); return MinimalFlightResult(**{k:tuple(v) if k=="cleanup_errors" else v for k,v in rk.items()})

    if not args.confirm_simulation_clearance:
        return _done(termination_reason="clearance_not_confirmed",exit_code=2)

    try: rt=load_lidar_runtime_config(args.config)
    except Exception as e: return _done(termination_reason=f"config:{e}",exit_code=2)
    vn,ln=rt.airsim.vehicle_name,rt.airsim.lidar_name
    try: pcfg=load_perception_config(args.perception_config)
    except Exception as e: return _done(termination_reason=f"perception:{e}",exit_code=2)
    try: fc=_load_flight_config(args.flight_config)
    except Exception as e: return _done(termination_reason=f"flight_config:{e}",exit_code=2)

    sp=Path(args.settings_json)
    if not sp.is_file(): return _done(termination_reason=f"settings_not_found:{sp}",exit_code=2)
    try: fov=load_lidar_fov(str(sp),vn,ln)
    except Exception as e: return _done(termination_reason=f"fov:{e}",exit_code=2)
    if check_max_range_against_fov(pcfg,fov): return _done(termination_reason="max_range_error",exit_code=2)
    if [s.legacy_name for s in validate_sector_fov_coverage(pcfg,fov) if not s.fully_observable]:
        return _done(termination_reason="fov_incompatible",exit_code=2)

    adapter=adapter_factory(); client=adapter.get_raw_client()
    try:
        if vn not in [str(v) for v in client.listVehicles()]:
            return _done(termination_reason=f"vehicle_not_found:{vn}",exit_code=2)
    except Exception as e: return _done(termination_reason=f"listVehicles:{e}",exit_code=3)

    lidar=LidarReader(adapter,frame_timeout_seconds=rt.frame_timeout_seconds)
    sr=StateReader(adapter,vehicle_name=vn); cr=CollisionReader(adapter,vehicle_name=vn)
    vc=VelocityController(adapter,config_path=args.config)

    _WARMUP_MAX=10; _WARMUP_INTERVAL=0.15; _WARMUP_CLEAR=5
    _FLOOR_OK=frozenset({"Floor","Floor_3"})
    saw_floor=False; cons_clean=0; saw_new_event_after_first=False; initial_floor_ts=0
    for i in range(_WARMUP_MAX):
        try:
            lf=lidar.read(); col=cr.read()
        except Exception as e:
            return _done(termination_reason=f"warmup_read_error_{i}:{e}",exit_code=2)
        if not lf.frame_valid:
            return _done(termination_reason=f"preflight_lidar_{i}:{lf.invalid_reason}",exit_code=2)
        if col.has_collided:
            if col.object_name not in _FLOOR_OK:
                return _done(termination_reason=f"preflight_collision_{i}:{col.object_name}",exit_code=2)
            if not saw_floor:
                saw_floor=True
                initial_floor_ts=col.raw_timestamp
            elif col.is_new_collision_event and col.raw_timestamp!=initial_floor_ts:
                saw_new_event_after_first=True
            cons_clean=0
        else:
            cons_clean+=1
        if cons_clean>=_WARMUP_CLEAR:
            break
        sleeper(_WARMUP_INTERVAL)
    if saw_new_event_after_first:
        return _done(termination_reason="preflight_new_collision_event",exit_code=2)
    if saw_floor and cons_clean<_WARMUP_CLEAR:
        return _done(termination_reason="preflight_floor_persists",exit_code=2)
    if saw_floor:
        rk["startup_floor_contact_baseline"]=True
    st0=sr.read()
    if not st0.ready or not st0.can_arm: return _done(termination_reason="not_ready",exit_code=2)
    spawn=(st0.position_ned_m[0],st0.position_ned_m[1])

    cleanup_errs=[]; target_z=float(fc["target_z_ned"]); max_dur=float(fc["max_flight_duration_s"])
    cmd_dur=float(fc["command_duration_s"]); geofence_r=float(fc["geofence_radius_m"])
    land_fr=int(fc["landing_confirmation_frames"])
    sz_cfg=pcfg.sectorization; pc_cfg=pcfg.pointcloud; sdefs=list(sz_cfg.sectors)
    fov_obs={}
    for sts in validate_sector_fov_coverage(pcfg,fov):
        for sd in sdefs:
            if sd.legacy_name==sts.legacy_name: fov_obs[sd.name]=(sts.fully_observable,1.0)

    # Output file
    out_fh=open(args.output,"w",encoding="utf-8") if args.output else None
    def _log(line):
        print(line)
        if out_fh: out_fh.write(line+"\n"); out_fh.flush()

    term="unknown"; fn=0; t0=0.0

    exit_code=1; early_exit=False
    try:
        try: client.enableApiControl(True,vehicle_name=vn); rk["api_control_acquired"]=True
        except Exception as e: term=f"enableApiControl:{e}"; exit_code=4; early_exit=True

        if not early_exit:
            try: client.armDisarm(True,vehicle_name=vn); rk["armed"]=True
            except Exception as e: term=f"arm:{e}"; exit_code=4; early_exit=True

        if not early_exit:
            try: client.takeoffAsync(timeout_sec=float(fc["takeoff_timeout_s"]),vehicle_name=vn).join(); rk["takeoff_completed"]=True
            except Exception as e: term=f"takeoff:{e}"; exit_code=4; early_exit=True

        if not early_exit:
            try: client.moveToZAsync(z=target_z,velocity=float(fc["max_vertical_speed_mps"]),timeout_sec=30,vehicle_name=vn).join()
            except Exception as e: term=f"moveToZ:{e}"; exit_code=4; early_exit=True

        if not early_exit:
            try: client.hoverAsync(vehicle_name=vn).join()
            except Exception as e: term=f"hover:{e}"; exit_code=4; early_exit=True

        if not early_exit:
            st_climb=sr.read()
            az=st_climb.position_ned_m[2]; vs_abs=abs(st_climb.linear_velocity_ned_mps[2])
            if (not math.isfinite(az) or not all(math.isfinite(v) for v in st_climb.position_ned_m+st_climb.linear_velocity_ned_mps)):
                term="post_climb_nonfinite"; exit_code=4; early_exit=True
            elif st_climb.landed_state==0 or abs(az-target_z)>0.3 or vs_abs>0.3:
                term=f"altitude_not_verified:z={az:.2f},vs={vs_abs:.2f},ls={st_climb.landed_state}"
                exit_code=4; early_exit=True

        if not early_exit: rk["airborne"]=True

        if not early_exit:
            _log(f"{'Frm':>4s} | {'t':>6s} | {'pos_x':>7s} | {'pos_y':>7s} | {'pos_z':>7s} | "
             f"{'front':>7s} | {'left':>7s} | {'right':>7s} | {'minD':>6s} | "
             f"{'vx':>6s} | {'vy':>6s} | {'col':>5s}")
        _log("-"*110)

        t0=clock(); term="time_limit"
        while True:
            fn+=1; ts=clock()
            if ts-t0>=max_dur: term="time_limit"; break
            try: lf=lidar.read(); st=sr.read(); col=cr.read()
            except Exception: term="rpc_error"; break
            if not lf.frame_valid: term=f"lidar_invalid:{lf.invalid_reason}"; break
            if col.has_collided: term=f"collision:{col.object_name}"; break
            if math.hypot(st.position_ned_m[0]-spawn[0],st.position_ned_m[1]-spawn[1])>geofence_r:
                term="geofence"; break

            fr=filter_pointcloud(lf.point_cloud_sensor,min_range_m=pc_cfg.min_range_m,max_range_m=pc_cfg.max_range_m,
                self_exclusion={"enabled":pc_cfg.self_exclusion.enabled,"x_min_m":pc_cfg.self_exclusion.x_min_m,
                    "x_max_m":pc_cfg.self_exclusion.x_max_m,"y_min_m":pc_cfg.self_exclusion.y_min_m,
                    "y_max_m":pc_cfg.self_exclusion.y_max_m,"z_min_m":pc_cfg.self_exclusion.z_min_m,
                    "z_max_m":pc_cfg.self_exclusion.z_max_m},
                voxel_downsample=pc_cfg.voxel_downsample.enabled,voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m)
            if not fr.valid: term=f"filter:{fr.invalid_reason}"; break
            try: dd=pointcloud_to_directional_distances(fr.filtered_points_sensor,sector_defs=sdefs,
                default_max_range_m=sz_cfg.default_max_range_m,default_min_points=sz_cfg.default_min_points,
                distance_strategy=sz_cfg.default_distance_strategy,nearest_k=sz_cfg.nearest_k,
                percentile=sz_cfg.percentile,frame_valid=True,fov_compatible=True,fov_observability=fov_obs)
            except Exception: term="sector_error"; break
            if not dd.frame_valid: term=f"dd:{dd.invalid_reason}"; break
            try: rays=dd.to_legacy_ray_distances()
            except Exception: term="legacy_error"; break

            dec=choose_reactive_command(rays.get("front",float("inf")),rays.get("left",float("inf")),
                                        rays.get("right",float("inf")),dd.minimum_distance_m,fc)
            if dec.should_terminate: term=dec.termination_reason; break

            try: vc.send_velocity_body_frd(dec.vx_body_mps,dec.vy_body_mps,0.0,duration=cmd_dur,vehicle_name=vn)
            except Exception: term="velocity_send_error"; break

            _log(f"{fn:4d} | {ts-t0:6.1f} | {st.position_ned_m[0]:7.2f} | {st.position_ned_m[1]:7.2f} | {st.position_ned_m[2]:7.2f} | "
                 f"{rays.get('front',float('inf')):7.2f} | {rays.get('left',float('inf')):7.2f} | {rays.get('right',float('inf')):7.2f} | "
                 f"{dd.minimum_distance_m:6.2f} | {dec.vx_body_mps:6.2f} | {dec.vy_body_mps:6.2f} | {str(col.has_collided):>5s}")
            sleeper(cmd_dur)

    except Exception as e: term=f"exception:{e}"
    finally:
        rk.update(termination_reason=term,frames_completed=fn,flight_duration_s=clock()-t0 if t0>0 else 0.0)

        # Phased cleanup based on state reached
        api_ok=rk.get("api_control_acquired",False)
        arm_ok=rk.get("armed",False)
        airborne_ok=rk.get("airborne",False) or rk.get("takeoff_completed",False)

        if not api_ok:
            pass
        elif not arm_ok:
            try: client.enableApiControl(False,vehicle_name=vn); rk["api_control_released"]=True
            except Exception as e: cleanup_errs.append(f"release:{e}")
        elif airborne_ok:
            try: client.hoverAsync(vehicle_name=vn).join()
            except Exception as e: cleanup_errs.append(f"hover:{e}")
            try: client.landAsync(timeout_sec=15,vehicle_name=vn).join()
            except Exception as e: cleanup_errs.append(f"land:{e}")
            for _ in range(land_fr):
                try:
                    if sr.read().landed_state==0: rk["landing_confirmed"]=True; break
                except Exception: pass
                sleeper(0.5)
            if rk.get("landing_confirmed"):
                try: client.armDisarm(False,vehicle_name=vn); rk["disarmed"]=True
                except Exception as e: cleanup_errs.append(f"disarm:{e}")
                if rk.get("disarmed"):
                    try: client.enableApiControl(False,vehicle_name=vn); rk["api_control_released"]=True
                    except Exception as e: cleanup_errs.append(f"release:{e}")
            else: cleanup_errs.append("landing_not_confirmed:no_disarm")
        else:
            try: client.armDisarm(False,vehicle_name=vn); rk["disarmed"]=True
            except Exception as e: cleanup_errs.append(f"disarm:{e}")
            if rk.get("disarmed"):
                try: client.enableApiControl(False,vehicle_name=vn); rk["api_control_released"]=True
                except Exception as e: cleanup_errs.append(f"release:{e}")

    rk["cleanup_errors"]=tuple(cleanup_errs)
    rk["success"]=(term=="time_limit" and not cleanup_errs)
    if exit_code==1: exit_code=0 if rk["success"] else (2 if not rk["api_control_acquired"] else 1)
    rk["exit_code"]=exit_code

    _log(f"\nSummary: frames={fn} duration={rk['flight_duration_s']:.1f}s term={term}")
    _log(f"  landing_confirmed={rk['landing_confirmed']} disarmed={rk['disarmed']} released={rk['api_control_released']}")
    if cleanup_errs: _log(f"  cleanup_errors={list(cleanup_errs)}")
    if out_fh: out_fh.close()
    return _done()


def main():
    p=argparse.ArgumentParser(description="Minimal LiDAR flight — SIMULATION ONLY")
    p.add_argument("--config",default=str(_PROJECT_ROOT/"configs"/"vehicle.yaml"))
    p.add_argument("--perception-config",default=str(_PROJECT_ROOT/"configs"/"perception.yaml"))
    p.add_argument("--flight-config",default=str(_PROJECT_ROOT/"configs"/"minimal_flight.yaml"))
    p.add_argument("--settings-json",required=True)
    p.add_argument("--confirm-simulation-clearance",action="store_true")
    p.add_argument("--output",default=None)
    args=p.parse_args()
    def _af():
        from adapters.airsim_client import AirSimClientAdapter
        a=AirSimClientAdapter(config_path=args.config,readonly=False); a.connect(); return a
    result=run_minimal_flight(args,_af,time.monotonic,time.sleep)
    print(f"\nResult: {result}"); sys.exit(result.exit_code)

if __name__=="__main__": main()

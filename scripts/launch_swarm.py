#!/usr/bin/env python
"""Launch independent flight_mode processes for configured AirSim drones.

Each drone keeps its own planner, perception/map workers, flight lock and
log. The swarm configuration selects ORCA or deterministic priority-based
horizontal yielding at the final command boundary.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "swarm.yaml"),
                        help="Swarm member YAML configuration.")
    parser.add_argument("--settings-json", required=True,
                        help="AirSim settings.json containing all configured vehicles.")
    parser.add_argument("--confirm-simulation-clearance", action="store_true",
                        help="Pass the required auto-flight confirmation to each member.")
    parser.add_argument("--startup-delay", type=float, default=0.5,
                        help="Delay between process launches in seconds.")
    parser.add_argument("--with-formation", action="store_true", help="Also start B's formation runner.")
    parser.add_argument("--formation-config", default=str(ROOT / "configs/formation.yaml"))
    parser.add_argument('--map-manifest',help='Optional offline known-map endpoint check before any flight control; does not replace runtime map/sensor verification.')
    parser.add_argument("--log-dir", help="Explicit run output directory for reproducible acceptance runs.")
    parser.add_argument("--stop-file", help="Cooperative fleet stop request file.")
    return parser.parse_args()


def _build_command(member: dict, args, log_dir: Path) -> list[str]:
    required = ("vehicle_name", "lidar_name")
    missing = [key for key in required if key not in member]
    if missing:
        raise ValueError(f"swarm member missing {', '.join(missing)}: {member!r}")
    has_goal_xy = member.get("goal_x") is not None and member.get("goal_y") is not None
    if not member.get("goal_actor") and not has_goal_xy:
        raise ValueError(
            f"swarm member needs goal_actor or both goal_x and goal_y: {member!r}"
        )
    if (member.get("goal_x") is None) != (member.get("goal_y") is None):
        raise ValueError(f"goal_x and goal_y must be set together: {member!r}")
    if member.get("goal_actor") and not member.get("start_actor"):
        raise ValueError(f"goal_actor requires a matching start_actor: {member!r}")
    name = str(member["vehicle_name"])
    cmd = [
        sys.executable, str(ROOT / "scripts" / "flight_mode.py"),
        "--mode", "auto",
        "--vehicle-name", name,
        "--lidar-name", str(member["lidar_name"]),
        "--settings-json", str(Path(args.settings_json).resolve()),
        "--swarm-config", str(Path(args.config).resolve()),
        "--flight-config", str(member.get("flight_config", ROOT / "configs" / "trajectory_flight.yaml")),
        "--perception-config", str(member.get("perception_config", ROOT / "configs" / "perception.yaml")),
        "--trajectory-config", str(member.get("trajectory_config", ROOT / "configs" / "trajectory_planner.yaml")),
        "--local-navigation-mode", str(member.get("local_navigation_mode", "trajectory")),
        "--log-file", str(log_dir / f"{name}.log"),
        "--result-json", str(log_dir / f"{name}.result.json"),
        "--stop-file", str(log_dir / f"{name}.stop"),
        "--mission-complete-file", str(log_dir / "mission.complete"),
        "--quiet",
    ]
    if member.get("start_actor"):
        cmd.extend(("--mission-start-actor", str(member["start_actor"])))
    if member.get("goal_actor"):
        cmd.extend(("--mission-end-actor", str(member["goal_actor"])))
    elif has_goal_xy:
        cmd.extend(("--goal-x", str(float(member["goal_x"])),
                    "--goal-y", str(float(member["goal_y"]))))
    if args.confirm_simulation_clearance:
        cmd.append("--confirm-simulation-clearance")
    if member.get("target_z") is not None:
        cmd.extend(("--target-z", str(float(member["target_z"]))))
    return cmd


def _stop_group(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _, proc in processes:
        if proc.poll() is None:
            try:
                stop_file = getattr(proc, "stop_file", None)
                if stop_file is not None:
                    Path(stop_file).touch()
                else:
                    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
            except Exception as exc:
                print(
                    f"Could not request graceful shutdown for process {proc.pid}: {exc}. "
                    "It was left running.", file=sys.stderr,
                )
    for _, proc in processes:
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            print(
                f"Process {proc.pid} did not stop in time; leaving it running for safety.",
                file=sys.stderr,
            )


def _request_graceful_stop(processes: list[tuple[str, subprocess.Popen]]) -> None:
    infrastructure = {"A-state-service", "B-formation-service"}
    controllers = [(name, proc) for name, proc in processes if name not in infrastructure]
    _stop_group(controllers)
    if any(proc.poll() is None for _, proc in controllers):
        print("Controllers still cleaning up; keeping A/B services alive.", file=sys.stderr)
        return
    # Keep authoritative state available throughout landing/disarm/release.
    _stop_group([(name, proc) for name, proc in processes if name in infrastructure])


def main() -> int:
    args = _parse_args()
    if not args.confirm_simulation_clearance:
        print("ERROR: --confirm-simulation-clearance is required.", file=sys.stderr)
        return 2
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    members = config.get("drones", [])
    if not isinstance(members, list) or len(members) < 2:
        print("ERROR: swarm config must define at least two drones.", file=sys.stderr)
        return 2
    if any(not isinstance(member, dict) for member in members):
        print("ERROR: every swarm entry must be a mapping.", file=sys.stderr)
        return 2
    names = [str(member.get("vehicle_name", "")) for member in members]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        print("ERROR: each drone needs a unique, non-empty AirSim vehicle_name.", file=sys.stderr)
        return 2

    log_dir = (Path(args.log_dir).resolve() if args.log_dir else
               ROOT / "runs" / f"swarm_{time.strftime('%Y%m%d_%H%M%S')}_{time.monotonic_ns()}")
    # A reused directory can contain an old ready/mission/stop marker.
    log_dir.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[str, subprocess.Popen]] = []
    log_handles = []
    from coordination.runtime_config import load_config
    _, service_cfg = load_config(config_path)
    if args.with_formation and not service_cfg.enabled:
        print("ERROR: --with-formation requires state_service.enabled", file=sys.stderr)
        return 2
    if args.with_formation and config.get('navigation', {}).get('independent_goals', False):
        print('ERROR: independent-goal missions must not start B formation control.', file=sys.stderr)
        return 2
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    try:
        if getattr(args,'map_manifest',None):
            from scripts.check_mission_endpoints import check
            import json
            endpoint_report=check(args.settings_json,config_path,args.formation_config,args.map_manifest)
            (log_dir/'endpoint_preflight.json').write_text(json.dumps(endpoint_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            if not endpoint_report['valid']:
                print('ERROR: endpoint preflight rejected: '+endpoint_report.get('message',endpoint_report['reason']),file=sys.stderr)
                return 2
        if service_cfg.enabled:
            from control.takeoff_preflight import verify_fleet_takeoff
            import json
            preflight=verify_fleet_takeoff(members,args.settings_json)
            (log_dir/'takeoff_preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
        if service_cfg.enabled:
            ready_file = log_dir / "state_service.ready"
            handle = (log_dir / "state_service.log").open("w", encoding="utf-8")
            log_handles.append(handle)
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/swarm_state_service.py"),
                "--settings-json", str(Path(args.settings_json).resolve()), "--swarm-config", str(config_path),
                "--ready-file", str(ready_file)], cwd=ROOT, creationflags=creationflags, stdout=handle, stderr=handle)
            processes.append(("A-state-service", proc))
            deadline = time.monotonic() + service_cfg.startup_timeout_s
            while not ready_file.exists():
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"A state service not ready; see {log_dir / 'state_service.log'}")
                time.sleep(0.1)
        for member in members:
            name = str(member["vehicle_name"])
            command = _build_command(member, args, log_dir)
            print(f"Launching {name}: log={log_dir / (name + '.log')}", flush=True)
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            proc = subprocess.Popen(command, cwd=ROOT, creationflags=creationflags)
            proc.stop_file = log_dir / f"{name}.stop"
            processes.append((name, proc))
            time.sleep(max(0.0, args.startup_delay))
        if args.with_formation:
            handle = (log_dir / "formation.log").open("w", encoding="utf-8")
            log_handles.append(handle)
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/formation_service.py"),
                "--swarm-config", str(config_path), "--formation-config", args.formation_config,
                '--stop-file',str(log_dir/'formation.stop')],
                cwd=ROOT, creationflags=creationflags, stdout=handle, stderr=handle)
            proc.stop_file=log_dir/'formation.stop'
            processes.append(("B-formation-service", proc))
        while processes:
            if (log_dir / "mission.complete").exists():
                print("Fleet mission completed; requesting coordinated landing.",flush=True)
                _request_graceful_stop(processes)
                return 0
            if args.stop_file and Path(args.stop_file).exists():
                _request_graceful_stop(processes)
                return 0
            remaining = []
            for name, proc in processes:
                code = proc.poll()
                if code is None:
                    remaining.append((name, proc))
                else:
                    print(f"{name} exited with code {code}.", flush=True)
                    if service_cfg.enabled:
                        _request_graceful_stop([(n, p) for n, p in processes if p.poll() is None])
                        return 0 if code == 0 else 1
            processes = remaining
            if processes:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping swarm processes; each flight process will run its cleanup.", flush=True)
        _request_graceful_stop(processes)
    except Exception as exc:
        print(f"Swarm launcher failed: {exc}", file=sys.stderr)
        _request_graceful_stop(processes)
        return 1
    finally:
        for handle in log_handles:
            handle.close()
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    raise SystemExit(main())

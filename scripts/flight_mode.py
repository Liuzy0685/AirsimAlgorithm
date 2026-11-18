#!/usr/bin/env python
"""
Unified flight mode entry point.

Usage::

    py scripts/flight_mode.py --mode manual

    py scripts/flight_mode.py ^
      --mode auto ^
      --confirm-simulation-clearance ^
      --settings-json "D:\\30817\\Adrone\\Project\\reference\\airsim_runtime\\settings_working.json"

Single-instance enforcement via lock file — only one flight mode can control
Drone1 at a time.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("flight_mode")

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """Configure compact console output and optional full diagnostic logging."""
    console = logging.StreamHandler()
    # In quiet mode, keep the console focused on actionable failures. The
    # complete INFO/WARNING stream is still written to --log-file when used.
    console.setLevel(logging.ERROR if quiet else logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    console.setFormatter(formatter)

    handlers = [console]
    resolved_log_file = None
    if log_file:
        resolved_log_file = Path(log_file)
        if not resolved_log_file.is_absolute():
            resolved_log_file = _PROJECT_ROOT / resolved_log_file
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        # Start a fresh diagnostic file for each flight run.
        file_handler = logging.FileHandler(
            resolved_log_file, mode="w", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )

    # Multiprocessing workers are spawned as fresh Python processes on Windows.
    # Pass the console/file policy through the environment so their logging
    # setup matches the parent process too.
    os.environ["AIRSIM_FLIGHT_QUIET"] = "1" if quiet else "0"
    os.environ["AIRSIM_FLIGHT_LOG_FILE"] = (
        str(resolved_log_file) if resolved_log_file else ""
    )

    if resolved_log_file:
        print(f"Detailed log file: {resolved_log_file}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified flight mode — manual or LiDAR-autonomous.")
    p.add_argument("--mode", required=True, choices=["manual", "auto"],
                   help="Flight mode: manual (keyboard) or auto (LiDAR avoidance).")
    p.add_argument("--manual-control", choices=["velocity", "attitude"], default="velocity",
                   help="Manual control type (default: velocity).")
    p.add_argument("--manual-speed", type=float, default=None,
                   help="Manual linear speed (m/s) for WASD. Default 0.5.")
    p.add_argument("--manual-vertical-speed", type=float, default=None,
                   help="Manual climb/descend speed (m/s) for R/F. Default 0.3.")
    p.add_argument("--manual-yaw-rate", type=float, default=None,
                   help="Manual yaw rate (rad/s) for Q/E. Default 0.5.")
    p.add_argument("--manual-input", choices=["keyboard", "gamepad"], default="keyboard",
                   help="Manual control input device: keyboard (WASD) or gamepad "
                        "(Xbox/XInput). Default: keyboard.")
    p.add_argument("--manual-gamepad-config", default=None,
                   help="Path to manual_gamepad.yaml for gamepad manual mode.")
    p.add_argument("--manual-disable-safety", action="store_true", default=False,
                   help="Disable the gamepad collision-guard (debug/testing only).")
    p.add_argument("--settings-json", default=None,
                   help="Path to AirSim settings.json for FOV validation (required for auto).")
    p.add_argument("--vehicle-name", default="Drone1",
                   help="AirSim 无人机名称（对应 vehicle_name，默认 Drone1）。")
    p.add_argument("--lidar-name", default="LidarSensor1",
                   help="LiDAR sensor name on the selected vehicle.")
    p.add_argument("--swarm-config", default=None,
                   help="Enable inter-UAV prediction and configured priority yielding.")
    p.add_argument("--result-json", help="Write automatic flight result and verified cleanup outcome.")
    p.add_argument("--stop-file", help="Request cooperative stop by creating this file; cleanup is not interrupted.")
    p.add_argument("--mission-complete-file", help="Notify fleet supervisor before starting leader landing.")
    p.add_argument("--confirm-simulation-clearance", action="store_true",
                   help="Required for auto mode.")
    p.add_argument("--perception-config",
                   default=str(_PROJECT_ROOT / "configs" / "perception.yaml"),
                   help="Path to perception config YAML.")
    p.add_argument("--flight-config",
                   default=str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml"),
                   help="Path to minimal_flight.yaml.")
    p.add_argument("--target-z", type=float, default=None,
                   help="Override target_z_ned from flight config.")
    p.add_argument("--max-duration", type=float, default=None,
                   help="Override max_flight_duration_s from flight config.")
    p.add_argument("--goal-x", type=float, default=None,
                   help="Explicit world-NED goal X; overrides MissionEnd actor.")
    p.add_argument("--goal-y", type=float, default=None,
                   help="Explicit world-NED goal Y; overrides MissionEnd actor.")
    p.add_argument("--mission-start-actor", default=None,
                   help="Named AirSim scene marker for this vehicle's local origin, e.g. PlayerStart1.")
    p.add_argument("--mission-end-actor", default=None,
                   help="Named AirSim scene marker for this vehicle's goal, e.g. MissionEnd1.")
    p.add_argument("--planner-mode", choices=["reactive", "apf_shadow", "apf"],
                   default="reactive",
                   help="Planner mode for auto flight: reactive, APF shadow logging, or APF control.")
    p.add_argument("--local-navigation-mode",
                   choices=["reactive", "guided_apf", "trajectory"],
                   default="reactive",
                   help="Local navigation layer for auto flight (A/B testing): "
                        "reactive (legacy sector rule), guided_apf (CBMBA-guided APF), "
                        "or trajectory (receding-horizon trajectory-centric planner).")
    p.add_argument("--trajectory-config",
                   default=None,
                   help="Path to trajectory_planner.yaml (default: "
                        "configs/trajectory_planner.yaml). Overrides the trajectory "
                        "planner configuration when --local-navigation-mode trajectory.")
    p.add_argument("--recovery-test-trigger", choices=["stuck", "oscillation"],
                   default=None,
                   help="One-shot test: inject a synthetic recovery condition "
                        "(stuck or oscillation) into the dispatch path. "
                        "Only fires once per process after takeoff + stable APF.")
    p.add_argument("--guided-apf-control", action="store_true",
                   default=False,
                   help="Enable guided APF real takeover (requires --planner-mode apf). "
                        "When active and conditions are met, CBMBA-guided lateral APF "
                        "becomes the dispatch source instead of normal APF.")
    p.add_argument("--cbmba-resolution", type=float, default=None,
                   help="Override CBMBA planner resolution (default: 0.75). "
                        "Must be > 0 and finite. "
                        "Example: --cbmba-resolution 1.5")
    p.add_argument("--quiet", action="store_true",
                   help="Show only ERROR logs on the console; full diagnostics "
                        "remain in --log-file.")
    p.add_argument("--log-file", default=None,
                   help="Save full INFO+ diagnostics to this file. Relative paths "
                        "are resolved from the project root.")
    return p.parse_args()


def _acquire_lock_or_die(mode_label: str, vehicle_name: str):
    from flight_modes.shared_flight_session import SharedFlightSession
    fh = SharedFlightSession.acquire_lock(mode_label, vehicle_name)
    if fh is None:
        print("\n  ERROR: Another flight mode instance is already controlling "
              f"无人机 {vehicle_name!r}.\n")
        sys.exit(5)
    return fh


def _run_manual(args: argparse.Namespace) -> int:
    if args.manual_input == "gamepad":
        return _run_manual_gamepad(args)

    from flight_modes.manual_mode import ManualMode, ManualControlType
    from flight_modes.shared_flight_session import SharedFlightSession

    lock_fh = _acquire_lock_or_die("manual", args.vehicle_name)
    mode_label = f"manual-{args.manual_control}"
    logger.info("Starting %s mode.", mode_label)

    ctrl_type = (
        ManualControlType.ATTITUDE if args.manual_control == "attitude"
        else ManualControlType.VELOCITY
    )

    session = SharedFlightSession(
        settings_json=args.settings_json or "",
        mode=mode_label,
        target_z_ned=args.target_z or -1.0,
        vehicle_name_override=args.vehicle_name,
        lidar_name_override=args.lidar_name,
    )
    session._lock_fh = lock_fh
    session._owns_lock = True

    try:
        session.initialize()
        # Manual mode does NOT auto-takeoff. User presses T.
        from flight_modes.manual_mode import ManualModeParams
        _mp = ManualModeParams()
        if args.manual_speed is not None:
            _mp.linear_speed_mps = args.manual_speed
        if args.manual_vertical_speed is not None:
            _mp.vertical_speed_mps = args.manual_vertical_speed
        if args.manual_yaw_rate is not None:
            _mp.yaw_rate_radps = args.manual_yaw_rate
        logger.info(
            "manual_params  linear=%.2f  vertical=%.2f  yaw_rate=%.2f",
            _mp.linear_speed_mps, _mp.vertical_speed_mps, _mp.yaw_rate_radps,
        )
        manual = ManualMode(session, control_type=ctrl_type, params=_mp)
        manual.run()
        # Landing handled by finally → session.cleanup() → land_and_disarm()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error("Manual mode error: %s", e)
        return 1
    finally:
        session.cleanup()


def _run_manual_gamepad(args: argparse.Namespace) -> int:
    from flight_modes.gamepad_config import load_manual_gamepad_config
    from flight_modes.manual_gamepad_mode import ManualGamepadMode
    from flight_modes.shared_flight_session import SharedFlightSession

    lock_fh = _acquire_lock_or_die("manual-gamepad", args.vehicle_name)
    logger.info("Starting manual-gamepad mode.")

    config = load_manual_gamepad_config(args.manual_gamepad_config)
    if args.manual_disable_safety:
        logger.warning("Collision guard DISABLED via --manual-disable-safety.")
        config = config.with_collision_guard(False)

    session = SharedFlightSession(
        settings_json=args.settings_json or "",
        mode="manual-gamepad",
        target_z_ned=args.target_z or -1.0,
        vehicle_name_override=args.vehicle_name,
        lidar_name_override=args.lidar_name,
    )
    session._lock_fh = lock_fh
    session._owns_lock = True

    try:
        session.initialize()
        # Gamepad mode does NOT auto-takeoff.  User presses A.
        mode = ManualGamepadMode(session, config=config)
        mode.run()
        # Landing handled by finally → session.cleanup() → land_and_disarm()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error("Manual gamepad mode error: %s", e)
        return 1
    finally:
        session.cleanup()


def _run_auto(args: argparse.Namespace) -> int:
    from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
    from flight_modes.shared_flight_session import SharedFlightSession

    if not args.confirm_simulation_clearance:
        print("\n  ERROR: --confirm-simulation-clearance is required.\n")
        return 2
    if not args.settings_json:
        print("\n  ERROR: --settings-json is required for auto mode.\n")
        return 2
    if (args.goal_x is None) != (args.goal_y is None):
        print("\n  ERROR: --goal-x and --goal-y must be provided together.\n")
        return 2

    lock_fh = _acquire_lock_or_die("auto", args.vehicle_name)
    logger.info("Starting auto mode (LiDAR avoidance).")

    # Build CLI overrides (only non-None values win over YAML)
    cli_overrides = {}
    if args.target_z is not None:
        cli_overrides["target_z_ned"] = float(args.target_z)
    if args.max_duration is not None:
        cli_overrides["max_flight_duration_s"] = float(args.max_duration)
    if args.goal_x is not None:
        cli_overrides["goal_x"] = float(args.goal_x)
        cli_overrides["goal_y"] = float(args.goal_y)
    if args.mission_start_actor is not None:
        cli_overrides["mission_start_actor"] = args.mission_start_actor
    if args.mission_end_actor is not None:
        cli_overrides["mission_end_actor"] = args.mission_end_actor
    cli_overrides["planner_mode"] = args.planner_mode
    cli_overrides["guided_apf_control"] = args.guided_apf_control
    cli_overrides["local_navigation_mode"] = args.local_navigation_mode
    if args.local_navigation_mode == "guided_apf":
        # Alias: "guided_apf" drives the existing CBMBA-guided APF takeover path.
        cli_overrides["planner_mode"] = "apf"
        cli_overrides["guided_apf_control"] = True
    if args.recovery_test_trigger is not None:
        cli_overrides["recovery_test_trigger"] = args.recovery_test_trigger
    if args.trajectory_config is not None:
        cli_overrides["trajectory_config_path"] = args.trajectory_config

    # ── CBMBA resolution override (validate early; fail before connecting) ──
    if args.cbmba_resolution is not None:
        _cbmba_res = args.cbmba_resolution
        if not math.isfinite(_cbmba_res):
            print(f"\n  ERROR: --cbmba-resolution must be finite, got {_cbmba_res}.\n")
            return 2
        if _cbmba_res <= 0:
            print(f"\n  ERROR: --cbmba-resolution must be > 0, got {_cbmba_res}.\n")
            return 2
        cli_overrides["cbmba_resolution"] = float(_cbmba_res)
        logger.info("cbmba_resolution_override=%.2f", _cbmba_res)

    logger.info("planner_mode=%s", args.planner_mode)

    # Load from YAML first, then merge CLI (planner_mode is not a YAML key)
    params = AutomaticModeParams.from_yaml(
        args.flight_config, cli_overrides if cli_overrides else None
    )

    session = SharedFlightSession(
        settings_json=args.settings_json,
        mode="auto",
        target_z_ned=params.target_z_ned,
        vehicle_name_override=args.vehicle_name,
        lidar_name_override=args.lidar_name,
    )
    session._lock_fh = lock_fh
    session._owns_lock = True

    bridge = None
    if getattr(args, "stop_file", None):
        session.stop_file = Path(args.stop_file)
    result_data = {"success": False, "termination_reason": "startup_not_completed"}
    try:
        session.initialize()

        if args.swarm_config:
            from coordination.runtime_config import load_config
            config, service_cfg = load_config(args.swarm_config)
            if service_cfg.enabled:
                if not config.get("coordination", {}).get("enabled", True):
                    raise ValueError("state service requires coordination.enabled")
                from control.swarm_flight_bridge import SwarmFlightBridge
                bridge = SwarmFlightBridge(session, config, service_cfg, params)
                bridge.start()

        auto = AutomaticMode(
            session,
            perception_config_path=args.perception_config,
            flight_config_path=args.flight_config,
            params=params,
            cli_overrides=cli_overrides if cli_overrides else None,
            swarm_config_path=args.swarm_config,
        )
        result = auto.run()
        from dataclasses import asdict
        result_data = asdict(result)
        if result.termination_reason == "mission_complete" and getattr(args,"mission_complete_file",None):
            Path(args.mission_complete_file).touch()
        if bridge is not None and result.termination_reason not in ("mission_complete", "time_limit", "coordinated_stop", "ctrl_c"):
            bridge._unverified_emergency = True
            reason = bridge.last_rejection_reason or result.termination_reason or "automatic_mode_failed"
            result_data["rejection_reason"] = reason
            bridge.link.report("EMERGENCY",reason)
        print(f"\n  Result: {result}")
        return 0 if result.success or result.termination_reason == "coordinated_stop" else 1
    except KeyboardInterrupt:
        result_data = {"success": False, "termination_reason": "coordinated_stop"}
        return 0
    except Exception as e:
        result_data = {"success": False, "termination_reason": f"exception:{e}"}
        logger.error("Auto mode error: %s", e)
        return 1
    finally:
        if bridge is not None:
            bridge.prepare_cleanup()
        session.cleanup()
        if bridge is not None:
            bridge.close()
        if getattr(args, "result_json", None):
            import json
            output = Path(args.result_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            result_data["cleanup"] = {"phase": session.state.phase.name,
                "landing_attempted": bool(session._land_command_sent),
                "landing_confirmed": session.state.phase.name in {"LANDED","DISARMED","CONTROL_RELEASED"},
                "control_released": session.state.phase.name == "CONTROL_RELEASED",
                "manual_intervention_required": session.state.phase.name == "MANUAL_INTERVENTION_REQUIRED"}
            result_data.update(landing_confirmed=result_data['cleanup']['landing_confirmed'],
                disarmed=session.state.phase.name in {'DISARMED','CONTROL_RELEASED'},
                api_control_released=result_data['cleanup']['control_released'])
            output.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _configure_logging(quiet=args.quiet, log_file=args.log_file)
    if args.mode == "manual":
        return _run_manual(args)
    else:
        return _run_auto(args)


if __name__ == "__main__":
    import signal
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    sys.exit(main())

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flight_mode")


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
    p.add_argument("--reset-vehicle-on-start", action="store_true",
                   default=False,
                   help="Reset the AirSim vehicle to its configured spawn pose "
                        "after connecting and before takeoff. Useful when UE "
                        "stays open between repeated auto runs.")
    return p.parse_args()


def _acquire_lock_or_die(mode_label: str):
    from flight_modes.shared_flight_session import SharedFlightSession
    fh = SharedFlightSession.acquire_lock(mode_label)
    if fh is None:
        print("\n  ERROR: Another flight mode instance is already running.\n"
              "  Only one flight mode can control Drone1 at a time.\n")
        sys.exit(5)
    return fh


def _run_manual(args: argparse.Namespace) -> int:
    if args.manual_input == "gamepad":
        return _run_manual_gamepad(args)

    from flight_modes.manual_mode import ManualMode, ManualControlType
    from flight_modes.shared_flight_session import SharedFlightSession

    lock_fh = _acquire_lock_or_die("manual")
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

    lock_fh = _acquire_lock_or_die("manual-gamepad")
    logger.info("Starting manual-gamepad mode.")

    config = load_manual_gamepad_config(args.manual_gamepad_config)
    if args.manual_disable_safety:
        logger.warning("Collision guard DISABLED via --manual-disable-safety.")
        config = config.with_collision_guard(False)

    session = SharedFlightSession(
        settings_json=args.settings_json or "",
        mode="manual-gamepad",
        target_z_ned=args.target_z or -1.0,
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

    lock_fh = _acquire_lock_or_die("auto")
    logger.info("Starting auto mode (LiDAR avoidance).")

    flight_config = args.flight_config
    default_flight_config = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
    flight_config_name = Path(flight_config).name.lower() if flight_config else ""
    if (
        args.local_navigation_mode == "trajectory"
        and (
            args.flight_config == default_flight_config
            or flight_config_name == "minimal_flight.yaml"
        )
    ):
        flight_config = str(_PROJECT_ROOT / "configs" / "trajectory_flight.yaml")
        logger.info(
            "trajectory_mode_default_flight_config  path=%s",
            flight_config,
        )

    # Build CLI overrides (only non-None values win over YAML)
    cli_overrides = {}
    if args.target_z is not None:
        cli_overrides["target_z_ned"] = float(args.target_z)
    if args.max_duration is not None:
        cli_overrides["max_flight_duration_s"] = float(args.max_duration)
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
        flight_config, cli_overrides if cli_overrides else None
    )
    logger.info(
        "auto_flight_params  config=%s  target_z=%.2f  max_duration=%.1f  "
        "geofence=%.1f  command_duration=%.2f",
        flight_config,
        params.target_z_ned,
        params.max_flight_duration_s,
        params.geofence_radius_m,
        params.command_duration_s,
    )

    session = SharedFlightSession(
        settings_json=args.settings_json,
        mode="auto",
        target_z_ned=params.target_z_ned,
    )
    session._lock_fh = lock_fh
    session._owns_lock = True

    try:
        session.initialize()
        if args.reset_vehicle_on_start:
            session.reset_vehicle()

        auto = AutomaticMode(
            session,
            perception_config_path=args.perception_config,
            flight_config_path=flight_config,
            params=params,
            cli_overrides=cli_overrides if cli_overrides else None,
        )
        result = auto.run()
        print(f"\n  Result: {result}")
        return 0 if result.success else 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error("Auto mode error: %s", e)
        return 1
    finally:
        session.cleanup()


def main() -> int:
    args = _parse_args()
    if args.mode == "manual":
        return _run_manual(args)
    else:
        return _run_auto(args)


if __name__ == "__main__":
    sys.exit(main())

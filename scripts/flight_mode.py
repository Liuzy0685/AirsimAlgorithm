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
        manual = ManualMode(session, control_type=ctrl_type)
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

    # Build CLI overrides (only non-None values win over YAML)
    cli_overrides = {}
    if args.target_z is not None:
        cli_overrides["target_z_ned"] = float(args.target_z)
    if args.max_duration is not None:
        cli_overrides["max_flight_duration_s"] = float(args.max_duration)
    cli_overrides["planner_mode"] = args.planner_mode
    cli_overrides["guided_apf_control"] = args.guided_apf_control
    if args.recovery_test_trigger is not None:
        cli_overrides["recovery_test_trigger"] = args.recovery_test_trigger

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
    )
    session._lock_fh = lock_fh
    session._owns_lock = True

    try:
        session.initialize()

        auto = AutomaticMode(
            session,
            perception_config_path=args.perception_config,
            flight_config_path=args.flight_config,
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

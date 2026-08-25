#!/usr/bin/env python
"""
Read-only sensor smoke test.

Reads state, LiDAR, and collision data repeatedly WITHOUT:
- enableApiControl
- armDisarm
- takeoff
- any velocity commands

Usage::

    python scripts/sensor_smoke_test.py [--frames N] [--interval S]

Exit with Ctrl+C at any time — no flight commands are sent on exit.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so intra-package imports work.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from configs.runtime_config import load_lidar_runtime_config
from sensors.lidar_reader import LidarReader
from sensors.state_reader import StateReader
from sensors.collision_reader import CollisionReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "vehicle.yaml")
DEFAULT_FRAMES = 50
DEFAULT_INTERVAL = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only AirSim sensor smoke test"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to vehicle.yaml (default: configs/vehicle.yaml)",
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help=f"Number of frames to read (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL,
        help=f"Seconds between frames (default: {DEFAULT_INTERVAL})",
    )
    args = parser.parse_args()

    # Graceful shutdown on Ctrl+C.
    running = True

    def _on_sigint(_sig, _frame):
        nonlocal running
        print("\nCtrl+C received — stopping (no flight commands sent).")
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    # --- Validate config BEFORE connecting (ROUND 2.3) -----------------------
    try:
        lidar_config = load_lidar_runtime_config(args.config)
    except Exception as exc:
        print(f"\nERROR in LiDAR config: {exc}")
        sys.exit(1)
    print(f"LiDAR config: timeout={lidar_config.frame_timeout_seconds}s, "
          f"max_consecutive_invalid={lidar_config.max_consecutive_invalid}")

    # --- Connect -----------------------------------------------------------
    adapter = AirSimClientAdapter(
        config_path=args.config,
        readonly=True,
    )

    try:
        adapter.connect()
    except Exception as exc:
        logger.error("Connection failed: %s", exc)
        print(f"\nFAILED to connect: {exc}")
        print("Is UE4 + AirSim running? Is port 41451 reachable?")
        sys.exit(1)

    # --- Validate vehicle presence ------------------------------------------
    try:
        vehicles = adapter.list_vehicles()
        print(f"Vehicles found: {vehicles}")
        if adapter.vehicle_name not in vehicles:
            print(
                f"ERROR: {adapter.vehicle_name!r} NOT in vehicle list! "
                f"Check settings.json."
            )
            adapter.close()
            sys.exit(1)
        print(f"Vehicle {adapter.vehicle_name!r} confirmed present.")
    except Exception as exc:
        logger.error("listVehicles() failed: %s", exc)
        adapter.close()
        sys.exit(1)

    # --- Sensors ------------------------------------------------------------
    lidar = LidarReader(adapter, frame_timeout_seconds=lidar_config.frame_timeout_seconds)
    state_reader = StateReader(adapter)
    collision_reader = CollisionReader(adapter)

    print(f"\nStarting smoke test: {args.frames} frames @ {args.interval}s interval\n")
    print(
        f"{'Frame':>5s} | {'LiDAR ts_ns':>20s} | {'Pts':>5s} | "
        f"{'Valid':>5s} | {'Invalid reason':>16s} | "
        f"{'Position NED (m)':>28s} | {'Roll/Pitch/Yaw (rad)':>26s} | {'Collision':>9s}"
    )
    print("-" * 145)

    frame_num = 0
    while running and frame_num < args.frames:
        frame_num += 1
        loop_start = time.monotonic()

        # Read all sensors.
        try:
            state = state_reader.read()
        except Exception as exc:
            logger.error("getMultirotorState() error: %s", exc)
            state = None

        try:
            lidar_frame = lidar.read()
        except Exception as exc:
            logger.error("getLidarData() error: %s", exc)
            lidar_frame = None

        try:
            collision = collision_reader.read()
        except Exception as exc:
            logger.error("simGetCollisionInfo() error: %s", exc)
            collision = None

        # --- Format output ---------------------------------------------------
        if lidar_frame is not None:
            ts_str = str(lidar_frame.raw_timestamp_ns)
            pts = lidar_frame.point_count
            valid_str = "OK" if lidar_frame.frame_valid else "FAIL"
            reason = lidar_frame.invalid_reason or "-"
        else:
            ts_str = "N/A"
            pts = 0
            valid_str = "FAIL"
            reason = "rpc_error"

        if state is not None:
            pos_str = f"[{state.position_ned_m[0]:7.2f}, {state.position_ned_m[1]:7.2f}, {state.position_ned_m[2]:7.2f}]"
            rpy_str = f"[{state.roll_rad:7.3f}, {state.pitch_rad:7.3f}, {state.yaw_rad:7.3f}]"
        else:
            pos_str = "[   N/A,    N/A,    N/A]"
            rpy_str = "[   N/A,    N/A,    N/A]"

        if collision is not None:
            coll_str = "YES" if collision.has_collided else "no"
        else:
            coll_str = "N/A"

        print(
            f"{frame_num:5d} | {ts_str:>20s} | {pts:5d} | "
            f"{valid_str:>5s} | {reason:>16s} | "
            f"{pos_str:>28s} | {rpy_str:>26s} | {coll_str:>9s}"
        )

        # Sleep for the remainder of the interval.
        elapsed = time.monotonic() - loop_start
        if elapsed < args.interval:
            time.sleep(args.interval - elapsed)

    # --- Clean exit (NO flight commands) ------------------------------------
    print(f"\n{frame_num} frames collected.")
    adapter.close()
    print("Done.  No flight commands were sent.")
    sys.exit(0)


if __name__ == "__main__":
    main()

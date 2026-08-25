#!/usr/bin/env python
"""Read-only LiDAR axis calibration helper.  Sends NO flight commands.

ROUND 3.3: Full rewrite supporting:
  --config       Path to vehicle.yaml
  --frames       Number of LiDAR frames to collect
  --direction    Which axis to calibrate: right|left|up|down
  --max-distance Maximum distance to consider for nearest-cluster (m)
  --output       Path to save calibration results

Algorithm:
  1. Collect N LiDAR frames (read-only, no control commands).
  2. For each frame, estimate a representative obstacle position by
     taking the median x/y/z of the nearest cluster of points within
     ``--max-distance`` in the specified direction.
  3. Report the per-frame and aggregate median positions.
  4. Save UTF-8 output to --output if provided.

The user MUST place exactly one nearby obstacle in the specified
direction relative to the drone body frame.

IMPORTANT (yaw≈π in current UE4 scene):
  "right" means the drone's actual body-right as seen in the UE
  viewport.  "left" means body-left.  "up" is body-up.  "down" is
  body-down.  Because the drone spawns facing -Y (world), these do
  NOT align with world NED.  Place obstacles by looking at the drone
  in the UE viewport.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from adapters.airsim_client import AirSimClientAdapter
from configs.runtime_config import load_lidar_runtime_config
from sensors.lidar_reader import LidarReader

DIRECTION_AXIS = {
    "right": 1,   # +Y in SensorLocalFrame
    "left": 1,    # -Y in SensorLocalFrame (negated later)
    "up": 2,      # -Z in SensorLocalFrame (negated later... actually +Z=down, so up=-Z)
    "down": 2,    # +Z in SensorLocalFrame
}

DIRECTION_SIGN = {
    "right": 1,   # positive Y
    "left": -1,   # negative Y
    "up": -1,     # negative Z (SensorLocalFrame: +Z=down, so up=-Z)
    "down": 1,    # positive Z (down=+Z)
}


def main():
    p = argparse.ArgumentParser(
        description="Read-only LiDAR axis calibration — ROUND 3.3"
    )
    p.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "configs" / "vehicle.yaml"),
        help="Path to vehicle.yaml",
    )
    p.add_argument("--frames", type=int, default=10, help="Number of LiDAR frames")
    p.add_argument(
        "--direction",
        required=True,
        choices=["right", "left", "up", "down"],
        help="Which drone body direction has the obstacle",
    )
    p.add_argument(
        "--max-distance",
        type=float,
        default=5.0,
        help="Max distance (m) to consider for nearest obstacle cluster",
    )
    p.add_argument("--output", default=None, help="Output file path (UTF-8)")
    args = p.parse_args()

    # --- Load config ---
    try:
        runtime_cfg = load_lidar_runtime_config(args.config)
    except Exception as e:
        print(f"ERROR loading config: {e}", file=sys.stderr)
        sys.exit(1)

    vehicle_name = runtime_cfg.airsim.vehicle_name
    lidar_name = runtime_cfg.airsim.lidar_name
    direction = args.direction

    print(f"LiDAR Axis Calibration — ROUND 3.3")
    print(f"  Vehicle: {vehicle_name}")
    print(f"  LiDAR:   {lidar_name}")
    print(f"  Direction: {direction}")
    print(f"  Max distance: {args.max_distance} m")
    print(f"  Frames: {args.frames}")
    print()
    print("⚠️  INSTRUCTIONS:")
    print(f"   Place exactly ONE nearby obstacle in the drone's")
    print(f"   body-{direction} direction as seen in the UE viewport.")
    print(f"   (yaw≈π in the current scene, so body-right ≠ world +Y)")
    print(f"   Keep all other directions clear.")
    print()

    # --- Connect (read-only) ---
    adapter = AirSimClientAdapter(config_path=args.config, readonly=True)
    try:
        adapter.connect()
    except Exception as e:
        print(f"Connect failed: {e}", file=sys.stderr)
        sys.exit(1)

    lidar = LidarReader(adapter, frame_timeout_seconds=runtime_cfg.frame_timeout_seconds)
    axis = DIRECTION_AXIS[direction]
    sign = DIRECTION_SIGN[direction]
    axis_label = {1: "Y", 2: "Z"}[axis]

    output_lines = []
    header = (
        f"{'Frame':>5s} | "
        f"{'median_x':>10s} | {'median_y':>10s} | {'median_z':>10s} | "
        f"{axis_label}_{'pos' if sign > 0 else 'neg'}_{'median':>10s} | "
        f"{'n_pts':>6s}"
    )
    print(header)
    output_lines.append(header)

    all_medians = []

    for i in range(args.frames):
        try:
            frame = lidar.read()
            if not frame.frame_valid:
                print(f"{i+1:5d} | INVALID: {frame.invalid_reason}")
                continue

            pts = frame.point_cloud_sensor  # N×3 array, SensorLocalFrame
            if pts.size == 0:
                print(f"{i+1:5d} | Empty point cloud")
                continue

            # Filter: only points in the specified direction within max_distance
            dir_values = pts[:, axis] * sign
            dist = np.linalg.norm(pts, axis=1)

            mask = (dir_values > 0) & (dist <= args.max_distance)
            if not np.any(mask):
                print(
                    f"{i+1:5d} | No points in {direction} direction "
                    f"within {args.max_distance}m"
                )
                continue

            cluster = pts[mask]

            # Take median of the cluster (more robust than mean)
            median_x = float(np.median(cluster[:, 0]))
            median_y = float(np.median(cluster[:, 1]))
            median_z = float(np.median(cluster[:, 2]))
            dir_median = float(np.median(cluster[:, axis] * sign))
            n_pts = int(np.sum(mask))

            all_medians.append((median_x, median_y, median_z))

            line = (
                f"{i+1:5d} | "
                f"{median_x:10.4f} | {median_y:10.4f} | {median_z:10.4f} | "
                f"{dir_median:10.4f} | "
                f"{n_pts:6d}"
            )
            print(line)
            output_lines.append(line)

        except Exception as e:
            print(f"{i+1:5d} | RPC error: {e}")
        time.sleep(0.2)

    adapter.close()

    # --- Summary ---
    if all_medians:
        arr = np.array(all_medians)
        agg_x = float(np.median(arr[:, 0]))
        agg_y = float(np.median(arr[:, 1]))
        agg_z = float(np.median(arr[:, 2]))
        summary = (
            f"\nAggregate median (n={len(all_medians)} frames):\n"
            f"  x = {agg_x:.4f} m\n"
            f"  y = {agg_y:.4f} m\n"
            f"  z = {agg_z:.4f} m\n"
        )
    else:
        summary = "\nNo valid frames collected."

    print(summary)
    output_lines.append(summary)
    print("\nDone. No flight commands sent.")

    # --- Save output ---
    if args.output:
        out_path = Path(args.output)
        out_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

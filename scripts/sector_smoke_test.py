#!/usr/bin/env python
"""Read-only sector smoke test — ROUND 3.3.  Sends NO flight commands.

FOV validation is FAIL-CLOSED: if --settings-json is not provided,
the file is missing/malformed, the LiDAR config is invalid, required
legacy sectors are not fully observable, or max_range exceeds the
physical Range, the script exits with a non-zero code BEFORE any
RPC connection is attempted.

FOV fully compatible → read-only connection → LiDAR → filter →
sectorization → legacy distances.
"""
from __future__ import annotations
import argparse, logging, math, signal, sys, time, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter
from configs.runtime_config import load_lidar_runtime_config
from perception.perception_config import load_perception_config, SectorDef
from perception.pointcloud_filter import filter_pointcloud
from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
from perception.sensor_fov import (
    load_lidar_fov,
    validate_sector_fov_coverage,
    check_max_range_against_fov,
)
from sensors.lidar_reader import LidarReader
from utils.consecutive_tracker import ConsecutiveInvalidTracker

DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "vehicle.yaml")
DEFAULT_PERCEPTION = str(_PROJECT_ROOT / "configs" / "perception.yaml")
DEFAULT_FRAMES = 20
DEFAULT_INTERVAL = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sector_smoke_test")


def _fail(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    parser = argparse.ArgumentParser(description="Read-only sector smoke test")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--perception-config", default=DEFAULT_PERCEPTION)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--settings-json",
        required=True,
        help="Path to AirSim settings.json for FOV validation (REQUIRED)",
    )
    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Load vehicle config (for vehicle_name, lidar_name)
    # ═══════════════════════════════════════════════════════════════
    try:
        lidar_config = load_lidar_runtime_config(args.config)
    except Exception as e:
        _fail(f"Vehicle config error: {e}")

    vehicle_name = lidar_config.airsim.vehicle_name
    lidar_name = lidar_config.airsim.lidar_name

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Load perception config
    # ═══════════════════════════════════════════════════════════════
    try:
        perception_cfg = load_perception_config(args.perception_config)
    except Exception as e:
        _fail(f"Perception config error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Validate settings JSON exists
    # ═══════════════════════════════════════════════════════════════
    settings_path = Path(args.settings_json)
    if not settings_path.is_file():
        _fail(f"Settings JSON not found: {settings_path}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Load and validate FOV from settings JSON
    # ═══════════════════════════════════════════════════════════════
    try:
        fov = load_lidar_fov(str(settings_path), vehicle_name, lidar_name)
    except Exception as e:
        _fail(f"FOV load failed: {e}")

    print(f"LiDAR FOV loaded: vehicle={vehicle_name!r}, lidar={lidar_name!r}")
    print(
        f"  Horizontal: [{fov.horizontal_start_deg}, {fov.horizontal_end_deg}] "
        f"({'full circle' if fov.horizontal_full_circle else 'partial'})"
    )
    print(f"  Vertical:   [{fov.vertical_lower_deg}, {fov.vertical_upper_deg}]")
    print(f"  Range:      {fov.range_m} m")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Validate max_range does not exceed LiDAR Range
    # ═══════════════════════════════════════════════════════════════
    range_errors = check_max_range_against_fov(perception_cfg, fov)
    if range_errors:
        for err in range_errors:
            print(f"  RANGE ERROR: {err}", file=sys.stderr)
        _fail(
            f"max_range exceeds LiDAR Range ({fov.range_m}m). "
            f"Fix perception.yaml or increase LiDAR Range."
        )

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Validate sector FOV coverage
    # ═══════════════════════════════════════════════════════════════
    fov_statuses = validate_sector_fov_coverage(perception_cfg, fov)

    # Build observability map — unknown legacy name is an internal error
    fov_by_legacy = {}
    for s in fov_statuses:
        fov_by_legacy[s.legacy_name] = s

    # Extract the REQUIRED 16 legacy names from perception config
    required_legacy = {s.legacy_name for s in perception_cfg.sectorization.sectors}

    unobservable = [
        name for name in required_legacy
        if name in fov_by_legacy and not fov_by_legacy[name].fully_observable
    ]
    partially = [
        name for name in required_legacy
        if name in fov_by_legacy and fov_by_legacy[name].partially_observable
    ]
    missing = [name for name in required_legacy if name not in fov_by_legacy]

    if missing:
        _fail(f"Internal error: sectors missing from FOV status: {missing}")

    if unobservable or partially:
        print("\nFOV INCOMPATIBLE", file=sys.stderr)
        if unobservable:
            print(f"  Unobservable sectors: {unobservable}", file=sys.stderr)
        if partially:
            print(f"  Partially observable sectors: {partially}", file=sys.stderr)
        for name in sorted(set(unobservable + partially)):
            st = fov_by_legacy[name]
            print(f"  - {name}: {st.note}", file=sys.stderr)
        print(
            f"\n  LiDAR vertical FOV: [{fov.vertical_lower_deg}, {fov.vertical_upper_deg}]",
            file=sys.stderr,
        )
        print(
            f"  Suggested: extend vertical FOV to at least ±30°\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nFOV FULLY COMPATIBLE — all 16 required legacy sectors observable.\n")

    # ═══════════════════════════════════════════════════════════════
    # Step 7: Build FOV observability dict for sector conversion
    # ═══════════════════════════════════════════════════════════════
    fov_observability = {}
    for s in fov_statuses:
        # Map by internal sector name
        # Find matching sector from perception config
        for sdef in perception_cfg.sectorization.sectors:
            if sdef.legacy_name == s.legacy_name:
                fov_observability[sdef.name] = (
                    s.fully_observable,
                    min(s.horizontal_coverage_fraction, s.vertical_coverage_fraction),
                )
                break

    # ═══════════════════════════════════════════════════════════════
    # Step 8: Connect to AirSim (read-only)
    # ═══════════════════════════════════════════════════════════════
    adapter = AirSimClientAdapter(config_path=args.config, readonly=True)
    try:
        adapter.connect()
    except Exception as e:
        _fail(f"Failed to connect to AirSim: {e}")

    try:
        vehicles = adapter.list_vehicles()
        if vehicle_name not in vehicles:
            _fail(f"Vehicle {vehicle_name!r} not in {vehicles}")
    except Exception as e:
        _fail(f"listVehicles failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Step 9: Set up pipeline components
    # ═══════════════════════════════════════════════════════════════
    lidar = LidarReader(adapter, frame_timeout_seconds=lidar_config.frame_timeout_seconds)
    pc = perception_cfg.pointcloud
    sz = perception_cfg.sectorization

    # Use validated SectorDef objects directly
    sector_defs: list = list(sz.sectors)

    max_inv = lidar_config.max_consecutive_invalid
    tracker = ConsecutiveInvalidTracker(threshold=max_inv)

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None

    def _w(line):
        print(line)
        if out_fh:
            out_fh.write(line + "\n")
            out_fh.flush()

    running = True

    def _sig(_, __):
        nonlocal running
        _w("\nCtrl+C — stopping.")
        running = False

    signal.signal(signal.SIGINT, _sig)

    _w(f"\nSector smoke test: {args.frames} frames @ {args.interval}s")
    _w(f"Strategy: {sz.default_distance_strategy}, k={sz.nearest_k}")
    hdr = f"{'Frm':>4s} | {'Raw':>5s} | {'Flt':>5s} | {'minD':>6s} | "
    for k in ["front", "fLeft", "fRight", "left", "right", "up", "down", "back"]:
        hdr += f"{k:>7s} | "
    hdr += "Valid"
    _w(hdr)
    _w("-" * 140)

    frame_num = 0
    while running and frame_num < args.frames:
        frame_num += 1
        t0 = time.monotonic()

        # --- LiDAR ---
        try:
            lidar_frame = lidar.read()
        except Exception as e:
            cnt = tracker.record_failure()
            _w(f"  INVALID: LiDAR RPC: {e}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        if not lidar_frame.frame_valid:
            cnt = tracker.record_failure()
            _w(f"  INVALID: LiDAR: {lidar_frame.invalid_reason}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        # --- Filter ---
        filter_result = filter_pointcloud(
            lidar_frame.point_cloud_sensor,
            min_range_m=pc.min_range_m,
            max_range_m=pc.max_range_m,
            self_exclusion={
                "enabled": pc.self_exclusion.enabled,
                "x_min_m": pc.self_exclusion.x_min_m,
                "x_max_m": pc.self_exclusion.x_max_m,
                "y_min_m": pc.self_exclusion.y_min_m,
                "y_max_m": pc.self_exclusion.y_max_m,
                "z_min_m": pc.self_exclusion.z_min_m,
                "z_max_m": pc.self_exclusion.z_max_m,
            },
            voxel_downsample=pc.voxel_downsample.enabled,
            voxel_size_m=pc.voxel_downsample.voxel_size_m,
        )
        if not filter_result.valid:
            cnt = tracker.record_failure()
            _w(f"  INVALID: Filter: {filter_result.invalid_reason}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        # --- Sector ---
        try:
            dd = pointcloud_to_directional_distances(
                filter_result.filtered_points_sensor,
                sector_defs=sector_defs,
                default_max_range_m=sz.default_max_range_m,
                default_min_points=sz.default_min_points,
                distance_strategy=sz.default_distance_strategy,
                nearest_k=sz.nearest_k,
                percentile=sz.percentile,
                frame_valid=lidar_frame.frame_valid,
                raw_timestamp_ns=lidar_frame.raw_timestamp_ns,
                received_monotonic_seconds=lidar_frame.received_monotonic_seconds,
                fov_compatible=True,
                fov_invalid_sectors=(),
                fov_observability=fov_observability,
            )
        except Exception as e:
            cnt = tracker.record_failure()
            _w(f"  INVALID: Sectorization: {e}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        if not dd.frame_valid:
            cnt = tracker.record_failure()
            _w(f"  INVALID: Sector: {dd.invalid_reason}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        # --- Legacy ---
        try:
            legacy = dd.to_legacy_ray_distances()
        except Exception as e:
            cnt = tracker.record_failure()
            _w(f"  INVALID: Legacy mapping: {e}  [{cnt}/{max_inv}]")
            if tracker.should_stop:
                _w("MAX INVALID — stopping.")
                break
            _sleep(t0, args.interval)
            continue

        # Full success → reset consecutive invalid counter
        tracker.record_success()

        # --- Output ---
        def _d(k):
            return f"{legacy.get(k, float('inf')):6.2f}" if k in legacy else "    NA"

        _w(
            f"{frame_num:4d} | {lidar_frame.point_count:5d} | "
            f"{filter_result.output_point_count:5d} | {dd.minimum_distance_m:6.2f} | "
            f"{_d('front')} | {_d('frontLeft')} | {_d('frontRight')} | "
            f"{_d('left')} | {_d('right')} | "
            f"{_d('up')} | {_d('down')} | {_d('back')} | OK"
        )

        # Enhanced vertical detail with FOV observability
        for k in [
            "up", "down",
            "frontUp", "frontDown",
            "leftUp", "rightUp",
            "leftDown", "rightDown",
        ]:
            sector = dd.sectors.get(k)
            # Look up FOV status from fov_by_legacy
            obs_name = None
            for sdef in sector_defs:
                if sdef.name == k:
                    obs_name = sdef.legacy_name
                    break
            obs = False
            if obs_name and obs_name in fov_by_legacy:
                obs = fov_by_legacy[obs_name].fully_observable

            if sector:
                _w(
                    f"  {k:>10s}: dist={sector.distance_m:6.2f}  "
                    f"has_return={str(sector.has_return):5s}  "
                    f"pts={sector.point_count:4d}  "
                    f"observable_by_fov={str(obs):5s}"
                )
            else:
                _w(f"  {k:>10s}: (missing sector)  observable_by_fov={str(obs):5s}")

        _sleep(t0, args.interval)

    _w(f"\n{frame_num} frames. Done. No flight commands sent.")
    if out_fh:
        out_fh.close()
    sys.exit(0)


def _sleep(t0, interval):
    elapsed = time.monotonic() - t0
    if elapsed < interval:
        time.sleep(interval - elapsed)


if __name__ == "__main__":
    main()

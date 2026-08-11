# UAV AirSim Avoidance

**Round 10** — CBMBA A* Migration Phase 1 ✅ PASS.
Pure Python A* planner migrated from old JS, 69 unit tests, shadow integration.
748 tests, 0 failures.  Next: Phase 2 obstacle pipeline + path tracking.

**Round 9** — Recovery Takeover Phase 1 ✅ PASS.
Real AirSim verified: stuck → backward escape → APF handoff;
oscillation → lateral sidestep → APF recovery.

## Project Goal

Build a Python-based autonomous obstacle-avoidance pipeline that interacts
with a Microsoft AirSim multirotor simulation running inside Unreal Engine
4.27.  The system reads LiDAR point clouds, drone state, and collision
info, and will eventually produce safe velocity commands.

## Directory Structure

```
uav-airsim-avoidance/
├── adapters/
│   ├── __init__.py
│   └── airsim_client.py           # Lazy-connecting AirSim RPC wrapper
├── models/
│   ├── __init__.py
│   ├── lidar_frame.py              # LiDAR frame data model (SensorLocalFrame)
│   ├── vehicle_state.py            # NED vehicle state data model
│   ├── collision_state.py          # Collision info data model
│   ├── sector_measurement.py       # Per-sector distance + FOV observability
│   └── directional_distances.py    # Frame-level result with legacy mapping
├── perception/
│   ├── __init__.py
│   ├── perception_config.py        # Strict YAML config loader (SectorDef)
│   ├── pointcloud_filter.py        # NaN/range/self-exclusion/voxel filter
│   ├── pointcloud_to_sectors.py    # Point-cloud → 16-sector conversion
│   └── sensor_fov.py               # LiDAR FOV loader + sector-coverage validator
├── sensors/
│   ├── __init__.py
│   ├── lidar_reader.py             # getLidarData() → LidarFrame
│   ├── state_reader.py             # getMultirotorState() → VehicleState
│   └── collision_reader.py         # simGetCollisionInfo() → CollisionState
├── control/
│   ├── __init__.py
│   └── velocity_controller.py      # Safe velocity send (read-only by default)
├── utils/
│   ├── __init__.py
│   └── consecutive_tracker.py      # ConsecutiveInvalidTracker (ROUND 3.3)
├── planners/
│   ├── __init__.py
│   ├── improved_potential_field.py  # APF reactive avoidance (from OldProject JS)
│   ├── local_recovery.py            # Stuck + oscillation detection (pure calc)
│   ├── recovery_commander.py        # Recovery command + state machine
│   └── cbmba_astar.py               # CBMBA A* path planner (from OldProject JS)
├── configs/
│   ├── vehicle.yaml                # Runtime configuration
│   ├── perception.yaml             # 16-sector definitions
│   └── settings.example.json       # Example AirSim settings (DO NOT overwrite real)
├── scripts/
│   ├── sensor_smoke_test.py        # Read-only sensor acquisition test
│   ├── sector_smoke_test.py        # Read-only sector smoke test (FOV-gated)
│   └── lidar_axis_calibration.py   # Read-only LiDAR axis calibration
├── tests/
│   ├── fixtures/                   # Self-contained test settings JSON files
│   ├── test_airsim_client_config.py
│   ├── test_collision_reader.py
│   ├── test_consecutive_tracker.py
│   ├── test_fov_fail_closed.py
│   ├── test_legacy_ray_compatibility.py
│   ├── test_lidar_parsing.py
│   ├── test_perception_config.py
│   ├── test_pointcloud_filter.py
│   ├── test_pointcloud_to_sectors.py
│   ├── test_runtime_config.py
│   ├── test_sensor_fov.py
│   ├── test_state_parsing.py
│   └── test_velocity_limits.py
├── requirements.txt
├── README.md                       # This file
└── ROUND_3_FIX3_REPORT.md          # Round 3.3 report
```

## Relationship to the Old Web Project

The old project lives under `OldProject/Drone-feature-yu/` and is a
JavaScript / Three.js / Web Worker application that simulates
a drone in a Rapier physics environment.

**This project is a from-scratch Python rewrite** that replaces the
simulated physics with real AirSim RPC calls.  Old source files are
never modified.  The audit documents in
`OldProject/Drone-feature-yu/audit_results/` are the source of truth
for coordinate systems, migration plans, and safety constraints.

## Round 3.3 Capabilities

- ✅ Point-cloud sectorization (16 spatial sectors matching old 16-ray layout)
- ✅ FOV fail-closed validation (no RPC connection unless FOV is fully compatible)
- ✅ Float32/float64 boundary snapping for deterministic sector classification
- ✅ LiDAR metadata validation (SensorType==6, Enabled, DataFrame, zero-width FOV)
- ✅ Horizontal + vertical FOV coverage with full/partial/unobservable classification
- ✅ ConsecutiveInvalidTracker for safety gating
- ✅ All tests self-contained (no external file dependencies)

## AirSim Settings

The **real** AirSim settings file is maintained by the user at:

```
C:\Users\Liuziyi\Documents\AirSim\settings.json
```

An example is provided at `configs/settings.example.json` for reference
only — **do NOT copy it over the real settings file** without a backup.

Key configuration values for this project:

| Parameter       | Value              |
|-----------------|--------------------|
| RPC IP          | 127.0.0.1          |
| RPC Port        | 41451              |
| Vehicle Name    | Drone1             |
| LiDAR Name      | LidarSensor1       |
| LiDAR DataFrame | SensorLocalFrame   |

**Important FOV note:** The real LiDAR currently has ±15° vertical FOV.
For full sector coverage, the vertical FOV must be at least ±30°.
See `LIDAR_FOV_COMPATIBILITY.md` for details.

## How to Set the AirSim PythonClient Path

This project does **not** require `pip install airsim`.  The `airsim`
package comes from the AirSim source tree.  Two loading methods are
supported:

### Method 1 — PYTHONPATH (recommended)

```powershell
# In PowerShell:
$env:PYTHONPATH = "D:\30817\Adrone\AirSim\AirSim-main\PythonClient;" + $env:PYTHONPATH
python scripts/sector_smoke_test.py --settings-json "C:\Users\Liuziyi\Documents\AirSim\settings.json"
```

### Method 2 — Environment Variable

```powershell
$env:AIRSIM_PYTHONCLIENT_PATH = "D:\30817\Adrone\AirSim\AirSim-main\PythonClient"
python scripts/sector_smoke_test.py --settings-json "C:\Users\Liuziyi\Documents\AirSim\settings.json"
```

## Running Tests

### Unit Tests (no UE4 required)

```bash
cd uav-airsim-avoidance
set PYTHONDONTWRITEBYTECODE=1
set AIRSIM_PYTHONCLIENT_PATH=
set PYTHONPATH=
py -m pytest -p no:cacheprovider tests -v
```

All 748 tests pass without any external dependencies — no AirSim,
no UE4, no reference files, no user settings.json required.

### Read-Only Sector Smoke Test (requires UE4 + AirSim)

```bash
# FOV validation REQUIRED:
python scripts/sector_smoke_test.py \
  --settings-json "C:\Users\Liuziyi\Documents\AirSim\settings.json" \
  --frames 20 --interval 0.2
```

The script will:
1. Validate LiDAR FOV from settings.json BEFORE connecting
2. Exit with FOV INCOMPATIBLE if any required sector is not fully observable
3. Exit if max_range exceeds LiDAR Range
4. Only connect to AirSim (read-only) when FOV is fully compatible
5. Read LiDAR frames, filter, sectorize, and produce legacy ray distances
6. Output enhanced per-sector detail with FOV observability flags

### LiDAR Axis Calibration

```bash
python scripts/lidar_axis_calibration.py \
  --direction right --max-distance 5.0 --frames 10 --output calib_right.txt
```

See `LIDAR_AXIS_CALIBRATION.md` for the full procedure.

## Current Capabilities (Round 10)

- ✅ Automatic API control acquisition / arming / takeoff / landing (SharedFlightSession)
- ✅ 16-sector LiDAR perception with FOV validation
- ✅ Reactive planner (front-blocked → sidestep)
- ✅ APF reactive avoidance (ImprovedPotentialField, from OldProject JS)
- ✅ Recovery detection: stuck (XY delta < 0.15m for ≥2.5s) + oscillation (vy sign flips)
- ✅ Recovery Takeover Phase 1: APF → Recovery → cooldown → APF (679 tests, 0 failures)
- ✅ CLI test trigger: `--recovery-test-trigger {stuck,oscillation}` (default disabled)
- ✅ CBMBA A* shadow planner: pure Python migration from old JS, computes + logs paths (no dispatch)

## What This Round Does NOT Do (yet)

- ❌ CBMBA A* takeover or command dispatch (shadow only, Phase 1)
- ❌ CBMBA LiDAR→obstacle pipeline (Phase 2)
- ❌ CBMBA path tracking / following (Phase 2+)
- ❌ AvoidanceSupervisor full mode scheduler
- ❌ ThreatAssessor
- ❌ YOLO / camera
- ❌ D3QN / RL inference

## Coordinate Systems

### World: NED (North-East-Down)

```
+X = North / forward
+Y = East  / right
+Z = Down
```

Used by: `VehicleState`, `CollisionState`, `send_velocity_world_ned()`.

### LiDAR: SensorLocalFrame

```
+X = forward
+Y = right
+Z = down
```

Used by: `LidarFrame.point_cloud_sensor`.

Heading=0 means the drone body axes align with world NED axes.

### Body: FRD (Forward-Right-Down)

```
vx = Forward
vy = Right
vz = Down
```

Used by: `send_velocity_body_frd()`.

## License

MIT (matches the AirSim PythonClient license).

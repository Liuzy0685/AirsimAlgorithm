"""Phase 2F v2 — Resolution Sensitivity Replay with Fidelity Audit.

Standalone diagnostic. Does NOT modify any repository file.

Steps:
  1. Feed exact real logged path through CbmbaGuidance (bypass A*)
  2. Trace why the v1 replay A* path diverged from real path
  3. Re-run A* at 4 resolutions with corrected inputs

Run:  py tests/phase2f_resolution_replay.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.cbmba_astar import (
    CbmbaAStarPlanner,
    CbmbaParams,
    _obstacle_half_extents,
    _vec3_to_cell,
    _cell_to_vec3,
)
from planners.cbmba_guidance import CbmbaGuidance, CbmbaGuidanceParams, _body_xy

# ═══════════════════════════════════════════════════════════════════════
# Frame snapshot (real AirSim, Phase 2E 20s run, late frame)
# ═══════════════════════════════════════════════════════════════════════

START = [1.11, -2.20, -0.41]
GOAL = [15.00, 0.00, -0.54]

REAL_PATH_XY = [
    (1.11, -2.20),
    (0.27, -4.20),
    (4.77, -4.20),
    (6.27, -2.70),
    (15.27, 0.30),
    (15.00, 0.00),
]
# Z values are diagnostic-only for guidance; use drone Z for all
REAL_PATH_WORLD = [[p[0], p[1], START[2]] for p in REAL_PATH_XY]

# LiDAR obstacles from real frame log
OBSTACLES = [
    {"position": [7.92, -2.20, -0.41],  # front
     "footprint_half_extents": [0.0, 0.0, 0.0], "type": "lidar"},
    {"position": [1.11, 0.77, -0.41],   # right
     "footprint_half_extents": [0.0, 0.0, 0.0], "type": "lidar"},
    {"position": [3.53, -1.20, -0.41],  # frontRight
     "footprint_half_extents": [0.0, 0.0, 0.0], "type": "lidar"},
    {"position": [-5.23, 0.43, -0.41],  # backRight
     "footprint_half_extents": [0.0, 0.0, 0.0], "type": "lidar"},
]

REAL_YAW = 0.0  # body≈world (heading toward goal at (15,0))
RESOLUTIONS = [1.50, 1.00, 0.75, 0.50]
INFLATION = 1.5

# Real guidance from log:
REAL_BODY_TARGET = (1.00, -2.00)
REAL_LATERAL_OFFSET = -2.00
REAL_DIRECTION = (0.447, -0.894)


# ── helpers ──

def _max_lateral_dev(path_world, start, goal):
    sx, sy = start[0], start[1]
    gx, gy = goal[0], goal[1]
    seg_dx, seg_dy = gx - sx, gy - sy
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return 0.0
    return max(
        abs((pt[0] - sx) * seg_dy - (pt[1] - sy) * seg_dx) / seg_len
        for pt in path_world
    )


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Feed exact real path through Guidance (bypass A* entirely)
# ═══════════════════════════════════════════════════════════════════════

def step1_guidance_on_real_path():
    print("=" * 72)
    print("STEP 1 — Guidance on exact real path (no A*)")
    print("=" * 72)
    print(f"  drone_position=({START[0]:.2f},{START[1]:.2f},{START[2]:.2f})")
    print(f"  yaw={REAL_YAW:.3f} rad")
    print(f"  real_path_xy={REAL_PATH_XY}")

    guidance = CbmbaGuidance(CbmbaGuidanceParams(
        min_forward_progress=0.25,
        min_waypoint_distance=0.5,
    ))

    result = guidance.select_waypoint(
        (START[0], START[1], START[2]),
        REAL_YAW,
        REAL_PATH_WORLD,
    )

    # Also compute body_xy for each path waypoint to show crossing geometry
    px, py = START[0], START[1]
    cos_yaw = math.cos(REAL_YAW)
    sin_yaw = math.sin(REAL_YAW)
    lookahead = 1.0  # guidance_lookahead_x default

    print()
    print("  Path segments in body frame:")
    for i in range(len(REAL_PATH_WORLD) - 1):
        a = REAL_PATH_WORLD[i]
        b = REAL_PATH_WORLD[i + 1]
        ba = _body_xy(a[0], a[1], px, py, cos_yaw, sin_yaw)
        bb = _body_xy(b[0], b[1], px, py, cos_yaw, sin_yaw)
        crosses = (ba[0] < lookahead <= bb[0]) or (bb[0] < lookahead <= ba[0])
        marker = " <-- CROSSES lookahead" if crosses else ""
        print(f"    seg({i},{i + 1}): "
              f"world ({a[0]:.2f},{a[1]:.2f})→({b[0]:.2f},{b[1]:.2f})  "
              f"body ({ba[0]:.2f},{ba[1]:.2f})→({bb[0]:.2f},{bb[1]:.2f}){marker}")

    print()
    print(f"  result.valid              = {result.valid}")
    print(f"  result.source_segment     = {result.source_segment}")
    if result.target_body_xy:
        print(f"  result.body_target        = "
              f"({result.target_body_xy[0]:.2f}, {result.target_body_xy[1]:.2f})")
    print(f"  result.lateral_offset     = {result.lateral_offset_m:.2f}")
    if result.direction_body_xy:
        print(f"  result.direction          = "
              f"({result.direction_body_xy[0]:.3f}, {result.direction_body_xy[1]:.3f})")
    print(f"  result.reason             = {result.reason}")

    # Compare with real
    if result.target_body_xy:
        dx = result.target_body_xy[0] - REAL_BODY_TARGET[0]
        dy = result.target_body_xy[1] - REAL_BODY_TARGET[1]
        match = abs(dx) < 0.02 and abs(dy) < 0.02
        print()
        print(f"  Real body_target  = ({REAL_BODY_TARGET[0]:.2f}, {REAL_BODY_TARGET[1]:.2f})")
        print(f"  Match: {'YES' if match else 'NO'}  (delta=({dx:.3f},{dy:.3f}))")

    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Trace A* grid build to understand why v1 path diverged
# ═══════════════════════════════════════════════════════════════════════

def step2_trace_astar_grid():
    print()
    print("=" * 72)
    print("STEP 2 — Trace A* grid build at resolution=1.5 (same as real)")
    print("=" * 72)

    res = 1.50
    params = CbmbaParams(
        resolution=res,
        inflation_radius=INFLATION,
        max_search_nodes=2000,
        wall_penalty_radius=0,
        adaptive_long_step_cells=1,
        free_cell_search_radius=3,
        goal_layer_count=2,
        max_goal_vertical_offset=4.0,
        map_padding=8.0,
    )

    planner = CbmbaAStarPlanner(params)

    # Compute origin exactly as the planner does
    origin = planner._compute_origin(START, GOAL, OBSTACLES)
    print(f"  origin = ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    print(f"  map_padding = {params.map_padding}")

    # Build occupancy grid
    occupied, occupied_cells = planner._build_occupancy_grid(
        OBSTACLES, origin, res, INFLATION, params,
    )
    print(f"  occupied_cells = {len(occupied)} total")

    # Show each obstacle's cell footprint
    for obs in OBSTACLES:
        ext = _obstacle_half_extents(obs)
        pos = obs["position"]
        min_x = pos[0] - ext.x - INFLATION
        max_x = pos[0] + ext.x + INFLATION
        min_y = pos[1] - ext.y - INFLATION
        max_y = pos[1] + ext.y + INFLATION
        min_z = pos[2] - ext.z - INFLATION
        max_z = pos[2] + ext.z + INFLATION
        x0 = math.floor((min_x - origin[0]) / res)
        x1 = math.ceil((max_x - origin[0]) / res)
        y0 = math.floor((min_y - origin[1]) / res)
        y1 = math.ceil((max_y - origin[1]) / res)
        z0 = math.floor((min_z - origin[2]) / res)
        z1 = math.ceil((max_z - origin[2]) / res)
        n_cells = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
        obs_name = obs.get("_diag_sector", f"({pos[0]:.1f},{pos[1]:.1f})")
        print(f"  obstacle {obs_name}: world=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  "
              f"x=[{x0},{x1}] y=[{y0},{y1}] z=[{z0},{z1}]  ({n_cells} cells)")

    # Start cell
    raw_start_cell = _vec3_to_cell(START, origin, res)
    raw_start_world = _cell_to_vec3(raw_start_cell, origin, res)
    start_free = raw_start_cell.key() not in occupied
    print()
    print(f"  raw start cell = ({raw_start_cell.x},{raw_start_cell.y},{raw_start_cell.z})"
          f"  key={raw_start_cell.key()}")
    print(f"  raw start cell center world = "
          f"({raw_start_world[0]:.2f}, {raw_start_world[1]:.2f}, {raw_start_world[2]:.2f})")
    print(f"  raw start cell occupied = {not start_free}")

    # ensure_free
    start_cell = planner._ensure_free_cell(raw_start_cell, params.free_cell_search_radius)
    start_cell_world = _cell_to_vec3(start_cell, origin, res)
    moved = (start_cell.x != raw_start_cell.x or
             start_cell.y != raw_start_cell.y or
             start_cell.z != raw_start_cell.z)
    print(f"  ensure_free cell = ({start_cell.x},{start_cell.y},{start_cell.z})"
          f"  moved={moved}")
    print(f"  ensure_free cell world = "
          f"({start_cell_world[0]:.2f}, {start_cell_world[1]:.2f}, {start_cell_world[2]:.2f})")

    # Goal cell
    raw_goal_cell = _vec3_to_cell(GOAL, origin, res)
    goal_free = raw_goal_cell.key() not in occupied
    print(f"  raw goal cell = ({raw_goal_cell.x},{raw_goal_cell.y},{raw_goal_cell.z})"
          f"  occupied={not goal_free}")

    # Goal layers
    goal_cells = planner._build_goal_cells(raw_goal_cell, params)
    print(f"  goal_cells ({len(goal_cells)}): "
          f"{[(c.x,c.y,c.z) for c in goal_cells]}")

    # ── run A* ──
    result = planner.plan_with_result(OBSTACLES, START, GOAL)
    path = result.path_world

    print()
    print(f"  A* success={result.success}  nodes={result.nodes_expanded}"
          f"  time={result.planning_time_ms:.2f}ms")
    print(f"  path_len={len(path)}")
    print(f"  path_world:")
    for i, p in enumerate(path):
        cell = _vec3_to_cell(p, origin, res)
        occ = cell.key() in occupied
        print(f"    [{i}] world=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  "
              f"cell=({cell.x},{cell.y},{cell.z})  occupied={occ}")

    # Print real path for side-by-side
    print()
    print(f"  Real path: {REAL_PATH_XY}")
    print(f"  Replay:    {[(round(p[0],2), round(p[1],2)) for p in path]}")

    return result, planner, origin, occupied


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Guidance on replay path (v1 replay)
# ═══════════════════════════════════════════════════════════════════════

def step3_guidance_on_replay_path(replay_path):
    print()
    print("=" * 72)
    print("STEP 3 — Guidance on replay path (resolution=1.5)")
    print("=" * 72)

    guidance = CbmbaGuidance(CbmbaGuidanceParams(
        min_forward_progress=0.25,
        min_waypoint_distance=0.5,
    ))

    result = guidance.select_waypoint(
        (START[0], START[1], START[2]),
        REAL_YAW,
        replay_path,
    )

    px, py = START[0], START[1]
    cos_yaw = math.cos(REAL_YAW)
    sin_yaw = math.sin(REAL_YAW)
    lookahead = 1.0

    print("  Path segments in body frame:")
    for i in range(len(replay_path) - 1):
        a = replay_path[i]
        b = replay_path[i + 1]
        ba = _body_xy(a[0], a[1], px, py, cos_yaw, sin_yaw)
        bb = _body_xy(b[0], b[1], px, py, cos_yaw, sin_yaw)
        crosses = (ba[0] < lookahead <= bb[0]) or (bb[0] < lookahead <= ba[0])
        marker = " <-- CROSSES" if crosses else ""
        print(f"    seg({i},{i + 1}): "
              f"body ({ba[0]:.2f},{ba[1]:.2f})→({bb[0]:.2f},{bb[1]:.2f}){marker}")

    print()
    print(f"  result.valid              = {result.valid}")
    print(f"  result.source_segment     = {result.source_segment}")
    if result.target_body_xy:
        print(f"  result.body_target        = "
              f"({result.target_body_xy[0]:.2f}, {result.target_body_xy[1]:.2f})")
    print(f"  result.lateral_offset     = {result.lateral_offset_m:.2f}")
    if result.direction_body_xy:
        print(f"  result.direction          = "
              f"({result.direction_body_xy[0]:.3f}, {result.direction_body_xy[1]:.3f})")

    # Explain the offset
    print()
    print(f"  Real lateral_offset = {REAL_LATERAL_OFFSET:.2f}  "
          f"(from segment crossing at y=-4.20)")
    print(f"  Replay lateral_offset = {result.lateral_offset_m:.2f}  "
          f"(from a different crossing segment)")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Resolution comparison (1.50, 1.00, 0.75, 0.50)
# ═══════════════════════════════════════════════════════════════════════

def step4_resolution_comparison():
    print()
    print("=" * 72)
    print("STEP 4 — Resolution comparison (corrected baseline)")
    print("=" * 72)

    guidance = CbmbaGuidance(CbmbaGuidanceParams(
        min_forward_progress=0.25,
        min_waypoint_distance=0.5,
    ))

    results = {}
    for res in RESOLUTIONS:
        print()
        print("-" * 72)
        print(f"resolution = {res:.2f}")
        print("-" * 72)

        params = CbmbaParams(
            resolution=res,
            inflation_radius=INFLATION,
            max_search_nodes=2000,
            wall_penalty_radius=0,
            adaptive_long_step_cells=1,
        )
        planner = CbmbaAStarPlanner(params)
        t0 = time.perf_counter()
        result = planner.plan_with_result(OBSTACLES, START, GOAL)
        elapsed = (time.perf_counter() - t0) * 1000

        path = result.path_world
        mld = _max_lateral_dev(path, START, GOAL)

        print(f"  success             = {result.success}")
        print(f"  nodes_expanded      = {result.nodes_expanded}")
        print(f"  planning_time_ms    = {result.planning_time_ms:.2f}  "
              f"(wall={elapsed:.2f})")
        print(f"  path_len            = {len(path)}")
        print(f"  grid_size           = {result.grid_size}")

        pts = [(round(p[0], 2), round(p[1], 2)) for p in path]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        print(f"  path_xy             = {pts}")
        print(f"  min_x={min(xs):.2f}  max_x={max(xs):.2f}  "
              f"min_y={min(ys):.2f}  max_y={max(ys):.2f}")
        print(f"  max_lateral_dev     = {mld:.3f}")

        # Guidance
        g_result = guidance.select_waypoint(
            (START[0], START[1], START[2]), REAL_YAW, path,
        )
        print(f"  guidance.valid              = {g_result.valid}")
        print(f"  guidance.source_segment     = {g_result.source_segment}")
        if g_result.target_body_xy:
            print(f"  guidance.body_target        = "
                  f"({g_result.target_body_xy[0]:.2f}, {g_result.target_body_xy[1]:.2f})")
        print(f"  guidance.lateral_offset     = {g_result.lateral_offset_m:.2f}")
        if g_result.direction_body_xy:
            print(f"  guidance.direction          = "
                  f"({g_result.direction_body_xy[0]:.3f}, {g_result.direction_body_xy[1]:.3f})")
        print(f"  guidance.reason             = {g_result.reason}")

        results[res] = {
            "path": path, "mld": mld, "guidance": g_result,
            "nodes": result.nodes_expanded, "time": result.planning_time_ms,
        }

    # ── summary table ──
    print()
    print("=" * 72)
    print("Summary table")
    print("=" * 72)
    print(f"{'Res':>6s}  {'Nodes':>6s}  {'Time(ms)':>9s}  {'max_lat_dev':>11s}  "
          f"{'lat_off':>8s}  {'body_target':>20s}")
    print("-" * 72)
    for res in RESOLUTIONS:
        r = results[res]
        g = r["guidance"]
        bt = f"({g.target_body_xy[0]:.2f},{g.target_body_xy[1]:.2f})" if g.target_body_xy else "None"
        print(f"{res:6.2f}  {r['nodes']:>6d}  {r['time']:>9.2f}  {r['mld']:>11.3f}  "
              f"{g.lateral_offset_m:>8.2f}  {bt:>20s}")

    # Reference: real
    print()
    print(f"  REAL (res=1.50):  lateral_offset={REAL_LATERAL_OFFSET:.2f}  "
          f"body_target={REAL_BODY_TARGET}")


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Step 1: Guidance on real path
    step1_guidance_on_real_path()

    # Step 2: Trace A* grid + get replay path
    replay_result, planner, origin, occupied = step2_trace_astar_grid()

    # Step 3: Guidance on replay path
    step3_guidance_on_replay_path(replay_result.path_world)

    # Step 4: Full resolution comparison
    step4_resolution_comparison()


if __name__ == "__main__":
    main()

"""
CBMBA A* Path Planner — pure Python migration from old JavaScript project.

Migrated from ``OldProject/Drone-feature-yu/src/navigation/planning/CbmbaAStarPlanner.js``.
See ``CBMBA_ASTAR_AUDIT.md`` for the full algorithm audit.

This module is a **pure computation** module:
- No external RPC calls
- No drone state access
- Independently testable
- Coordinates: world-frame [x, y, z] in meters, Z-up convention (matching old project)

Grid convention (matching old JS exactly):
- ``vec3_to_cell`` uses ``round()`` (not floor)
- Cell key: ``"x|y|z"`` string
- 26-connectivity + adaptive long-step neighbors
- Open set: linear-scan Map (NOT heapq — matches old behavior)
- fScore = gScore + h * weightedHeuristic (weighted A*)

Usage::

    planner = CbmbaAStarPlanner(params)
    path = planner.plan(obstacles, start, goal)
    # path is List[List[float]] — world coordinate waypoints
    # returns [start, goal] if A* fails or planner is disabled
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── data classes ──


@dataclass
class CbmbaPlanResult:
    """Output of one CBMBA plan call.

    Attributes:
        path_world:
            Ordered waypoints in world coordinates [x, y, z].
            Always contains at least [start, goal] (even on failure).
        success:
            True if A* found a valid path (not just the fallback).
        nodes_expanded:
            Number of A* iterations executed.
        planning_time_ms:
            Wall-clock planning time in milliseconds (approximate).
        start_cell:
            Grid cell used for start (after ensure-free).
        goal_cell:
            Grid cell used for goal (the one that was reached).
        origin:
            Grid origin [x, y, z] used for this plan.
        grid_size:
            Number of occupied cells in the occupancy grid.
    """

    path_world: List[List[float]] = field(default_factory=list)
    success: bool = False
    nodes_expanded: int = 0
    planning_time_ms: float = 0.0
    start_cell: Optional[_Cell] = None
    goal_cell: Optional[_Cell] = None
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grid_size: int = 0
    max_lateral_deviation_m: float = 0.0


@dataclass
class CbmbaParams:
    """Configurable CBMBA A* parameters (matching old JS defaults)."""

    enabled: bool = True
    resolution: float = 0.75
    inflation_radius: float = 1.5
    # LiDAR/map obstacles are already measured on the obstacle surface, so
    # applying the full generic inflation to them makes narrow passages look
    # closed.  Explicit geometric obstacles keep ``inflation_radius``.
    surface_observation_inflation_radius: float = 0.75
    max_search_nodes: int = 16000
    max_planning_time_ms: float = 0.0
    """Hard wall-clock budget (ms) for one A* search.  0 disables the budget.
    When exceeded, the search aborts early and returns no path, so the caller
    can fall back to the previous valid path instead of blocking the realtime
    control loop."""
    replan_distance_threshold: float = 2.0
    replan_time_threshold: float = 0.5
    map_padding: float = 8.0
    weighted_heuristic: float = 1.15
    vertical_move_cost: float = 1.4
    vertical_heuristic_weight: float = 1.35
    turn_penalty: float = 0.2
    wall_penalty_radius: int = 2
    wall_penalty_weight: float = 0.3
    goal_layer_count: int = 2
    max_goal_vertical_offset: float = 4.0
    line_of_sight_samples: int = 20
    line_of_sight_inflation: float = 0.8
    free_cell_search_radius: int = 3
    adaptive_long_step_cells: int = 2
    sector_bias_weight: float = 0.4
    building_min_height: float = 1.2
    building_downward_seal_depth: float = 6.0
    # ── planning bounds (not in old JS) ──
    # Maximum lateral (perpendicular) distance in meters a cell may be from
    # the straight-line start→goal axis before it is treated as blocked.
    # Prevents the A* search from making arbitrarily large sideways detours
    # through unobserved space (Failure B: maze/irregular-pillar excursions).
    planning_bounds_xy_m: float = 12.0


# ── internal types ──


@dataclass
class _Cell:
    """Grid cell with integer coordinates."""
    x: int
    y: int
    z: int

    def key(self) -> str:
        return f"{self.x}|{self.y}|{self.z}"

    @staticmethod
    def from_key(key: str) -> _Cell:
        x, y, z = key.split("|")
        return _Cell(int(x), int(y), int(z))


@dataclass
class _ObstacleExtents:
    """Parsed obstacle extents (half-sizes in each axis)."""
    x: float
    y: float
    z: float


# ── helpers ──


def _obstacle_half_extents(obstacle: dict) -> _ObstacleExtents:
    """Extract half-extents from an obstacle dict (matching old JS ``obstacleHalfExtents``)."""
    fp = obstacle.get("footprint_half_extents")
    if fp is not None and isinstance(fp, (list, tuple)) and len(fp) >= 3:
        return _ObstacleExtents(
            x=max(fp[0] or 0, 0),
            y=max(fp[1] or 0, 0),
            z=max(fp[2] or 0, 0),
        )
    radius = max(obstacle.get("size", 0) or 0, 0)
    return _ObstacleExtents(x=radius, y=radius, z=radius)


def _effective_obstacle_inflation(
    obstacle: dict,
    default_inflation: float,
    params: Optional[CbmbaParams] = None,
) -> float:
    """Select inflation for a geometric obstacle or a surface observation."""
    if params is not None and obstacle.get("type") in {"lidar", "map"}:
        return max(0.0, float(params.surface_observation_inflation_radius))
    return max(0.0, float(default_inflation))


def _vec3_to_cell(vec: List[float], origin: List[float], resolution: float) -> _Cell:
    """World → grid cell (matching old JS ``vec3ToCell`` — uses round)."""
    return _Cell(
        x=round((vec[0] - origin[0]) / resolution),
        y=round((vec[1] - origin[1]) / resolution),
        z=round((vec[2] - origin[2]) / resolution),
    )


def _cell_to_vec3(cell: _Cell, origin: List[float], resolution: float) -> List[float]:
    """Grid cell → world (matching old JS ``cellToVec3``)."""
    return [
        origin[0] + cell.x * resolution,
        origin[1] + cell.y * resolution,
        origin[2] + cell.z * resolution,
    ]


def _heuristic(a: _Cell, b: _Cell, vertical_weight: float = 1.0) -> float:
    """Euclidean distance with vertical weighting (matching old JS ``heuristic``)."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = (a.z - b.z) * vertical_weight
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _normalize_cell_direction(from_cell: _Cell, to_cell: _Cell) -> str:
    """Direction sign vector as string key (matching old JS ``normalizeCellDirection``)."""
    dx = _sign(to_cell.x - from_cell.x)
    dy = _sign(to_cell.y - from_cell.y)
    dz = _sign(to_cell.z - from_cell.z)
    return f"{dx}|{dy}|{dz}"


def _sign(val: float) -> int:
    """Return -1, 0, or +1."""
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _distance_between(a: List[float], b: List[float]) -> float:
    """3D Euclidean distance between two world coordinate points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _point_to_segment_distance_xy(
    point: List[float], seg_start: List[float], seg_end: List[float],
) -> float:
    """Perpendicular XY distance from point to the line segment seg_start→seg_end.

    Used by planning-bounds enforcement: cells whose perpendicular distance
    to the start→goal axis exceeds ``planning_bounds_xy_m`` are treated as
    blocked, preventing unbounded lateral search excursions (Failure B).
    """
    px, py = point[0], point[1]
    ax, ay = seg_start[0], seg_start[1]
    bx, by = seg_end[0], seg_end[1]
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        # Degenerate segment: distance to point
        return math.hypot(px - ax, py - ay)
    # Project point onto line, clamp to segment
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    """Linear interpolation between two 3D points."""
    return [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]


def _vec3_length(v: List[float]) -> float:
    """Length of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec3_normalize(v: List[float]) -> List[float]:
    """Normalize a 3D vector in place (returns new list)."""
    length = _vec3_length(v)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def _vec3_sub(a: List[float], b: List[float]) -> List[float]:
    """a - b."""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec3_dot(a: List[float], b: List[float]) -> float:
    """Dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec3_angle_between(a: List[float], b: List[float]) -> float:
    """Angle in radians between two 3D vectors."""
    dot = _vec3_dot(a, b)
    len_a = _vec3_length(a)
    len_b = _vec3_length(b)
    if len_a < 1e-12 or len_b < 1e-12:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (len_a * len_b)))
    return math.acos(cos_angle)


# ── main planner ──


class CbmbaAStarPlanner:
    """CBMBA A* path planner — pure Python, no AirSim dependencies.

    Migrated from ``CbmbaAStarPlanner.js``.  See ``CBMBA_ASTAR_AUDIT.md``.

    Stateful: stores ``last_path``, ``last_start``, ``last_goal``, ``last_plan_time``,
    ``last_origin`` for replan checks via ``maybe_replan()``.
    """

    def __init__(self, params: Optional[CbmbaParams] = None) -> None:
        self.params = params or CbmbaParams()
        # ── state (for replan) ──
        self._grid: Set[str] = set()
        self._occupied_cells: List[_Cell] = []
        self.last_path: List[List[float]] = []
        self.last_plan_time: float = -float("inf")
        self.last_start: Optional[List[float]] = None
        self.last_goal: Optional[List[float]] = None
        self.last_origin: List[float] = [0.0, 0.0, 0.0]
        # ── planning corridor (for bounds enforcement) ──
        self._corridor_start: Optional[List[float]] = None
        self._corridor_end: Optional[List[float]] = None

    # ── public API ──

    def plan(
        self,
        obstacles: List[dict],
        start: List[float],
        goal: List[float],
    ) -> List[List[float]]:
        """Compute a path from start to goal around obstacles.

        Args:
            obstacles:
                List of obstacle dicts.  Each dict must have at least
                ``position`` [x,y,z].  May also have ``size`` (fallback radius),
                ``footprint_half_extents`` [hx,hy,hz], ``type`` (e.g. "building").
            start:
                World-coordinate start position [x, y, z].
            goal:
                World-coordinate goal position [x, y, z].

        Returns:
            List of world-coordinate waypoints [x, y, z].  Always contains at
            least [start, goal].  On success, the path is A* cells (cell centers)
            with start prepended and goal appended, smoothed via LOS compaction.
        """
        p = self.params
        if not p.enabled:
            self.last_path = [list(start), list(goal)]
            return self.last_path

        # ── set planning corridor for bounds enforcement ──
        self._corridor_start = list(start)
        self._corridor_end = list(goal)

        origin = self._compute_origin(start, goal, obstacles)
        self.last_origin = origin
        occupied, occupied_cells = self._build_occupancy_grid(
            obstacles, origin, p.resolution, p.inflation_radius, p,
        )
        self._grid = occupied
        self._occupied_cells = occupied_cells

        raw_start_cell = _vec3_to_cell(start, origin, p.resolution)
        raw_goal_cell = _vec3_to_cell(goal, origin, p.resolution)
        start_cell = self._ensure_free_cell(raw_start_cell, p.free_cell_search_radius)
        goal_cells = self._build_goal_cells(raw_goal_cell, p)
        path_cells = self._run_astar(start_cell, goal_cells, p.max_search_nodes, p)

        if not path_cells:
            self.last_path = [list(start), list(goal)]
            self.last_plan_time = _now_seconds()
            self.last_start = list(start)
            self.last_goal = list(goal)
            return self.last_path

        sampled_path = [_cell_to_vec3(c, origin, p.resolution) for c in path_cells]
        self.last_path = self._smooth_path(
            [list(start)] + sampled_path + [list(goal)], p,
        )
        self.last_plan_time = _now_seconds()
        self.last_start = list(start)
        self.last_goal = list(goal)
        return self.last_path

    def plan_with_result(
        self,
        obstacles: List[dict],
        start: List[float],
        goal: List[float],
    ) -> CbmbaPlanResult:
        """Like ``plan()`` but returns a structured ``CbmbaPlanResult`` with diagnostics.

        This is the recommended API for shadow integration — it provides
        planning metadata (node count, timing, success flag) in addition
        to the path.
        """
        import time as _time_module
        t0 = _time_module.perf_counter()

        result = CbmbaPlanResult()
        p = self.params
        result.path_world = [list(start), list(goal)]  # fallback

        # ── set planning corridor for bounds enforcement ──
        self._corridor_start = list(start)
        self._corridor_end = list(goal)

        if not p.enabled:
            result.planning_time_ms = (_time_module.perf_counter() - t0) * 1000.0
            self.last_path = result.path_world
            return result

        origin = self._compute_origin(start, goal, obstacles)
        result.origin = list(origin)
        self.last_origin = origin

        occupied, occupied_cells = self._build_occupancy_grid(
            obstacles, origin, p.resolution, p.inflation_radius, p,
        )
        self._grid = occupied
        self._occupied_cells = occupied_cells
        result.grid_size = len(occupied)

        raw_start_cell = _vec3_to_cell(start, origin, p.resolution)
        raw_goal_cell = _vec3_to_cell(goal, origin, p.resolution)
        start_cell = self._ensure_free_cell(raw_start_cell, p.free_cell_search_radius)
        goal_cells = self._build_goal_cells(raw_goal_cell, p)
        result.start_cell = start_cell

        path_cells, nodes_expanded = self._run_astar_with_stats(
            start_cell, goal_cells, p.max_search_nodes, p,
        )
        result.nodes_expanded = nodes_expanded

        if not path_cells:
            result.planning_time_ms = (_time_module.perf_counter() - t0) * 1000.0
            self.last_path = result.path_world
            self.last_plan_time = _now_seconds()
            self.last_start = list(start)
            self.last_goal = list(goal)
            return result

        # Record which goal cell was reached (last cell in path)
        result.goal_cell = path_cells[-1]
        result.success = True

        sampled_path = [_cell_to_vec3(c, origin, p.resolution) for c in path_cells]
        smoothed = self._smooth_path(
            [list(start)] + sampled_path + [list(goal)], p,
        )
        result.path_world = smoothed
        result.planning_time_ms = (_time_module.perf_counter() - t0) * 1000.0
        result.max_lateral_deviation_m = self.path_max_lateral_deviation(smoothed)

        self.last_path = result.path_world
        self.last_plan_time = _now_seconds()
        self.last_start = list(start)
        self.last_goal = list(goal)
        return result

    def maybe_replan(
        self,
        start: List[float],
        goal: List[float],
        obstacles: List[dict],
    ) -> List[List[float]]:
        """Replan only if conditions warrant (matching old JS ``maybeReplan``).

        Triggers replan when:
        - No previous path
        - Start or goal moved > replan_distance_threshold
        - Time since last plan > replan_time_threshold
        - Existing path is blocked by any obstacle
        """
        p = self.params
        now = _now_seconds()
        should = (
            len(self.last_path) == 0
            or self.last_start is None
            or self.last_goal is None
            or _distance_between(start, self.last_start) > p.replan_distance_threshold
            or _distance_between(goal, self.last_goal) > p.replan_distance_threshold
            or (now - self.last_plan_time) > p.replan_time_threshold
            or self._is_path_blocked(
                obstacles, self.last_path, p.inflation_radius, p,
            )
        )
        if not should:
            return self.last_path
        return self.plan(obstacles, start, goal)

    # ── occupancy grid ──

    def _build_occupancy_grid(
        self,
        obstacles: List[dict],
        origin: List[float],
        resolution: float,
        inflation_radius: float,
        params: CbmbaParams,
    ) -> Tuple[Set[str], List[_Cell]]:
        """Build a binary occupancy grid from obstacle list.

        Returns (occupied_set, occupied_cells_list).
        """
        occupied: Set[str] = set()
        occupied_cells: List[_Cell] = []
        p = params

        for obstacle in (obstacles or []):
            extents = _obstacle_half_extents(obstacle)
            is_building_volume = (
                obstacle.get("type") == "building"
                and (extents.z * 2) >= p.building_min_height
            )

            pos = obstacle["position"]
            # ── guard: NaN / Inf in obstacle position ──
            if any(not math.isfinite(v) for v in pos):
                continue

            obstacle_inflation = _effective_obstacle_inflation(
                obstacle, inflation_radius, p,
            )
            min_x = pos[0] - extents.x - obstacle_inflation
            max_x = pos[0] + extents.x + obstacle_inflation
            min_y = pos[1] - extents.y - obstacle_inflation
            max_y = pos[1] + extents.y + obstacle_inflation
            min_z = pos[2] - extents.z - obstacle_inflation
            max_z = (
                pos[2] + extents.z + obstacle_inflation
                + (p.building_downward_seal_depth if is_building_volume else 0)
            )

            x0 = math.floor((min(min_x, max_x) - origin[0]) / resolution)
            x1 = math.ceil((max(min_x, max_x) - origin[0]) / resolution)
            y0 = math.floor((min(min_y, max_y) - origin[1]) / resolution)
            y1 = math.ceil((max(min_y, max_y) - origin[1]) / resolution)
            z0 = math.floor((min(min_z, max_z) - origin[2]) / resolution)
            z1 = math.ceil((max(min_z, max_z) - origin[2]) / resolution)

            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        cell = _Cell(x, y, z)
                        key = cell.key()
                        if key not in occupied:
                            occupied.add(key)
                            occupied_cells.append(cell)

        return occupied, occupied_cells

    # ── goal cells ──

    def _build_goal_cells(self, goal_cell: _Cell, params: CbmbaParams) -> List[_Cell]:
        """Build multi-layer goal cells (matching old JS ``buildGoalCells``)."""
        cells: List[_Cell] = []
        step_count = max(1, params.goal_layer_count)
        vertical_cell_offset = max(
            1, round(params.max_goal_vertical_offset / params.resolution),
        )
        for dz in range(-step_count, step_count + 1):
            offset = round((dz / step_count) * vertical_cell_offset)
            candidate = self._ensure_free_cell(
                _Cell(goal_cell.x, goal_cell.y, goal_cell.z + offset),
                params.free_cell_search_radius,
            )
            if not any(c.x == candidate.x and c.y == candidate.y and c.z == candidate.z for c in cells):
                cells.append(candidate)
        return cells

    # ── free cell search ──

    def _ensure_free_cell(self, cell: _Cell, max_radius: int = 2) -> _Cell:
        """Find a free cell near ``cell`` (matching old JS ``ensureFreeCell``).

        If ``cell`` is free, return it.  Otherwise search expanding
        radius 1..max_radius for first free neighbor.
        """
        if cell.key() not in self._grid:
            return cell

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        candidate = _Cell(cell.x + dx, cell.y + dy, cell.z + dz)
                        if candidate.key() not in self._grid:
                            return candidate

        return cell  # fallback: return original even though occupied

    # ── ordered neighbors ──

    def _ordered_neighbors(
        self,
        cell: _Cell,
        goal_cell: _Cell,
        current_dir: Optional[str],
        params: CbmbaParams,
    ) -> List[_Cell]:
        """Generate and sort neighbors (matching old JS ``orderedNeighbors``)."""
        candidates: Dict[str, _Cell] = {}
        long_step = max(1, params.adaptive_long_step_cells)
        step_set = [1, long_step] if long_step > 1 else [1]
        goal_dir = _Cell(
            x=_sign(goal_cell.x - cell.x),
            y=_sign(goal_cell.y - cell.y),
            z=_sign(goal_cell.z - cell.z),
        )

        for step in step_set:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = _Cell(
                            x=cell.x + dx * step,
                            y=cell.y + dy * step,
                            z=cell.z + dz * step,
                        )
                        candidates[neighbor.key()] = neighbor

        result = list(candidates.values())

        def _sort_key(n: _Cell) -> float:
            dir_n = _normalize_cell_direction(cell, n)
            parts = [int(x) for x in dir_n.split("|")]
            bias = (
                (1 if parts[0] == goal_dir.x else 0)
                + (1 if parts[1] == goal_dir.y else 0)
                + (1 if parts[2] == goal_dir.z else 0)
                + (
                    params.sector_bias_weight
                    if current_dir is not None and dir_n == current_dir
                    else 0
                )
            )
            return _heuristic(n, goal_cell) - bias

        result.sort(key=_sort_key)
        return result

    # ── A* search ──

    def _run_astar(
        self,
        start_cell: _Cell,
        goal_cells: List[_Cell],
        max_search_nodes: int,
        params: CbmbaParams,
    ) -> List[_Cell]:
        """Run weighted A* search. Returns list of Cells or empty list."""
        path, _ = self._run_astar_with_stats(start_cell, goal_cells, max_search_nodes, params)
        return path

    def _run_astar_with_stats(
        self,
        start_cell: _Cell,
        goal_cells: List[_Cell],
        max_search_nodes: int,
        params: CbmbaParams,
    ) -> Tuple[List[_Cell], int]:
        """Run weighted A* with node count. Returns (path, nodes_expanded)."""
        import time as _time_module
        _deadline = None
        if getattr(params, "max_planning_time_ms", 0.0) > 0.0:
            _deadline = _time_module.perf_counter() + params.max_planning_time_ms / 1000.0

        # ── data structures (matching old JS exactly) ──
        open_set: Dict[str, _Cell] = {}             # Map<string, Cell>
        came_from: Dict[str, str] = {}              # Map<string, string>
        g_score: Dict[str, float] = {}              # Map<string, number>
        f_score: Dict[str, float] = {}              # Map<string, number>
        closed: Set[str] = set()

        start_key = start_cell.key()
        goal_keys = {c.key() for c in goal_cells}
        goal_reference = goal_cells[0]

        open_set[start_key] = start_cell
        g_score[start_key] = 0.0
        f_score[start_key] = _heuristic(
            start_cell, goal_reference, params.vertical_heuristic_weight,
        )

        iterations = 0
        while open_set and iterations < max_search_nodes:
            iterations += 1

            # ── planning time budget: abort early so the realtime loop is
            # never blocked past max_planning_time_ms.  Returns no path; the
            # caller reuses the previous valid path.
            if _deadline is not None and _time_module.perf_counter() > _deadline:
                return [], iterations

            # ── linear scan for best fScore (matching old JS) ──
            current_key: Optional[str] = None
            current_cell: Optional[_Cell] = None
            best_score = float("inf")
            for key, cell in open_set.items():
                score = f_score.get(key, float("inf"))
                if score < best_score:
                    best_score = score
                    current_key = key
                    current_cell = cell

            if current_cell is None or current_key is None:
                break

            # ── goal check ──
            if current_key in goal_keys:
                return self._reconstruct_path(came_from, current_key), iterations

            del open_set[current_key]
            closed.add(current_key)

            # ── nearest goal for heuristic ──
            nearest_goal = goal_reference
            best_h = _heuristic(current_cell, goal_reference, params.vertical_heuristic_weight)
            for gc in goal_cells:
                h = _heuristic(current_cell, gc, params.vertical_heuristic_weight)
                if h < best_h:
                    best_h = h
                    nearest_goal = gc

            # ── current direction for sector bias ──
            current_dir: Optional[str] = None
            parent_key = came_from.get(current_key)
            if parent_key is not None:
                current_dir = _normalize_cell_direction(
                    _Cell.from_key(parent_key), current_cell,
                )

            # ── expand neighbors ──
            for neighbor in self._ordered_neighbors(
                current_cell, nearest_goal, current_dir, params,
            ):
                neighbor_key = neighbor.key()
                if neighbor_key in closed or neighbor_key in self._grid:
                    continue
                # ── planning bounds: skip cells outside the corridor ──
                if not self._cell_in_corridor(
                    neighbor, self.last_origin, params.resolution,
                ):
                    continue

                current_g = g_score.get(current_key, float("inf"))
                traversal_cost = self._compute_traversal_cost(
                    current_cell, neighbor, current_key, came_from, params,
                )
                tentative_g = current_g + traversal_cost

                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                came_from[neighbor_key] = current_key
                g_score[neighbor_key] = tentative_g
                h = _heuristic(neighbor, nearest_goal, params.vertical_heuristic_weight)
                f_score[neighbor_key] = tentative_g + h * params.weighted_heuristic
                open_set[neighbor_key] = neighbor

        return [], iterations

    # ── traversal cost ──

    def _compute_traversal_cost(
        self,
        current_cell: _Cell,
        neighbor_cell: _Cell,
        current_key: str,
        came_from: Dict[str, str],
        params: CbmbaParams,
    ) -> float:
        """Compute cost of moving from current_cell to neighbor_cell."""
        dx = neighbor_cell.x - current_cell.x
        dy = neighbor_cell.y - current_cell.y
        dz = neighbor_cell.z - current_cell.z
        base = math.sqrt(
            dx * dx + dy * dy + (dz * params.vertical_move_cost) ** 2,
        )

        # ── turn penalty ──
        turn_penalty = 0.0
        parent_key = came_from.get(current_key)
        if parent_key is not None:
            parent_cell = _Cell.from_key(parent_key)
            prev_dir = _normalize_cell_direction(parent_cell, current_cell)
            next_dir = _normalize_cell_direction(current_cell, neighbor_cell)
            if prev_dir != next_dir:
                turn_penalty = params.turn_penalty

        # ── wall proximity penalty ──
        wall_penalty = (
            self._local_obstacle_penalty(neighbor_cell, params.wall_penalty_radius)
            * params.wall_penalty_weight
        )

        return base + turn_penalty + wall_penalty

    # ── local obstacle penalty ──

    def _local_obstacle_penalty(self, cell: _Cell, radius: int) -> float:
        """Compute wall proximity penalty (matching old JS ``localObstaclePenalty``)."""
        if radius <= 0:
            return 0.0

        penalty = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    sample = _Cell(cell.x + dx, cell.y + dy, cell.z + dz)
                    if sample.key() not in self._grid:
                        continue
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
                    penalty += 1.0 / distance
        return penalty

    # ── path reconstruction ──

    def _reconstruct_path(
        self, came_from: Dict[str, str], current_key: str,
    ) -> List[_Cell]:
        """Back-trace from goal to start, then reverse."""
        path = [_Cell.from_key(current_key)]
        while current_key in came_from:
            current_key = came_from[current_key]
            path.append(_Cell.from_key(current_key))
        path.reverse()
        return path

    # ── path smoothing ──

    def _smooth_path(
        self, points: List[List[float]], params: CbmbaParams,
    ) -> List[List[float]]:
        """Line-of-sight compaction + angle filtering (matching old JS ``smoothPath``)."""
        if not points or len(points) < 3:
            return points

        # ── Step 1: LOS compaction ──
        compact = [points[0]]
        anchor_index = 0
        while anchor_index < len(points) - 1:
            furthest_index = anchor_index + 1
            for i in range(len(points) - 1, anchor_index + 1, -1):
                if self._has_line_of_sight(points[anchor_index], points[i], params):
                    furthest_index = i
                    break
            compact.append(points[furthest_index])
            anchor_index = furthest_index

        # ── Step 2: angle filtering ──
        smooth = [compact[0]]
        for i in range(1, len(compact) - 1):
            prev = _vec3_normalize(_vec3_sub(compact[i - 1], compact[i]))  # actually prev→curr direction reversed
            # Fix: compute incoming and outgoing vectors correctly
            v1 = _vec3_normalize(_vec3_sub(compact[i], compact[i - 1]))       # incoming
            v2 = _vec3_normalize(_vec3_sub(compact[i + 1], compact[i]))       # outgoing
            angle = _vec3_angle_between(v1, v2)
            if angle > 0.12:  # keep points where angle > ~6.9°
                smooth.append(compact[i])
        smooth.append(compact[-1])
        return smooth

    # ── line of sight ──

    def _has_line_of_sight(
        self, start: List[float], end: List[float], params: CbmbaParams,
    ) -> bool:
        """Check if there's a clear line between start and end (matching old JS)."""
        samples = max(4, params.line_of_sight_samples)
        for i in range(1, samples):
            t = i / samples
            point = _lerp(start, end, t)
            cell = _vec3_to_cell(point, self.last_origin, params.resolution)
            if cell.key() in self._grid:
                return False

            if params.line_of_sight_inflation > 0:
                radius = math.ceil(params.line_of_sight_inflation / params.resolution)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        expanded = _Cell(cell.x + dx, cell.y + dy, cell.z)
                        if expanded.key() in self._grid:
                            return False
        return True

    # ── is path blocked ──

    def _is_path_blocked(
        self,
        obstacles: List[dict],
        path: List[List[float]],
        inflation_radius: float,
        params: Optional[CbmbaParams] = None,
    ) -> bool:
        """Check if any obstacle intersects the path (matching old JS ``isPathBlocked``)."""
        if not path or len(path) < 2:
            return False
        for obstacle in (obstacles or []):
            center = obstacle["position"]
            size = obstacle.get("size", 0) or 0
            obstacle_inflation = _effective_obstacle_inflation(
                obstacle, inflation_radius, params,
            )
            for i in range(len(path) - 1):
                a = path[i]
                b = path[i + 1]
                proj = _project_to_segment(center, a, b)
                clearance = _vec3_length(_vec3_sub(proj, center)) - (
                    size + obstacle_inflation
                )
                if clearance < 0:
                    return True
        return False

    # ── planning bounds ──

    def _cell_in_corridor(self, cell: _Cell, origin: List[float], resolution: float) -> bool:
        """Return True if *cell* is within the planning-bounds corridor.

        The corridor is the region whose perpendicular XY distance to the
        start→goal line segment is ≤ ``planning_bounds_xy_m``.
        """
        if self._corridor_start is None or self._corridor_end is None:
            return True  # no corridor set → no bounds
        p = self.params
        if p.planning_bounds_xy_m <= 0:
            return True  # bounds disabled
        world = _cell_to_vec3(cell, origin, resolution)
        dist = _point_to_segment_distance_xy(
            world, self._corridor_start, self._corridor_end,
        )
        return dist <= p.planning_bounds_xy_m

    def is_path_in_bounds(self, path: List[List[float]]) -> bool:
        """Return True if every waypoint is within the planning corridor.

        A waypoint is out of bounds if its perpendicular XY distance to the
        start→goal axis exceeds ``planning_bounds_xy_m``.
        """
        if self.params.planning_bounds_xy_m <= 0:
            return True
        if self._corridor_start is None or self._corridor_end is None:
            return True
        if not path or len(path) < 2:
            return True
        limit = self.params.planning_bounds_xy_m
        for wp in path:
            d = _point_to_segment_distance_xy(
                wp, self._corridor_start, self._corridor_end,
            )
            if d > limit:
                return False
        return True

    def path_max_lateral_deviation(self, path: List[List[float]]) -> float:
        """Return the maximum perpendicular XY distance of any waypoint from
        the start→goal axis.  Used for diagnostics (Failure B monitoring).
        """
        if self._corridor_start is None or self._corridor_end is None:
            return 0.0
        if not path:
            return 0.0
        return max(
            _point_to_segment_distance_xy(wp, self._corridor_start, self._corridor_end)
            for wp in path
        )

    # ── origin computation ──

    def _compute_origin(
        self,
        start: List[float],
        goal: List[float],
        obstacles: List[dict],
    ) -> List[float]:
        """Compute grid origin from bounding box (matching old JS ``computeOrigin``)."""
        p = self.params
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        zs = [start[2], goal[2]]
        for obstacle in (obstacles or []):
            pos = obstacle["position"]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
        return [
            min(xs) - p.map_padding,
            min(ys) - p.map_padding,
            min(zs) - p.map_padding,
        ]


def _now_seconds() -> float:
    """Return monotonic seconds (replaces ``performance.now()/1000``)."""
    import time
    return time.monotonic()


def _project_to_segment(
    point: List[float], start: List[float], end: List[float],
) -> List[float]:
    """Project point onto line segment start→end (matching old JS ``projectToSegment``)."""
    seg = _vec3_sub(end, start)
    seg_len_sq = _vec3_dot(seg, seg)
    if seg_len_sq < 1e-12:
        return list(start)
    t = _vec3_dot(_vec3_sub(point, start), seg) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return [
        start[0] + seg[0] * t,
        start[1] + seg[1] * t,
        start[2] + seg[2] * t,
    ]

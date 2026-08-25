"""
Persistent 2D occupancy grid map.

A geometric map-memory layer for the trajectory-centric navigation upgrade.
Pose comes from AirSim ground truth, so this module does **mapping only**
(no SLAM / localisation loop).

Design
------
- Grid is a square region of world-NED XY space around a fixed origin
  (the drone spawn point, set on the first ``update``).
- Each cell holds a **log-odds** score.  Three states are derived:
  ``UNKNOWN`` (never observed), ``FREE`` (observed empty), ``OCCUPIED``.
- LiDAR returns are integrated with **ray casting**: the hit cell gets a
  positive log-odds increment, every cell along the ray between the drone
  and the hit gets a negative (free) increment.  This follows the standard
  occupancy-grid mapping formulation (Thrun et al. 2005).

Coordinate system
-----------------
- World NED: +X = North, +Y = East, +Z = Down.
- LiDAR input is SensorLocalFrame: +X = forward, +Y = right, +Z = down.
- Only **horizontal** LiDAR returns (|sensor_z| within a band) contribute
  to the 2D grid — the drone flies at roughly constant altitude.

Read-before-write semantics
---------------------------
Callers should (1) plan with the map as it stands (``M_<t``), (2) execute,
and (3) only then call ``update`` with the current observation to obtain
``M_<t+1``.  This keeps historical map state and current observation
semantically distinct (DreamFly causal-memory principle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ── cell states ──

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

_LOG_ODDS_MIN = -6.0
_LOG_ODDS_MAX = 6.0


# ── parameters ──


@dataclass
class OccupancyGridParams:
    """Configuration for the persistent occupancy grid.

    Attributes:
        resolution_m: Cell edge length (m).  Smaller = finer but more cells.
        map_radius_m: Half-size of the square map around the fixed origin.
        max_range_m: LiDAR returns beyond this range are ignored.
        min_range_m: LiDAR returns closer than this are ignored.
        occupied_log_odds: Log-odds increment for a hit cell.
        free_log_odds: Log-odds decrement for a free cell along a ray.
        occupied_threshold: A cell is OCCUPIED when log-odds >= this.
        inflation_cells: Extra cells marked occupied around each hit (dilation).
        horizontal_band_half_height_m: |sensor_z| threshold for a LiDAR point
            to be treated as a horizontal (in-plane) obstacle.
        ray_sample_spacing_m: Distance between free-cell samples along a ray.
    """

    resolution_m: float = 0.5
    map_radius_m: float = 40.0
    max_range_m: float = 15.0
    min_range_m: float = 0.2
    occupied_log_odds: float = 0.85
    free_log_odds: float = -0.4
    occupied_threshold: float = 0.0
    inflation_cells: int = 1
    horizontal_band_half_height_m: float = 1.0
    ray_sample_spacing_m: float = 0.25
    self_filter_radius_m: float = 0.5


# ── map ──


class OccupancyGridMap:
    """Persistent 2D occupancy grid (world-NED XY).

    Stateful: the grid accumulates log-odds across ``update`` calls.  The
    origin is fixed at the first update position so the map stays in a
    consistent world frame as the drone moves.
    """

    def __init__(self, params: Optional[OccupancyGridParams] = None) -> None:
        self.params = params or OccupancyGridParams()
        self._log_odds: Dict[Tuple[int, int], float] = {}
        self._origin: Optional[Tuple[float, float]] = None
        self.version: int = 0  # increments once per successfully written frame

    # ── public API ──

    @property
    def cell_count(self) -> int:
        """Total observed cells currently tracked (telemetry; no map semantics)."""
        return len(self._log_odds)

    def update(
        self,
        points_sensor: np.ndarray,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
    ) -> None:
        """Integrate one LiDAR frame into the map (write step).

        Args:
            points_sensor: N×3 array in SensorLocalFrame (+X fwd, +Y right, +Z down).
            drone_position_ned: Drone world position ``(x, y, z)`` in NED.
            yaw_rad: Drone yaw (0 = North, π/2 = East).
        """
        if self._origin is None:
            self._origin = (drone_position_ned[0], drone_position_ned[1])

        p = self.params
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        px = drone_position_ned[0]
        py = drone_position_ned[1]

        if points_sensor is None or getattr(points_sensor, "size", 0) == 0:
            return
        if getattr(points_sensor, "ndim", 2) != 2 or points_sensor.shape[1] != 3:
            return

        for sx, sy, sz in points_sensor:
            if abs(float(sz)) > p.horizontal_band_half_height_m:
                continue
            d = math.hypot(float(sx), float(sy))
            if d < max(p.min_range_m, p.self_filter_radius_m):
                continue  # self / noise returns near the vehicle centre
            if d > p.max_range_m:
                continue

            # SensorLocalFrame → world NED (forward=(cos,sin), right=(-sin,cos))
            wx = px + float(sx) * cos_yaw - float(sy) * sin_yaw
            wy = py + float(sx) * sin_yaw + float(sy) * cos_yaw

            self._raycast_update(px, py, wx, wy)

        self.version += 1

    def state_at(self, x: float, y: float) -> int:
        """Return UNKNOWN / FREE / OCCUPIED for a world-NED XY point."""
        cell = self._world_to_cell(x, y)
        if cell is None:
            return UNKNOWN
        lo = self._log_odds.get(cell)
        if lo is None:
            return UNKNOWN
        return OCCUPIED if lo >= self.params.occupied_threshold else FREE

    def is_occupied(self, x: float, y: float) -> bool:
        return self.state_at(x, y) == OCCUPIED

    def get_occupied_points(self) -> List[Tuple[float, float]]:
        """Return world-NED XY of every OCCUPIED cell centre."""
        p = self.params
        out: List[Tuple[float, float]] = []
        for (ix, iy), lo in self._log_odds.items():
            if lo >= p.occupied_threshold:
                out.append(self._cell_center(ix, iy))
        return out

    def get_occupied_points_in_radius(
        self, center_x: float, center_y: float, radius_m: float,
    ) -> List[Tuple[float, float]]:
        """Return world-NED XY of OCCUPIED cells within ``radius_m`` of a centre.

        Only the cells in the local bounding box are scanned (O(local cells)),
        so the cost is independent of the total map size.
        """
        p = self.params
        if self._origin is None:
            return []
        ox, oy = self._origin
        cx_cell = int(math.floor((center_x - ox) / p.resolution_m))
        cy_cell = int(math.floor((center_y - oy) / p.resolution_m))
        half = int(math.ceil(radius_m / p.resolution_m))
        r2 = radius_m * radius_m
        out: List[Tuple[float, float]] = []
        for ix in range(cx_cell - half, cx_cell + half + 1):
            for iy in range(cy_cell - half, cy_cell + half + 1):
                lo = self._log_odds.get((ix, iy))
                if lo is None or lo < p.occupied_threshold:
                    continue
                x, y = self._cell_center(ix, iy)
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= r2:
                    out.append((x, y))
        return out

    def to_obstacles(self, z_ned: float) -> List[dict]:
        """Convert occupied cells into CBMBA-compatible obstacle dicts.

        Each occupied cell becomes a point obstacle at the cell centre with
        ``footprint_half_extents`` equal to half the grid resolution (plus
        no inflation — CBMBA applies its own ``inflation_radius``).
        """
        half = self.params.resolution_m / 2.0
        return [
            {
                "position": [x, y, z_ned],
                "footprint_half_extents": [half, half, half],
                "type": "map",
                "velocity": [0.0, 0.0, 0.0],
                "dynamic": False,
                "confidence": 0.9,
            }
            for (x, y) in self.get_occupied_points()
        ]

    # ── exploration-ready interfaces (not used this round) ──

    def get_frontier_cells(self) -> List[Tuple[float, float]]:
        """Return world-NED XY of FREE cells adjacent to at least one UNKNOWN cell."""
        p = self.params
        frontiers: List[Tuple[float, float]] = []
        for (ix, iy), lo in self._log_odds.items():
            if lo >= p.occupied_threshold:
                continue  # only free cells can be frontiers
            if self._has_unknown_neighbor(ix, iy):
                frontiers.append(self._cell_center(ix, iy))
        return frontiers

    def get_unknown_ratio(self) -> float:
        """Fraction of the (bounded) map that is still UNKNOWN."""
        p = self.params
        half_cells = int(round(p.map_radius_m / p.resolution_m))
        total = (2 * half_cells + 1) ** 2
        if total == 0:
            return 1.0
        unknown = total - len(self._log_odds)
        return max(0.0, min(1.0, unknown / total))

    def get_explored_area(self) -> float:
        """Approximate explored area in m² (number of observed cells × cell area)."""
        cell_area = self.params.resolution_m * self.params.resolution_m
        return len(self._log_odds) * cell_area

    # ── internals ──

    def _world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        p = self.params
        ox, oy = self._origin
        ix = int(math.floor((x - ox) / p.resolution_m))
        iy = int(math.floor((y - oy) / p.resolution_m))
        half = int(round(p.map_radius_m / p.resolution_m))
        if abs(ix) > half or abs(iy) > half:
            return None
        return (ix, iy)

    def _cell_center(self, ix: int, iy: int) -> Tuple[float, float]:
        p = self.params
        ox, oy = self._origin
        return (
            ox + (ix + 0.5) * p.resolution_m,
            oy + (iy + 0.5) * p.resolution_m,
        )

    def _raycast_update(self, sx: float, sy: float, ex: float, ey: float) -> None:
        """Mark cells FREE along (sx,sy)→(ex,ey), OCCUPIED at the end (+ inflation)."""
        p = self.params
        dx = ex - sx
        dy = ey - sy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return

        # Free samples along the ray (exclusive of the endpoint).
        n_free = max(1, int(dist / p.ray_sample_spacing_m))
        for i in range(1, n_free + 1):
            t = i / (n_free + 1)
            self._mark_free(sx + dx * t, sy + dy * t)

        # Occupied at the hit point (with inflation).
        hit_cell = self._world_to_cell(ex, ey)
        if hit_cell is None:
            return
        for dxi in range(-p.inflation_cells, p.inflation_cells + 1):
            for dyi in range(-p.inflation_cells, p.inflation_cells + 1):
                self._mark_occupied(hit_cell[0] + dxi, hit_cell[1] + dyi)

    def _mark_free(self, x: float, y: float) -> None:
        cell = self._world_to_cell(x, y)
        if cell is None:
            return
        self._log_odds[cell] = self._clamp(
            self._log_odds.get(cell, 0.0) + self.params.free_log_odds
        )

    def _mark_occupied(self, ix: int, iy: int) -> None:
        p = self.params
        half = int(round(p.map_radius_m / p.resolution_m))
        if abs(ix) > half or abs(iy) > half:
            return
        cell = (ix, iy)
        self._log_odds[cell] = self._clamp(
            self._log_odds.get(cell, 0.0) + p.occupied_log_odds
        )

    def _has_unknown_neighbor(self, ix: int, iy: int) -> bool:
        for dxi in (-1, 0, 1):
            for dyi in (-1, 0, 1):
                if dxi == 0 and dyi == 0:
                    continue
                if (ix + dxi, iy + dyi) not in self._log_odds:
                    return True
        return False

    @staticmethod
    def _clamp(v: float) -> float:
        return max(_LOG_ODDS_MIN, min(_LOG_ODDS_MAX, v))

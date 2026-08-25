"""
Local 2D distance field (ESDF) for clearance-aware trajectory evaluation.

The trajectory planner needs to know, for every point along a candidate
trajectory, how far away the nearest obstacle is.  A full 3D TSDF is not
needed because the drone flies at roughly constant altitude, so this module
builds a **2D** Euclidean signed distance field from a set of obstacle points
(occupied cells from the persistent map + current LiDAR returns).

Implementation
--------------
Rather than a voxel sweep, this module computes exact (brute-force) Euclidean
distance to a local obstacle set.  The obstacle set is small (a local window
around the drone), so ``distance_at`` is O(n_obstacles) per query.  For a
trajectory of ~17 samples evaluated against a few hundred obstacles this is
negligible.  The module is pure Python (numpy optional) and dependency-free.

Current-obstacle priority
-------------------------
``set_obstacles`` receives obstacles merged by the caller, with **current
LiDAR obstacles already mixed in**.  The caller must ensure current LiDAR
obstacles are present so a historical map marking an area free can never
override a fresh LiDAR return showing it blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class DistanceField:
    """2D Euclidean distance field over a local obstacle set (world-NED XY)."""

    _obstacles: List[Tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._obstacles is None:
            self._obstacles = []

    # ── public API ──

    def set_obstacles(self, points: Iterable[Tuple[float, float]]) -> None:
        """Replace the obstacle set.  Each element is ``(x_world, y_world)``."""
        self._obstacles = [(float(px), float(py)) for px, py in points]

    @property
    def obstacle_count(self) -> int:
        return len(self._obstacles)

    def distance_at(self, x: float, y: float) -> float:
        """Euclidean distance (m) to the nearest obstacle, or ``inf`` if empty."""
        best = float("inf")
        for ox, oy in self._obstacles:
            dx = x - ox
            dy = y - oy
            d = math.hypot(dx, dy)
            if d < best:
                best = d
        return best

    def gradient_at(self, x: float, y: float, eps: float = 0.1) -> Tuple[float, float]:
        """Numerical gradient of the distance field, pointing **away** from obstacles.

        Returns a unit-length 2D vector ``(gx, gy)``.  If the field is empty
        or flat, returns ``(0.0, 0.0)``.
        """
        d_xp = self.distance_at(x + eps, y)
        d_xm = self.distance_at(x - eps, y)
        d_yp = self.distance_at(x, y + eps)
        d_ym = self.distance_at(x, y - eps)
        gx = (d_xp - d_xm) / (2.0 * eps)
        gy = (d_yp - d_ym) / (2.0 * eps)
        mag = math.hypot(gx, gy)
        if mag < 1e-9:
            return (0.0, 0.0)
        return (gx / mag, gy / mag)

    def trajectory_min_clearance(self, trajectory: Iterable[Tuple[float, float]]) -> float:
        """Minimum distance-to-nearest-obstacle over all trajectory points."""
        best = float("inf")
        for x, y in trajectory:
            d = self.distance_at(x, y)
            if d < best:
                best = d
        return best

    def trajectory_mean_clearance(self, trajectory: Iterable[Tuple[float, float]]) -> float:
        """Mean distance-to-nearest-obstacle over all trajectory points."""
        total = 0.0
        n = 0
        for x, y in trajectory:
            total += self.distance_at(x, y)
            n += 1
        return total / n if n > 0 else float("inf")

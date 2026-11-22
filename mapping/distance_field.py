"""
Local 2D distance field (ESDF) for clearance-aware trajectory evaluation.

The trajectory planner needs to know, for every point along a candidate
trajectory, how far away the nearest obstacle is.  A full 3D TSDF is not
needed because the drone flies at roughly constant altitude, so this module
builds a **2D** Euclidean signed distance field from a set of obstacle points
(occupied cells from the persistent map + current LiDAR returns).

Implementation
--------------
Rather than a voxel sweep, this module computes Euclidean distance to a local
observed obstacle set. Queries are O(n_obstacles); numpy accelerates larger
sets and chunked polyline checks. Numpy is already a project dependency.

Current-obstacle priority
-------------------------
``set_obstacles`` receives obstacles merged by the caller, with **current
LiDAR obstacles already mixed in**.  The caller must ensure current LiDAR
obstacles are present so a historical map marking an area free can never
override a fresh LiDAR return showing it blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class DistanceField:
    """2D Euclidean distance field over a local obstacle set (world-NED XY)."""

    _obstacles: List[Tuple[float, float]] = None  # type: ignore[assignment]
    _array: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.set_obstacles(self._obstacles or [])

    # ── public API ──

    def set_obstacles(self, points: Iterable[Tuple[float, float]]) -> None:
        """Replace the obstacle set.  Each element is ``(x_world, y_world)``."""
        self._obstacles = [(float(px), float(py)) for px, py in points]
        self._array = np.asarray(self._obstacles, dtype=float).reshape(-1, 2)

    @property
    def obstacle_count(self) -> int:
        return len(self._obstacles)

    def distance_at(self, x: float, y: float) -> float:
        """Euclidean distance (m) to the nearest obstacle, or ``inf`` if empty."""
        if len(self._obstacles) >= 32:
            delta = self._array - (x, y)
            return math.sqrt(float(np.einsum("ij,ij->i", delta, delta).min()))
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
        if not self._obstacles:
            return (0., 0.)
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

    def trajectory_clearances(self, trajectory, *, check_budget=None):
        """Exact clearance of the polyline to observed points, plus sample mean.

        Unlike sample-only checks this also catches an obstacle between samples.
        Chunking bounds temporary memory and permits cooperative cancellation;
        neither this nor a planner deadline can preempt an individual numpy call.
        """
        points = np.asarray(trajectory, dtype=float).reshape(-1, 2)
        if not len(points) or not self._obstacles:
            return float("inf"), float("inf")
        obstacles = self._array
        sample_min = np.full(len(points), np.inf)
        segment_min = float("inf")
        start, delta = points[:-1], np.diff(points, axis=0)
        lengths = np.einsum("ij,ij->i", delta, delta)
        for offset in range(0, len(obstacles), 512):
            if check_budget is not None:
                check_budget()
            block = obstacles[offset:offset + 512]
            diff = points[:, None, :] - block[None, :, :]
            sample_min = np.minimum(sample_min, np.einsum("ijk,ijk->ij", diff, diff).min(axis=1))
            if len(delta):
                relative = block[None, :, :] - start[:, None, :]
                projection = np.einsum("ijk,ik->ij", relative, delta)
                projection /= np.maximum(lengths[:, None], 1e-30)
                projection = np.clip(projection, 0., 1.)
                error = relative - projection[:, :, None] * delta[:, None, :]
                segment_min = min(segment_min, float(np.einsum("ijk,ijk->ij", error, error).min()))
        if check_budget is not None:
            check_budget()
        return math.sqrt(min(segment_min, float(sample_min.min()))), float(np.sqrt(sample_min).mean())

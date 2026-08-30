"""
AirSim in-simulator debug drawing + HUD (Phase C0, sec 19/20).

Draws the trajectory navigator's internal state into the Unreal viewport so a
real column-test flight can be inspected visually:

- global reference path (CBMBA)
- the selected local trajectory
- the rejected/invalid candidate family set
- obstacle points (downsampled LiDAR + map)
- the fixed mission goal

Every call is best-effort: it is wrapped in try/except and disabled by default
so a missing ``airsim`` import or a drawing RPC hiccup can never break the
control loop.  All drawing is configured via ``trajectory_debug`` YAML.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# AirSim severity levels for simPrintLogMessage.
HUD_SEVERITY_INFO = 0
HUD_SEVERITY_IMPORTANT = 1
HUD_SEVERITY_WARN = 2
HUD_SEVERITY_ERROR = 3


def _v3(x: float, y: float, z: float):
    import airsim  # lazy: only needed when drawing is enabled
    return airsim.Vector3r(float(x), float(y), float(z))


class AirSimDebugDrawer:
    """Thin, guarded wrapper over the AirSim ``simPlot*`` / ``simPrintLogMessage`` RPCs."""

    def __init__(
        self,
        adapter: Any,
        enabled: bool = True,
        vehicle_name: Optional[str] = None,
        line_thickness: float = 5.0,
        point_size: float = 10.0,
        duration_s: float = 0.3,
    ) -> None:
        self._adapter = adapter
        self.enabled = bool(enabled)
        self._vehicle_name = vehicle_name or adapter.vehicle_name
        self.line_thickness = line_thickness
        self.point_size = point_size
        self.duration_s = duration_s

    # ── low-level ──

    def _client(self):
        return self._adapter.get_raw_client()

    def _safe(self, fn, *args) -> None:
        if not self.enabled:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — drawing must never crash flight
            logger.debug("airsim_debug_draw skipped: %s", exc)

    def _line(self, pts: Sequence[Tuple[float, float, float]], rgba, persistent: bool) -> None:
        if len(pts) < 2:
            return
        vectors = [_v3(p[0], p[1], p[2]) for p in pts]
        self._safe(
            self._client().simPlotLineList,
            vectors, rgba, self.line_thickness, self.duration_s, persistent,
        )

    def _points(self, pts: Sequence[Tuple[float, float, float]], rgba, persistent: bool) -> None:
        if not pts:
            return
        vectors = [_v3(p[0], p[1], p[2]) for p in pts]
        self._safe(
            self._client().simPlotPoints,
            vectors, rgba, self.point_size, self.duration_s, persistent,
        )

    # ── primitives ──

    def draw_global_path(self, path, z: float) -> None:
        """Draw the CBMBA global reference path (world NED, z fixed)."""
        pts = [(float(p[0]), float(p[1]), z) for p in path if p is not None and len(p) >= 2]
        self._line(pts, [0.0, 1.0, 0.0, 1.0], persistent=False)  # green

    def draw_selected_trajectory(self, points, z: float) -> None:
        """Draw the selected local trajectory (world NED XY)."""
        pts = [(float(x), float(y), z) for (x, y) in points]
        self._line(pts, [0.0, 0.6, 1.0, 1.0], persistent=False)  # light blue

    def draw_obstacles(self, points, z: float) -> None:
        """Draw obstacle points (LiDAR + map) as red dots."""
        pts = [(float(x), float(y), z) for (x, y) in points]
        self._points(pts, [1.0, 0.0, 0.0, 1.0], persistent=False)

    def draw_mission_goal(self, goal_xy: Tuple[float, float], z: float) -> None:
        gx = float(goal_xy[0])
        gy = float(goal_xy[1])
        gz = float(z)
        size = 1.0
        self._points([(gx, gy, gz)], [1.0, 1.0, 0.0, 1.0], persistent=False)
        self._line([
            (gx - size, gy, gz), (gx + size, gy, gz),
            (gx, gy - size, gz), (gx, gy + size, gz),
            (gx, gy, gz - size), (gx, gy, gz + size),
        ], [1.0, 1.0, 0.0, 1.0], persistent=False)

    def draw_goal_alignment(
        self,
        drone_xyz: Tuple[float, float, float],
        goal_xy: Tuple[float, float],
        goal_z: float,
    ) -> None:
        """Draw the current drone-to-goal alignment plus a vertical height cue."""
        dx = float(drone_xyz[0])
        dy = float(drone_xyz[1])
        dz = float(drone_xyz[2])
        gx = float(goal_xy[0])
        gy = float(goal_xy[1])
        gz = float(goal_z)
        self._line([(dx, dy, dz), (gx, gy, dz)], [0.0, 1.0, 1.0, 1.0], persistent=False)  # cyan
        self._line([(gx, gy, dz), (gx, gy, gz)], [0.0, 1.0, 1.0, 1.0], persistent=False)  # cyan
        self._points([(dx, dy, dz)], [0.0, 1.0, 1.0, 1.0], persistent=False)

    def draw_drone_path(self, history: List[Tuple[float, float, float]]) -> None:
        """Leave a persistent trace of the drone path."""
        pts = [(float(x), float(y), float(z)) for (x, y, z) in history]
        self._line(pts, [1.0, 1.0, 1.0, 0.6], persistent=True)

    # ── HUD (sec 20) ──

    def hud_status(self, message: str, severity: int = HUD_SEVERITY_INFO) -> None:
        """Print a one-line status to the AirSim HUD / sim log."""
        self._safe(
            self._client().simPrintLogMessage,
            message, self._vehicle_name, severity,
        )

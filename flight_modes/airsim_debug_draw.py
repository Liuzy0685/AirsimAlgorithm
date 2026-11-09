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
import queue
import threading
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
        goal_marker_size_m: float = 2.0,
        goal_marker_height_m: float = 0.0,
        duration_s: float = 0.3,
        async_mode: bool = True,
        queue_size: int = 1,
    ) -> None:
        self._adapter = adapter
        self.enabled = bool(enabled)
        self._vehicle_name = vehicle_name or adapter.vehicle_name
        self.line_thickness = line_thickness
        self.point_size = point_size
        self.goal_marker_size_m = max(0.5, float(goal_marker_size_m))
        self.goal_marker_height_m = max(0.0, float(goal_marker_height_m))
        self.duration_s = duration_s
        self._async_mode = bool(async_mode)
        self._draw_adapter = None
        self._draw_queue: Optional[queue.Queue] = None
        self._draw_stop: Optional[threading.Event] = None
        self._draw_thread: Optional[threading.Thread] = None
        self._warned_rpcs = set()
        self._legacy_rpcs = set()
        self._submitted_frames = 0
        self._rendered_frames = 0

        # AirSim simPlot* calls are synchronous and may wait on Unreal's game
        # thread.  In flight mode they must not share the control-loop client.
        # A one-slot latest-frame queue keeps drawing useful without allowing
        # old diagnostic frames to build up and delay navigation.
        if self.enabled and self._async_mode:
            self._draw_queue = queue.Queue(maxsize=max(1, int(queue_size)))
            self._draw_stop = threading.Event()
            self._draw_thread = threading.Thread(
                target=self._draw_worker,
                name="airsim-debug-drawer",
                daemon=True,
            )
            self._draw_thread.start()

    # ── low-level ──

    def _client(self):
        adapter = self._draw_adapter or self._adapter
        return adapter.get_raw_client()

    def _safe(
        self,
        fn,
        *args,
        rpc_name: Optional[str] = None,
        legacy_args=None,
    ) -> None:
        if not self.enabled:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — drawing must never crash flight
            name = rpc_name or getattr(fn, "__name__", "unknown_rpc")
            # Older AirSim builds expose simPlot* without the final
            # is_persistent argument. Retry only for an argument-count
            # mismatch; other errors should not be sent twice.
            if legacy_args is not None and "argument count" in str(exc).lower():
                try:
                    raw_client = self._client()
                    rpc_client = getattr(raw_client, "client", None)
                    raw_call = getattr(rpc_client, "call", None)
                    if callable(raw_call):
                        # Calling the generated Python wrapper again would
                        # still append is_persistent. Use msgpack-rpc
                        # directly so old servers receive exactly 4 args.
                        raw_call(name, *legacy_args)
                    else:
                        # Keep lightweight test doubles and custom adapters
                        # compatible when they expose no raw RPC object.
                        fn(*legacy_args)
                    if name not in self._legacy_rpcs:
                        self._legacy_rpcs.add(name)
                        logger.info(
                            "airsim_debug_draw legacy_signature  rpc=%s  args=%d",
                            name, len(legacy_args),
                        )
                    return
                except Exception as legacy_exc:  # noqa: BLE001
                    exc = legacy_exc
            if name not in self._warned_rpcs:
                self._warned_rpcs.add(name)
                logger.warning(
                    "airsim_debug_draw_rpc_failed  rpc=%s  error=%s",
                    name, exc,
                )
            logger.debug("airsim_debug_draw skipped: %s", exc, exc_info=True)

    def _line(self, pts: Sequence[Tuple[float, float, float]], rgba, persistent: bool) -> None:
        if len(pts) < 2:
            return
        vectors = [_v3(p[0], p[1], p[2]) for p in pts]
        self._safe(
            self._client().simPlotLineList,
            vectors, rgba, self.line_thickness, self.duration_s, persistent,
            rpc_name="simPlotLineList",
            legacy_args=(vectors, rgba, self.line_thickness, self.duration_s),
        )

    def _points(self, pts: Sequence[Tuple[float, float, float]], rgba, persistent: bool) -> None:
        if not pts:
            return
        vectors = [_v3(p[0], p[1], p[2]) for p in pts]
        self._safe(
            self._client().simPlotPoints,
            vectors, rgba, self.point_size, self.duration_s, persistent,
            rpc_name="simPlotPoints",
            legacy_args=(vectors, rgba, self.point_size, self.duration_s),
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
        # Keep this guard at the RPC boundary as well as in the flight loop:
        # plotting thousands of points is a synchronous AirSim call.
        points = list(points)
        if len(points) > 200:
            step = max(1, len(points) // 200)
            points = points[::step][:200]
        pts = [(float(x), float(y), z) for (x, y) in points]
        self._points(pts, [1.0, 0.0, 0.0, 1.0], persistent=False)

    def draw_mission_goal(self, goal_xy: Tuple[float, float], z: float) -> None:
        """Draw a large yellow X at the mission goal altitude."""
        x, y = float(goal_xy[0]), float(goal_xy[1])
        half = self.goal_marker_size_m / 2.0
        top_z = float(z) - self.goal_marker_height_m
        marker_segments = [
            (x - half, y - half, z), (x + half, y + half, z),
            (x - half, y + half, z), (x + half, y - half, z),
        ]
        yellow = [1.0, 1.0, 0.0, 1.0]
        marker_points = [(x, y, z)]
        if self.goal_marker_height_m > 0.0:
            marker_segments.extend([(x, y, z), (x, y, top_z)])
            marker_points.append((x, y, top_z))
        self._line(marker_segments, yellow, persistent=False)
        self._points(marker_points, yellow, persistent=False)

    def draw_goal_line(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        z: float,
    ) -> None:
        """Draw the direct current-position to mission-goal guide line."""
        self._line(
            [
                (float(start_xy[0]), float(start_xy[1]), float(z)),
                (float(goal_xy[0]), float(goal_xy[1]), float(z)),
            ],
            [1.0, 1.0, 0.0, 1.0],
            persistent=False,
        )

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

    # ── asynchronous flight-safe drawing ──

    def submit_frame(
        self,
        *,
        global_path=None,
        selected_trajectory=None,
        obstacles=None,
        mission_goal: Optional[Tuple[float, float]] = None,
        goal_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        z: float = 0.0,
        goal_z: Optional[float] = None,
        hud_message: Optional[str] = None,
    ) -> bool:
        """Queue a diagnostic frame without waiting for an AirSim RPC.

        The payload is copied because the producer owns mutable planner/map
        containers.  If the renderer is behind, the older queued frame is
        discarded; rendering stale diagnostics is less useful than keeping
        the control loop on time.
        """
        if not self.enabled:
            return False
        frame = {
            "global_path": list(global_path or []),
            "selected_trajectory": list(selected_trajectory or []),
            "obstacles": list(obstacles or []),
            "mission_goal": tuple(mission_goal) if mission_goal is not None else None,
            "goal_line": goal_line,
            "z": float(z),
            "goal_z": float(z if goal_z is None else goal_z),
            "hud_message": hud_message,
        }
        if not self._async_mode or self._draw_queue is None:
            self._draw_frame(frame)
            self._submitted_frames += 1
            return True
        try:
            self._draw_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._draw_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._draw_queue.put_nowait(frame)
            except queue.Full:
                return False
        self._submitted_frames += 1
        if self._submitted_frames == 1:
            logger.info("airsim_debug_draw frame_queued")
        return True

    def _draw_frame(self, frame: dict) -> None:
        z = frame["z"]
        goal_z = frame.get("goal_z", z)
        self.draw_global_path(frame["global_path"], z)
        if frame["selected_trajectory"]:
            self.draw_selected_trajectory(frame["selected_trajectory"], z)
        if frame["obstacles"]:
            self.draw_obstacles(frame["obstacles"], z)
        if frame["mission_goal"] is not None:
            self.draw_mission_goal(frame["mission_goal"], goal_z)
        if frame["goal_line"] is not None:
            start_xy, goal_xy = frame["goal_line"]
            self.draw_goal_line(start_xy, goal_xy, goal_z)
        if frame["hud_message"]:
            self.hud_status(frame["hud_message"])

    def _draw_worker(self) -> None:
        clone = None
        try:
            # Keep plotting RPCs off the control client's msgpack-rpc socket.
            clone = self._adapter.clone_readonly()
            clone.connect()
            self._draw_adapter = clone
            logger.info("airsim_debug_draw worker_connected")
            while self._draw_stop is not None and not self._draw_stop.is_set():
                try:
                    frame = self._draw_queue.get(timeout=0.2)  # type: ignore[union-attr]
                except queue.Empty:
                    continue
                if frame is None:
                    break
                self._draw_frame(frame)
                self._rendered_frames += 1
                if self._rendered_frames == 1:
                    logger.info("airsim_debug_draw first_frame_rendered")
        except Exception as exc:  # drawing must never affect flight
            logger.warning("airsim_async_debug_draw stopped: %s", exc)
        finally:
            self._draw_adapter = None
            if clone is not None:
                try:
                    clone.close()
                except Exception:
                    pass

    def close(self) -> None:
        """Stop the renderer promptly without waiting on an AirSim RPC."""
        if self._draw_stop is None:
            return
        self._draw_stop.set()
        if self._draw_queue is not None:
            try:
                self._draw_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._draw_queue.get_nowait()
                    self._draw_queue.put_nowait(None)
                except queue.Empty:
                    pass
        if self._draw_thread is not None and self._draw_thread.is_alive():
            self._draw_thread.join(timeout=0.5)
        self._draw_thread = None

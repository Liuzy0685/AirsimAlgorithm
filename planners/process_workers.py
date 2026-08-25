"""
Cross-process planner workers (Phase C3-R).

The CBMBA global A* and the LocalTrajectoryPlanner are both CPU-bound pure
Python.  Running them on a ``threading.Thread`` (as ``GlobalPlannerWorker``
did) is not enough: a pure-Python loop holds the **GIL** for its entire
search, so the main control thread's ``perf_counter`` keeps advancing while it
is starved of CPU.  The result was a falsely-inflated ``dispatch_ms`` and a
20 Hz loop that collapsed to a few Hz whenever a 1-3 s A* ran.

These workers move each planner into its **own OS process** so the 20 Hz
control loop is never blocked:

* ``CbmbaProcessWorker``            — global CBMBA A* (1-3 s per search)
* ``LocalTrajectoryPlannerWorker``  — distance-field build + candidate gen
                                      + scoring (100-250 ms per plan)

Both follow the same contract:

* The worker *owns* its planner instance and every piece of cross-frame state
  (``TrajectoryMemory``, the mirror ``OccupancyGridMap``, the ``DistanceField``).
  No planner instance is ever shared or passed across the process boundary —
  only picklable snapshots are.
* The control loop **never blocks**: no ``future.result()``, no ``join()``,
  no blocking ``queue.get()``.  It only ever *polls* ``poll_latest_result()``
  (``queue.get_nowait()``) and keeps using the cached path / trajectory when
  no new result is ready.
* **Latest-request-wins**: at most one in-flight request plus one pending
  slot.  A new request overwrites the pending slot; a stale snapshot is never
  allowed to build up into a long queue.

Windows note: the default ``spawn`` start method re-imports this module in the
child, so the worker entry points MUST be module-level functions (they are:
``_cbmba_worker_main`` / ``_local_worker_main``).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("process_workers")


def _configure_child_logging() -> None:
    """Give a spawned child process the same console logging as the parent.

    ``spawn`` re-imports modules but does NOT re-run the parent's
    ``logging.basicConfig``, so the child's root logger would otherwise sit at
    the default WARNING level.  This is a separate OS process, so configuring
    it can never contend with the main loop.
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    except Exception:  # noqa: BLE001 — never let logging setup kill a worker
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (verbatim mirrors of AutomaticMode._sensor_points_to_world_xy
# / _downsample_xy so the worker's distance field matches the main loop exactly).
# ─────────────────────────────────────────────────────────────────────────────


def _sensor_points_to_world_xy(
    points,
    drone_position_ned,
    yaw_rad,
    max_range: float = 15.0,
    horizontal_band: float = 1.0,
) -> list:
    out = []
    if points is None or getattr(points, "size", 0) == 0:
        return out
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    px = drone_position_ned[0]
    py = drone_position_ned[1]
    for row in points:
        try:
            sx, sy, sz = float(row[0]), float(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if abs(sz) > horizontal_band:
            continue
        if math.hypot(sx, sy) > max_range:
            continue
        wx = px + sx * cos_y - sy * sin_y
        wy = py + sx * sin_y + sy * cos_y
        out.append((wx, wy))
    return out


def _downsample_xy(points, res_m: float):
    if not points or res_m <= 0.0:
        return points
    seen = set()
    out = []
    inv = 1.0 / res_m
    for x, y in points:
        key = (int(math.floor(x * inv)), int(math.floor(y * inv)))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CBMBA global planner — separate process
# ─────────────────────────────────────────────────────────────────────────────


def _cbmba_worker_main(req_q, res_q, planner_config: dict, poll_interval_s: float) -> None:
    """Module-level entry point for the CBMBA planner process (spawn-safe)."""
    _configure_child_logging()
    wlog = logging.getLogger("cbmba_process_worker")

    # The worker owns its own planner instance — never shared with the parent.
    from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams

    planner = CbmbaAStarPlanner(CbmbaParams(**planner_config))
    wlog.info("cbmba_process_worker  started  resolution=%.2f",
              planner.params.resolution)

    while True:
        req = None
        # Drain the queue, keeping only the latest request (latest-request-wins).
        while True:
            try:
                item = req_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return  # shutdown sentinel
            req = item

        if req is None:
            time.sleep(poll_interval_s)
            continue

        t0 = time.perf_counter()
        # Phase C3-R GIL verification: this line prints from the *separate
        # process* at the moment the A* starts.  If ``running=true`` appears
        # interleaved with 50 ms ``control_scheduler`` frames (rather than
        # freezing them), the loop is not blocked by the search.
        wlog.info("cbmba_runtime  request_id=%d  running=true",
                  req["request_id"])
        try:
            res = planner.plan_with_result(
                req["obstacles"], req["start"], req["goal"],
            )
            wlog.info("cbmba_runtime  request_id=%d  running=false  time_ms=%.2f",
                      req["request_id"], res.planning_time_ms)
            result = {
                "request_id": req["request_id"],
                "reason": req["reason"],
                "success": res.success,
                "path_world": res.path_world,
                "nodes_expanded": res.nodes_expanded,
                "planning_time_ms": res.planning_time_ms,
                "grid_size": res.grid_size,
                "max_lateral_deviation_m": res.max_lateral_deviation_m,
            }
        except Exception as exc:  # noqa: BLE001
            wlog.warning("cbmba_process_worker_error: %s", exc)
            result = {
                "request_id": req["request_id"],
                "reason": req["reason"],
                "success": False,
                "path_world": [list(req["start"]), list(req["goal"])],
                "nodes_expanded": 0,
                "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
                "grid_size": 0,
                "max_lateral_deviation_m": 0.0,
                "error": str(exc),
            }
        res_q.put(result)


class CbmbaProcessWorker:
    """Main-thread handle for the CBMBA planner **process**.

    Mirrors the ``GlobalPlannerWorker`` public API (``request_replan`` /
    ``get_latest_result`` / ``has_in_flight_request`` / ``search_count`` /
    ``shutdown``) so the control loop's call sites change minimally, but the
    heavy A* now runs in a separate process instead of a GIL-sharing thread.
    """

    def __init__(self, planner_config: dict, poll_interval_s: float = 0.005):
        self._planner_config = dict(planner_config)
        self._poll_interval_s = poll_interval_s
        self._req_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._res_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._proc: Optional[multiprocessing.Process] = None
        self._lock = threading.Lock()
        self._request_counter = 0
        self._pending: Optional[dict] = None      # latest unsent request (1 slot)
        self._running_id: Optional[int] = None    # request id currently in-flight
        self._latest: Optional[dict] = None       # latest finished result
        self.search_count = 0
        self._started = False
        # Phase C4: IPC / backpressure instrumentation.
        self.submitted_count = 0
        self.coalesced_count = 0
        self._ipc_put_ms = 0.0
        self._ipc_poll_ms = 0.0
        self._ipc_last_log_mono = 0.0
        self._last_obstacle_count = 0
        self.time_ms_sum = 0.0
        self.time_ms_max = 0.0

    # ── main-thread API ──

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = multiprocessing.Process(
            target=_cbmba_worker_main,
            args=(self._req_queue, self._res_queue,
                  self._planner_config, self._poll_interval_s),
            name="cbmba-global-planner-proc",
            daemon=True,
        )
        self._proc.start()
        self._started = True
        logger.info(
            "process_worker_started  worker=cbmba  pid=%s",
            self._proc.pid,
        )

    def request_replan(self, obstacles, start, goal, reason: str = "") -> int:
        with self._lock:
            self._service_locked()
            if self._pending is not None:
                self.coalesced_count += 1  # latest-request-wins: overwrite stale pending
            self._request_counter += 1
            rid = self._request_counter
            self._last_obstacle_count = len(obstacles)
            self._pending = {
                "request_id": rid,
                "reason": reason,
                "obstacles": list(obstacles),
                "start": list(start),
                "goal": list(goal),
            }
            self._service_locked()  # submit the just-set pending slot
            return rid

    def poll_latest_result(self) -> Optional[dict]:
        """Non-blocking read of the latest finished result (never blocks)."""
        return self.get_latest_result()

    def get_latest_result(self) -> Optional[dict]:
        with self._lock:
            self._service_locked()
            return self._latest

    def has_in_flight_request(self) -> bool:
        with self._lock:
            self._service_locked()
            return self._pending is not None or self._running_id is not None

    def is_running(self) -> bool:
        """True while a search is executing inside the worker process."""
        with self._lock:
            self._service_locked()
            return self._running_id is not None

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        _pid = proc.pid
        try:
            self._req_queue.put(None, block=False)
        except Exception:  # noqa: BLE001
            pass
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        logger.info(
            "process_worker_shutdown  worker=cbmba  pid=%s  alive=%s",
            _pid, "false" if not proc.is_alive() else "true",
        )

    # ── internal (callers hold self._lock) ──

    def _service_locked(self) -> None:
        # Drain finished results (never block).
        while True:
            try:
                _poll_t0 = time.perf_counter()
                result = self._res_queue.get_nowait()
                self._ipc_poll_ms = (time.perf_counter() - _poll_t0) * 1000.0
            except queue.Empty:
                break
            self.search_count += 1
            self._latest = result
            _pt = result.get("planning_time_ms", 0.0)
            self.time_ms_sum += _pt
            self.time_ms_max = max(self.time_ms_max, _pt)
            rid = result.get("request_id", -1)
            if self._running_id is not None and rid >= self._running_id:
                self._running_id = None
            logger.info(
                "cbmba_search_invocation  request_id=%d  count=%d  "
                "reason=%s  time_ms=%.2f  success=%s",
                rid, self.search_count, result.get("reason", ""),
                result.get("planning_time_ms", 0.0),
                "true" if result.get("success") else "false",
            )
        # Submit the pending slot if the worker is idle (backpressure: never
        # more than one request in flight).
        if self._pending is not None and self._running_id is None:
            req = self._pending
            self._pending = None
            self._running_id = req["request_id"]
            try:
                _put_t0 = time.perf_counter()
                self._req_queue.put(req, block=False)
                self._ipc_put_ms = (time.perf_counter() - _put_t0) * 1000.0
                self.submitted_count += 1
            except Exception:  # noqa: BLE001
                self._pending = req
                self._running_id = None
        # Low-frequency IPC / backpressure telemetry (Phase C4).
        _now = time.monotonic()
        if _now - self._ipc_last_log_mono >= 1.0:
            self._ipc_last_log_mono = _now
            logger.info(
                "worker_ipc_timing  worker=cbmba  put_ms=%.3f  poll_ms=%.3f  "
                "obstacle_count=%d  submitted=%d  coalesced=%d  completed=%d",
                self._ipc_put_ms, self._ipc_poll_ms, self._last_obstacle_count,
                self.submitted_count, self.coalesced_count, self.search_count,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Local trajectory planner — separate process
# ─────────────────────────────────────────────────────────────────────────────


def _local_worker_main(
    req_q,
    res_q,
    traj_config: dict,
    occ_config: dict,
    memory_history_length: int,
    dfield_radius_m: float,
    downsample_m: float,
    poll_interval_s: float,
) -> None:
    """Module-level entry point for the local-trajectory process (spawn-safe)."""
    _configure_child_logging()
    wlog = logging.getLogger("local_traj_process_worker")

    from planners.local_trajectory_planner import (
        LocalTrajectoryPlanner, TrajectoryMemory, TrajectoryPlannerParams,
    )
    from mapping.occupancy_grid import UNKNOWN, OccupancyGridMap, OccupancyGridParams
    from mapping.distance_field import DistanceField

    params = TrajectoryPlannerParams(**traj_config)
    memory = TrajectoryMemory(history_length=memory_history_length)
    planner = LocalTrajectoryPlanner(params=params, memory=memory)
    # Mirror occupancy grid — kept in sync read-before-write so the worker's
    # distance field matches the main loop's historical map + current LiDAR.
    occ_grid = OccupancyGridMap(OccupancyGridParams(**occ_config))
    dfield = DistanceField()
    wlog.info("local_traj_process_worker  started  planning_hz=%.1f",
              params.planning_hz)

    while True:
        req = None
        reset_requested = False
        while True:
            try:
                item = req_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return  # shutdown sentinel
            if isinstance(item, dict) and item.get("reset"):
                reset_requested = True
                continue
            req = item

        # Recovery-exit reset: forget the last trajectory before the next plan
        # so the stale family/memory can't pull the drone back to a dead end.
        if reset_requested:
            memory.reset()
            wlog.info("local_traj_memory_reset  reason=worker_reset_command")

        if req is None:
            time.sleep(poll_interval_s)
            continue

        pos = (float(req["drone_position_ned"][0]),
               float(req["drone_position_ned"][1]),
               float(req["drone_position_ned"][2]))
        yaw = float(req["yaw_rad"])
        goal_xy = (float(req["goal_xy"][0]), float(req["goal_xy"][1]))
        global_path = req["global_path"] or []
        gv = int(req["global_path_version"])
        lidar = req["lidar_points"]

        # ── read-before-write: query mirror map BEFORE integrating current lidar ──
        _df_t0 = time.perf_counter()
        world_xy = _sensor_points_to_world_xy(
            lidar, pos, yaw,
            max_range=occ_grid.params.max_range_m,
            horizontal_band=occ_grid.params.horizontal_band_half_height_m,
        )
        lidar_n_raw = len(world_xy)
        world_xy = _downsample_xy(world_xy, downsample_m)
        map_obstacles = occ_grid.get_occupied_points_in_radius(
            pos[0], pos[1], dfield_radius_m,
        )
        obstacles = list(map_obstacles) + world_xy
        dfield.set_obstacles(obstacles)
        build_ms = (time.perf_counter() - _df_t0) * 1000.0
        wlog.info(
            "distance_field_build  map_version=%d  global_occupied=%d  "
            "local_occupied=%d  current_lidar_points=%d  "
            "downsampled_lidar_points=%d  build_ms=%.2f",
            occ_grid.version, len(occ_grid.get_occupied_points()),
            len(map_obstacles), lidar_n_raw, len(world_xy), build_ms,
        )

        def _is_unknown(x: float, y: float) -> bool:
            try:
                return occ_grid.state_at(x, y) == UNKNOWN
            except Exception:  # noqa: BLE001
                return False

        try:
            _plan_t0 = time.perf_counter()
            result = planner.plan(
                drone_position_ned=pos,
                yaw_rad=yaw,
                goal_xy=goal_xy,
                global_path=global_path,
                distance_field=dfield,
                unknown_query=_is_unknown,
                global_path_version=gv,
            )
            compute_ms = (time.perf_counter() - _plan_t0) * 1000.0
            wlog.info(
                "trajectory_worker  request_id=%d  compute_ms=%.2f",
                req["request_id"], compute_ms,
            )
            path_xy = [
                (float(wp[0]), float(wp[1]))
                for wp in global_path if wp is not None and len(wp) >= 2
            ]
            gp_min_clear = (
                dfield.trajectory_min_clearance(path_xy) if path_xy else float("inf")
            )
            envelope = {
                "request_id": req["request_id"],
                "result": result,                     # TrajectoryPlanResult (picklable)
                "global_path_min_clearance": gp_min_clear,
                "distance_field_build_ms": build_ms,
                "planning_time_ms": compute_ms,
                "map_obstacle_count": len(map_obstacles),
                "lidar_n_raw": lidar_n_raw,
                "downsampled_n": len(world_xy),
            }
        except Exception as exc:  # noqa: BLE001
            wlog.warning("local_traj_process_worker_error: %s", exc)
            envelope = {
                "request_id": req["request_id"],
                "result": None,
                "global_path_min_clearance": float("inf"),
                "distance_field_build_ms": build_ms,
                "planning_time_ms": 0.0,
                "map_obstacle_count": len(map_obstacles),
                "lidar_n_raw": lidar_n_raw,
                "downsampled_n": len(world_xy),
                "error": str(exc),
            }
        res_q.put(envelope)

        # ── write step: integrate current lidar into the mirror map ──
        try:
            occ_grid.update(lidar, pos, yaw)
        except Exception as exc:  # noqa: BLE001
            wlog.warning("local_worker_map_update_error: %s", exc)


class LocalTrajectoryPlannerWorker:
    """Main-thread handle for the local-trajectory planner **process**.

    The worker owns the ``LocalTrajectoryPlanner`` + ``TrajectoryMemory`` +
    ``DistanceField`` + mirror ``OccupancyGridMap``, so the control loop only
    ever sends a compact snapshot and receives back a selected trajectory.
    """

    def __init__(
        self,
        traj_config: dict,
        occ_config: dict,
        memory_history_length: int,
        dfield_radius_m: float,
        downsample_m: float,
        poll_interval_s: float = 0.005,
    ):
        self._traj_config = dict(traj_config)
        self._occ_config = dict(occ_config)
        self._memory_history_length = int(memory_history_length)
        self._dfield_radius_m = float(dfield_radius_m)
        self._downsample_m = float(downsample_m)
        self._poll_interval_s = poll_interval_s
        self._req_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._res_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._proc: Optional[multiprocessing.Process] = None
        self._lock = threading.Lock()
        self._request_counter = 0
        self._pending: Optional[dict] = None
        self._running_id: Optional[int] = None
        self._latest: Optional[dict] = None
        self.plan_count = 0
        self._started = False
        # Phase C4: IPC / backpressure instrumentation.
        self.submitted_count = 0
        self.coalesced_count = 0
        self._ipc_put_ms = 0.0
        self._ipc_poll_ms = 0.0
        self._ipc_last_log_mono = 0.0
        self._last_lidar_points = 0
        self._last_global_path_points = 0
        self.plan_time_ms_sum = 0.0
        self.plan_time_ms_max = 0.0

    # ── main-thread API ──

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = multiprocessing.Process(
            target=_local_worker_main,
            args=(self._req_queue, self._res_queue,
                  self._traj_config, self._occ_config,
                  self._memory_history_length,
                  self._dfield_radius_m, self._downsample_m,
                  self._poll_interval_s),
            name="local-trajectory-planner-proc",
            daemon=True,
        )
        self._proc.start()
        self._started = True
        logger.info(
            "process_worker_started  worker=trajectory  pid=%s",
            self._proc.pid,
        )

    def request_plan(self, snapshot: Dict[str, Any]) -> int:
        with self._lock:
            self._service_locked()
            if self._pending is not None:
                self.coalesced_count += 1  # latest-request-wins: overwrite stale pending
            self._request_counter += 1
            rid = self._request_counter
            _lp = snapshot.get("lidar_points")
            _gp = snapshot.get("global_path")
            self._last_lidar_points = len(_lp) if _lp is not None else 0
            self._last_global_path_points = len(_gp) if _gp is not None else 0
            self._pending = {"request_id": rid, **snapshot}
            self._service_locked()
            return rid

    def reset_memory(self) -> None:
        """Ask the worker process to forget its trajectory memory.

        Phase C3-R: ``TrajectoryMemory`` now lives in the worker process, so the
        Recovery-exit reset must be forwarded across the process boundary (the
        main loop's own memory copy is no longer the one the planner reads).
        Non-blocking: the reset command is queued and applied before the next
        plan.
        """
        try:
            self._req_queue.put({"reset": True}, block=False)
        except Exception:  # noqa: BLE001
            pass

    def poll_latest_result(self) -> Optional[dict]:
        """Non-blocking read of the latest finished plan (never blocks)."""
        return self.get_latest_result()

    def get_latest_result(self) -> Optional[dict]:
        with self._lock:
            self._service_locked()
            return self._latest

    def has_in_flight_request(self) -> bool:
        with self._lock:
            self._service_locked()
            return self._pending is not None or self._running_id is not None

    def is_running(self) -> bool:
        with self._lock:
            self._service_locked()
            return self._running_id is not None

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        _pid = proc.pid
        try:
            self._req_queue.put(None, block=False)
        except Exception:  # noqa: BLE001
            pass
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        logger.info(
            "process_worker_shutdown  worker=trajectory  pid=%s  alive=%s",
            _pid, "false" if not proc.is_alive() else "true",
        )

    # ── internal (callers hold self._lock) ──

    def _service_locked(self) -> None:
        while True:
            try:
                _poll_t0 = time.perf_counter()
                envelope = self._res_queue.get_nowait()
                self._ipc_poll_ms = (time.perf_counter() - _poll_t0) * 1000.0
            except queue.Empty:
                break
            self.plan_count += 1
            self._latest = envelope
            _pt = envelope.get("planning_time_ms", 0.0)
            self.plan_time_ms_sum += _pt
            self.plan_time_ms_max = max(self.plan_time_ms_max, _pt)
            rid = envelope.get("request_id", -1)
            if self._running_id is not None and rid >= self._running_id:
                self._running_id = None
        if self._pending is not None and self._running_id is None:
            req = self._pending
            self._pending = None
            self._running_id = req["request_id"]
            try:
                _put_t0 = time.perf_counter()
                self._req_queue.put(req, block=False)
                self._ipc_put_ms = (time.perf_counter() - _put_t0) * 1000.0
                self.submitted_count += 1
            except Exception:  # noqa: BLE001
                self._pending = req
                self._running_id = None
        # Low-frequency IPC / backpressure telemetry (Phase C4).
        _now = time.monotonic()
        if _now - self._ipc_last_log_mono >= 1.0:
            self._ipc_last_log_mono = _now
            logger.info(
                "worker_ipc_timing  worker=trajectory  put_ms=%.3f  poll_ms=%.3f  "
                "lidar_points=%d  global_path_points=%d  "
                "submitted=%d  coalesced=%d  completed=%d",
                self._ipc_put_ms, self._ipc_poll_ms,
                self._last_lidar_points, self._last_global_path_points,
                self.submitted_count, self.coalesced_count, self.plan_count,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Persistent occupancy-grid mapping — separate process (Phase C4-R)
# ─────────────────────────────────────────────────────────────────────────────


def _mapping_worker_main(
    req_q, res_q, occ_config: dict, poll_interval_s: float,
) -> None:
    """Module-level entry point for the persistent-map process (spawn-safe).

    Owns the **only** ``OccupancyGridMap`` the main loop no longer touches
    directly: the 300-800 ms pure-Python ray-casting ``update()`` runs here, so
    the realtime control loop (and the GIL-sharing PerceptionWorker thread) are
    never starved.  Only a picklable LiDAR-frame snapshot crosses the process
    boundary — never the grid itself.
    """
    _configure_child_logging()
    wlog = logging.getLogger("mapping_process_worker")

    from mapping.occupancy_grid import OccupancyGridMap, OccupancyGridParams

    occ_grid = OccupancyGridMap(OccupancyGridParams(**occ_config))
    wlog.info("mapping_process_worker  started  resolution=%.2f",
              occ_grid.params.resolution_m)

    while True:
        req = None
        # Drain the queue, keeping only the latest request (latest-request-wins).
        while True:
            try:
                item = req_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return  # shutdown sentinel
            req = item

        if req is None:
            time.sleep(poll_interval_s)
            continue

        t0 = time.perf_counter()
        rid = req["request_id"]
        wlog.info("map_worker_runtime  request_id=%d  running=true", rid)
        points = req.get("points_sensor")
        try:
            occ_grid.update(
                points,
                tuple(req["drone_position_ned"]),
                float(req["yaw_rad"]),
            )
            compute_ms = (time.perf_counter() - t0) * 1000.0
            occupied = occ_grid.get_occupied_points()
            result = {
                "request_id": rid,
                "map_version": occ_grid.version,
                "sensor_timestamp": req.get("sensor_timestamp", -1.0),
                "compute_ms": compute_ms,
                "cells_updated": occ_grid.cell_count,
                "rays": len(points) if points is not None else 0,
                "occupied_points": [[float(x), float(y)] for (x, y) in occupied],
                "obstacle_count": len(occupied),
            }
        except Exception as exc:  # noqa: BLE001
            wlog.warning("mapping_process_worker_error: %s", exc)
            result = {
                "request_id": rid,
                "map_version": -1,
                "sensor_timestamp": req.get("sensor_timestamp", -1.0),
                "compute_ms": (time.perf_counter() - t0) * 1000.0,
                "cells_updated": 0,
                "rays": 0,
                "occupied_points": [],
                "obstacle_count": 0,
                "error": str(exc),
            }
        res_q.put(result)
        wlog.info(
            "map_worker_runtime  request_id=%d  running=false  time_ms=%.2f  "
            "cells_updated=%d  rays=%d  map_version=%d",
            rid, result["compute_ms"], result["cells_updated"],
            result["rays"], result["map_version"],
        )


class MappingProcessWorker:
    """Main-thread handle for the persistent occupancy-map **process**.

    The worker owns the ``OccupancyGridMap``; the control loop only ever sends a
    picklable LiDAR-frame snapshot (``request_update``) and reads back a compact
    snapshot (occupied points + monotonic ``map_version``), never the grid.
    """

    def __init__(self, occ_config: dict, poll_interval_s: float = 0.005):
        self._occ_config = dict(occ_config)
        self._poll_interval_s = poll_interval_s
        self._req_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._res_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._proc: Optional[multiprocessing.Process] = None
        self._lock = threading.Lock()
        self._request_counter = 0
        self._pending: Optional[dict] = None
        self._running_id: Optional[int] = None
        self._latest: Optional[dict] = None
        self.update_count = 0
        self._started = False
        # Phase C4-R: IPC / backpressure + compute instrumentation.
        self.submitted_count = 0
        self.coalesced_count = 0
        self._ipc_put_ms = 0.0
        self._ipc_poll_ms = 0.0
        self._ipc_put_ms_sum = 0.0
        self._ipc_put_ms_n = 0
        self._ipc_poll_ms_sum = 0.0
        self._ipc_poll_ms_n = 0
        self._ipc_last_log_mono = 0.0
        self._last_points = 0
        self.compute_ms_samples: list = []

    # ── main-thread API ──

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = multiprocessing.Process(
            target=_mapping_worker_main,
            args=(self._req_queue, self._res_queue,
                  self._occ_config, self._poll_interval_s),
            name="mapping-proc",
            daemon=True,
        )
        self._proc.start()
        self._started = True
        logger.info(
            "map_process_worker_started  worker=mapping  pid=%s",
            self._proc.pid,
        )

    def request_update(self, snapshot: Dict[str, Any]) -> int:
        with self._lock:
            self._service_locked()
            if self._pending is not None:
                self.coalesced_count += 1  # latest-request-wins: overwrite stale pending
            self._request_counter += 1
            rid = self._request_counter
            _pts = snapshot.get("points_sensor")
            self._last_points = len(_pts) if _pts is not None else 0
            self._pending = {"request_id": rid, **snapshot}
            self._service_locked()
            return rid

    def poll_latest_result(self) -> Optional[dict]:
        """Non-blocking read of the latest finished map snapshot."""
        return self.get_latest_result()

    def get_latest_result(self) -> Optional[dict]:
        with self._lock:
            self._service_locked()
            return self._latest

    def has_in_flight_request(self) -> bool:
        with self._lock:
            self._service_locked()
            return self._pending is not None or self._running_id is not None

    def is_running(self) -> bool:
        with self._lock:
            self._service_locked()
            return self._running_id is not None

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        _pid = proc.pid
        try:
            self._req_queue.put(None, block=False)
        except Exception:  # noqa: BLE001
            pass
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        logger.info(
            "map_process_worker_shutdown  worker=mapping  pid=%s  alive=%s",
            _pid, "false" if not proc.is_alive() else "true",
        )

    # ── internal (callers hold self._lock) ──

    def _service_locked(self) -> None:
        # Drain finished results (never block).  A result with a stale
        # (non-increasing) map_version is dropped so it can never overwrite a
        # newer snapshot (causal map semantics).
        while True:
            try:
                _poll_t0 = time.perf_counter()
                result = self._res_queue.get_nowait()
                self._ipc_poll_ms = (time.perf_counter() - _poll_t0) * 1000.0
                self._ipc_poll_ms_sum += self._ipc_poll_ms
                self._ipc_poll_ms_n += 1
            except queue.Empty:
                break
            rid = result.get("request_id", -1)
            if self._running_id is not None and rid >= self._running_id:
                self._running_id = None
            mv = result.get("map_version", -1)
            if mv < 0:
                logger.warning(
                    "map_update_error  request_id=%d  error=%s",
                    rid, result.get("error", ""),
                )
                continue  # keep the previous good snapshot
            if self._latest is not None and self._latest.get("map_version", -1) >= mv:
                logger.info(
                    "MAP_STALE_RESULT_IGNORED  request_id=%d  map_version=%d  "
                    "latest_map_version=%d",
                    rid, mv, self._latest.get("map_version", -1),
                )
                continue
            self._latest = result
            self.update_count += 1
            self.compute_ms_samples.append(result.get("compute_ms", 0.0))
            logger.info(
                "map_update_completed  request_id=%d  map_version=%d  "
                "compute_ms=%.2f  obstacle_count=%d",
                rid, mv, result.get("compute_ms", 0.0),
                result.get("obstacle_count", 0),
            )
        # Submit the pending slot if the worker is idle (backpressure: never
        # more than one request in flight).
        if self._pending is not None and self._running_id is None:
            req = self._pending
            self._pending = None
            self._running_id = req["request_id"]
            try:
                _put_t0 = time.perf_counter()
                self._req_queue.put(req, block=False)
                self._ipc_put_ms = (time.perf_counter() - _put_t0) * 1000.0
                self._ipc_put_ms_sum += self._ipc_put_ms
                self._ipc_put_ms_n += 1
                self.submitted_count += 1
            except Exception:  # noqa: BLE001
                self._pending = req
                self._running_id = None
        # Low-frequency IPC / backpressure telemetry (Phase C4-R).
        _now = time.monotonic()
        if _now - self._ipc_last_log_mono >= 1.0:
            self._ipc_last_log_mono = _now
            logger.info(
                "worker_ipc_timing  worker=mapping  put_ms=%.3f  poll_ms=%.3f  "
                "lidar_points=%d  submitted=%d  coalesced=%d  completed=%d",
                self._ipc_put_ms, self._ipc_poll_ms, self._last_points,
                self.submitted_count, self.coalesced_count, self.update_count,
            )

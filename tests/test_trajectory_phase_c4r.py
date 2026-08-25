"""Phase C4-R unit tests — persistent-map process (occupancy-grid hotspot fix).

Covers the pure-mechanism units introduced in Phase C4-R:

  A. mapping process is non-blocking (``request_update`` returns without waiting)
  B. latest-request-wins (one running + one pending slot; stale pending dropped)
  C. same LiDAR frame dedup (one update per sensor timestamp)
  D. stale map result ignored (monotonic map_version)
  E. compact result (no grid object crosses the process boundary)
  F. mapping process shutdown leaves no orphan
  G. CBMBA request uses the latest compact map snapshot
  H. hover summary counters bucket each safety source correctly

Tests A/E/F spawn a real mapping subprocess; each is bounded by ``shutdown()``
so a failure can never leave a stray process behind.
"""

import math
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import AutomaticMode
from mapping.occupancy_grid import OccupancyGridParams
from planners.process_workers import MappingProcessWorker


# ── helpers ──

def _mapping_worker(poll_interval_s=0.001):
    return MappingProcessWorker(
        occ_config=asdict(OccupancyGridParams()), poll_interval_s=poll_interval_s,
    )


def _points(n):
    """Deterministic in-plane LiDAR points (SensorLocalFrame), all valid."""
    pts = []
    for i in range(n):
        ang = (i % 360) * math.pi / 180.0
        d = 1.0 + (i % 14)
        pts.append([d * math.cos(ang), d * math.sin(ang), 0.0])
    return np.array(pts, dtype=np.float64)


def _map_snapshot(points=None, ts=0.0):
    return {
        "sensor_timestamp": ts,
        "drone_position_ned": [0.0, 0.0, -1.0],
        "yaw_rad": 0.0,
        "points_sensor": points if points is not None else np.zeros((0, 3)),
    }


def _wait_for(predicate, timeout_s=8.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = predicate()
        if val is not None and val is not False:
            return val
        time.sleep(0.01)
    return predicate()


class _FakeMappingWorker:
    def __init__(self):
        self.request_count = 0
        self.coalesced_count = 0
        self.submitted_count = 0
        self._counter = 0

    def request_update(self, snapshot):
        self.request_count += 1
        self._counter += 1
        return self._counter

    def get_latest_result(self):
        return None


class _FakeCbmbaWorkerCapturing:
    def __init__(self):
        self.reasons = []
        self.coalesced_count = 0
        self.submitted_count = 0
        self.search_count = 0
        self.last_obstacles = []
        self._counter = 0

    def has_in_flight_request(self):
        return False

    def request_replan(self, obstacles, start, goal, reason=""):
        self.reasons.append(reason)
        self.last_obstacles = list(obstacles)
        self._counter += 1
        return self._counter

    def get_latest_result(self):
        return None


# ── A. mapping process is non-blocking ──


class TestMappingProcessNonBlocking:
    def test_request_update_returns_without_waiting(self):
        w = _mapping_worker()
        w.start()
        try:
            t0 = time.perf_counter()
            rid = w.request_update(_map_snapshot(_points(4000)))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert rid == 1
            # Enqueueing must be ~instant; it must NOT wait for the ray casting.
            assert dt_ms < 200.0
        finally:
            w.shutdown()

    def test_result_arrives_later_without_blocking_loop(self):
        w = _mapping_worker()
        w.start()
        try:
            w.request_update(_map_snapshot(_points(4000)))
            res = _wait_for(lambda: w.get_latest_result())
            assert res is not None
            assert res["request_id"] == 1
            assert res["map_version"] >= 1
            assert "occupied_points" in res
        finally:
            w.shutdown()


# ── B. latest-request-wins ──


class TestMappingLatestRequestWins:
    def test_pending_slot_holds_only_latest(self):
        # Without starting the process, requests pile up deterministically:
        # one running + one pending slot, and the pending slot keeps only the
        # LATEST request (older ones are overwritten).
        w = _mapping_worker()
        w.request_update(_map_snapshot(ts=1.0))
        w.request_update(_map_snapshot(ts=2.0))
        w.request_update(_map_snapshot(ts=3.0))
        assert w._running_id == 1
        assert w._pending is not None
        assert w._pending["request_id"] == 3  # r2 overwritten by r3
        assert w.coalesced_count == 1
        assert w.has_in_flight_request()
        assert w.update_count == 0  # no process → nothing actually completed


# ── C. same LiDAR frame dedup ──


class TestSameLidarFrameDedup:
    def _auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._mapping_worker = _FakeMappingWorker()
        a._last_map_sensor_timestamp = None
        a._last_applied_map_version = -1
        a._map_snapshot = {"occupied_points": [], "map_version": -1}
        return a

    def test_same_sensor_timestamp_triggers_one_update(self):
        a = self._auto()
        lf = types.SimpleNamespace(received_monotonic_seconds=123.456)
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0], yaw_rad=0.0)
        fr = types.SimpleNamespace(filtered_points_sensor=np.zeros((5, 3)))
        assert a._request_map_update(lf, st, fr) is True
        assert a._request_map_update(lf, st, fr) is False  # same frame → dedup
        assert a._mapping_worker.request_count == 1


# ── D. stale map result ignored ──


class TestStaleMapResultIgnored:
    def test_stale_map_version_does_not_overwrite(self):
        w = _mapping_worker()  # not started
        w._res_queue.put({
            "request_id": 1, "map_version": 5, "occupied_points": [[1.0, 2.0]],
            "compute_ms": 1.0, "obstacle_count": 1, "cells_updated": 10,
            "rays": 5, "sensor_timestamp": 1.0,
        })
        w._res_queue.put({
            "request_id": 2, "map_version": 3, "occupied_points": [[9.0, 9.0]],
            "compute_ms": 1.0, "obstacle_count": 1, "cells_updated": 10,
            "rays": 5, "sensor_timestamp": 2.0,
        })
        res = w.get_latest_result()
        assert res["map_version"] == 5  # stale 3 dropped
        assert res["occupied_points"] == [[1.0, 2.0]]
        assert w.update_count == 1  # only the newest counted


# ── E. compact result ──


class TestCompactMapResult:
    def test_result_has_no_grid_object(self):
        w = _mapping_worker()
        w.start()
        try:
            w.request_update(_map_snapshot(_points(500)))
            res = _wait_for(lambda: w.get_latest_result())
            assert res is not None
            from mapping.occupancy_grid import OccupancyGridMap
            for v in res.values():
                assert not isinstance(v, OccupancyGridMap)
            assert isinstance(res["occupied_points"], list)
            # each occupied point is a compact 2-elt [x, y] list, not a dict.
            for p in res["occupied_points"]:
                assert len(p) == 2
        finally:
            w.shutdown()


# ── F. mapping process shutdown leaves no orphan ──


class TestMappingShutdownNoOrphan:
    def test_mapping_shutdown_stops_process(self):
        w = _mapping_worker()
        w.start()
        proc = w._proc
        assert proc is not None and proc.is_alive()
        w.shutdown()
        assert w._proc is None
        proc.join(timeout=3.0)
        assert not proc.is_alive()


# ── G. CBMBA request uses the latest compact map snapshot ──


class TestCbmbaUsesMapSnapshot:
    def _auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._traj_global_path = []
        a._traj_global_path_version = 0
        a._traj_last_replan_time = -float("inf")
        a._traj_global_replan_hz = 1.5
        a._traj_global_replan_requested = False
        a._traj_global_path_min_clearance = float("inf")
        a._initial_replan_requested = False
        a._traj_applied_replan_id = -1
        a._cbmba_search_total = 0
        a._traj_params = types.SimpleNamespace(hard_clearance_m=1.0)
        a._traj_path_switch_min_improvement = 0.10
        a._occ_grid_params = types.SimpleNamespace(resolution_m=0.5)
        a._map_snapshot = {
            "occupied_points": [[3.0, 4.0], [5.0, 6.0]], "map_version": 2,
        }
        a._prune_global_path = lambda path, xy: path
        a._path_length_xy = lambda path: len(path)
        a._path_blocked_by_field = lambda path: False
        a._record_cbmba_search = lambda reason: None
        a._global_planner_worker = _FakeCbmbaWorkerCapturing()
        return a

    def test_cbmba_receives_map_obstacles(self):
        a = self._auto()
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0])
        goal = (42.0, 4.0, -1.0)
        a._tick_global_replan(st, goal, [], 10.0)
        obstacles = a._global_planner_worker.last_obstacles
        positions = [o["position"] for o in obstacles]
        assert [3.0, 4.0, -1.0] in positions
        assert [5.0, 6.0, -1.0] in positions
        assert all(o["type"] == "map" for o in obstacles)


# ── H. hover summary counters ──


class TestHoverCounterSemantics:
    def _auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._trajectory_dispatch_count = 0
        a._hover_dispatch_count = 0
        a._hover_due_perception_stale = 0
        a._hover_due_control_overrun = 0
        a._hover_due_trajectory_stale = 0
        a._hover_due_no_feasible = 0
        a._hover_due_other = 0
        return a

    def test_dispatch_sources_bucket_correctly(self):
        a = self._auto()
        a._record_dispatch_source("trajectory")
        a._record_dispatch_source("trajectory")
        a._record_dispatch_source("control_loop_hover")
        a._record_dispatch_source("trajectory_stale")
        a._record_dispatch_source("trajectory_no_feasible")
        a._record_dispatch_source("recovery")
        assert a._trajectory_dispatch_count == 2
        assert a._hover_due_control_overrun == 1
        assert a._hover_due_trajectory_stale == 1
        assert a._hover_due_no_feasible == 1
        assert a._hover_dispatch_count == 3  # recovery is a maneuver, not a hover

    def test_perception_stale_hover_counted(self):
        a = self._auto()
        a._record_perception_stale_hover()
        assert a._hover_due_perception_stale == 1
        assert a._hover_dispatch_count == 1

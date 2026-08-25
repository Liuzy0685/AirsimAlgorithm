"""Phase C3-R unit tests — GIL / planner-blocking fix.

Covers the pure-mechanism units introduced in Phase C3-R:

  A. CBMBA process is non-blocking (``request_replan`` returns without waiting)
  B. latest-request-wins (one running + one pending slot; stale pending dropped)
  C. no duplicate "initial" replan (dedicated ``_initial_replan_requested`` flag)
  D. stale local-plan result ignored (monotonic request id)
  E. local trajectory process is non-blocking (``request_plan`` returns immediately)
  F. process shutdown leaves no orphan (join + terminate)
  G. recovery position source (detector reports real motion, never (0,0,0))
  H. altitude startup gate (climb confirmation returns a real bool)

No AirSim RPC is exercised.  Tests A/B/E/F spawn real subprocesses (the
planner workers); each is bounded by ``shutdown()`` and a timeout so a
failure can never leave a stray process behind.
"""

import math
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import AutomaticMode
from flight_modes.shared_flight_session import SharedFlightSession
from planners.cbmba_astar import CbmbaParams
from planners.local_recovery import LocalRecovery, RecoveryParams
from planners.local_trajectory_planner import TrajectoryMemory, TrajectoryPlannerParams
from planners.process_workers import CbmbaProcessWorker, LocalTrajectoryPlannerWorker
from mapping.occupancy_grid import OccupancyGridParams


# ── helpers ──

def _cbmba_worker(poll_interval_s=0.001):
    return CbmbaProcessWorker(
        planner_config=asdict(CbmbaParams()), poll_interval_s=poll_interval_s,
    )


def _local_worker():
    return LocalTrajectoryPlannerWorker(
        traj_config=asdict(TrajectoryPlannerParams()),
        occ_config=asdict(OccupancyGridParams()),
        memory_history_length=TrajectoryMemory().history_length,
        dfield_radius_m=10.0,
        downsample_m=0.25,
    )


def _local_snapshot():
    return {
        "drone_position_ned": [0.0, 0.0, -1.0],
        "yaw_rad": 0.0,
        "goal_xy": [10.0, 0.0],
        "global_path": [[0.0, 0.0, -1.0], [10.0, 0.0, -1.0]],
        "global_path_version": 0,
        "lidar_points": [],
    }


def _wait_for(predicate, timeout_s=8.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = predicate()
        if val is not None and val is not False:
            return val
        time.sleep(0.01)
    return predicate()


class _FakeLocalWorker:
    def __init__(self, envelope):
        self._envelope = envelope

    def poll_latest_result(self):
        return self._envelope


# ── A. CBMBA process is non-blocking ──


class TestCbmbaProcessNonBlocking:
    def test_request_replan_returns_without_waiting_for_search(self):
        w = _cbmba_worker()
        w.start()
        try:
            t0 = time.perf_counter()
            rid = w.request_replan(
                [[0, 0, 0], [5, 0, 0]], [0, 0, 0], [30, 0, 0], reason="initial",
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert rid == 1
            # Enqueueing must be ~instant; it must NOT wait for the A* result.
            assert dt_ms < 200.0
        finally:
            w.shutdown()

    def test_result_arrives_later_without_blocking_loop(self):
        w = _cbmba_worker()
        w.start()
        try:
            w.request_replan(
                [[0, 0, 0], [5, 0, 0]], [0, 0, 0], [30, 0, 0], reason="initial",
            )
            res = _wait_for(lambda: w.get_latest_result())
            assert res is not None
            assert res["request_id"] == 1
            assert "success" in res
        finally:
            w.shutdown()


# ── B. latest-request-wins ──


class TestLatestRequestWins:
    def test_pending_slot_holds_only_latest(self):
        # Without starting the process, the worker never drains → requests pile
        # up deterministically: one running + one pending slot, and the pending
        # slot keeps only the LATEST request (older ones are overwritten, so a
        # stale-snapshot queue can never build up).
        w = _cbmba_worker()
        w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="r1")
        w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="r2")
        w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="r3")
        assert w._running_id == 1
        assert w._pending is not None
        assert w._pending["request_id"] == 3  # r2 overwritten by r3
        assert w.has_in_flight_request()
        assert w.search_count == 0  # no process → nothing actually searched


# ── C. no duplicate "initial" replan ──


class TestNoDuplicateInitial:
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
        a._map_snapshot = {"occupied_points": [], "map_version": -1}
        a._prune_global_path = lambda path, xy: path
        a._path_length_xy = lambda path: len(path)
        a._path_blocked_by_field = lambda path: False
        a._record_cbmba_search = lambda reason: None
        a._global_planner_worker = _FakeCbmbaWorker()
        return a

    def test_second_initial_request_is_skipped(self):
        a = self._auto()
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0])
        goal = (42.0, 4.0, -1.0)
        a._tick_global_replan(st, goal, [], 10.0)
        a._tick_global_replan(st, goal, [], 10.0)
        initial_count = sum(
            1 for r in a._global_planner_worker.reasons if r == "initial"
        )
        assert initial_count == 1


class _FakeCbmbaWorker:
    def __init__(self):
        self.reasons = []
        self.search_count = 0
        self.coalesced_count = 0
        self.submitted_count = 0
        self._counter = 0

    def has_in_flight_request(self):
        return False

    def request_replan(self, obstacles, start, goal, reason=""):
        self.reasons.append(reason)
        self._counter += 1
        return self._counter

    def get_latest_result(self):
        return None


# ── D. stale local-plan result ignored ──


class TestStaleResultIgnored:
    def _auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._traj_local_plan_seq = 5
        a._traj_global_path_min_clearance = float("inf")
        return a

    def test_local_poll_ignores_stale_request_id(self):
        a = self._auto()
        a._local_traj_worker = _FakeLocalWorker({"request_id": 3})
        assert a._poll_local_plan() is None  # 3 <= 5 → stale
        assert a._traj_local_plan_seq == 5

    def test_local_poll_applies_newer_and_updates_clearance(self):
        a = self._auto()
        res = types.SimpleNamespace(selected=None)
        a._local_traj_worker = _FakeLocalWorker({
            "request_id": 7, "result": res, "global_path_min_clearance": 0.6,
        })
        out = a._poll_local_plan()
        assert out is res
        assert a._traj_local_plan_seq == 7
        assert a._traj_global_path_min_clearance == 0.6


# ── E. local trajectory process is non-blocking ──


class TestLocalTrajectoryProcessNonBlocking:
    def test_request_plan_returns_without_waiting(self):
        w = _local_worker()
        w.start()
        try:
            t0 = time.perf_counter()
            rid = w.request_plan(_local_snapshot())
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert rid == 1
            assert dt_ms < 200.0
        finally:
            w.shutdown()

    def test_plan_result_arrives_later(self):
        w = _local_worker()
        w.start()
        try:
            w.request_plan(_local_snapshot())
            env = _wait_for(lambda: w.get_latest_result())
            assert env is not None
            assert env["request_id"] == 1
            assert "global_path_min_clearance" in env
        finally:
            w.shutdown()


# ── F. process shutdown leaves no orphan ──


class TestProcessShutdownNoOrphan:
    def test_cbmba_shutdown_stops_process(self):
        w = _cbmba_worker()
        w.start()
        proc = w._proc
        assert proc is not None and proc.is_alive()
        w.shutdown()
        assert w._proc is None
        proc.join(timeout=3.0)
        assert not proc.is_alive()

    def test_local_shutdown_stops_process(self):
        w = _local_worker()
        w.start()
        proc = w._proc
        assert proc is not None and proc.is_alive()
        w.shutdown()
        assert w._proc is None
        proc.join(timeout=3.0)
        assert not proc.is_alive()


# ── G. recovery position source ──


class TestRecoveryPositionSource:
    def test_detector_reports_real_position_before_window_fills(self):
        # Before the fix, a window shorter than stuck_min_frames made the
        # detector return the (0,0,0) sentinel for latest/oldest position.
        r = LocalRecovery(RecoveryParams(stuck_min_frames=10))
        d = r.update(0.0, (42.0, 4.0, -1.0), (0.1, 0.0, 0.0), yaw_rad=0.0)
        assert not d.is_stuck
        assert d.stuck_latest_position == (42.0, 4.0, -1.0)
        assert d.stuck_oldest_position == (42.0, 4.0, -1.0)
        assert d.stuck_latest_position != (0.0, 0.0, 0.0)


# ── H. altitude startup gate ──


def _make_session(z_values):
    s = SharedFlightSession(settings_json="fake.json", mode="auto")
    s.target_z_ned = -1.0
    s.max_vertical_speed_mps = 0.5
    client = MagicMock()

    def _state(z):
        st = MagicMock()
        kin = MagicMock()
        kin.position.z_val = z
        st.kinematics_estimated = kin
        return st

    client.getMultirotorState.side_effect = [_state(z) for z in z_values]
    s._client = client
    return s, client


class TestAltitudeStartupGate:
    def test_confirm_altitude_returns_true_when_reached(self):
        s, client = _make_session([-1.02])  # error 0.02 ≤ 0.3
        assert s._confirm_altitude("Drone1") is True
        assert client.moveToZAsync.call_count == 0

    def test_confirm_altitude_returns_false_when_short(self):
        s, client = _make_session([-0.42, -0.5])  # both short of -1.0
        assert s._confirm_altitude("Drone1") is False
        assert client.moveToZAsync.call_count == 1

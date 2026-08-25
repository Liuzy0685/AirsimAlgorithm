"""Phase C5-R unit tests — control scheduler catch-up fix + worker prep timing.

Covers the pure-mechanism units introduced in Phase C5-R:

  A. normal frame: work=15 ms → scheduler sleeps the remaining ~35 ms
  B. overrun frame: work=200 ms → re-anchor now+period, sleep one period
  C. continuous overrun re-anchors one period ahead of ``now`` each time
  D. target=20 Hz → no <25 ms tight-loop burst across a simulated run
  E. worker request preparation is timed exclusive of the enqueue/put
  F. exclusive sub-timings stay within their parent request window
  G. same LiDAR frame is not re-built (map-update dedup happens before build)
  H. PerceptionWorker timing metrics are observation-only (data payload intact)
"""

import math
import sys
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import flight_modes.automatic_mode as am
from flight_modes.automatic_mode import AutomaticMode, PerceptionWorker


# ── helpers ──

def _sched_auto(period_s=0.05, target_hz=20.0):
    """Minimal AutomaticMode with only the scheduler fields set."""
    a = AutomaticMode.__new__(AutomaticMode)
    a._control_period_s = period_s
    a._control_loop_target_hz = target_hz
    return a


def _call_sleep_to_next(a, next_tick, now):
    """Call ``_sleep_to_next_period`` under a controlled clock; return
    ``(5_tuple, fake_sleep)`` so tests can assert the sleep was/wasn't called.
    """
    with mock.patch.object(am.time, "monotonic", return_value=now), \
            mock.patch.object(am.time, "sleep") as fake_sleep:
        result = a._sleep_to_next_period(next_tick)
    return result, fake_sleep


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = predicate()
        if val is not None and val is not False:
            return val
        time.sleep(0.005)
    return predicate()


class _FakeLocalWorker:
    def __init__(self):
        self.last_snapshot = None

    def request_plan(self, snapshot):
        self.last_snapshot = snapshot
        return 7


class _FakeMapWorker:
    def __init__(self):
        self.requests = []
        self.coalesced_count = 0
        self._counter = 0

    def request_update(self, snapshot):
        self.requests.append(snapshot)
        self._counter += 1
        return self._counter

    def get_latest_result(self):
        return None


class _FakeCbmbaWorker:
    def __init__(self):
        self.last_obstacles = []
        self.coalesced_count = 0
        self._counter = 0

    def has_in_flight_request(self):
        return False

    def request_replan(self, obstacles, start, goal, reason=""):
        self.last_obstacles = list(obstacles)
        self._counter += 1
        return self._counter

    def get_latest_result(self):
        return None


# ── A. normal frame sleeps the rest of the period ──


class TestNormalFrameSleep:
    def test_15ms_work_sleeps_35ms_at_20hz(self):
        a = _sched_auto()
        now = 1000.0
        next_tick = now - 0.015  # 15 ms of work already done this frame
        (nxt, sleep_ms, late, missed, resynced), fake_sleep = _call_sleep_to_next(
            a, next_tick, now,
        )
        assert sleep_ms == pytest.approx(35.0, abs=0.001)
        assert late == 0.0
        assert missed == 0
        assert resynced is False
        # deadline advances by exactly one period, not one period + catch-up
        assert nxt == pytest.approx(now + 0.035, abs=1e-9)
        assert fake_sleep.call_count == 1


# ── B. overrun frame does not catch up ──


class TestOverrunNoCatchUp:
    def test_200ms_work_produces_single_resync_not_burst(self):
        a = _sched_auto()
        now = 1000.0
        next_tick = now - 0.200  # 200 ms of work blew the 50 ms deadline
        (nxt, sleep_ms, late, missed, resynced), fake_sleep = _call_sleep_to_next(
            a, next_tick, now,
        )
        assert sleep_ms == pytest.approx(50.0, abs=0.001)  # one full period
        assert late == pytest.approx(150.0, abs=0.001)  # 3 full periods late
        assert missed == 3
        assert resynced is True
        # re-anchored one period ahead of now — the deadline is NOT carried
        # forward as debt, and the next tick is spaced ~period after now.
        assert nxt == pytest.approx(now + 0.05, abs=1e-9)
        # exactly one full-period sleep is issued to space the next tick — never
        # a tight sleep(0) burst that would "repay" the missed periods.
        fake_sleep.assert_called_once_with(0.05)


# ── C. continuous overrun re-anchors each time ──


class TestContinuousOverrunReanchors:
    def test_debt_never_accumulates(self):
        a = _sched_auto()
        for i in range(3):
            now = 2000.0 + i  # each frame is its own "now"
            next_tick = now - 0.200  # always 200 ms late again
            (nxt, sleep_ms, late, missed, resynced), _ = _call_sleep_to_next(
                a, next_tick, now,
            )
            assert resynced is True
            assert sleep_ms == pytest.approx(50.0, abs=0.001)
            assert missed == 3
            # every overrun re-anchors one period ahead of the *current* now, so
            # the debt from frame i does not pile onto frame i+1.
            assert nxt == pytest.approx(now + 0.05, abs=1e-9)


# ── D. target 20 Hz → no tight-loop burst ──


class TestNoTightLoopBurst:
    def test_steady_20hz_never_sleeps_below_25ms(self):
        a = _sched_auto()
        now = 1000.0
        next_tick = now
        sleeps = []
        for _ in range(50):
            now += 0.015  # 15 ms work each frame
            with mock.patch.object(am.time, "monotonic", return_value=now), \
                    mock.patch.object(am.time, "sleep"):
                nxt, sleep_ms, _, _, _ = a._sleep_to_next_period(next_tick)
            next_tick = nxt
            now += sleep_ms / 1000.0
            sleeps.append(sleep_ms)
        assert sleeps
        # the scheduler sleeps the ~35 ms remainder every frame — never a
        # catch-up burst of sleep=0 (<25 ms) ticks.
        assert all(s >= 25.0 for s in sleeps)
        assert all(s <= 40.0 for s in sleeps)


# ── E. worker request prepare is exclusive of the put ──


class TestWorkerRequestPrepExclusive:
    def _local_auto(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._traj_global_path = [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0)]
        a._traj_global_path_version = 3
        a._traj_path_copy_ms = 0.0
        a._traj_snapshot_build_ms = 0.0
        a._traj_request_prepare_ms = 0.0
        a._traj_request_put_ms = 0.0
        a._local_traj_worker = _FakeLocalWorker()
        return a

    def test_local_plan_copies_path_but_passes_lidar_by_reference(self):
        a = self._local_auto()
        st = types.SimpleNamespace(position_ned_m=[1.0, 2.0, -3.0], yaw_rad=0.5)
        lidar = np.zeros((100, 3))
        a._request_local_plan(st, (10.0, 11.0, -1.0), lidar)
        snap = a._local_traj_worker.last_snapshot
        assert snap is not None
        # LiDAR points are the shared numpy array — no Python list/tuple copy.
        assert snap["lidar_points"] is lidar
        # The global path IS snapshot-copied (a new list, not the live list).
        assert snap["global_path"] == a._traj_global_path
        assert snap["global_path"] is not a._traj_global_path
        # prepare == snapshot build; put is a separate, non-negative stage.
        assert a._traj_path_copy_ms >= 0.0
        assert a._traj_snapshot_build_ms >= 0.0
        assert a._traj_request_prepare_ms == a._traj_snapshot_build_ms
        assert a._traj_request_put_ms >= 0.0

    def test_map_request_prepare_and_put_are_separate(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._mapping_worker = _FakeMapWorker()
        a._last_map_sensor_timestamp = None
        a._map_snapshot_build_ms = 0.0
        a._map_request_prepare_ms = 0.0
        a._map_request_put_ms = 0.0
        lf = types.SimpleNamespace(received_monotonic_seconds=123.0)
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0], yaw_rad=0.0)
        fr = types.SimpleNamespace(filtered_points_sensor=np.zeros((5, 3)))
        assert a._request_map_update(lf, st, fr) is True
        # prepare == snapshot build; put is separate and non-negative.
        assert a._map_request_prepare_ms == a._map_snapshot_build_ms
        assert a._map_snapshot_build_ms >= 0.0
        assert a._map_request_put_ms >= 0.0
        # LiDAR points in the snapshot are the shared numpy array (no copy).
        assert a._mapping_worker.requests[0]["points_sensor"] is fr.filtered_points_sensor

    def test_cbmba_obstacle_snapshot_timed_separate_from_put(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._traj_global_path = []
        a._traj_global_path_version = 0
        a._traj_global_replan_requested = False
        a._initial_replan_requested = False
        a._traj_last_replan_time = -float("inf")
        a._traj_global_replan_hz = 1.5
        a._traj_global_path_min_clearance = float("inf")
        a._traj_applied_replan_id = -1
        a._traj_params = types.SimpleNamespace(hard_clearance_m=1.0)
        a._traj_path_switch_min_improvement = 0.10
        a._occ_grid_params = types.SimpleNamespace(resolution_m=0.5)
        a._map_snapshot = {"occupied_points": [[3.0, 4.0]], "map_version": 2}
        a._global_planner_worker = _FakeCbmbaWorker()
        a._cbmba_obstacle_snapshot_ms = 0.0
        a._cbmba_request_prepare_ms = 0.0
        a._cbmba_request_put_ms = 0.0
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0])
        a._tick_global_replan(st, (42.0, 4.0, -1.0), [], 10.0)
        # prepare == obstacle snapshot build; put is separate and non-negative.
        assert a._cbmba_request_prepare_ms == a._cbmba_obstacle_snapshot_ms
        assert a._cbmba_obstacle_snapshot_ms >= 0.0
        assert a._cbmba_request_put_ms >= 0.0
        # the worker received the combined obstacles without a redundant list().
        assert len(a._global_planner_worker.last_obstacles) == 1


# ── F. exclusive sub-timings stay within the parent request window ──


class TestExclusiveTimingInvariant:
    def test_local_plan_subtimings_sum_within_parent(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._traj_global_path = [(0.0, 0.0, -1.0)]
        a._traj_global_path_version = 1
        a._traj_path_copy_ms = 0.0
        a._traj_snapshot_build_ms = 0.0
        a._traj_request_prepare_ms = 0.0
        a._traj_request_put_ms = 0.0
        a._local_traj_worker = _FakeLocalWorker()
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0], yaw_rad=0.0)
        lidar = np.zeros((50, 3))
        t0 = time.perf_counter()
        a._request_local_plan(st, (9.0, 9.0, -1.0), lidar)
        parent_ms = (time.perf_counter() - t0) * 1000.0
        # prepare + put are exclusive sub-stages: their sum can never exceed the
        # parent request window (which also contains the trailing log line).
        sub_sum = a._traj_request_prepare_ms + a._traj_request_put_ms
        assert 0.0 <= sub_sum <= parent_ms + 1.0


# ── G. same LiDAR frame is not re-built ──


class TestSameFrameNotRebuilt:
    def test_map_update_dedup_skips_snapshot_build(self):
        a = AutomaticMode.__new__(AutomaticMode)
        a._mapping_worker = _FakeMapWorker()
        a._last_map_sensor_timestamp = None
        a._map_snapshot_build_ms = 0.0
        a._map_request_prepare_ms = 0.0
        a._map_request_put_ms = 0.0
        lf = types.SimpleNamespace(received_monotonic_seconds=7.0)
        st = types.SimpleNamespace(position_ned_m=[0.0, 0.0, -1.0], yaw_rad=0.0)
        fr = types.SimpleNamespace(filtered_points_sensor=np.zeros((8, 3)))
        assert a._request_map_update(lf, st, fr) is True
        first_build = a._map_snapshot_build_ms
        # Same frame → dedup returns False BEFORE the snapshot is re-built.
        assert a._request_map_update(lf, st, fr) is False
        assert a._mapping_worker._counter == 1
        assert a._map_snapshot_build_ms == first_build


# ── H. PerceptionWorker timing is observation-only ──


class TestPerceptionTimingObservationOnly:
    def test_timing_does_not_alter_snapshot_payload(self):
        captured = {}

        def perceive(lf):
            fr = types.SimpleNamespace(filtered_points_sensor=np.zeros((10, 3)))
            dd = types.SimpleNamespace(minimum_distance_m=2.0)
            rays = 128
            captured["fr"], captured["dd"], captured["rays"] = fr, dd, rays
            return fr, dd, rays

        class _FakeLidar:
            point_count = 64
            consecutive_stale_count = 0

            def read(self):
                return types.SimpleNamespace(point_count=64)

        w = PerceptionWorker(_FakeLidar(), perceive, poll_hz=100.0)
        w.start()
        try:
            snap = _wait_for(lambda: w.get_latest_snapshot())
            assert snap is not None
            # the timing instrumentation must not rewrite the data payload.
            assert snap.fr is captured["fr"]
            assert snap.dd is captured["dd"]
            assert snap.rays == captured["rays"]
            assert w.points_raw == 64
            assert w.points_filtered == 10
            # observation-only metrics are populated but never gate behaviour.
            assert w.lidar_rpc_ms >= 0.0
            assert w.processing_ms >= 0.0
            assert w.publish_ms >= 0.0
        finally:
            w.shutdown()


# ── I. scheduler sequence: no post-resync tight tick ──


class TestSchedulerSequence:
    def test_post_resync_tick_is_spaced_one_full_period(self):
        """Simulate work → overrun → work.  The overrun re-anchors one period
        ahead of ``now`` and sleeps a full period, so the following loop-start
        interval is never a tight <40 ms catch-up tick."""
        a = _sched_auto()  # 20 Hz → period 0.05

        class _Clock:
            def __init__(self):
                self.t = 1000.0

            def monotonic(self):
                return self.t

            def sleep(self, s):
                self.t += s

        clock = _Clock()
        next_tick = clock.t
        starts = [clock.t]

        # Frame 0: on-time (15 ms work).
        clock.t += 0.015
        with mock.patch.object(am.time, "monotonic", clock.monotonic), \
                mock.patch.object(am.time, "sleep", clock.sleep):
            next_tick, _, _, _, resynced = a._sleep_to_next_period(next_tick)
        assert not resynced
        starts.append(clock.t)

        # Frame 1: overrun (200 ms work) → resync.
        clock.t += 0.200
        with mock.patch.object(am.time, "monotonic", clock.monotonic), \
                mock.patch.object(am.time, "sleep", clock.sleep):
            next_tick, sleep_ms, _, _, resynced = a._sleep_to_next_period(next_tick)
        assert resynced
        assert sleep_ms == pytest.approx(50.0)
        starts.append(clock.t)

        # Frame 2: on-time again (15 ms work).
        clock.t += 0.015
        with mock.patch.object(am.time, "monotonic", clock.monotonic), \
                mock.patch.object(am.time, "sleep", clock.sleep):
            next_tick, _, _, _, resynced = a._sleep_to_next_period(next_tick)
        assert not resynced
        starts.append(clock.t)

        iv01 = (starts[1] - starts[0]) * 1000.0
        iv12 = (starts[2] - starts[1]) * 1000.0
        iv23 = (starts[3] - starts[2]) * 1000.0
        # On-time intervals sit at the period; the frame right after the resync
        # is spaced ≥ period — never a tight <40 ms tick.
        assert iv01 == pytest.approx(50.0, abs=0.5)
        assert iv12 >= 40.0
        assert iv23 == pytest.approx(50.0, abs=0.5)

    def test_drifted_deadline_resyncs_with_full_period_sleep(self):
        """A deadline that has drifted behind ``now`` (from a prior hiccup) is
        missed by a SMALL-work frame; the re-anchor sleeps a full period so the
        frame's loop-start interval is work+period (≥40 ms), never a tight tick."""
        a = _sched_auto()
        now = 1000.0
        next_tick = now - 0.040  # deadline drifted 40 ms behind
        now_after = now + 0.015  # frame did only 15 ms of work
        (nxt, sleep_ms, late, missed, resynced), _ = _call_sleep_to_next(
            a, next_tick, now_after,
        )
        assert resynced is True
        assert late == pytest.approx(5.0, abs=0.001)
        assert missed == 0  # no full period missed, but the boundary was crossed
        assert sleep_ms == pytest.approx(50.0, abs=0.001)
        assert nxt == pytest.approx(now_after + 0.05, abs=1e-9)

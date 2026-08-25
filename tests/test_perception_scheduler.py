"""Phase C8-R — PerceptionWorker rate-capped poll scheduler tests.

The worker was originally a *fixed-delay* loop::

    while running:
        work()                 # getLidarData + filter + sectorise (~270 ms)
        time.sleep(period)     # fixed 100 ms @ poll_hz=10

which stretches the real poll cadence to ``work + period`` (~370 ms) instead of
the target ``period`` (~100 ms).

The correct perception semantics are *rate-capped*, not deadline-repaying:

* **work < period** (fast iteration): sleep ``period - elapsed`` so the cadence
  is capped at ``poll_hz`` (a light iteration must still sleep the remainder,
  or a ~0 ms RPC/processing path would tight-loop).
* **work >= period** (overrun): do **NOT** add a trailing fixed sleep — the
  long work itself already limits the rate, so the next iteration starts
  immediately.  A missed deadline is dropped (resync), never repaid, and never
  tight-loop caught up.

This is deliberately *not* the control loop's ``_sleep_to_next_period``: the
control loop re-anchors a full period on overrun to avoid tight-ticking flight
commands; the perception worker only reads the latest sensor, and long work
already throttles it.

These tests drive ``PerceptionWorker._run`` deterministically with a fake
monotonic clock (injected via ``clock=``) and a monkeypatched ``time.sleep``,
so NO real time passes and NO AirSim RPC is exercised.

Note: ``time.sleep`` is only *called* when the scheduler decides to sleep, so
an overrun iteration produces **no** entry in the recorded ``sleeps`` list —
that absence *is* the assertion that no trailing fixed sleep was added.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import PerceptionWorker


# ── helpers ──


class _FakeClock:
    """Controllable monotonic clock.  ``sleep`` advances it (no real sleep)."""

    def __init__(self, t=1000.0):
        self.t = t
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class _TimedLidar:
    """Fake LiDAR whose ``read()`` advances the clock by the per-poll ``work_s``
    and stops the worker after ``max_polls`` reads.

    ``work_s`` may be a scalar (same work every poll) or a sequence (cycled,
    one duration per poll) to exercise overrun → recovery transitions.
    """

    def __init__(self, clock, work_s=0.0):
        self.clock = clock
        self.polls = 0
        self.max_polls = 10
        self.worker = None
        self.rpc_calls_since_change = 0
        self.last_raw_timestamp_ns = 12345
        self.consecutive_stale_count = 0
        if isinstance(work_s, (list, tuple)):
            self._schedule = list(work_s)
            self._work_s = 0.0
        else:
            self._schedule = None
            self._work_s = float(work_s)

    def _work_this_poll(self):
        if self._schedule is not None:
            return self._schedule[self.polls % len(self._schedule)]
        return self._work_s

    def read(self):
        self.clock.t += self._work_this_poll()
        self.polls += 1
        if self.worker is not None and self.polls >= self.max_polls:
            self.worker._running = False
        return types.SimpleNamespace(point_count=64)


def _fake_perceive(lf):
    return (
        types.SimpleNamespace(filtered_points_sensor=np.zeros((8, 3))),
        types.SimpleNamespace(frame_valid=True),
        {"front": 10.0, "left": 10.0, "right": 10.0},
    )


def _run_scheduler(monkeypatch, work_s, period_s, n_polls):
    """Build a worker on a fake clock, run it to self-termination (the lidar
    stops it after ``n_polls``), and return ``(clock, worker, lidar)``."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "sleep", clock.sleep)

    lidar = _TimedLidar(clock, work_s=work_s)
    worker = PerceptionWorker(
        lidar, _fake_perceive, poll_hz=1.0 / period_s, clock=clock.monotonic,
    )
    lidar.worker = worker
    lidar.max_polls = n_polls

    worker.start()
    # The fake lidar sets _running=False after max_polls; wait for the thread
    # to finish deterministically instead of racing shutdown() clearing the
    # flag before the worker's first while-check.
    worker._thread.join(timeout=2.0)
    worker.shutdown()
    return clock, worker, lidar


# ── cases A–F ──


class TestPerceptionScheduler:
    def test_a_work_under_period_caps_at_target_cadence(self, monkeypatch):
        """work (20 ms) < period (100 ms): sleep ≈ period-work and the poll
        cadence is the target period — NOT work+period."""
        period_s, work_s, n = 0.100, 0.020, 8
        clock, worker, lidar = _run_scheduler(monkeypatch, work_s, period_s, n)

        assert lidar.polls == n
        assert len(clock.sleeps) == n
        for s in clock.sleeps:
            assert s == pytest.approx(period_s - work_s, abs=0.005)
        assert worker.loop_gap_ms == pytest.approx(period_s * 1000.0, abs=5.0)

    def test_b_mild_overrun_adds_no_fixed_sleep(self, monkeypatch):
        """work (150 ms) > period (100 ms): no trailing fixed sleep, so the
        interval is work (150 ms), not work+period (250 ms)."""
        period_s, work_s, n = 0.100, 0.150, 6
        clock, worker, lidar = _run_scheduler(monkeypatch, work_s, period_s, n)

        assert lidar.polls == n
        assert clock.sleeps == []  # every iteration overran → no sleep calls
        assert worker.loop_gap_ms == pytest.approx(work_s * 1000.0, abs=5.0)

    def test_c_heavy_overrun_starts_next_immediately(self, monkeypatch):
        """work (270 ms) >> period (100 ms): the residual fixed-delay bug would
        give work+period (370 ms); correct behaviour gives work (270 ms)."""
        period_s, work_s, n = 0.100, 0.270, 5
        clock, worker, lidar = _run_scheduler(monkeypatch, work_s, period_s, n)

        assert lidar.polls == n
        assert clock.sleeps == []
        assert worker.loop_gap_ms == pytest.approx(work_s * 1000.0, abs=5.0)

    def test_d_overrun_does_not_repay_missed_deadline(self, monkeypatch):
        """After a 270 ms overrun (170 ms deficit), a zero-work poll must still
        sleep a full period — the deficit is dropped, not repaid by a shortened
        or skipped sleep (no catch-up burst)."""
        period_s = 0.100
        schedule = [0.270, 0.0]  # overrun, then a zero-work poll
        n = 3
        clock, worker, lidar = _run_scheduler(monkeypatch, schedule, period_s, n)

        assert lidar.polls == n
        # iter 0 overruns (no sleep); iter 1 is zero-work → one full-period
        # sleep (NOT a deficit-repaying shortened/negative sleep); iter 2
        # overruns again (no sleep).
        assert len(clock.sleeps) == 1
        assert clock.sleeps[0] == pytest.approx(period_s, abs=0.005)

    def test_e_recovers_to_target_cadence_after_overrun(self, monkeypatch):
        """Once work drops back below the period, the cadence recovers to the
        target period (no lingering overrun penalty)."""
        period_s = 0.100
        schedule = [0.270, 0.270, 0.020, 0.020, 0.020, 0.020, 0.020]
        n = len(schedule)
        clock, worker, lidar = _run_scheduler(monkeypatch, schedule, period_s, n)

        assert lidar.polls == n
        # iters 0-1 overrun (no sleep); iters 2-6 sleep period-work.
        assert len(clock.sleeps) == 5
        for s in clock.sleeps:
            assert s == pytest.approx(period_s - 0.020, abs=0.005)
        assert worker.loop_gap_ms == pytest.approx(period_s * 1000.0, abs=5.0)

    def test_f_shutdown_exits_worker(self, monkeypatch):
        """shutdown() stops the daemon thread and clears the handle."""
        clock = _FakeClock()
        monkeypatch.setattr(time, "sleep", clock.sleep)

        lidar = _TimedLidar(clock, work_s=0.0)
        worker = PerceptionWorker(
            lidar, _fake_perceive, poll_hz=10.0, clock=clock.monotonic,
        )
        worker.start()
        assert worker._thread is not None and worker._thread.is_alive()
        worker.shutdown()
        assert worker._thread is None

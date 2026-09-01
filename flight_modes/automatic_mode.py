"""
Automatic flight mode — LiDAR-based autonomous obstacle avoidance.

All preflight checks execute BEFORE enableApiControl:
    1. Drone1 present
    2. LidarSensor1 consecutive valid frames
    3. No active collision
    4. FOV compatible (NOT hardcoded — validates sector coverage)
    5. Perception config valid
    6. minimal_flight.yaml loaded and parameters strictly validated
    7. target_z, speeds, time, geofence validated

CLI overrides take priority over YAML, but all values are strictly checked.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger("automatic_mode")


class GlobalPlannerWorker:
    """Minimal background worker for the CBMBA global planner (Phase C1-R sec 7-8).

    Decouples the heavy A* global search from the 20-30 Hz control loop.  The
    worker owns a **private** CBMBA instance (never shared with the main
    thread), receives immutable obstacle snapshots, and publishes the latest
    finished result through a lock.  The control loop only ever *reads* the
    published result — it never blocks waiting for a search.
    """

    def __init__(self, planner, poll_interval_s: float = 0.005):
        self._planner = planner
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._pending = None
        self._latest = None
        self._request_counter = 0
        self._busy = False
        self.search_count = 0
        self._thread = None
        self._running = False

    # ── main-thread API ──
    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="cbmba-global-planner", daemon=True,
        )
        self._thread.start()

    def request_replan(self, obstacles, start, goal, reason: str = "") -> int:
        """Queue a replan; the newest request overwrites any pending one."""
        with self._lock:
            self._request_counter += 1
            rid = self._request_counter
            self._pending = {
                "obstacles": list(obstacles),   # snapshot copy
                "start": list(start),
                "goal": list(goal),
                "request_id": rid,
                "reason": reason,
            }
            return rid

    def get_latest_result(self) -> Optional[dict]:
        with self._lock:
            return self._latest

    def has_in_flight_request(self) -> bool:
        """True while a request is queued (pending) or being searched (busy)."""
        with self._lock:
            return self._pending is not None or self._busy

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── worker thread ──
    def _run(self) -> None:
        while self._running:
            with self._lock:
                req = self._pending
                self._pending = None
                if req is not None:
                    self._busy = True
            if req is None:
                time.sleep(self._poll_interval_s)
                continue
            try:
                res = self._planner.plan_with_result(
                    req["obstacles"], req["start"], req["goal"],
                )
                with self._lock:
                    self.search_count += 1
                    self._latest = {
                        "request_id": req["request_id"],
                        "reason": req["reason"],
                        "success": res.success,
                        "path_world": res.path_world,
                        "nodes_expanded": res.nodes_expanded,
                        "planning_time_ms": res.planning_time_ms,
                        "grid_size": res.grid_size,
                        "max_lateral_deviation_m": res.max_lateral_deviation_m,
                    }
                logger.info(
                    "cbmba_search_invocation  request_id=%d  count=%d  "
                    "reason=%s  time_ms=%.2f  success=%s",
                    req["request_id"], self.search_count, req["reason"],
                    res.planning_time_ms, "true" if res.success else "false",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("global_planner_worker_error: %s", exc)
            finally:
                with self._lock:
                    self._busy = False

# ── bypass episode ──


@dataclass
class PerceptionSnapshot:
    """Immutable perception output published by ``PerceptionWorker``.

    The worker performs the slow ``getLidarData`` RPC plus the filter /
    sectorisation pipeline on its own thread with an independent AirSim
    client, then publishes one of these.  The control loop only ever reads
    the snapshot (never blocks on LiDAR).
    """

    lf: Any = None                    # LidarFrame (valid or not)
    fr: Any = None                    # FilteredPointCloud (None if filter failed)
    dd: Any = None                    # directional distances (None if sector failed)
    rays: Any = None                  # legacy ray-distance dict (None on failure)
    received_mono: float = 0.0        # time.monotonic() of the LiDAR read
    stale_count: int = 0              # worker LiDAR consecutive-stale counter


def _fov_fraction(start_deg: float, end_deg: float) -> float:
    """Horizontal FOV sweep as a fraction of a full 360° circle.

    ``-180..180`` (and other full-circle wraps) collapse to ``1.0``; a partial
    sweep yields ``span / 360``.  Diagnostic only — scales the *estimated*
    attempted-ray count, never the true AirSim ray count.
    """
    span = (float(end_deg) - float(start_deg)) % 360.0
    if span == 0.0:
        span = 360.0
    return min(1.0, span / 360.0)


class PerceptionWorker:
    """Background LiDAR → perception worker (Phase C2 sec 10-14).

    ``getLidarData`` (~100-200 ms on the msgpack RPC) and the filter +
    sectorisation pipeline used to run synchronously inside the control loop,
    which is what dragged it down to ~1.3 Hz.  This worker:

    1. owns an **independent** read-only AirSim client (``lidar_reader``),
    2. polls LiDAR at ``poll_hz``,
    3. runs the filter / sectorisation pipeline via ``perceive_fn``,
    4. publishes an immutable ``PerceptionSnapshot`` under a lock.

    The control loop reads ``get_latest_snapshot()`` without blocking and, if
    the newest snapshot is older than ``stale_stop_s``, hovers in place.
    """

    def __init__(
        self,
        lidar_reader: Any,
        perceive_fn: Any,
        poll_hz: float = 10.0,
        clock=time.monotonic,
        points_per_second: float = 150000.0,
        horizontal_fov_deg: Tuple[float, float] = (-180.0, 180.0),
    ) -> None:
        self._lidar = lidar_reader
        self._perceive = perceive_fn
        self._poll_interval_s = 1.0 / max(0.1, float(poll_hz))
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: Optional[PerceptionSnapshot] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.poll_count = 0
        self.error_count = 0
        # Phase C5-R: observation-only timing (distinguish AirSim RPC latency
        # from Python thread starvation).  Never affects scheduling behaviour.
        self.target_poll_hz = float(poll_hz)
        self._last_loop_mono: Optional[float] = None
        self.loop_gap_ms = 0.0
        self.lidar_rpc_ms = 0.0
        self.processing_ms = 0.0
        self.publish_ms = 0.0
        self.points_raw = 0
        self.points_filtered = 0
        # Phase C9-R: observation-only payload profile (raw → filtered point
        # counts per stage).  Derived from FilterResult counters; never affects
        # scheduling or the published snapshot.
        self.payload_raw_xyz = 0
        self.payload_after_finite = 0
        self.payload_after_range = 0
        self.payload_after_self = 0
        self.payload_after_voxel = 0
        self.payload_final_filtered = 0
        self._timing_last_log_mono = 0.0
        # Phase C6-R: RPC/timestamp discrimination (distinguish "AirSim scan
        # stuck" from "thread starved" from "RPC blocking" from "Python
        # processing").  Observation-only; never affects scheduling behaviour.
        self.last_lidar_timestamp: Optional[int] = None
        self.timestamp_changed = False
        self.new_timestamp_count = 0
        self.same_timestamp_count = 0
        self._rpc_window_start_mono: Optional[float] = None
        self._rpc_window_calls = 0
        self._rpc_window_new_ts = 0
        self._rpc_window_same_ts = 0
        self._rpc_window_sum_ms = 0.0
        self._rpc_window_max_ms = 0.0
        self._rpc_window_gap_max_ms = 0.0
        # Phase C10: scan interval + hit-ratio observation (observation-only).
        # scan_dt_ms is the gap between consecutive *different* LiDAR timestamps
        # (timestamp is ns).  estimated_attempted_rays = PPS * dt (scaled by the
        # horizontal-FOV fraction); estimated_hit_ratio = raw_xyz / attempted.
        # Never affects scheduling or the published snapshot.
        self.points_per_second = float(points_per_second)
        self.horizontal_fov_deg = (
            float(horizontal_fov_deg[0]), float(horizontal_fov_deg[1]),
        )
        self.fov_fraction = _fov_fraction(*self.horizontal_fov_deg)
        self._prev_new_timestamp_ns: Optional[int] = None
        self.scan_dt_ms = 0.0
        self.estimated_attempted_rays = 0.0
        self.estimated_hit_ratio = 0.0
        self.hit_ratio_anomaly_count = 0
        self.position_xy = (0.0, 0.0)

    # ── main-thread API ──

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="lidar-perception-worker", daemon=True,
        )
        self._thread.start()

    def get_latest_snapshot(self) -> Optional[PerceptionSnapshot]:
        with self._lock:
            return self._latest

    def snapshot_age_s(self, now: Optional[float] = None) -> float:
        """Age of the latest snapshot (or +inf if none yet)."""
        snap = self.get_latest_snapshot()
        if snap is None:
            return float("inf")
        t = now if now is not None else self._clock()
        return max(0.0, t - snap.received_mono)

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── worker thread ──

    def _run(self) -> None:
        # Rate-capped poll scheduling (no fixed-delay, no catch-up).  poll_hz
        # caps the cadence when an iteration is fast: sleep only until the poll
        # period elapses.  When a single iteration (LiDAR RPC + processing)
        # already exceeds the period, do NOT add a trailing fixed sleep — the
        # long work itself already limits the rate, so the next iteration starts
        # immediately.  Missed deadlines are dropped (resync to now), never
        # repaid, and never tight-loop caught up.
        while self._running:
            snap = None
            _loop_mono = self._clock()
            if self._last_loop_mono is not None:
                self.loop_gap_ms = (_loop_mono - self._last_loop_mono) * 1000.0
            self._last_loop_mono = _loop_mono
            # Phase C6-R: 1 s RPC-summary window (observation-only).
            if self._rpc_window_start_mono is None:
                self._rpc_window_start_mono = _loop_mono
            self._rpc_window_gap_max_ms = max(self._rpc_window_gap_max_ms, self.loop_gap_ms)
            try:
                _rpc_t0 = time.perf_counter()
                lf = self._lidar.read()
                self.lidar_rpc_ms = (time.perf_counter() - _rpc_t0) * 1000.0
                received = self._clock()
                # Phase C6-R: classify whether this RPC returned a NEW LiDAR
                # timestamp (rpc_calls_since_change == 0 on the reader) or a
                # repeat of the previous scan.  Discriminates "AirSim scan
                # stuck" (many repeats) from "RPC blocking" (rpc_ms high).
                _since_change = getattr(self._lidar, "rpc_calls_since_change", -1)
                self.timestamp_changed = (_since_change == 0)
                self.last_lidar_timestamp = getattr(
                    self._lidar, "last_raw_timestamp_ns", None,
                )
                if self.timestamp_changed:
                    self.new_timestamp_count += 1
                    # Phase C10: scan interval between consecutive NEW scans.
                    _ts = self.last_lidar_timestamp
                    if (_ts is not None and self._prev_new_timestamp_ns is not None
                            and _ts > self._prev_new_timestamp_ns):
                        self.scan_dt_ms = (_ts - self._prev_new_timestamp_ns) / 1e6
                    if _ts is not None:
                        self._prev_new_timestamp_ns = _ts
                else:
                    self.same_timestamp_count += 1
                # Phase C10: vehicle/sensor position for hit-ratio context.
                _sp = getattr(lf, "sensor_pose", None)
                if _sp is not None:
                    _pos = _sp.get("position", {}) or {}
                    self.position_xy = (
                        float(_pos.get("x", 0.0)), float(_pos.get("y", 0.0)),
                    )
                _proc_t0 = time.perf_counter()
                fr, dd, rays = self._perceive(lf)
                self.processing_ms = (time.perf_counter() - _proc_t0) * 1000.0
                _stale = getattr(self._lidar, "consecutive_stale_count", 0)
                if not isinstance(_stale, (int, float)):
                    _stale = 0
                snap = PerceptionSnapshot(
                    lf=lf, fr=fr, dd=dd, rays=rays, received_mono=received,
                    stale_count=int(_stale),
                )
                self.points_raw = getattr(lf, "point_count", 0)
                _fp = getattr(fr, "filtered_points_sensor", None)
                self.points_filtered = (
                    len(_fp) if _fp is not None and hasattr(_fp, "__len__") else 0
                )
                # Phase C9-R: derive per-stage point counts from FilterResult
                # counters (observation-only; no behaviour change).
                _ic = getattr(fr, "input_point_count", 0)
                _rnf = getattr(fr, "removed_nonfinite_count", 0)
                _rmin = getattr(fr, "removed_min_range_count", 0)
                _rmax = getattr(fr, "removed_max_range_count", 0)
                _rself = getattr(fr, "removed_self_body_count", 0)
                _rvox = getattr(fr, "voxel_reduction_count", 0)
                self.payload_raw_xyz = _ic
                self.payload_after_finite = _ic - _rnf
                self.payload_after_range = _ic - _rnf - _rmin - _rmax
                self.payload_after_self = _ic - _rnf - _rmin - _rmax - _rself
                self.payload_after_voxel = _ic - _rnf - _rmin - _rmax - _rself - _rvox
                self.payload_final_filtered = self.payload_after_voxel
                # Phase C10: estimated attempted rays + hit ratio (obs-only).
                if self.scan_dt_ms > 0.0:
                    self.estimated_attempted_rays = (
                        self.points_per_second * (self.scan_dt_ms / 1000.0)
                        * self.fov_fraction
                    )
                    if self.estimated_attempted_rays > 0.0 and self.payload_raw_xyz > 0:
                        _ratio = self.payload_raw_xyz / self.estimated_attempted_rays
                        self.estimated_hit_ratio = _ratio
                        if _ratio > 1.0:
                            self.hit_ratio_anomaly_count += 1
                self.poll_count += 1
                # Phase C6-R: accumulate the 1 s RPC summary window.
                self._rpc_window_calls += 1
                self._rpc_window_sum_ms += self.lidar_rpc_ms
                self._rpc_window_max_ms = max(self._rpc_window_max_ms, self.lidar_rpc_ms)
                if self.timestamp_changed:
                    self._rpc_window_new_ts += 1
                else:
                    self._rpc_window_same_ts += 1
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                logger.warning("perception_worker_error: %s", exc)
            if snap is not None:
                _pub_t0 = time.perf_counter()
                with self._lock:
                    self._latest = snap
                self.publish_ms = (time.perf_counter() - _pub_t0) * 1000.0
            # Observation-only timing (sec 16): emit ~1 Hz so a long AirSim RPC
            # (lidar_rpc_ms high) is distinguishable from thread starvation
            # (loop_gap_ms high with lidar_rpc_ms low).
            if _loop_mono - self._timing_last_log_mono >= 1.0:
                self._timing_last_log_mono = _loop_mono
                logger.info(
                    "perception_worker_timing  target_poll_hz=%.1f  "
                    "poll_count=%d  loop_gap_ms=%.2f  lidar_rpc_ms=%.2f  "
                    "processing_ms=%.2f  publish_ms=%.2f  points_raw=%d  "
                    "points_filtered=%d  lidar_timestamp=%s  timestamp_changed=%s",
                    self.target_poll_hz, self.poll_count,
                    self.loop_gap_ms, self.lidar_rpc_ms,
                    self.processing_ms, self.publish_ms,
                    self.points_raw, self.points_filtered,
                    self.last_lidar_timestamp,
                    "true" if self.timestamp_changed else "false",
                )
                # Phase C9-R: raw → filtered payload profile (observation-only).
                logger.info(
                    "lidar_payload_profile  raw_xyz=%d  after_finite=%d  "
                    "after_range=%d  after_self_filter=%d  after_voxel=%d  "
                    "final_filtered=%d  sector_input=%d  "
                    "raw_rpc_ms=%.2f  processing_total_ms=%.2f",
                    self.payload_raw_xyz, self.payload_after_finite,
                    self.payload_after_range, self.payload_after_self,
                    self.payload_after_voxel, self.payload_final_filtered,
                    self.payload_final_filtered,
                    self.lidar_rpc_ms, self.processing_ms,
                )
                # Phase C10: scan interval + hit ratio (observation-only).
                logger.info(
                    "lidar_scan_profile  timestamp=%s  scan_dt_ms=%.2f  "
                    "raw_xyz=%d  estimated_attempted_rays=%.1f  "
                    "estimated_hit_ratio=%.4f  hit_ratio_anomalies=%d  "
                    "position_xy=%.2f,%.2f  fov_fraction=%.3f  "
                    "points_per_second=%.0f",
                    self.last_lidar_timestamp, self.scan_dt_ms,
                    self.payload_raw_xyz, self.estimated_attempted_rays,
                    self.estimated_hit_ratio, self.hit_ratio_anomaly_count,
                    self.position_xy[0], self.position_xy[1],
                    self.fov_fraction, self.points_per_second,
                )
                # Phase C6-R: per-window RPC summary — the discriminator between
                # (A) AirSim not producing scans (same_timestamp high, rpc low),
                # (B) worker thread starved (loop_gap_max high, rpc low),
                # (C) getLidarData blocking (rpc_mean/max high),
                # (D) Python processing too slow (processing_ms high).
                _win_s = _loop_mono - self._rpc_window_start_mono
                _rpc_mean = (
                    self._rpc_window_sum_ms / self._rpc_window_calls
                    if self._rpc_window_calls else 0.0
                )
                logger.info(
                    "perception_lidar_rpc_summary  window_s=%.2f  rpc_calls=%d  "
                    "new_timestamp_count=%d  same_timestamp_count=%d  "
                    "rpc_mean_ms=%.2f  rpc_max_ms=%.2f  loop_gap_max_ms=%.2f  "
                    "last_timestamp=%s",
                    _win_s, self._rpc_window_calls,
                    self._rpc_window_new_ts, self._rpc_window_same_ts,
                    _rpc_mean, self._rpc_window_max_ms, self._rpc_window_gap_max_ms,
                    self.last_lidar_timestamp,
                )
                # reset the window
                self._rpc_window_start_mono = _loop_mono
                self._rpc_window_calls = 0
                self._rpc_window_new_ts = 0
                self._rpc_window_same_ts = 0
                self._rpc_window_sum_ms = 0.0
                self._rpc_window_max_ms = 0.0
                self._rpc_window_gap_max_ms = 0.0
            # Rate-cap only.  If this iteration finished before the poll period
            # elapsed, sleep the remainder (caps the cadence at poll_hz).  If it
            # overran (LiDAR RPC + processing > period), do NOT add a trailing
            # fixed sleep — the work itself already limits the rate; drop the
            # missed deadline and start the next iteration immediately.
            _elapsed = self._clock() - _loop_mono
            _sleep_s = self._poll_interval_s - _elapsed
            if _sleep_s > 0.0:
                time.sleep(_sleep_s)


# ── bypass episode ──


@dataclass
class BypassEpisode:
    """Side-commitment state for preventing left/right oscillation (Failure A).

    Once entered, the drone commits to moving *only* toward the chosen
    lateral direction until the release conditions are met or a safety
    veto fires.  This prevents the CBMBA→APF guidance feedback loop
    from flipping vy sign frame-by-frame in a narrowing corridor.
    """

    active: bool = False
    side: Optional[int] = None        # +1 = right, -1 = left
    start_time: float = 0.0           # time.monotonic() when entered
    reason: str = ""                  # diagnostic reason for entry
    min_duration_s: float = 2.5       # minimum hold time before release allowed
    entry_clearance_side_m: float = 0.0   # LiDAR clearance on chosen side at entry

    # Stable pre-bypass reference geometry.  Frozen ONCE when the bypass episode
    # is created (NOT at rejoin_enter), so REJOIN measures cross-track error
    # against the path that existed BEFORE the drone deviated around the
    # obstacle — independent of the drone's current instantaneous position.
    # ``()`` means "no valid reference" → REJOIN must NOT fall back to a live
    # path (path_error stays inf).
    reference_path_xy: Tuple[Tuple[float, float], ...] = ()
    reference_source: str = ""
    reference_generation_id: Optional[int] = None
    reference_first_xy: Optional[Tuple[float, float]] = None
    reference_frozen_position_xy: Optional[Tuple[float, float]] = None
    # Peak cross-track error to the FROZEN reference during this episode.
    # Used at release time to distinguish a real lateral excursion (→ REJOIN)
    # from a bypass that never deviated (→ NORMAL, no re-alignment needed).
    max_path_error_m: float = 0.0

    # A trajectory recovery from a dead end needs a stronger handoff than a
    # normal guided-APF bypass. Keep the selected wall side committed until
    # that wall's end is actually visible; front clearance alone is not enough
    # inside a large U-shaped obstacle.
    trajectory_dead_end: bool = False
    wall_end_clear_since: Optional[float] = None
    max_displacement_m: float = 0.0


@dataclass
class RejoinEpisode:
    """Post-bypass re-alignment state (Failure A follow-up).

    The bypass transition graph is NORMAL → BYPASS → REJOIN → NORMAL.
    When a bypass releases with reason ``obstacle_passed``, the drone drops
    the side commitment but has NOT yet re-aimed at the goal.  REJOIN holds
    that intermediate state: ``bypass_enforce`` and ``side_commit_hold`` are
    forbidden, and CBMBA guidance + Guided APF are allowed to pull the drone
    back onto the reference path.  It exits to NORMAL once the cross-track
    ``path_error`` drops below ``exit_path_error_m`` for at least
    ``min_duration_s`` — NOT merely because the yaw happens to point at the
    goal (which is nearly always true and caused instant REJOIN exits).
    """

    active: bool = False
    start_time: float = 0.0
    reason: str = ""
    exit_path_error_m: float = 1.5
    min_duration_s: float = 0.6            # minimum dwell before path-error exit
    start_path_error: float = float("inf")  # cross-track error to FROZEN reference at entry

    # Stable rejoin reference geometry.  INHERITED from the BypassEpisode at
    # ``rejoin_enter`` (frozen at bypass-episode creation), so cross-track error
    # is measured against the path that existed BEFORE the drone deviated around
    # the obstacle — NOT a path re-seeded from the current UAV position on every
    # CBMBA replan.  ``()`` means "no reference" → path_error stays inf → REJOIN
    # never falsely exits.
    reference_path_xy: Tuple[Tuple[float, float], ...] = ()
    reference_source: str = ""              # provenance (e.g. "cbmba_path_world")
    reference_generation_id: Optional[int] = None
    reference_first_xy: Optional[Tuple[float, float]] = None  # first point of reference
    reference_frozen_position_xy: Optional[Tuple[float, float]] = None  # where it was frozen


# ── forward-progress watchdog ──


@dataclass
class ForwardProgressWatchdog:
    """Detects when the drone is not making forward progress toward the goal.

    If the cumulative forward distance does not increase by at least
    ``min_progress_m`` within ``window_s``, the watchdog fires.
    Used to detect when CBMBA guidance is leading the drone into a
    dead end or an infeasible lateral excursion (Failures A & B).
    """

    window_s: float = 8.0             # evaluation window
    min_progress_m: float = 1.0       # minimum forward progress in window
    check_interval_s: float = 2.0     # how often to evaluate
    _start_time: float = 0.0
    _start_position: Tuple[float, float] = (0.0, 0.0)
    _last_check_time: float = 0.0
    _fired: bool = False
    _fired_count: int = 0

    def reset(self, now: float, position_xy: Tuple[float, float]) -> None:
        self._start_time = now
        self._start_position = position_xy
        self._last_check_time = now
        self._fired = False

    def update(self, now: float, position_xy: Tuple[float, float]) -> bool:
        """Return True if the watchdog has fired (insufficient progress)."""
        elapsed = now - self._start_time
        if elapsed < self.window_s:
            return False
        if now - self._last_check_time < self.check_interval_s:
            return self._fired  # return cached result
        self._last_check_time = now
        progress = math.hypot(
            position_xy[0] - self._start_position[0],
            position_xy[1] - self._start_position[1],
        )
        if progress < self.min_progress_m:
            self._fired = True
            self._fired_count += 1
            # Reset baseline so it can fire again after corrective action
            self._start_time = now
            self._start_position = position_xy
            return True
        # Progress is adequate — reset baseline
        self._start_time = now
        self._start_position = position_xy
        self._fired = False
        return False

# ── reactive decision ──


@dataclass(frozen=True)
class ReactiveDecision:
    vx_body_mps: float = 0.0
    vy_body_mps: float = 0.0
    should_terminate: bool = False
    termination_reason: str = ""


def choose_reactive_command(
    front_m: float, left_m: float, right_m: float,
    minimum_distance_m: float, config: Dict[str, float],
) -> ReactiveDecision:
    emerg = config["emergency_distance_m"]
    ft = config["front_threshold_m"]
    fwd = config["forward_speed_mps"]
    side = config["side_speed_mps"]
    if minimum_distance_m < emerg:
        return ReactiveDecision(should_terminate=True, termination_reason="emergency_distance")
    if front_m > ft:
        return ReactiveDecision(vx_body_mps=fwd)
    if left_m > right_m:
        return ReactiveDecision(vy_body_mps=-side)
    return ReactiveDecision(vy_body_mps=side)


# ── flight result ──


@dataclass
class AutomaticFlightResult:
    success: bool = False
    termination_reason: str = "unknown"
    frames_completed: int = 0
    flight_duration_s: float = 0.0
    api_control_acquired: bool = False
    armed: bool = False
    takeoff_completed: bool = False
    airborne: bool = False
    landing_confirmed: bool = False
    disarmed: bool = False
    api_control_released: bool = False
    startup_floor_contact_baseline: bool = False


# ── flight config loading ──


def _load_flight_config(path: str) -> Dict[str, Any]:
    """Load and validate minimal_flight.yaml. Returns validated dict."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    fc = raw.get("minimal_flight", {})
    if not fc:
        raise ValueError("minimal_flight section missing from flight config")

    params = {}
    validations = [
        ("target_z_ned", -10.0, -0.1),
        ("max_vertical_speed_mps", 0.1, 2.0),
        ("max_flight_duration_s", 0.5, 600.0),
        ("command_duration_s", 0.05, 1.0),
        ("forward_speed_mps", 0.05, 2.0),
        ("side_speed_mps", 0.05, 1.0),
        ("front_threshold_m", 0.5, 20.0),
        ("emergency_distance_m", 0.3, 5.0),
        ("geofence_radius_m", 0.5, 50.0),
        ("preflight_lidar_frames", 1, 20),
        ("takeoff_timeout_s", 5.0, 60.0),
    ]
    for key, lo, hi in validations:
        val = fc.get(key)
        if val is None:
            raise ValueError(f"minimal_flight.{key} is missing")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"minimal_flight.{key} must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"minimal_flight.{key} must be finite")
        if not (lo <= val <= hi):
            raise ValueError(f"minimal_flight.{key}={val} must be in [{lo}, {hi}]")
        params[key] = float(val)
    return params


def _merge_params(yaml_params: Dict[str, Any], cli_overrides: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """CLI overrides win over YAML. Both sources already validated."""
    merged = dict(yaml_params)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if k in merged:
                merged[k] = float(v)
    return merged


@dataclass
class AutomaticModeParams:
    """Resolved flight parameters (after YAML + CLI merge)."""
    target_z_ned: float = -1.0
    max_vertical_speed_mps: float = 0.5
    max_flight_duration_s: float = 10.0
    command_duration_s: float = 0.2
    forward_speed_mps: float = 0.2
    side_speed_mps: float = 0.15
    front_threshold_m: float = 2.5
    emergency_distance_m: float = 0.8
    geofence_radius_m: float = 2.0
    preflight_lidar_frames: int = 3
    takeoff_timeout_s: float = 20.0

    @classmethod
    def from_yaml(cls, path: str, cli_overrides: Optional[Dict[str, float]] = None) -> AutomaticModeParams:
        yp = _load_flight_config(path)
        merged = _merge_params(yp, cli_overrides)
        return cls(
            target_z_ned=merged["target_z_ned"],
            max_vertical_speed_mps=merged["max_vertical_speed_mps"],
            max_flight_duration_s=merged["max_flight_duration_s"],
            command_duration_s=merged["command_duration_s"],
            forward_speed_mps=merged["forward_speed_mps"],
            side_speed_mps=merged["side_speed_mps"],
            front_threshold_m=merged["front_threshold_m"],
            emergency_distance_m=merged["emergency_distance_m"],
            geofence_radius_m=merged["geofence_radius_m"],
            preflight_lidar_frames=int(merged["preflight_lidar_frames"]),
            takeoff_timeout_s=merged["takeoff_timeout_s"],
        )


# ── CBMBA obstacle conversion (pure function, no AirSim calls) ──


def _sector_distances_to_obstacles(
    rays: dict,
    drone_position_ned: Tuple[float, float, float],
    yaw_rad: float,
    max_range: float = 15.0,
    obstacle_radius: float = 0.8,
) -> list:
    """Convert LiDAR sector distances into CBMBA-compatible obstacle dicts.

    This is a compatibility layer — it does NOT modify the A* core.
    Each LiDAR ray that hits an obstacle (< max_range) produces one
    obstacle dict at the estimated world position.

    Sector directions (body-frame angles relative to forward/X):
        front  = 0°,  frontLeft  = -22.5°,  frontRight = +22.5°
        left   = -90°, right      = +90°
        backLeft = -157.5°,        backRight  = +157.5°
        back   = 180°

    Args:
        rays: Dict of sector_name → distance_m (from dd.to_legacy_ray_distances()).
        drone_position_ned: Drone NED position (x, y, z).
        yaw_rad: Drone yaw in radians (NED: 0=North, π/2=East).
        max_range: Max distance to consider (farther = no obstacle detected).
        obstacle_radius: Assigned obstacle radius (half-size).

    Returns:
        List of obstacle dicts suitable for CbmbaAStarPlanner.plan().
    """
    # Body-frame sector angles (radians): forward=0, right=π/2
    SECTOR_ANGLES = {
        "front": 0.0,
        "frontLeft": -math.pi / 8,        # -22.5°
        "frontRight": math.pi / 8,         # +22.5°
        "left": -math.pi / 2,              # -90°
        "right": math.pi / 2,              # +90°
        "backLeft": -math.pi * 7 / 8,      # -157.5°
        "backRight": math.pi * 7 / 8,      # +157.5°
        "back": math.pi,                    # 180°
    }

    obstacles = []
    px, py, pz = drone_position_ned

    for sector_name, distance in rays.items():
        if sector_name not in SECTOR_ANGLES:
            continue
        if not isinstance(distance, (int, float)):
            continue
        if not math.isfinite(distance):
            continue
        if distance >= max_range or distance <= 0:
            continue

        # Body-frame direction
        body_angle = SECTOR_ANGLES[sector_name]
        # Rotate by yaw to get world-frame (NED) direction
        world_angle = yaw_rad + body_angle
        dir_x = math.cos(world_angle)
        dir_y = math.sin(world_angle)

        # Estimated obstacle world position
        obs_x = px + dir_x * distance
        obs_y = py + dir_y * distance
        obs_z = pz  # LiDAR sectors are horizontal — assume at same altitude

        # Body-frame XY hit point (diagnostic — CBMBA ignores unknown keys)
        body_hit_x = math.cos(body_angle) * distance
        body_hit_y = math.sin(body_angle) * distance

        obstacles.append({
            "position": [obs_x, obs_y, obs_z],
            "footprint_half_extents": [0.0, 0.0, 0.0],
            "type": "lidar",
            "velocity": [0.0, 0.0, 0.0],
            "dynamic": False,
            "confidence": 0.7,
            # ── diagnostic metadata (not consumed by CBMBA) ──
            "_diag_sector": sector_name,
            "_diag_distance": distance,
            "_diag_body_xy": (body_hit_x, body_hit_y),
        })

    return obstacles


# ── trajectory-mode config loading ──


def _load_trajectory_config(path: str) -> Dict[str, Any]:
    """Load optional trajectory_planner.yaml (returns {} if missing/invalid)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _build_trajectory_params(cfg: Dict[str, Any]):
    from planners.local_trajectory_planner import TrajectoryPlannerParams
    s = (cfg or {}).get("trajectory_planner", {}) or {}
    w = s.get("weights") or {}

    def _f(key, default):
        v = s.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _i(key, default):
        v = s.get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(default)

    def _b(key, default):
        return bool(s.get(key, default))

    weights = {
        "goal_progress": float(w.get("goal_progress", 2.0)),
        "goal_heading_alignment": float(w.get("goal_heading_alignment", 6.0)),
        "global_path_alignment": float(w.get("global_path_alignment", 3.0)),
        "clearance": float(w.get("clearance", 3.0)),
        "smoothness": float(w.get("smoothness", 1.0)),
        "trajectory_consistency": float(w.get("trajectory_consistency", 1.5)),
        "curvature_penalty": float(w.get("curvature_penalty", 0.5)),
        "reverse_penalty": float(w.get("reverse_penalty", 2.0)),
        "unknown_penalty": float(w.get("unknown_penalty", 0.5)),
    }

    rejoin = s.get("rejoin") or {}
    reverse_hold = s.get("reverse_hold") or {}
    straight_goal_rejoin = s.get("straight_goal_rejoin") or {}
    narrow_passage = s.get("narrow_passage") or {}
    apf_safety = s.get("apf_safety") or {}
    family_switch = s.get("family_switch") or {}
    adaptive_horizon = s.get("adaptive_horizon") or {}

    def _nf(d, key, default):
        v = d.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    return TrajectoryPlannerParams(
        enabled=_b("enabled", True),
        num_candidates=_i("num_candidates", 9),
        horizon_m=_f("horizon_m", 4.0),
        sample_spacing_m=_f("sample_spacing_m", 0.25),
        planning_hz=_f("planning_hz", 8.0),
        max_compute_ms=_f("max_compute_ms", 20.0),
        hard_clearance_m=_f("hard_clearance_m", 1.0),
        preferred_clearance_m=_f("preferred_clearance_m", 1.8),
        enable_distance_field_refinement=_b("enable_distance_field_refinement", False),
        refinement_gain=_f("refinement_gain", 0.2),
        forward_speed_mps=_f("forward_speed_mps", 0.25),
        lateral_speed_mps=_f("lateral_speed_mps", 0.20),
        command_lookahead_m=_f("command_lookahead_m", 1.0),
        tracker_lookahead_m=_f("tracker_lookahead_m", 1.0),
        alignment_scale_m=_f("alignment_scale_m", 3.0),
        consistency_scale_m=_f("consistency_scale_m", 0.5),
        side_consistency_bonus=_f("side_consistency_bonus", 0.3),
        rejoin_trigger_lateral_error_m=_nf(rejoin, "trigger_lateral_error_m", 0.75),
        rejoin_completion_lateral_error_m=_nf(rejoin, "completion_lateral_error_m", 0.30),
        rejoin_alignment_bonus_weight=_nf(rejoin, "alignment_bonus_weight", 1.0),
        rejoin_clear_front_required_m=_nf(rejoin, "clear_front_required_m", 3.0),
        family_switch_min_score_improvement=_nf(family_switch, "min_score_improvement", 0.15),
        family_switch_min_hold_time_s=_nf(family_switch, "min_hold_time_s", 0.5),
        direct_opposite_switch_penalty=_f("direct_opposite_switch_penalty", 1.0),
        reverse_hold_enabled=bool(reverse_hold.get("enabled", True)),
        reverse_release_front_clearance_m=_nf(
            reverse_hold, "release_front_clearance_m", 4.0,
        ),
        reverse_hold_max_duration_s=_nf(
            reverse_hold, "max_duration_s", 2.0,
        ),
        reverse_hold_max_distance_m=_nf(
            reverse_hold, "max_distance_m", 1.0,
        ),
        straight_goal_rejoin_enabled=bool(
            straight_goal_rejoin.get("enabled", True)
        ),
        straight_goal_alignment_trigger=_nf(
            straight_goal_rejoin, "alignment_trigger", 0.82,
        ),
        straight_goal_alignment_min_gain=_nf(
            straight_goal_rejoin, "min_alignment_gain", 0.10,
        ),
        straight_goal_rejoin_max_score_loss=_nf(
            straight_goal_rejoin, "max_score_loss", 1.5,
        ),
        narrow_passage_enabled=bool(narrow_passage.get("enabled", True)),
        narrow_passage_side_probe_m=_nf(
            narrow_passage, "side_probe_m", 1.0,
        ),
        narrow_passage_side_obstacle_max_distance_m=_nf(
            narrow_passage, "side_obstacle_max_distance_m", 2.5,
        ),
        narrow_passage_max_center_clearance_m=_nf(
            narrow_passage, "max_center_clearance_m", 1.6,
        ),
        narrow_passage_hold_enabled=bool(
            narrow_passage.get("hold_enabled", True)
        ),
        narrow_passage_hold_max_duration_s=_nf(
            narrow_passage, "hold_max_duration_s", 6.0,
        ),
        narrow_passage_hold_max_distance_m=_nf(
            narrow_passage, "hold_max_distance_m", 4.5,
        ),
        adaptive_horizon_enabled=bool(adaptive_horizon.get("enabled", True)),
        min_horizon_m=_nf(adaptive_horizon, "min_horizon_m", 2.0),
        mid_horizon_m=_nf(adaptive_horizon, "mid_horizon_m", 3.0),
        max_horizon_m=_nf(adaptive_horizon, "max_horizon_m", 4.0),
        adaptive_near_threshold_m=_nf(adaptive_horizon, "near_threshold_m", 3.0),
        adaptive_mid_threshold_m=_nf(adaptive_horizon, "mid_threshold_m", 4.0),
        apf_max_lateral_correction_mps=_nf(apf_safety, "max_lateral_correction_mps", 0.25),
        apf_max_speed_reduction_ratio=_nf(apf_safety, "max_speed_reduction_ratio", 0.8),
        weights=weights,
    )


def _build_goal_termination_params(cfg: Dict[str, Any]):
    from planners.goal_termination import GoalTerminationParams
    s = (cfg or {}).get("goal_termination", {}) or {}

    def _f(key, default):
        v = s.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    return GoalTerminationParams(
        enabled=bool(s.get("enabled", True)),
        distance_tolerance_m=_f("distance_tolerance_m", 1.0),
        altitude_tolerance_m=_f("altitude_tolerance_m", 0.4),
        max_speed_mps=_f("max_speed_mps", 0.25),
        max_vertical_speed_mps=_f("max_vertical_speed_mps", 0.20),
        position_std_tolerance_m=_f("position_std_tolerance_m", 0.20),
        history_size_frames=int(_f("history_size_frames", 1)),
        dwell_time_s=_f("dwell_time_s", 1.0),
    )


def _build_occupancy_grid_params(cfg: Dict[str, Any]):
    from mapping.occupancy_grid import OccupancyGridParams
    s = (cfg or {}).get("occupancy_grid", {}) or {}

    def _f(key, default):
        v = s.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _i(key, default):
        v = s.get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(default)

    return OccupancyGridParams(
        resolution_m=_f("resolution_m", 0.5),
        map_radius_m=_f("map_radius_m", 40.0),
        max_range_m=_f("max_range_m", 15.0),
        min_range_m=_f("min_range_m", 0.2),
        occupied_log_odds=_f("occupied_log_odds", 0.85),
        free_log_odds=_f("free_log_odds", -0.4),
        occupied_threshold=_f("occupied_threshold", 0.0),
        inflation_cells=_i("inflation_cells", 1),
        horizontal_band_half_height_m=_f("horizontal_band_half_height_m", 1.0),
        ray_sample_spacing_m=_f("ray_sample_spacing_m", 0.25),
        self_filter_radius_m=_f("self_filter_radius_m", 0.5),
    )


def _sensor_points_to_world_xy(
    points,
    drone_position_ned,
    yaw_rad,
    max_range: float = 15.0,
    horizontal_band: float = 1.0,
) -> list:
    """Convert horizontal SensorLocalFrame points to world-NED XY obstacle points.

    Only points with |sensor_z| within ``horizontal_band`` and 2-D range within
    ``max_range`` are kept — these are the in-plane obstacles the 2D distance
    field cares about.  Returns a list of ``(x_world, y_world)`` tuples.
    """
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


def _resolve_mission_goal(
    actor_xyz,
    target_z_ned: float,
    fallback_start_ned,
    fallback_heading_rad: float,
    fallback_dist_m: float,
    goal_xy_override=None,
):
    """Return ``(goal_ned_xyz, source, actor_xyz_or_None)``.

    Phase C1-R: the MissionEnd actor's X/Y define the navigation endpoint, but
    its Z is scene metadata only (the marker sits on the ground, Z ≈ +0.4 NED).
    The actor goal's Z is the altitude-hold target (``target_z_ned``, the cruise
    altitude the drone climbed to), never the actor's ground-level Z — otherwise
    the goal-termination altitude check ``|dz| ≤ tol`` can never pass at cruise
    Z ≈ -1.0 against actor Z ≈ +0.4.

    The ``config_fixed`` fallback preserves the legacy goal Z = initial airborne
    Z (``fallback_start_ned[2]``), not the cruise target.
    """
    if goal_xy_override is not None:
        return (
            (
                float(goal_xy_override[0]),
                float(goal_xy_override[1]),
                float(target_z_ned),
            ),
            "cli_fixed",
            None,
        )
    if actor_xyz is not None:
        return (
            (float(actor_xyz[0]), float(actor_xyz[1]), float(target_z_ned)),
            "actor",
            actor_xyz,
        )
    return (
        (
            float(fallback_start_ned[0]) + math.cos(fallback_heading_rad) * fallback_dist_m,
            float(fallback_start_ned[1]) + math.sin(fallback_heading_rad) * fallback_dist_m,
            float(fallback_start_ned[2]),
        ),
        "config_fixed",
        None,
    )


def _replan_result_action(request_id: int, applied_id: int) -> str:
    """Classify a finished worker result against the last applied request id.

    The applied request id only ever increases (monotonic).  A result whose
    ``request_id`` is *less* than the applied id is a stale late-finish and must
    be ignored; a greater id is a newer result to apply.  Equal ids are a noop.
    """
    if request_id < applied_id:
        return "ignore_stale"
    if request_id > applied_id:
        return "apply"
    return "noop"


# ── automatic mode ──


class AutomaticMode:
    """LiDAR-based autonomous obstacle avoidance flight.

    All preflight checks execute BEFORE enableApiControl (in session.takeoff_and_climb).
    """

    def __init__(
        self,
        session: Any,
        perception_config_path: Optional[str] = None,
        flight_config_path: Optional[str] = None,
        params: Optional[AutomaticModeParams] = None,
        cli_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._session = session
        self._client = session.client
        self._adapter = session.adapter
        self._vn = session.vehicle_name
        self._cli_overrides = dict(cli_overrides or {})

        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self._perception_config_path = (
            perception_config_path or str(_PROJECT_ROOT / "configs" / "perception.yaml")
        )
        self._flight_config_path = (
            flight_config_path or str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        )

        # Load and validate flight config
        if params is not None:
            self._params = params
        else:
            self._params = AutomaticModeParams.from_yaml(
                self._flight_config_path, cli_overrides
            )
        logger.info("Flight config loaded from %s", self._flight_config_path)

        self._running = False
        self._last_velocity_future = None

        # Planner mode: "reactive" (default) | "apf_shadow" | "apf"
        self._planner_mode = cli_overrides.get("planner_mode", "reactive") if cli_overrides else "reactive"
        logger.info("planner_mode=%s", self._planner_mode)
        self._guided_apf_control = bool(cli_overrides.get("guided_apf_control", False)) if cli_overrides else False
        logger.info("guided_apf_control=%s", self._guided_apf_control)
        from planners.improved_potential_field import ImprovedPotentialField, ApfParams
        self._apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True,
            enable_per_sector_diagnostics=False,
        ))
        from planners.local_recovery import LocalRecovery, RecoveryParams, RecoveryDecision
        self._recovery = LocalRecovery(RecoveryParams(
            history_window_s=4.0,
            stuck_time_window_s=2.5,
            stuck_position_epsilon_m=0.15,
            stuck_min_frames=10,
            oscillation_time_window_s=2.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        from planners.recovery_commander import (
            RecoveryCommanderParams, RecoveryStateMachine,
        )
        self._recovery_sm = RecoveryStateMachine()

        # ── CBMBA A* shadow planner (compute + log only; never dispatches) ──
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams as _CbmbaParams

        # ── runtime resolution override (Phase 2G) ──
        _cbmba_resolution: float = 0.75
        if cli_overrides:
            _override_res = cli_overrides.get("cbmba_resolution")
            if _override_res is not None:
                if not math.isfinite(_override_res):
                    raise ValueError(
                        f"--cbmba-resolution must be finite, got {_override_res}"
                    )
                if _override_res <= 0:
                    raise ValueError(
                        f"--cbmba-resolution must be > 0, got {_override_res}"
                    )
                _cbmba_resolution = float(_override_res)

        self._cbmba = CbmbaAStarPlanner(_CbmbaParams(
            resolution=_cbmba_resolution,
            inflation_radius=1.5,
            max_search_nodes=2000,
            max_planning_time_ms=150.0,    # hard wall-clock budget → abort early, reuse cached path
            wall_penalty_radius=0,        # skip expensive proximity scan in shadow
            adaptive_long_step_cells=1,    # 26 neighbors instead of 52
        ))
        self._cbmba_enabled = True
        logger.info("cbmba_shadow  enabled=true  resolution=%.2f  max_search_nodes=%d",
                     self._cbmba.params.resolution, self._cbmba.params.max_search_nodes)

        # ── CBMBA guidance adapter (shadow only; never dispatches) ──
        from planners.cbmba_guidance import CbmbaGuidance, CbmbaGuidanceParams as _CbmbaGuidanceParams
        self._cbmba_guidance = CbmbaGuidance(_CbmbaGuidanceParams(
            min_forward_progress=0.25,
            min_waypoint_distance=0.5,
        ))
        self._cbmba_guidance_enabled = True
        logger.info("cbmba_guidance_shadow  enabled=true  min_forward_progress=%.2f  min_waypoint_distance=%.2f",
                     self._cbmba_guidance.params.min_forward_progress,
                     self._cbmba_guidance.params.min_waypoint_distance)

        # ── diagnostic rate-limit state ──
        self._diag_last_obstacle_count: int = -1
        self._diag_last_obstacle_log_time: float = -float("inf")
        self._diag_last_path_points: Optional[tuple] = None
        self._diag_last_path_log_time: float = -float("inf")

        # ── recovery test trigger (CLI only; one-shot) ──
        self._recovery_test_trigger = (
            cli_overrides.get("recovery_test_trigger") if cli_overrides else None
        )
        self._recovery_test_trigger_fired = False
        # Delay: ~1s after first airborne APF frame (≈10-15 frames at 10 Hz)
        self._recovery_test_trigger_delay_frames = 15
        if self._recovery_test_trigger is not None:
            logger.info(
                "recovery_test_trigger_enabled  type=%s  delay_frames=%d  once_per_process=true",
                self._recovery_test_trigger,
                self._recovery_test_trigger_delay_frames,
            )

        # ── bypass episode state (Failure A: prevents left/right oscillation) ──
        self._bypass = BypassEpisode()
        self._bypass_min_duration_s: float = 1.8
        self._bypass_entry_clearance_m: float = 1.5    # min LiDAR clearance to enter bypass
        self._bypass_release_clearance_m: float = 2.5   # clearance needed on both sides to release
        self._bypass_release_front_m: float = 2.5       # front clearance marking "obstacle passed"
        self._bypass_max_lateral_mps: float = 0.25      # max |vy| under bypass enforcement
        self._bypass_veto_unsafe_s: float = 1.5          # continuous unsafety before veto
        self._bypass_unsafe_start: Optional[float] = None  # when persistent unsafety began
        self._bypass_frame: int = 0                       # frame counter for bypass diagnostics
        # Peak bypass cross-track error must reach this to justify a REJOIN
        # handoff on "obstacle_passed".  Equals the REJOIN corridor half-width
        # (exit_path_error_m): a bypass whose peak error never left the corridor
        # has nothing to re-align, so release goes straight back to NORMAL.
        self._rejoin_excursion_m: float = 1.5

        # ── rejoin state (Failure A follow-up: BYPASS → REJOIN → NORMAL) ──
        self._rejoin = RejoinEpisode()

        # ── CBMBA shadow replan cadence (keeps A* off the realtime loop) ──
        self._cbmba_replan_interval_s: float = 0.5       # 2 Hz replan, not every control cycle
        self._cbmba_last_replan_time: float = -float("inf")
        self._cbmba_cached_result = None                 # latest valid path, reused between replans
        self._cbmba_path_generation: int = 0             # bumps on each fresh replan (diagnostic)

        # ── forward-progress watchdog (Failures A & B) ──
        self._progress_watchdog = ForwardProgressWatchdog(
            window_s=8.0,
            min_progress_m=1.0,
            check_interval_s=2.0,
        )

        # ── path validity gate state ──
        self._path_valid: bool = True
        self._path_fail_reason: str = ""
        self._consecutive_invalid_paths: int = 0
        self._max_consecutive_invalid_paths: int = 5

        # ── trajectory-centric local navigation (new layer) ──
        # "reactive" | "guided_apf" | "trajectory".  Default "reactive"
        # preserves the pre-existing behaviour exactly.
        self._local_navigation_mode = (
            cli_overrides.get("local_navigation_mode", "reactive")
            if cli_overrides else "reactive"
        )
        logger.info("local_navigation_mode=%s", self._local_navigation_mode)

        _traj_config_path_override = (
            cli_overrides.get("trajectory_config_path") if cli_overrides else None
        )
        self._traj_config_path = (
            _traj_config_path_override
            or str(_PROJECT_ROOT / "configs" / "trajectory_planner.yaml")
        )
        _traj_cfg = _load_trajectory_config(self._traj_config_path)
        _cbmba_cfg = _traj_cfg.get("cbmba", {}) or {}
        _dead_end_bypass_cfg = (
            (_traj_cfg.get("trajectory_planner", {}) or {}).get(
                "dead_end_bypass", {}
            ) or {}
        )
        self._dead_end_bypass_min_lateral_mps: float = float(
            _dead_end_bypass_cfg.get("min_lateral_speed_mps", 0.18)
        )
        self._dead_end_bypass_max_lateral_mps: float = float(
            _dead_end_bypass_cfg.get("max_lateral_speed_mps", 0.45)
        )
        self._dead_end_bypass_front_speed_mps: float = float(
            _dead_end_bypass_cfg.get("front_speed_mps", 0.25)
        )
        self._dead_end_bypass_min_distance_m: float = float(
            _dead_end_bypass_cfg.get("min_wall_follow_distance_m", 1.5)
        )
        self._dead_end_bypass_release_side_m: float = float(
            _dead_end_bypass_cfg.get("release_side_clearance_m", 3.5)
        )
        self._dead_end_bypass_release_front_m: float = float(
            _dead_end_bypass_cfg.get("release_front_clearance_m", 4.5)
        )
        self._dead_end_bypass_release_hold_s: float = float(
            _dead_end_bypass_cfg.get("release_hold_s", 0.8)
        )

        # LiDAR/map entries describe measured obstacle surfaces rather than
        # obstacle centres. Their passage inflation is configured separately
        # from the larger default used by explicit geometric obstacles.
        try:
            _surface_inflation = float(
                _cbmba_cfg.get("surface_observation_inflation_radius", 0.75)
            )
            if math.isfinite(_surface_inflation) and _surface_inflation >= 0.0:
                self._cbmba.params.surface_observation_inflation_radius = (
                    _surface_inflation
                )
        except (TypeError, ValueError):
            pass
        try:
            _los_inflation = float(
                _cbmba_cfg.get(
                    "line_of_sight_inflation",
                    self._cbmba.params.line_of_sight_inflation,
                )
            )
            if math.isfinite(_los_inflation) and _los_inflation >= 0.0:
                self._cbmba.params.line_of_sight_inflation = _los_inflation
        except (TypeError, ValueError):
            pass

        # Recovery is configured with the trajectory layer so its short escape
        # maneuver can keep up with the faster cruise command.  The directional
        # guard below still vetoes a command into a close obstacle.
        _recovery_cfg = (
            (_traj_cfg.get("trajectory_planner", {}) or {}).get("recovery", {})
            or {}
        )
        if _recovery_cfg:
            self._recovery_sm = RecoveryStateMachine(RecoveryCommanderParams(
                reverse_speed=float(_recovery_cfg.get("reverse_speed_mps", 0.25)),
                lateral_speed=float(_recovery_cfg.get("lateral_speed_mps", 0.25)),
                max_duration_s=float(_recovery_cfg.get("max_duration_s", 0.75)),
                cooldown_s=float(_recovery_cfg.get("cooldown_s", 1.5)),
                dead_end_escape_enabled=bool(
                    _recovery_cfg.get("dead_end_escape_enabled", True)
                ),
                dead_end_front_trigger_m=float(
                    _recovery_cfg.get("dead_end_front_trigger_m", 2.5)
                ),
                dead_end_side_trigger_m=float(
                    _recovery_cfg.get("dead_end_side_trigger_m", 2.0)
                ),
                vertical_climb_enabled=bool(
                    _recovery_cfg.get("vertical_climb_enabled", True)
                ),
                vertical_clearance_m=float(
                    _recovery_cfg.get("vertical_clearance_m", 2.5)
                ),
                vertical_climb_speed_mps=float(
                    _recovery_cfg.get("vertical_climb_speed_mps", 0.20)
                ),
                vertical_climb_duration_s=float(
                    _recovery_cfg.get("vertical_climb_duration_s", 1.2)
                ),
                vertical_climb_delta_m=float(
                    _recovery_cfg.get("vertical_climb_delta_m", 0.40)
                ),
                wall_follow_forward_speed_mps=float(
                    _recovery_cfg.get("wall_follow_forward_speed_mps", 0.15)
                ),
                wall_follow_duration_s=float(
                    _recovery_cfg.get("wall_follow_duration_s", 4.0)
                ),
                wall_follow_side_lock_enabled=bool(
                    _recovery_cfg.get("wall_follow_side_lock_enabled", True)
                ),
            ))

        # Initial heading alignment: before navigation starts, point the
        # vehicle toward the fixed MissionEnd XY.  The trajectory planner uses
        # forward body-frame primitives; without this gate a goal behind the
        # initial nose can make the vehicle drive forward while it gradually
        # bends toward the goal.
        _ha_cfg = (_traj_cfg.get("heading_alignment", {}) or {})
        self._heading_alignment_enabled = bool(_ha_cfg.get("enabled", True))
        self._heading_alignment_trigger_rad = math.radians(
            max(0.0, float(_ha_cfg.get("trigger_angle_deg", 20.0)))
        )
        self._heading_alignment_settle_rad = math.radians(
            max(0.0, float(_ha_cfg.get("settle_angle_deg", 8.0)))
        )
        self._heading_alignment_kp = max(
            0.0, float(_ha_cfg.get("kp", 1.2))
        )
        self._heading_alignment_max_rate = max(
            0.0, float(_ha_cfg.get("max_yaw_rate_radps", 0.5))
        )
        self._heading_alignment_timeout_s = max(
            0.0, float(_ha_cfg.get("timeout_s", 20.0))
        )

        # Runtime heading alignment handles the case where the vehicle has
        # already passed the goal while an old STRAIGHT trajectory is still
        # cached.  It is intentionally gated to the near-goal region so that
        # obstacle bypasses remain under the local planner's control.
        self._runtime_heading_alignment_enabled = bool(
            _ha_cfg.get("runtime_enabled", True)
        )
        self._runtime_heading_alignment_trigger_rad = math.radians(
            max(0.0, float(_ha_cfg.get("runtime_trigger_angle_deg", 100.0)))
        )
        self._runtime_heading_alignment_settle_rad = math.radians(
            max(0.0, float(_ha_cfg.get("runtime_settle_angle_deg", 12.0)))
        )
        self._runtime_heading_alignment_max_distance_m = max(
            0.0, float(_ha_cfg.get("runtime_max_distance_m", 8.0))
        )
        self._runtime_heading_alignment_kp = max(
            0.0, float(_ha_cfg.get("runtime_kp", self._heading_alignment_kp))
        )
        self._runtime_heading_alignment_max_rate = max(
            0.0,
            float(_ha_cfg.get(
                "runtime_max_yaw_rate_radps", self._heading_alignment_max_rate
            )),
        )
        self._runtime_heading_alignment_command_duration_s = max(
            0.05, float(_ha_cfg.get("runtime_command_duration_s", 0.05))
        )
        self._runtime_goal_behind_min_forward_m = max(
            0.1, float(_ha_cfg.get("runtime_goal_behind_min_forward_m", 0.5))
        )
        self._runtime_heading_alignment_active = False
        self._runtime_heading_alignment_started_mono: Optional[float] = None

        from planners.local_trajectory_planner import (
            LocalTrajectoryPlanner, TrajectoryMemory,
        )
        from planners.trajectory_tracker import TrajectoryTracker
        from planners.goal_termination import GoalTerminationChecker
        from mapping.occupancy_grid import OccupancyGridMap
        from mapping.distance_field import DistanceField

        self._traj_params = _build_trajectory_params(_traj_cfg)
        self._goal_term_params = _build_goal_termination_params(_traj_cfg)
        self._occ_grid_params = _build_occupancy_grid_params(_traj_cfg)

        self._occ_grid = OccupancyGridMap(self._occ_grid_params)
        self._distance_field = DistanceField()
        self._traj_memory = TrajectoryMemory(
            history_length=int(
                (_traj_cfg.get("trajectory_memory", {}) or {}).get("history_length", 10)
            )
        )
        self._traj_planner = LocalTrajectoryPlanner(
            params=self._traj_params, memory=self._traj_memory,
        )
        self._traj_tracker = TrajectoryTracker(
            lookahead_m=self._traj_params.tracker_lookahead_m,
            sample_spacing_m=self._traj_params.sample_spacing_m,
            forward_speed_mps=self._traj_params.forward_speed_mps,
            lateral_speed_mps=self._traj_params.lateral_speed_mps,
            command_lookahead_m=self._traj_params.command_lookahead_m,
            yaw_gain=float(((_traj_cfg.get("trajectory_tracking", {}) or {})
                            .get("yaw_control", {}) or {}).get("kp", 1.4)),
            max_yaw_rate_radps=float(((_traj_cfg.get("trajectory_tracking", {}) or {})
                                      .get("yaw_control", {}) or {})
                                     .get("max_yaw_rate_radps", 0.5)),
            goal_blend_distance_m=float(((_traj_cfg.get("trajectory_tracking", {}) or {})
                                         .get("yaw_control", {}) or {})
                                        .get("goal_blend_distance_m", 4.0)),
            goal_direct_distance_m=float(((_traj_cfg.get("trajectory_tracking", {}) or {})
                                          .get("yaw_control", {}) or {})
                                         .get("goal_direct_distance_m", 2.0)),
            goal_slowdown_distance_m=float(((_traj_cfg.get("trajectory_tracking", {}) or {})
                                            .get("yaw_control", {}) or {})
                                           .get("goal_slowdown_distance_m", 4.0)),
            terminal_goal_approach_radius_m=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_goal_approach_radius_m", 0.0
                )
            ),
            terminal_slowdown_radius_m=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_slowdown_radius_m", 0.0
                )
            ),
            terminal_goal_kp=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_goal_kp", 0.5
                )
            ),
            terminal_goal_max_speed_mps=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_goal_max_speed_mps", self._traj_params.forward_speed_mps
                )
            ),
            terminal_braking_accel_mps2=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_braking_accel_mps2", 0.35
                )
            ),
            terminal_capture_radius_m=float(
                (_traj_cfg.get("trajectory_tracking", {}) or {}).get(
                    "terminal_capture_radius_m", 0.02
                )
            ),
        )
        _yaw_control_cfg = (
            (_traj_cfg.get("trajectory_tracking", {}) or {}).get("yaw_control", {})
            or {}
        )
        self._trajectory_yaw_enabled = bool(_yaw_control_cfg.get("enabled", True))
        self._goal_term = GoalTerminationChecker(self._goal_term_params)

        # ── global path cache (CBMBA replans at ~1-2 Hz; local replans faster) ──
        self._traj_global_path: list = []
        self._traj_global_path_version: int = 0
        self._traj_last_replan_time: float = -float("inf")
        self._traj_global_replan_hz = float(_cbmba_cfg.get("global_replan_hz", 1.5))
        self._traj_path_switch_min_improvement = float(
            _cbmba_cfg.get("path_switch_min_improvement_ratio", 0.10)
        )

        # ── local planning / tracking split state ──
        self._traj_planning_hz = self._traj_params.planning_hz
        self._traj_last_plan_time: float = -float("inf")
        self._traj_last_apply_time: float = -float("inf")
        self._traj_cached_points: list = []
        self._traj_cached_family: Optional[str] = None
        self._traj_narrow_passage_active: bool = False
        self._traj_force_replan: bool = True

        # ── no-feasible → fast recovery ──
        _nf_cfg = (_traj_cfg.get("trajectory_planner", {}) or {}).get("no_feasible", {}) or {}
        self._traj_no_feasible_count: int = 0
        self._traj_no_feasible_start: Optional[float] = None
        self._traj_no_feasible_recovery_count = int(_nf_cfg.get("recovery_trigger_count", 3))
        self._traj_no_feasible_recovery_duration_s = float(
            _nf_cfg.get("recovery_trigger_duration_s", 0.5)
        )

        # ── local ESDF window + lidar downsample ──
        _df_cfg = _traj_cfg.get("distance_field", {}) or {}
        self._traj_dfield_radius_m = float(_df_cfg.get("radius_m", 10.0))
        self._traj_lidar_downsample_m = float(_df_cfg.get("lidar_downsample_resolution_m", 0.25))

        # ── tracking error → replan ──
        _tt_cfg = _traj_cfg.get("trajectory_tracking", {}) or {}
        self._traj_tracking_error_threshold_m = float(
            _tt_cfg.get("replan_error_threshold_m", 0.5)
        )
        self._traj_last_track_pose: Optional[Tuple[float, float]] = None
        self._traj_last_track_cmd: Tuple[float, float] = (0.0, 0.0)
        self._traj_tracking_error_sum: float = 0.0
        self._traj_tracking_error_max: float = 0.0
        self._traj_tracking_error_n: int = 0
        self._traj_recovery_was_active: bool = False

        # ── no-feasible → recovery request handoff, recovery-exit → replan ──
        self._traj_request_recovery: bool = False
        self._traj_escape_hint: Optional[dict] = None
        self._traj_global_replan_requested: bool = False

        # ── CBMBA global planner worker (Phase C3-R: separate process) ──
        # The worker owns a *private* CBMBA instance in its own OS process
        # (never shared, never passed across the process boundary) so the 20 Hz
        # control loop never blocks on a 1-3 s A* search — and, critically, the
        # pure-Python A* no longer holds the GIL that used to starve the loop.
        # Only trajectory mode uses the worker; other modes keep the sync shadow.
        self._global_planner_worker = None
        self._local_traj_worker = None
        self._mapping_worker = None                # Phase C4-R: persistent-map process
        self._map_snapshot: dict = {"occupied_points": [], "map_version": -1}
        self._last_map_sensor_timestamp: Optional[float] = None
        self._last_applied_map_version: int = -1
        self._traj_applied_replan_id: int = -1
        self._cbmba_search_total: int = 0          # total plan_with_result calls (all modes)
        self._cbmba_searches_this_frame: int = 0   # duplicate-search guard (per frame)
        self._initial_replan_requested: bool = False  # Phase C3-R: dedup "initial" replan
        self._traj_global_path_min_clearance: float = float("inf")
        self._traj_local_plan_seq: int = 0         # last applied local-plan request id
        self._navigation_ready: bool = False       # Phase C3-R: altitude startup gate
        if self._local_navigation_mode == "trajectory":
            from dataclasses import asdict
            from planners.process_workers import (
                CbmbaProcessWorker, LocalTrajectoryPlannerWorker, MappingProcessWorker,
            )
            self._global_planner_worker = CbmbaProcessWorker(
                planner_config=asdict(self._cbmba.params),
            )
            self._local_traj_worker = LocalTrajectoryPlannerWorker(
                traj_config=asdict(self._traj_params),
                occ_config=asdict(self._occ_grid_params),
                memory_history_length=self._traj_memory.history_length,
                dfield_radius_m=self._traj_dfield_radius_m,
                downsample_m=self._traj_lidar_downsample_m,
            )
            self._mapping_worker = MappingProcessWorker(
                occ_config=asdict(self._occ_grid_params),
            )

        # ── Phase C0: flight validation + closed-loop hardening ──
        from flight_modes.trajectory_flight_metrics import (
            TrajectoryFlightMetrics,
            ObstacleAvoidanceEpisodeTracker,
            FamilyTransitionLog,
            MissionProgressMonitor,
            SingleObstacleBehaviorMonitor,
        )

        _tp = (_traj_cfg.get("trajectory_planner", {}) or {})
        self._allow_guided_apf_fallback = bool(_tp.get("allow_guided_apf_fallback", False))

        # Stale cached-trajectory limits (sec 9).
        _tt_cfg2 = (_traj_cfg.get("trajectory_tracking", {}) or {})
        self._traj_stale_warn_s = float(_tt_cfg2.get("stale_warn_s", 0.25))
        self._traj_stale_stop_s = float(_tt_cfg2.get("stale_stop_s", 0.75))

        # Control-loop watchdog + scheduler (sec 10/11/34).  target_hz is the
        # control-loop rate; the AirSim command_duration_s is unrelated.  The
        # deadline scheduler sleeps to the next period boundary, never a fixed
        # command_duration_s.
        _cl_cfg = (_traj_cfg.get("control_loop", {}) or {})
        self._control_loop_target_hz = float(_cl_cfg.get("target_hz", 20.0))
        self._control_period_s = 1.0 / max(0.5, self._control_loop_target_hz)
        self._control_loop_overrun_warn_ms = float(_cl_cfg.get("overrun_warn_ms", 80.0))
        self._control_loop_overrun_stop_ms = float(_cl_cfg.get("overrun_stop_ms", 500.0))
        _as_cfg = (_traj_cfg.get("altitude_safety", {}) or {})
        self._altitude_safety_enabled = bool(_as_cfg.get("enabled", True))
        self._altitude_error_stop_m = max(
            0.2, float(_as_cfg.get("error_stop_m", 0.8))
        )
        self._loop_last_iter_mono: Optional[float] = None
        self._loop_overrun_count: int = 0
        self._loop_overrun_stop_count: int = 0
        self._loop_max_overrun_ms: float = 0.0
        self._loop_dt_sum_ms: float = 0.0
        self._loop_dt_n: int = 0
        self._loop_dt_samples_ms: list = []
        self._loop_current_overrun_ms: float = 0.0
        # Phase C4: realtime architecture summary accumulators.
        self._perception_age_samples_ms: list = []
        self._perception_stale_count: int = 0
        self._last_frame_stale_hold: bool = False  # stale→valid transition marker (P0-A diag)
        self._cbmba_completed: int = 0
        self._cbmba_time_ms_sum: float = 0.0
        self._cbmba_time_ms_max: float = 0.0
        self._traj_completed: int = 0
        self._traj_time_ms_sum: float = 0.0
        self._traj_time_ms_max: float = 0.0
        self._trajectory_dispatch_count: int = 0
        self._hover_dispatch_count: int = 0
        # Phase C4-R: hover dispatch breakdown by safety source (sec 19/26).
        self._hover_due_perception_stale: int = 0
        self._hover_due_control_overrun: int = 0
        self._hover_due_trajectory_stale: int = 0
        self._hover_due_no_feasible: int = 0
        self._hover_due_other: int = 0
        self._max_altitude_error_m: float = 0.0
        # Phase C4-R: worker-request prepare vs put split (sec 17, profile-only).
        self._cbmba_request_prepare_ms: float = 0.0
        self._traj_request_prepare_ms: float = 0.0
        self._map_request_prepare_ms: float = 0.0
        # Phase C5-R: request put (enqueue/pickle) split from prepare, plus
        # per-worker snapshot-substep timings (sec 6, profile-only).
        self._cbmba_request_put_ms: float = 0.0
        self._traj_request_put_ms: float = 0.0
        self._map_request_put_ms: float = 0.0
        self._cbmba_obstacle_snapshot_ms: float = 0.0
        self._traj_path_copy_ms: float = 0.0
        self._traj_snapshot_build_ms: float = 0.0
        self._map_snapshot_build_ms: float = 0.0
        self._lidar_stale_frames_total: int = 0
        # Phase C6-R: loop-start-interval tight-loop + resync accounting.
        # A "tight loop" is a loop-start interval < 25/40 ms (i.e. the scheduler
        # fired the next tick too soon after the previous one); a "post-resync
        # tight loop" is one that immediately follows a deadline re-anchor — the
        # signature of a catch-up burst.  Acceptance requires post_resync == 0.
        self._loop_tight_lt25_count: int = 0
        self._loop_tight_lt40_count: int = 0
        self._loop_deadline_resync_count: int = 0
        self._loop_post_resync_tight_count: int = 0
        self._loop_last_was_resync: bool = False

        # Background LiDAR/perception worker config (sec 10-14).
        _pr_cfg = (_traj_cfg.get("perception_runtime", {}) or {})
        self._perception_poll_hz = float(_pr_cfg.get("poll_hz", 10.0))
        self._perception_stale_warn_s = float(_pr_cfg.get("stale_warn_s", 0.25))
        self._perception_stale_stop_s = float(_pr_cfg.get("stale_stop_s", 0.75))
        self._perception_worker: Optional[PerceptionWorker] = None

        # Round 9: LiDAR empty-frame health state machine.  A FRESH timestamp
        # with an empty point cloud is a transient dropout → safety hold (never
        # an immediate abort, never "free space").  Only a persistent run of
        # empty frames beyond these limits terminates (lidar_invalid:persistent_empty).
        _lh_cfg = (_traj_cfg.get("lidar_health", {}) or {})
        self._lidar_empty_grace_s = float(_lh_cfg.get("empty_grace_s", 1.0))
        self._lidar_empty_max_frames = int(_lh_cfg.get("max_consecutive_empty_frames", 5))
        if self._lidar_empty_grace_s < 0.0:
            self._lidar_empty_grace_s = 1.0
        if self._lidar_empty_max_frames < 1:
            self._lidar_empty_max_frames = 5
        self._lidar_empty_hold_active: bool = False
        self._lidar_empty_since_mono: Optional[float] = None
        self._lidar_consecutive_empty: int = 0
        self._lidar_empty_max_consecutive: int = 0
        self._lidar_empty_frames_total: int = 0
        self._lidar_valid_nonempty_frames: int = 0
        self._lidar_invalid_frames: int = 0
        self._lidar_last_nonempty_ts_ns: Optional[int] = None
        self._lidar_last_valid_point_count: int = 0
        self._lidar_prev_ts_ns: Optional[int] = None
        self._lidar_empty_last_mono: Optional[float] = None
        self._lidar_health_log_mono: float = 0.0
        self._last_frame_empty_hold: bool = False  # empty→valid transition marker
        self._lidar_empty_last_run_frames: int = 0
        self._lidar_empty_last_run_duration_s: float = 0.0

        # Debug drawing / HUD / CSV trace (sec 19-21).
        self._traj_debug_cfg = (_traj_cfg.get("trajectory_debug", {}) or {})
        self._debug_drawer = None
        self._debug_draw_period_s = max(
            0.1, float(self._traj_debug_cfg.get("update_period_s", 0.5))
        )
        self._debug_draw_last_mono = 0.0
        self._trace_writer = None
        self._trace_csv_path: Optional[str] = None

        # Flight metrics + monitors (sec 1-3, 23-25, 31).
        self._traj_metrics = TrajectoryFlightMetrics()
        self._traj_family_log = FamilyTransitionLog()
        _ep_cfg = (_traj_cfg.get("obstacle_episode", {}) or {})
        self._traj_episode_tracker = ObstacleAvoidanceEpisodeTracker(
            start_distance_m=float(_ep_cfg.get("start_distance_m", 3.0)),
            end_distance_m=float(_ep_cfg.get("end_distance_m", 3.5)),
            hold_frames=int(_ep_cfg.get("hold_frames", 5)),
        )
        _mp_cfg = (_traj_cfg.get("mission_progress", {}) or {})
        self._mission_progress_monitor = MissionProgressMonitor(
            window_s=float(_mp_cfg.get("window_s", 10.0)),
            min_progress_m=float(_mp_cfg.get("min_progress_m", 1.0)),
            check_interval_s=float(_mp_cfg.get("check_interval_s", 2.0)),
            stuck_epsilon_m=float(_mp_cfg.get("stuck_epsilon_m", 0.2)),
        )
        _som_cfg = (_traj_cfg.get("single_obstacle_monitor", {}) or {})
        self._obstacle_behavior_monitor = SingleObstacleBehaviorMonitor(
            min_samples=int(_som_cfg.get("min_samples", 8)),
        )

        # ── safety-geometry audit (sec 17/18): single source of truth ──
        logger.info(
            "safety_geometry_audit  hard_clearance=%.2fm  "
            "preferred_clearance=%.2fm  emergency_distance=%.2fm  "
            "occ_grid_inflation_cells=%d  cbmba_inflation_radius=%.2fm  "
            "double_inflation=%s",
            self._traj_params.hard_clearance_m,
            self._traj_params.preferred_clearance_m,
            self._params.emergency_distance_m,
            self._occ_grid_params.inflation_cells,
            self._cbmba.params.inflation_radius,
            "true" if (self._occ_grid_params.inflation_cells > 0
                       and self._cbmba.params.inflation_radius > 0) else "false",
        )

        logger.info(
            "trajectory_layer  enabled=%s  num_candidates=%d  horizon=%.1fm  "
            "hard_clearance=%.1fm  planning_hz=%.1f  global_replan_hz=%.1f  "
            "dfield_radius=%.1fm  lidar_ds=%.2fm  goal_term=%s  "
            "adaptive_horizon=%s  allow_guided_apf_fallback=%s",
            self._traj_params.enabled, self._traj_params.num_candidates,
            self._traj_params.horizon_m, self._traj_params.hard_clearance_m,
            self._traj_planning_hz, self._traj_global_replan_hz,
            self._traj_dfield_radius_m, self._traj_lidar_downsample_m,
            self._goal_term_params.enabled,
            "true" if self._traj_params.adaptive_horizon_enabled else "false",
            "true" if self._allow_guided_apf_fallback else "false",
        )

    # ── public API ──

    # ── bypass episode lifecycle (Failure A: oscillation prevention) ──

    @staticmethod
    def _side_label(committed_side) -> str:
        """Safe side-name helper — handles MagicMock / non-int values."""
        if not isinstance(committed_side, int):
            return "unknown"
        if committed_side > 0:
            return "right"
        if committed_side < 0:
            return "left"
        return "hold"

    def _should_enter_bypass(
        self, rays: Dict[str, float],
        guidance_dir: Optional[Tuple[float, float]],
    ) -> Tuple[bool, str]:
        """Decide whether to enter a bypass episode.

        Entry conditions (all must be true):
        1. At least one side has LiDAR clearance < entry threshold (space is tight)
        2. CBMBA guidance has a clear lateral preference (|direction_y| > 0.15)
        3. The guidance-preferred side has adequate LiDAR clearance for safety
        """
        left = rays.get("left", float("inf")) or float("inf")
        right = rays.get("right", float("inf")) or float("inf")
        front = rays.get("front", float("inf")) or float("inf")

        # Condition 1: space is constrained on at least one side
        min_side = min(left, right)
        if min_side >= self._bypass_entry_clearance_m:
            return False, "both_sides_open"

        # Condition 2: CBMBA guidance has lateral preference
        if guidance_dir is None:
            return False, "no_guidance"
        gx, gy = guidance_dir
        if abs(gy) < 0.15:
            return False, f"guidance_forward_dominant(gy={gy:.3f})"

        # Condition 3: guidance-preferred side is safe enough
        guidance_side = 1 if gy > 0 else -1
        side_clearance = right if guidance_side == 1 else left
        min_entry_clearance = 1.0  # absolute minimum to enter
        if side_clearance < min_entry_clearance:
            return False, (
                f"guidance_side_unsafe(side={'right' if guidance_side == 1 else 'left'},"
                f" clearance={side_clearance:.2f})"
            )

        # Don't enter if front is wide open (no need to bypass)
        if front > 5.0 and min_side > 1.5:
            return False, "front_wide_open"

        return True, f"enter(side={'right' if guidance_side == 1 else 'left'})"

    def _inheritance_formal_entry(
        self, rays: Dict[str, float],
        guidance_dir: Optional[Tuple[float, float]],
    ) -> Tuple[bool, str]:
        """Formal entry gate for recovery→bypass side inheritance.

        Returns ``(allowed, reason)``.  Recovery may persist a committed side
        into a BYPASS episode ONLY when the formal entry gate passes (a real
        corridor constraint).  When there is no constraint — both sides open,
        or no guidance — recovery was a false trigger and there is no side to
        persist: skip the inheritance so the drone returns straight to NORMAL.
        """
        if not self._guided_apf_control or guidance_dir is None:
            return False, "guided_apf_unavailable"
        return self._should_enter_bypass(rays, guidance_dir)

    def _choose_bypass_side(
        self, rays: Dict[str, float],
        guidance_dir: Optional[Tuple[float, float]],
    ) -> int:
        """Choose bypass direction: +1 (right), -1 (left).

        Priority: CBMBA guidance > LiDAR openness.  LiDAR is used as a
        safety gate: the chosen side must have at least minimum clearance.
        """
        left = rays.get("left", float("inf")) or float("inf")
        right = rays.get("right", float("inf")) or float("inf")

        if guidance_dir is not None and abs(guidance_dir[1]) > 0.1:
            side = 1 if guidance_dir[1] > 0 else -1
            clearance = right if side == 1 else left
            if clearance >= 1.0:
                return side

        # Fallback: LiDAR alone
        if right > left:
            return 1
        if left > right:
            return -1
        return -1  # default left (conservative)

    def _should_release_bypass(
        self, rays: Dict[str, float], now: float,
    ) -> Tuple[bool, str]:
        """Decide whether to release an active bypass episode.

        Release conditions (any one triggers):
        1. Both sides have clearance ≥ release threshold (space opened up)
        2. Chosen side has been persistently unsafe (veto)
        3. Front has been wide open for sustained period
        """
        ep = self._bypass
        if not ep.active:
            return True, "not_active"

        left = rays.get("left", float("inf")) or float("inf")
        right = rays.get("right", float("inf")) or float("inf")
        front = rays.get("front", float("inf")) or float("inf")
        elapsed = now - ep.start_time
        chosen_clearance = right if ep.side == 1 else left

        # Condition 1: both sides clear — normal release
        if (not ep.trajectory_dead_end
                and elapsed >= ep.min_duration_s
                and left >= self._bypass_release_clearance_m
                and right >= self._bypass_release_clearance_m):
            return True, "both_sides_clear"

        # Condition 2: chosen side persistently unsafe — safety veto
        if chosen_clearance < 0.8:  # dangerously close
            if self._bypass_unsafe_start is None:
                self._bypass_unsafe_start = now
            elif now - self._bypass_unsafe_start >= self._bypass_veto_unsafe_s:
                return True, (
                    f"veto:chosen_side_unsafe(side={'right' if ep.side == 1 else 'left'},"
                    f" clearance={chosen_clearance:.2f}, duration={now - self._bypass_unsafe_start:.2f}s)"
                )
        else:
            self._bypass_unsafe_start = None  # safety restored, reset timer

        # A large U-shaped dead end can have a clear-looking forward ray while
        # the vehicle is still between its three walls. For this episode type
        # the chosen wall must open, together with the front, for a short
        # continuous interval before normal goal guidance is allowed back in.
        if ep.trajectory_dead_end:
            wall_end_clear = (
                front >= self._dead_end_bypass_release_front_m
                and chosen_clearance >= self._dead_end_bypass_release_side_m
                and ep.max_displacement_m >= self._dead_end_bypass_min_distance_m
            )
            if wall_end_clear:
                if ep.wall_end_clear_since is None:
                    ep.wall_end_clear_since = now
                elif now - ep.wall_end_clear_since >= self._dead_end_bypass_release_hold_s:
                    return True, "dead_end_wall_end"
            else:
                ep.wall_end_clear_since = None
            return False, "hold_dead_end_wall"

        # Condition 3: obstacle passed — front clear AND the committed side has
        # opened back up.  Fires WITHOUT min_duration: once the drone has
        # physically moved past the obstacle, holding the side commitment any
        # longer would carry it away from the goal (the permanent-hold bug).
        if (front > self._bypass_release_front_m
                and chosen_clearance >= self._bypass_release_clearance_m):
            return True, "obstacle_passed"

        # Condition 4: front wide open for sustained period
        if front > 5.0 and elapsed >= ep.min_duration_s:
            return True, "front_wide_open"

        return False, "hold"

    def _track_bypass_excursion(self, position_xy: Tuple[float, float]) -> None:
        """Record the peak cross-track error during an active BYPASS episode.

        Measured against the FROZEN reference so a live re-seeded path cannot
        hide the excursion.  Used at release time to decide REJOIN (real
        deviation) vs NORMAL (the drone never left the corridor).
        """
        if not self._bypass.active or not self._bypass.reference_path_xy:
            if self._bypass.active and self._bypass.trajectory_dead_end:
                frozen = self._bypass.reference_frozen_position_xy
                if frozen is not None:
                    self._bypass.max_displacement_m = max(
                        self._bypass.max_displacement_m,
                        math.hypot(position_xy[0] - frozen[0], position_xy[1] - frozen[1]),
                    )
            return
        if self._bypass.trajectory_dead_end:
            frozen = self._bypass.reference_frozen_position_xy
            if frozen is not None:
                self._bypass.max_displacement_m = max(
                    self._bypass.max_displacement_m,
                    math.hypot(position_xy[0] - frozen[0], position_xy[1] - frozen[1]),
                )
        err = self._rejoin_path_error(position_xy, self._bypass.reference_path_xy)
        if math.isfinite(err) and err > self._bypass.max_path_error_m:
            self._bypass.max_path_error_m = err

    def _bypass_release_destination(self, rel_reason: str) -> str:
        """Choose the BYPASS release destination: ``"rejoin"`` vs ``"normal"``.

        A bypass is only handed to REJOIN (post-bypass re-alignment) when it
        actually took the drone OUT of the reference corridor — i.e. its peak
        cross-track error reached ``_rejoin_excursion_m``.  A bypass that never
        deviated (e.g. a recovery-inherited side with no real obstacle, released
        in a single frame) must go straight back to NORMAL: there is nothing to
        re-align, and forcing REJOIN would enter it with a near-zero start error
        that the convergence gate can never distinguish from progress.
        """
        if rel_reason.startswith("obstacle_passed") and \
                self._bypass.max_path_error_m >= self._rejoin_excursion_m:
            return "rejoin"
        return "normal"

    def _reset_stale_hold_accumulators(
        self, now: float, position_xy: Tuple[float, float],
    ) -> None:
        """Exclude an intentional perception-stale safety hover from the stuck
        detector and forward-progress watchdog.

        A stale hold issues zero velocity on purpose; once perception recovers,
        the timestamp gap + unchanged position would otherwise read as "stuck"
        (stationary for a long wall-clock duration) and "no forward progress".
        Re-anchor both accumulators at the hold position so the intentional
        hover never feeds a false recovery or a false progress-watchdog fire.
        """
        self._recovery.reset()
        self._progress_watchdog.reset(now, position_xy)

    def _handle_lidar_empty_frame(
        self, lf, now_mono: float, position_ned, velocity_ned,
    ) -> Optional[str]:
        """Round 9: handle one LiDAR frame whose point cloud is empty.

        Returns ``"lidar_invalid:persistent_empty"`` when the empty condition has
        persisted past the grace limits (the caller must terminate); otherwise
        returns ``None`` and the caller issues a zero-velocity safety hold and
        keeps polling.

        Side effects (all deterministic):
          * advance empty-frame counters (consecutive / total / max),
          * re-anchor the stuck detector + progress watchdog so the intentional
            zero-velocity hold never later reads as "stuck" / "no progress",
          * set the empty→valid transition marker for ``post_empty_recovery``,
          * emit ``empty_hold_enter`` / ``empty_hold_active`` / ``lidar_empty``.
        """
        raw_ts = getattr(lf, "raw_timestamp_ns", None)
        prev_ts = self._lidar_prev_ts_ns
        ts_advanced = (raw_ts is not None and prev_ts is not None and raw_ts != prev_ts)
        self._lidar_prev_ts_ns = raw_ts

        first_empty = self._lidar_consecutive_empty == 0
        self._lidar_consecutive_empty += 1
        self._lidar_empty_frames_total += 1
        self._lidar_empty_max_consecutive = max(
            self._lidar_empty_max_consecutive, self._lidar_consecutive_empty,
        )
        if first_empty:
            self._lidar_empty_since_mono = now_mono
        empty_elapsed = (
            now_mono - self._lidar_empty_since_mono
            if self._lidar_empty_since_mono is not None else 0.0
        )
        poll_interval = (
            now_mono - self._lidar_empty_last_mono
            if self._lidar_empty_last_mono is not None else 0.0
        )
        self._lidar_empty_last_mono = now_mono
        self._lidar_empty_hold_active = True
        self._last_frame_empty_hold = True

        # Re-anchor stuck + progress accumulators (P0-A pattern) so the hold is
        # never mistaken for a genuine stall.
        self._reset_stale_hold_accumulators(
            now_mono, (position_ned[0], position_ned[1]),
        )

        if first_empty:
            logger.info(
                "empty_hold_enter  consecutive_empty=%d  point_count=%d  "
                "position=(%.3f,%.3f,%.3f)  timestamp=%s  timestamp_advanced=%s",
                self._lidar_consecutive_empty, getattr(lf, "point_count", 0),
                position_ned[0], position_ned[1], position_ned[2],
                raw_ts, "true" if ts_advanced else "false",
            )
        else:
            self._log_throttled(
                "empty_hold_active", 1.0,
                "empty_hold_active  consecutive_empty=%d  empty_elapsed=%.3f  "
                "position=(%.3f,%.3f,%.3f)",
                self._lidar_consecutive_empty, empty_elapsed,
                position_ned[0], position_ned[1], position_ned[2],
            )

        self._log_throttled(
            "lidar_empty", 1.0,
            "lidar_empty  sensor_timestamp=%s  previous_timestamp=%s  "
            "timestamp_advanced=%s  point_count=0  "
            "position=(%.3f,%.3f,%.3f)  velocity=(%.3f,%.3f,%.3f)  "
            "wall_poll_interval=%.3f  consecutive_empty=%d  empty_elapsed=%.3f",
            raw_ts, prev_ts,
            "true" if ts_advanced else "false",
            position_ned[0], position_ned[1], position_ned[2],
            velocity_ned[0], velocity_ned[1], velocity_ned[2],
            poll_interval,
            self._lidar_consecutive_empty, empty_elapsed,
        )

        # Persistent-empty fail-safe (sec 7): terminate only when the empty
        # condition has been continuous beyond the frame-count OR duration limit.
        if self._lidar_consecutive_empty >= self._lidar_empty_max_frames:
            return "lidar_invalid:persistent_empty"
        if empty_elapsed >= self._lidar_empty_grace_s:
            return "lidar_invalid:persistent_empty"
        return None

    def _enforce_bypass_side(
        self, vx: float, vy: float, bypass_side: int,
    ) -> Tuple[float, float]:
        """Clamp vy to enforce the bypass side commitment.

        - vy sign MUST match bypass_side (if side=+1, vy ≥ 0; if side=-1, vy ≤ 0)
        - |vy| is clamped to [0, _bypass_max_lateral_mps]
        - vx is NOT modified (forward_sign_guard handles that separately)
        """
        max_lat = self._bypass_max_lateral_mps
        if bypass_side > 0:
            vy = max(0.0, min(vy, max_lat))
        elif bypass_side < 0:
            vy = max(-max_lat, min(vy, 0.0))
        return vx, vy

    def _enforce_trajectory_dead_end_bypass(
        self, vx: float, vy: float, rays: Dict[str, float],
    ) -> Tuple[float, float]:
        """Keep trajectory control moving around a committed dead-end wall.

        The local trajectory may still point at the mission goal while the
        vehicle is inside a U-shaped obstacle. Preserve the chosen side,
        suppress forward motion into a close front wall, and provide a small
        sideward bias when the selected side has usable clearance.
        """
        ep = self._bypass
        if not ep.active or not ep.trajectory_dead_end or ep.side not in (-1, 1):
            return vx, vy

        vx, vy = self._enforce_bypass_side(vx, vy, ep.side)
        hard = self._traj_params.hard_clearance_m
        front = float(rays.get("front", float("inf")) or float("inf"))
        chosen_clearance = float(
            (rays.get("right", float("inf")) if ep.side == 1
             else rays.get("left", float("inf"))) or float("inf")
        )

        if front < hard and vx > 0.0:
            vx = 0.0
        elif front < self._dead_end_bypass_release_front_m:
            # Keep the motion wall-follow dominant while the three-wall trap
            # is still nearby. A stale reverse family must not send the drone
            # back toward the map boundary after recovery has handed off.
            vx = max(0.0, min(vx, self._dead_end_bypass_front_speed_mps))

        # Never force lateral motion into the selected wall. Once there is a
        # little margin, avoid a zero/small tracker command that would leave
        # the vehicle oscillating at the mouth of the dead end.
        if chosen_clearance <= hard:
            vy = 0.0
        elif chosen_clearance >= hard + 0.20:
            vy = ep.side * min(
                self._dead_end_bypass_max_lateral_mps,
                max(abs(vy), self._dead_end_bypass_min_lateral_mps),
            )

        return vx, vy

    @staticmethod
    def _rejoin_path_error(position_xy: Tuple[float, float], cbmba_path) -> float:
        """Cross-track error: min perpendicular XY distance to the reference path.

        Uses segment (not waypoint) distance so a sparse path — e.g. a long
        straight final segment — reports a small error for a drone sitting on
        the path line between two far-apart waypoints.  Returns ``inf`` when
        the path is empty.
        """
        from planners.cbmba_astar import _point_to_segment_distance_xy

        if not cbmba_path:
            return float("inf")
        px, py = position_xy
        if len(cbmba_path) == 1:
            wp = cbmba_path[0]
            if wp is not None and len(wp) >= 2:
                return math.hypot(wp[0] - px, wp[1] - py)
            return float("inf")

        min_d = float("inf")
        for i in range(len(cbmba_path) - 1):
            a = cbmba_path[i]
            b = cbmba_path[i + 1]
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                continue
            d = _point_to_segment_distance_xy(
                [px, py], [a[0], a[1]], [b[0], b[1]],
            )
            if d < min_d:
                min_d = d
        return min_d

    @staticmethod
    def _rejoin_heading_error(st, mission_goal) -> float:
        """|bearing(goal) - yaw| wrapped to [0, π].

        Diagnostic ONLY — never an exit condition.  A small value merely means
        the nose happens to point at the goal, not that the drone has actually
        rejoined the reference path.
        """
        px = st.position_ned_m[0]
        py = st.position_ned_m[1]
        yaw = st.yaw_rad
        _he = math.atan2(mission_goal[1] - py, mission_goal[0] - px) - yaw
        _he = (_he + math.pi) % (2.0 * math.pi) - math.pi
        return abs(_he)

    @staticmethod
    def _wrapped_heading_error(position_ned, yaw_rad, goal_xy) -> float:
        """Signed shortest yaw error from the vehicle nose to ``goal_xy``.

        Positive means turn right/increase AirSim NED yaw; negative means turn
        left.  All values are radians.
        """
        dx = float(goal_xy[0]) - float(position_ned[0])
        dy = float(goal_xy[1]) - float(position_ned[1])
        bearing = math.atan2(dy, dx)
        return (bearing - float(yaw_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _align_heading_to_goal(self, state_reader, collision_reader, velocity_controller,
                               mission_goal) -> None:
        """Turn in place until the fixed mission goal is in front.

        This is deliberately a pre-navigation gate.  Once aligned, the normal
        trajectory planner remains responsible for obstacle avoidance and does
        not receive a continuous heading override while detouring.
        """
        if not self._heading_alignment_enabled:
            return

        st = state_reader.read()
        distance_xy = math.hypot(
            mission_goal[0] - st.position_ned_m[0],
            mission_goal[1] - st.position_ned_m[1],
        )
        if distance_xy < 1e-6:
            return
        initial_error = self._wrapped_heading_error(
            st.position_ned_m, st.yaw_rad, mission_goal,
        )
        if abs(initial_error) <= self._heading_alignment_trigger_rad:
            logger.info(
                "heading_alignment  state=not_needed  error_deg=%.1f  "
                "trigger_deg=%.1f",
                math.degrees(initial_error),
                math.degrees(self._heading_alignment_trigger_rad),
            )
            return

        logger.info(
            "heading_alignment  state=start  error_deg=%.1f  distance=%.2f  "
            "kp=%.2f  max_rate=%.2f",
            math.degrees(initial_error), distance_xy,
            self._heading_alignment_kp, self._heading_alignment_max_rate,
        )
        start = time.monotonic()
        last_error = initial_error
        while time.monotonic() - start < self._heading_alignment_timeout_s:
            st = state_reader.read()
            col = collision_reader.read()
            if col.has_collided:
                raise RuntimeError(
                    f"collision_during_heading_alignment:{col.object_name}"
                )
            error = self._wrapped_heading_error(
                st.position_ned_m, st.yaw_rad, mission_goal,
            )
            last_error = error
            if abs(error) <= self._heading_alignment_settle_rad:
                # Cancel the last rate command while preserving cruise altitude.
                velocity_controller.send_velocity_body_frd_z(
                    0.0, 0.0, self._params.target_z_ned,
                    duration=self._params.command_duration_s,
                    vehicle_name=self._vn,
                    yaw_rate=0.0,
                )
                logger.info(
                    "heading_alignment  state=complete  error_deg=%.1f  "
                    "elapsed=%.2f",
                    math.degrees(error), time.monotonic() - start,
                )
                return

            yaw_rate = self._heading_alignment_kp * error
            yaw_rate = max(
                -self._heading_alignment_max_rate,
                min(self._heading_alignment_max_rate, yaw_rate),
            )
            velocity_controller.send_velocity_body_frd_z(
                0.0, 0.0, self._params.target_z_ned,
                duration=self._params.command_duration_s,
                vehicle_name=self._vn,
                yaw_rate=yaw_rate,
            )
            self._log_throttled(
                "heading_alignment_progress", 1.0,
                "heading_alignment  state=turning  error_deg=%.1f  "
                "yaw_rate=%.2f",
                math.degrees(error), yaw_rate,
            )
            time.sleep(min(0.05, max(0.01, self._params.command_duration_s / 4.0)))

        # A timeout is a fail-safe: do not start horizontal navigation while the
        # desired heading is still unknown.  The outer cleanup path will land.
        velocity_controller.send_velocity_body_frd_z(
            0.0, 0.0, self._params.target_z_ned,
            duration=self._params.command_duration_s,
            vehicle_name=self._vn,
            yaw_rate=0.0,
        )
        raise RuntimeError(
            "heading_alignment_timeout: error_deg=%.1f timeout_s=%.1f"
            % (math.degrees(last_error), self._heading_alignment_timeout_s)
        )

    def _runtime_heading_alignment_command(
        self, state, mission_goal, now: float,
    ) -> Tuple[bool, bool, Optional[float]]:
        """Return ``(active, just_completed, yaw_rate)`` for runtime alignment.

        A trajectory is generated in world coordinates, but its tracker uses a
        forward body-frame command.  If the drone crosses the goal, that cached
        command can keep moving it away from MissionEnd.  Near the goal and
        with the goal clearly behind the nose, stop horizontal motion and turn
        in place.  Once aligned, invalidate the old trajectory so the next
        local plan is built for the new heading.
        """
        if not self._runtime_heading_alignment_enabled:
            return False, False, None

        distance_xy = math.hypot(
            float(mission_goal[0]) - float(state.position_ned_m[0]),
            float(mission_goal[1]) - float(state.position_ned_m[1]),
        )
        goal_dx = float(mission_goal[0]) - float(state.position_ned_m[0])
        goal_dy = float(mission_goal[1]) - float(state.position_ned_m[1])
        goal_forward = (
            goal_dx * math.cos(float(state.yaw_rad))
            + goal_dy * math.sin(float(state.yaw_rad))
        )
        goal_is_behind = (
            distance_xy > 1.0
            and goal_forward < -getattr(
                self, "_runtime_goal_behind_min_forward_m", 0.5
            )
        )
        if (
            not self._runtime_heading_alignment_active
            and distance_xy > self._runtime_heading_alignment_max_distance_m
            and not goal_is_behind
        ):
            return False, False, None

        error = self._wrapped_heading_error(
            state.position_ned_m, state.yaw_rad, mission_goal,
        )
        abs_error = abs(error)

        if not self._runtime_heading_alignment_active:
            if abs_error < self._runtime_heading_alignment_trigger_rad and not goal_is_behind:
                return False, False, None
            self._runtime_heading_alignment_active = True
            self._runtime_heading_alignment_started_mono = now
            # The cached blue line is no longer trustworthy after an
            # overshoot.  Do not let it run again while the vehicle turns.
            self._traj_cached_points = []
            self._traj_cached_family = None
            self._traj_narrow_passage_active = False
            self._traj_force_replan = True
            logger.warning(
                "runtime_heading_alignment  state=start  error_deg=%.1f  "
                "distance=%.2f  action=hover_and_turn",
                math.degrees(error), distance_xy,
            )

        # Keep the visualization and the tracker from showing/reusing a plan
        # while the vehicle is rotating.  A fresh plan is requested after the
        # settle frame below.
        if self._runtime_heading_alignment_active:
            self._traj_cached_points = []
            self._traj_cached_family = None
            self._traj_narrow_passage_active = False
            self._traj_force_replan = True

        if abs_error <= self._runtime_heading_alignment_settle_rad:
            self._runtime_heading_alignment_active = False
            self._runtime_heading_alignment_started_mono = None
            # Keep this frame stationary; the planner gets a clean frame on
            # the next tick and cannot reuse the pre-overshoot path.
            self._traj_cached_points = []
            self._traj_cached_family = None
            self._traj_narrow_passage_active = False
            self._traj_force_replan = True
            logger.info(
                "runtime_heading_alignment  state=complete  error_deg=%.1f  "
                "distance=%.2f  action=replan",
                math.degrees(error), distance_xy,
            )
            return False, True, 0.0

        yaw_rate = self._runtime_heading_alignment_kp * error
        yaw_rate = max(
            -self._runtime_heading_alignment_max_rate,
            min(self._runtime_heading_alignment_max_rate, yaw_rate),
        )
        self._log_throttled(
            "runtime_heading_alignment_progress", 1.0,
            "runtime_heading_alignment  state=turning  error_deg=%.1f  "
            "yaw_rate=%.2f  distance=%.2f",
            math.degrees(error), yaw_rate, distance_xy,
        )
        return True, False, yaw_rate

    def _freeze_reference_xy(
        self, position_xy: Tuple[float, float], path_world,
    ) -> Tuple[Tuple[Tuple[float, float], ...], str, Optional[int], Optional[Tuple[float, float]]]:
        """Freeze a stable XY reference path snapshot from a CBMBA ``path_world``.

        Returns ``(ref_xy, source, generation_id, first_xy)``.  ``path_world``
        is the planner output whose first waypoint is re-seeded from the start
        position at every replan; freezing a snapshot here (at BYPASS episode
        creation, BEFORE the drone deviates) is what keeps REJOIN's
        ``path_error`` from degenerating to ~0 against a live self-referential
        path.  Empty/None ``path_world`` yields an empty reference so callers
        never silently fall back to a live path.
        """
        if not path_world:
            return (), "", None, None
        ref_xy = tuple((float(p[0]), float(p[1])) for p in path_world)
        return ref_xy, "cbmba_path_world", self._cbmba_path_generation, ref_xy[0]

    @staticmethod
    def _reference_fingerprint(ref_xy) -> str:
        """Compact, run-stable identity string for a frozen reference snapshot.

        Deterministic (no Python ``hash()``, no time/randomness) so two logs
        carrying the same fingerprint provably refer to the SAME snapshot —
        used to show BYPASS's reference is bit-identical to REJOIN's.  Embeds
        first→last[count] plus a FNV-1a checksum over rounded coordinates.
        """
        if not ref_xy:
            return "empty"
        _h = 2166136261
        for _x, _y in ref_xy:
            for _v in (round(_x, 3), round(_y, 3)):
                for _b in str(_v).encode("utf-8"):
                    _h ^= _b
                    _h = (_h * 16777619) & 0xFFFFFFFF
        _first = ref_xy[0]
        _last = ref_xy[-1]
        return (
            f"({_first[0]:.3f},{_first[1]:.3f})->"
            f"({_last[0]:.3f},{_last[1]:.3f})[{len(ref_xy)}]:{_h:08x}"
        )

    def _should_exit_rejoin(
        self, st, mission_goal, cbmba_path, now: float,
    ) -> Tuple[bool, str]:
        """Decide whether to exit REJOIN back to NORMAL.

        Exit requires BOTH:
        1. ``elapsed >= min_duration_s`` — REJOIN must not exit on the same
           frame it entered (prevents enter→exit thrash).
        2. ``path_error < exit_path_error_m`` — measured against the FROZEN
           ``ep.reference_path_xy`` (NOT the live ``cbmba_path``, whose first
           waypoint is re-seeded from the current UAV position each replan and
           would make ``path_error`` trivially 0).

        ``cbmba_path`` is retained for diagnostics only (to observe whether the
        live path is anchored to the current position).  The goal-heading error
        is diagnostic only and can never trigger an exit on its own.
        """
        ep = self._rejoin
        if not ep.active:
            return True, "not_active"

        elapsed = now - ep.start_time
        if elapsed < ep.min_duration_s:
            return False, (
                f"dwell(elapsed={elapsed:.3f} < {ep.min_duration_s:.3f})"
            )

        path_error = self._rejoin_path_error(
            (st.position_ned_m[0], st.position_ned_m[1]), ep.reference_path_xy,
        )
        # P1-D: exit requires CONVERGENCE, not merely "inside the corridor".
        # A REJOIN entered with a near-zero start error (a bypass that never
        # actually deviated) would otherwise "exit" while the error is still
        # INCREASING (e.g. start=0.011 → now=0.023).  Require the error to have
        # dropped below BOTH the exit threshold AND its entry value.
        if path_error < ep.exit_path_error_m and path_error <= ep.start_path_error:
            return True, (
                f"path_error(={path_error:.3f} < {ep.exit_path_error_m:.3f}"
                f" and <= start={ep.start_path_error:.3f})"
            )

        return False, "hold"

    def _build_rejoin_from_bypass(
        self, position_xy: Tuple[float, float], start_time: float, reason: str,
    ) -> "RejoinEpisode":
        """BYPASS → REJOIN handoff: inherit the BYPASS episode's frozen reference.

        The reference was frozen at bypass_enter (BEFORE the drone deviated),
        so its first waypoint is NOT the current position.  This method copies
        that SAME snapshot into the new ``RejoinEpisode`` and computes
        ``start_path_error`` against it.  If the BYPASS episode has an empty
        reference (no valid path was ever available), the empty snapshot is
        preserved — we NEVER silently fall back to the live path here.
        """
        _byp_ep = self._bypass
        _ref_xy = _byp_ep.reference_path_xy
        _path_err = (
            self._rejoin_path_error(position_xy, _ref_xy)
            if _ref_xy else float("inf")
        )
        return RejoinEpisode(
            active=True,
            start_time=start_time,
            reason=reason,
            start_path_error=_path_err,
            reference_path_xy=_ref_xy,
            reference_source=_byp_ep.reference_source or "none",
            reference_generation_id=_byp_ep.reference_generation_id,
            reference_first_xy=_byp_ep.reference_first_xy,
            reference_frozen_position_xy=_byp_ep.reference_frozen_position_xy,
        )

    # ── trajectory-centric planning helpers ──

    @staticmethod
    def _downsample_xy(points, res_m: float):
        """Voxel-grid downsample of world-NED XY points (O(n))."""
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

    @staticmethod
    def _path_xy(path) -> List[Tuple[float, float]]:
        return [
            (float(wp[0]), float(wp[1]))
            for wp in path if wp is not None and len(wp) >= 2
        ]

    def _path_length_xy(self, path) -> float:
        xy = self._path_xy(path)
        if len(xy) < 2:
            return float("inf")
        return sum(
            math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
            for i in range(len(xy) - 1)
        )

    def _discover_mission_goal_actor(self):
        """Return ``(goal_ned_xyz, actor_name)`` for a MissionEnd actor, else None.

        Phase C1 item #1: the real MissionEnd actor placed in the AirSim world
        must win over any test-only fixed goal.  We probe
        ``simListSceneObjects`` + ``simGetObjectPose`` (both return world-NED
        metres, the same frame as the state reader).  Any RPC failure or a mock
        adapter degrades to ``None`` so the caller falls back to the fixed goal.
        """
        try:
            client = self._adapter.get_raw_client()
        except Exception:
            return None
        names = []
        try:
            names = client.simListSceneObjects(".*")
        except Exception:
            try:
                names = client.simListSceneObjects("MissionEnd.*")
            except Exception:
                return None
        if not isinstance(names, (list, tuple)):
            return None

        def _is_mission_end(name: str) -> bool:
            s = name.lower().replace("_", "").replace(" ", "")
            return "missionend" in s

        candidates = [n for n in names if isinstance(n, str) and _is_mission_end(n)]
        if not candidates:
            return None
        name = candidates[0]
        try:
            pose = client.simGetObjectPose(name)
            pos = pose.position
            goal = (float(pos.x_val), float(pos.y_val), float(pos.z_val))
        except Exception as exc:  # noqa: BLE001
            logger.warning("mission_goal_actor_pose_failed  actor=%s  err=%s", name, exc)
            return None
        return (goal, name)

    def _path_blocked_by_field(self, path) -> bool:
        """True if the cached global path's clearance is below hard clearance.

        Phase C3-R: the distance field now lives in the local planner **process**,
        so the main loop cannot rebuild it here.  Instead it reads the
        ``global_path_min_clearance`` the worker reported alongside its last
        plan (``_traj_global_path_min_clearance``).  Until the first plan lands,
        ``inf`` means "unknown" → not blocked (do not force a replan on empty
        knowledge).
        """
        if not path:
            return True
        _min_clear = self._traj_global_path_min_clearance
        if not math.isfinite(_min_clear):
            return False
        return _min_clear < self._traj_params.hard_clearance_m

    def _prune_global_path(self, path, drone_xy: Tuple[float, float]):
        """Drop passed waypoints (keep from the nearest waypoint onward)."""
        if not path or len(path) <= 2:
            return path
        best_i = 0
        best_d = float("inf")
        for i, wp in enumerate(path):
            if wp is None or len(wp) < 2:
                continue
            d = math.hypot(float(wp[0]) - drone_xy[0], float(wp[1]) - drone_xy[1])
            if d < best_d:
                best_d = d
                best_i = i
        # Never keep only the (already-reached) start point.
        start = max(best_i, 1)
        if start >= len(path):
            start = len(path) - 1
        return path[start:]

    def _record_cbmba_search(self, reason: str) -> None:
        """Bookkeep one CBMBA A* call and warn on any duplicate within a frame.

        Phase C1-R sec 6: the shadow CBMBA ran unconditionally every frame in
        trajectory mode, so a single frame could trigger 2+ A* searches.  This
        counter makes that observable and lets us keep the trajectory path to
        exactly one search per replan request.
        """
        self._cbmba_search_total += 1
        self._cbmba_searches_this_frame += 1
        if self._cbmba_searches_this_frame > 1:
            logger.warning(
                "CBMBA_DUPLICATE_SEARCH  frame_searches=%d  reason=%s",
                self._cbmba_searches_this_frame, reason,
            )

    def _tick_global_replan(
        self,
        st,
        mission_goal: Tuple[float, float, float],
        cbmba_obstacles: list,
        now: float,
    ) -> None:
        """Global reference-path replan tick (Phase C3-R).

        Runs every control frame but is internally rate-limited to
        ``global_replan_hz``.  The heavy CBMBA A* executes on a **separate
        process** (``CbmbaProcessWorker``); the loop only *requests* a replan
        and *reads* the latest finished result (non-blocking).  The
        "current-obstacle blocks the path" check uses the clearance the local
        planner worker reported for the cached global path
        (``_traj_global_path_min_clearance``) instead of rebuilding a distance
        field in the main loop.
        """
        pos = (st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2])
        drone_xy = (pos[0], pos[1])

        # ── current-obstacle block → immediate global replan ──
        if self._traj_global_path and not self._traj_global_replan_requested:
            _min_clear = self._traj_global_path_min_clearance
            if math.isfinite(_min_clear) and _min_clear < self._traj_params.hard_clearance_m:
                self._traj_global_replan_requested = True
                logger.info(
                    "trajectory_global_replan  reason=current_lidar_block  "
                    "min_clear=%.2f  hard_clearance=%.2f",
                    _min_clear, self._traj_params.hard_clearance_m,
                )

        # ── global reference path replan (process-decoupled; rate-limited) ──
        replan_interval = 1.0 / max(0.1, self._traj_global_replan_hz)
        _worker = self._global_planner_worker
        if _worker is not None:
            _needs_replan = (
                not self._traj_global_path
                or now - self._traj_last_replan_time >= replan_interval
                or self._traj_global_replan_requested
            )
            if _needs_replan:
                # Phase C3-R: ≤1 request in flight + dedup "initial".  The first
                # "initial" request stays pending until its result is APPLIED; a
                # second "initial" before then would only overwrite the pending
                # slot.  The dedicated flag closes the request-before-apply gap.
                if self._initial_replan_requested and not self._traj_global_path:
                    logger.info(
                        "cbmba_replan_request_skipped  reason=initial_already_requested",
                    )
                elif _worker.has_in_flight_request():
                    logger.info(
                        "cbmba_replan_request_skipped  reason=request_already_in_flight",
                    )
                else:
                    # Phase C5-R: obstacle snapshot build (the only real prep
                    # work here) is timed separately from the enqueue/pickle.
                    _prep_t0 = time.perf_counter()
                    _combined = cbmba_obstacles + self._map_obstacles(pos[2])
                    self._cbmba_obstacle_snapshot_ms = (time.perf_counter() - _prep_t0) * 1000.0
                    self._cbmba_request_prepare_ms = self._cbmba_obstacle_snapshot_ms
                    _reason = ("initial" if not self._traj_global_path else "scheduled")
                    _prev_coalesced = _worker.coalesced_count
                    _put_t0 = time.perf_counter()
                    _rid = _worker.request_replan(
                        _combined,
                        [pos[0], pos[1], pos[2]],
                        [mission_goal[0], mission_goal[1], mission_goal[2]],
                        reason=_reason,
                    )
                    self._cbmba_request_put_ms = (time.perf_counter() - _put_t0) * 1000.0
                    if _worker.coalesced_count > _prev_coalesced:
                        logger.info(
                            "cbmba_request_coalesced  request_id=%d  reason=%s",
                            _rid, _reason,
                        )
                    else:
                        logger.info(
                            "cbmba_request_submitted  request_id=%d  reason=%s",
                            _rid, _reason,
                        )
                    if _reason == "initial":
                        self._initial_replan_requested = True
                    self._traj_last_replan_time = now
                    self._traj_global_replan_requested = False

            # Apply the latest finished worker result (non-blocking read).
            # A result with request_id < applied is stale (an older request that
            # finished late); ignore it and record.  The applied id only ever
            # increases, so a stale result can never overwrite a newer path.
            _wres = _worker.get_latest_result()
            _action = (
                _replan_result_action(
                    _wres["request_id"], self._traj_applied_replan_id
                )
                if _wres is not None else "noop"
            )
            if _action == "ignore_stale":
                logger.info(
                    "cbmba_stale_result_ignored  request_id=%d  applied=%d",
                    _wres["request_id"], self._traj_applied_replan_id,
                )
            if _action == "apply":
                self._traj_applied_replan_id = _wres["request_id"]
                self._cbmba_search_total = _worker.search_count
                self._initial_replan_requested = False
                logger.info(
                    "cbmba_result_applied  request_id=%d  success=%s  "
                    "path_points=%d  planning_time_ms=%.2f",
                    _wres["request_id"],
                    "true" if _wres.get("success") else "false",
                    len(_wres.get("path_world", [])) if _wres.get("success") else 0,
                    _wres.get("planning_time_ms", 0.0),
                )
                if _wres["success"] and len(_wres["path_world"]) >= 2:
                    new_path = self._prune_global_path(_wres["path_world"], drone_xy)
                    new_len = self._path_length_xy(new_path)
                    old_len = self._path_length_xy(self._traj_global_path)
                    old_blocked = self._path_blocked_by_field(self._traj_global_path)
                    _imp_pct = (
                        (old_len - new_len) / old_len * 100.0 if old_len > 0 else 0.0
                    )
                    if (not self._traj_global_path
                            or old_blocked
                            or new_len < old_len * (1.0 - self._traj_path_switch_min_improvement)):
                        _reason = ("empty_path" if not self._traj_global_path
                                   else ("old_path_blocked" if old_blocked
                                         else "shorter_path"))
                        self._traj_global_path = new_path
                        self._traj_global_path_version += 1
                        # Path changed → cached clearance is for the OLD path.
                        self._traj_global_path_min_clearance = float("inf")
                        logger.info(
                            "trajectory_global_replan  waypoints=%d  nodes=%d  "
                            "time_ms=%.2f  version=%d  path_len=%.2f  "
                            "old_len=%.2f  improvement_pct=%.2f  "
                            "old_blocked=%s  switched=true  reason=%s  request_id=%d",
                            len(new_path), _wres["nodes_expanded"],
                            _wres["planning_time_ms"],
                            self._traj_global_path_version, new_len,
                            old_len, _imp_pct,
                            "true" if old_blocked else "false", _reason,
                            _wres["request_id"],
                        )
                    else:
                        logger.info(
                            "trajectory_global_replan  switched=false  "
                            "reason=insufficient_improvement  "
                            "new_len=%.2f  old_len=%.2f  improvement_pct=%.2f  "
                            "min_improvement=%.2f  request_id=%d",
                            new_len, old_len, _imp_pct,
                            self._traj_path_switch_min_improvement,
                            _wres["request_id"],
                        )
        else:
            # Defensive synchronous fallback (no worker) — keeps the old path.
            if (not self._traj_global_path
                    or now - self._traj_last_replan_time >= replan_interval
                    or self._traj_global_replan_requested):
                try:
                    combined = list(cbmba_obstacles) + list(self._map_obstacles(pos[2]))
                    self._record_cbmba_search("trajectory_global_replan_sync")
                    res = self._cbmba.plan_with_result(
                        combined,
                        [pos[0], pos[1], pos[2]],
                        [mission_goal[0], mission_goal[1], mission_goal[2]],
                    )
                    self._traj_last_replan_time = now
                    self._traj_global_replan_requested = False
                    if res.success and len(res.path_world) >= 2:
                        new_path = self._prune_global_path(res.path_world, drone_xy)
                        self._traj_global_path = new_path
                        self._traj_global_path_version += 1
                        logger.info(
                            "trajectory_global_replan  waypoints=%d  nodes=%d  "
                            "time_ms=%.2f  version=%d  switched=true  reason=sync",
                            len(new_path), res.nodes_expanded, res.planning_time_ms,
                            self._traj_global_path_version,
                        )
                except Exception as e:
                    logger.warning("trajectory_global_replan_error: %s", e)

    def _request_local_plan(
        self,
        st,
        mission_goal: Tuple[float, float, float],
        filtered_points,
    ) -> None:
        """Send a compact snapshot to the local-trajectory planner **process**.

        Non-blocking: the snapshot is queued and the loop continues immediately.
        The worker owns the distance field + planner + ``TrajectoryMemory``, so
        only a picklable snapshot crosses the boundary (never a planner
        instance, never the map).
        """
        _worker = self._local_traj_worker
        if _worker is None:
            return
        # Phase C5-R: the LiDAR points are a shared numpy array passed by
        # reference (no per-worker Python copy); only the global-path list is
        # snapshot-copied.  Time path copy and dict build separately from the
        # enqueue/pickle so a slow step can be isolated.
        _prep_t0 = time.perf_counter()
        _path_copy_t0 = time.perf_counter()
        _path_copy = list(self._traj_global_path)
        self._traj_path_copy_ms = (time.perf_counter() - _path_copy_t0) * 1000.0
        _snapshot = {
            "drone_position_ned": [
                st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2],
            ],
            "yaw_rad": float(st.yaw_rad),
            "goal_xy": [mission_goal[0], mission_goal[1]],
            "global_path": _path_copy,
            "global_path_version": int(self._traj_global_path_version),
            "lidar_points": filtered_points,
        }
        self._traj_snapshot_build_ms = (time.perf_counter() - _prep_t0) * 1000.0
        self._traj_request_prepare_ms = self._traj_snapshot_build_ms
        _put_t0 = time.perf_counter()
        _rid = _worker.request_plan(_snapshot)
        self._traj_request_put_ms = (time.perf_counter() - _put_t0) * 1000.0
        logger.info(
            "trajectory_worker_request  request_id=%d  lidar_points=%d  "
            "global_path_points=%d",
            _rid, len(filtered_points), len(self._traj_global_path),
        )

    def _poll_local_plan(self):
        """Non-blocking poll of the local planner worker's latest result.

        Returns a ``TrajectoryPlanResult`` only when a *newer* result (higher
        request id) has finished; otherwise returns ``None`` and the loop keeps
        using the cached trajectory.  Also caches the reported global-path
        clearance for ``_path_blocked_by_field``.
        """
        _worker = self._local_traj_worker
        if _worker is None:
            return None
        _env = _worker.poll_latest_result()
        if _env is None:
            return None
        _rid = _env.get("request_id", -1)
        if _rid <= self._traj_local_plan_seq:
            return None  # already applied (or stale)
        self._traj_local_plan_seq = _rid
        self._traj_global_path_min_clearance = _env.get(
            "global_path_min_clearance", float("inf"),
        )
        _result = _env.get("result")
        if _result is None:
            logger.warning(
                "local_plan_worker_error  request_id=%d  error=%s",
                _rid, _env.get("error", ""),
            )
            return None
        return _result

    def _request_map_update(self, lf, st, fr) -> bool:
        """Non-blocking: send the current LiDAR frame to the mapping process.

        Dedups on the LiDAR frame receive timestamp so a 20 Hz control loop
        re-using a ~10 Hz LiDAR snapshot never runs ``update()`` twice on the
        same sensor frame.  Returns True if a request was actually submitted.
        """
        _worker = self._mapping_worker
        if _worker is None:
            return False
        _ts = getattr(lf, "received_monotonic_seconds", None)
        if _ts is None or _ts == self._last_map_sensor_timestamp:
            return False
        self._last_map_sensor_timestamp = _ts
        _pts = fr.filtered_points_sensor if fr is not None else None
        _prev_coalesced = _worker.coalesced_count
        # Phase C5-R: `_pts` is the shared numpy array (no copy); only the
        # snapshot dict is built here, timed separately from the enqueue/pickle.
        _prep_t0 = time.perf_counter()
        _snapshot = {
            "sensor_timestamp": _ts,
            "drone_position_ned": [
                st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2],
            ],
            "yaw_rad": float(st.yaw_rad),
            "points_sensor": _pts,
        }
        self._map_snapshot_build_ms = (time.perf_counter() - _prep_t0) * 1000.0
        self._map_request_prepare_ms = self._map_snapshot_build_ms
        _put_t0 = time.perf_counter()
        _rid = _worker.request_update(_snapshot)
        self._map_request_put_ms = (time.perf_counter() - _put_t0) * 1000.0
        if _worker.coalesced_count > _prev_coalesced:
            logger.info(
                "map_update_coalesced  request_id=%d  sensor_timestamp=%.3f",
                _rid, _ts,
            )
        else:
            logger.info(
                "map_update_submitted  request_id=%d  sensor_timestamp=%.3f",
                _rid, _ts,
            )
        return True

    def _poll_map_update(self) -> None:
        """Non-blocking: apply the latest finished map snapshot (monotonic version)."""
        _worker = self._mapping_worker
        if _worker is None:
            return
        _res = _worker.get_latest_result()
        if _res is None:
            return
        _mv = _res.get("map_version", -1)
        if _mv <= self._last_applied_map_version:
            return
        self._last_applied_map_version = _mv
        self._map_snapshot = _res
        logger.info(
            "map_update_applied  request_id=%d  map_version=%d  "
            "obstacle_count=%d  compute_ms=%.2f",
            _res.get("request_id", -1), _mv,
            _res.get("obstacle_count", 0), _res.get("compute_ms", 0.0),
        )

    def _map_obstacles(self, z_ned: float) -> list:
        """Convert the latest map snapshot's occupied points into CBMBA obstacles.

        Mirrors ``OccupancyGridMap.to_obstacles`` exactly, but reads the compact
        ``occupied_points`` the mapping process returned (no grid on this side).
        """
        half = self._occ_grid_params.resolution_m / 2.0
        return [
            {
                "position": [x, y, z_ned],
                "footprint_half_extents": [half, half, half],
                "type": "map",
                "velocity": [0.0, 0.0, 0.0],
                "dynamic": False,
                "confidence": 0.9,
            }
            for (x, y) in self._map_snapshot.get("occupied_points", [])
        ]

    def _record_dispatch_source(self, command_source: str) -> None:
        """Classify one dispatched command for the realtime summary.

        Only ``trajectory`` counts as a trajectory dispatch; every hold/hover
        source is counted into the appropriate hover bucket (sec 19/26).
        """
        if command_source == "trajectory":
            self._trajectory_dispatch_count += 1
        elif command_source == "control_loop_hover":
            self._hover_dispatch_count += 1
            self._hover_due_control_overrun += 1
        elif command_source == "trajectory_stale":
            self._hover_dispatch_count += 1
            self._hover_due_trajectory_stale += 1
        elif command_source == "trajectory_no_feasible":
            self._hover_dispatch_count += 1
            self._hover_due_no_feasible += 1
        elif command_source == "recovery":
            pass  # recovery is a maneuver, not a hover
        else:
            self._hover_dispatch_count += 1
            self._hover_due_other += 1

    def _record_perception_stale_hover(self) -> None:
        """Perception-stale holds happen before dispatch (a ``continue``), so they
        are counted here, not in ``_record_dispatch_source``."""
        self._hover_dispatch_count += 1
        self._hover_due_perception_stale += 1

    @staticmethod
    def _swept_command_clearance(
        points_sensor,
        vx: float,
        vy: float,
        command_duration_s: float,
        horizontal_band_m: float = 1.0,
    ) -> float:
        """Estimate clearance from the current body-frame command segment."""
        speed = math.hypot(vx, vy)
        if points_sensor is None or speed < 1e-6:
            return float("inf")
        try:
            count = len(points_sensor)
        except TypeError:
            return float("inf")
        if count == 0:
            return float("inf")

        # Include the command duration plus a short response margin. Downsample
        # only for this final guard; the planner still receives the full scan.
        lookahead = max(0.65, speed * max(0.15, command_duration_s) + 0.35)
        ux, uy = vx / speed, vy / speed
        step = max(1, count // 1000)
        best = float("inf")
        for row in points_sensor[::step]:
            try:
                sx, sy, sz = float(row[0]), float(row[1]), float(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(sz) > horizontal_band_m:
                continue
            along = sx * ux + sy * uy
            if along < 0.0 or along > lookahead:
                continue
            cross = abs(sx * uy - sy * ux)
            if cross < best:
                best = cross
        return best

    def _apf_safety_filter(
        self, vx: float, vy: float, rays: Dict[str, float], minimum_distance_m: float,
        points_sensor=None, preserve_centerline: bool = False,
    ) -> Tuple[float, float]:
        """Limited APF safety layer over a selected trajectory command.

        The trajectory planner already decided the direction; this is a
        **safety filter**, never a second navigator.  It may only:
          (a) slow forward motion (bounded by ``apf_max_speed_reduction_ratio``),
          (b) add a small lateral repulsion (bounded by
              ``apf_max_lateral_correction_mps``), and
          (c) emergency-stop when the obstacle is inside ``emergency_distance_m``.

        It MUST NOT reverse the trajectory's lateral sign — if the small nudge
        would cross zero, the lateral command is clamped to 0 (direction is
        preserved, never flipped).
        """
        p = self._traj_params
        emerg = self._params.emergency_distance_m

        # Sector medians can miss a pillar between sector centres. Check the
        # actual point cloud along the command direction before dispatching.
        swept_clearance = self._swept_command_clearance(
            points_sensor, vx, vy, self._params.command_duration_s,
            self._occ_grid_params.horizontal_band_half_height_m,
        )
        if math.isfinite(swept_clearance):
            # A close point behind or beside the commanded swept corridor must
            # not stop a vehicle that is correctly threading a gap. Only the
            # corridor itself uses the emergency threshold.
            if swept_clearance < emerg:
                return 0.0, 0.0
        elif minimum_distance_m < emerg:
            # Preserve the old fail-safe when no point cloud is available.
            return 0.0, 0.0
        if math.isfinite(swept_clearance) and swept_clearance < p.hard_clearance_m + 0.20:
            scale = max(
                0.25,
                min(1.0, (swept_clearance - emerg) / max(0.05, p.hard_clearance_m + 0.20 - emerg)),
            )
            vx *= scale
            vy *= scale

        front = rays.get("front", float("inf")) or float("inf")
        left = rays.get("left", float("inf")) or float("inf")
        right = rays.get("right", float("inf")) or float("inf")
        ft = self._params.front_threshold_m

        # (a) forward speed reduction, clamped to keep ≥ ratio of the command.
        if front < ft:
            scale = max(0.0, min(1.0, (front - emerg) / max(0.01, ft - emerg)))
            ratio = p.apf_max_speed_reduction_ratio
            vx *= ratio + (1.0 - ratio) * scale
        # A speed reduction alone is not a collision barrier. Enforce the
        # local planner hard-clearance contract at the final command boundary.
        if front < p.hard_clearance_m and vx > 0.0:
            vx = 0.0

        # (b) small lateral repulsion, bounded by the correction cap.
        nudge = 0.0
        if not preserve_centerline:
            if left < 1.0 and left < right:
                nudge = p.apf_max_lateral_correction_mps * (1.0 - left)
            elif right < 1.0 and right < left:
                nudge = -p.apf_max_lateral_correction_mps * (1.0 - right)
        vy_filtered = vy + nudge

        # (c) sign guard — never reverse the trajectory's chosen lateral side.
        if vy != 0.0 and vy_filtered * vy < 0.0:
            vy_filtered = 0.0

        max_lat = p.lateral_speed_mps
        vy_filtered = max(-max_lat, min(max_lat, vy_filtered))
        return vx, vy_filtered

    def _recovery_directional_guard(
        self, vx: float, vy: float, rays: Dict[str, float]
    ) -> Tuple[float, float]:
        """Prevent recovery from commanding into a measured obstacle face."""
        hard = self._traj_params.hard_clearance_m
        front = float(rays.get("front", float("inf")) or float("inf"))
        back = float(rays.get("back", float("inf")) or float("inf"))
        left = float(rays.get("left", float("inf")) or float("inf"))
        right = float(rays.get("right", float("inf")) or float("inf"))
        if front < hard and vx > 0.0:
            vx = 0.0
        if back < hard and vx < 0.0:
            vx = 0.0
        if left < hard and vy < 0.0:
            vy = 0.0
        if right < hard and vy > 0.0:
            vy = 0.0
        return vx, vy

    def _traj_end_body_y(self, points, st) -> float:
        """Body-frame lateral (Y) offset of the trajectory endpoint from the drone.

        Body FRD: +Y = right.  LEFT trajectories end with a negative value.
        """
        if not points:
            return 0.0
        ex, ey = points[-1]
        dx = ex - st.position_ned_m[0]
        dy = ey - st.position_ned_m[1]
        yaw = st.yaw_rad
        return -dx * math.sin(yaw) + dy * math.cos(yaw)

    def _log_throttled(self, key: str, interval_s: float, msg: str, *args) -> None:
        """Log an INFO line at most once per ``interval_s`` per ``key``.

        Windows console stdout is a synchronous, blocking write; several
        per-frame diagnostic INFO logs would otherwise dominate the control
        loop's frame budget at 20 Hz (sec 19/20).  Warnings/errors are NOT
        throttled — only this helper is, and it only handles INFO.
        """
        now = time.monotonic()
        store = getattr(self, "_rate_limit_store", None)
        if store is None:
            store = {}
            self._rate_limit_store = store
        last = store.get(key)
        if last is not None and (now - last) < interval_s:
            return
        store[key] = now
        logger.info(msg, *args)

    @staticmethod
    def _altitude_hold_velocity(
        current_z: float, target_z: float, max_speed: float, gain: float = 1.0,
    ) -> float:
        """Return a bounded NED vertical velocity toward ``target_z``.

        NED Z is positive downward, so the error is ``target - current``.
        Using ``current - target`` would command a descent when the vehicle is
        already below the requested altitude.
        """
        error = float(target_z) - float(current_z)
        limit = max(0.0, float(max_speed))
        return max(-limit, min(limit, error * float(gain)))

    def _sleep_to_next_period(self, next_tick: float):
        """Advance the deadline scheduler and sleep to the next period boundary.

        No-catch-up semantics (Phase C5-R / C6-R): a fixed-rate control loop is
        a *maximum* cadence, not a debt to be repaid.  On a missed deadline the
        scheduler skips the missed periods and re-anchors one full period ahead
        of ``now``, then sleeps that full period — so the *next* control tick is
        spaced ~``period`` after the resync.  It never tight-loops ``sleep=0``
        to "catch up" on late ticks (that starves the GIL, the PerceptionWorker
        thread and the simulator).

        Returns ``(next_tick, sleep_ms, deadline_late_ms, missed_periods,
        resynced)``.  ``missed_periods`` is the number of full period boundaries
        that passed since the deadline (0 when on time); ``resynced`` is True
        only when the deadline was re-anchored.
        """
        period = self._control_period_s
        now = time.monotonic()
        next_tick += period
        sleep_s = next_tick - now
        if sleep_s >= 0:
            deadline_late_ms = 0.0
            sleep_ms = sleep_s * 1000.0
            missed_periods = 0
            resynced = False
            time.sleep(sleep_s)
        else:
            deadline_late_ms = -sleep_s * 1000.0
            # Number of full period boundaries between the (advanced) deadline
            # and now.  ``floor`` (not ``ceil``) counts only whole missed
            # periods; the +1e-9 guards the float ratio against a 3.000…00004
            # representation of an exact 3.0.
            missed_periods = int(math.floor((-sleep_s) / period + 1e-9))
            # Re-anchor one full period ahead of now and sleep that period, so
            # the next loop-start is ≥ period after the resync (no immediate
            # tight tick that would appear as a post-resync <40ms interval).
            sleep_ms = period * 1000.0
            next_tick = now + period
            resynced = True
            time.sleep(period)
        return next_tick, sleep_ms, deadline_late_ms, missed_periods, resynced

    def _update_trajectory_tracking(self, st, cmd_vx: float, cmd_vy: float) -> None:
        """Planned-vs-actual displacement error; force replan when too large."""
        pos = (st.position_ned_m[0], st.position_ned_m[1])
        if self._traj_last_track_pose is not None:
            prev_x, prev_y = self._traj_last_track_pose
            actual_delta = math.hypot(pos[0] - prev_x, pos[1] - prev_y)
            planned_delta = (
                math.hypot(cmd_vx, cmd_vy) * self._params.command_duration_s
            )
            error_m = abs(planned_delta - actual_delta)
            self._traj_tracking_error_sum += error_m
            self._traj_tracking_error_max = max(self._traj_tracking_error_max, error_m)
            self._traj_tracking_error_n += 1
            self._log_throttled(
                "trajectory_tracking", 1.0,
                "trajectory_tracking  planned_delta=%.3f  actual_delta=%.3f  "
                "error_m=%.3f  threshold=%.3f",
                planned_delta, actual_delta, error_m,
                self._traj_tracking_error_threshold_m,
            )
            if error_m > self._traj_tracking_error_threshold_m:
                self._traj_force_replan = True
        self._traj_last_track_pose = pos
        self._traj_last_track_cmd = (cmd_vx, cmd_vy)

    # ── main flight loop ──

    def run(self) -> AutomaticFlightResult:
        rk: Dict[str, Any] = dict(
            success=False, termination_reason="unknown",
            frames_completed=0, flight_duration_s=0.0,
            api_control_acquired=False, armed=False,
            takeoff_completed=False, airborne=False,
            landing_confirmed=False, disarmed=False,
            api_control_released=False,
            startup_floor_contact_baseline=False,
        )

        try:
            # ────────────────────────────────────────────────
            # PREFLIGHT (BEFORE enableApiControl — all checks here)
            # ────────────────────────────────────────────────

            from sensors.lidar_reader import LidarReader
            from sensors.state_reader import StateReader
            from sensors.collision_reader import CollisionReader

            lidar = LidarReader(self._adapter)
            sr = StateReader(self._adapter, vehicle_name=self._vn)
            cr = CollisionReader(self._adapter, vehicle_name=self._vn)

            # 1. Drone1 existence (via state read)
            try:
                st0 = sr.read()
            except Exception as e:
                rk["termination_reason"] = f"preflight_state:{e}"
                return AutomaticFlightResult(**rk)

            # 2. LiDAR consecutive valid frames
            pf = self._params.preflight_lidar_frames
            for i in range(pf):
                try:
                    lf = lidar.read()
                except Exception as e:
                    rk["termination_reason"] = f"preflight_lidar_read_{i}:{e}"
                    return AutomaticFlightResult(**rk)
                if not lf.frame_valid:
                    rk["termination_reason"] = f"preflight_lidar_{i}:{lf.invalid_reason}"
                    return AutomaticFlightResult(**rk)

            # 3. Collision warm-up (before enableApiControl)
            #    First ground contact → accepted as startup candidate
            #    (even with non-zero ts or is_new_collision_event=True).
            #    Subsequent new collision events → reject.
            #
            #    Ground object names vary by UE environment:
            #      - "Floor" / "Floor_3" : named floor meshes (reference env)
            #      - "(null)" / ""       : unnamed ground (AirSim reports a
            #                              nameless collision object as "(null)")
            _WARMUP_MAX = 10
            _WARMUP_INTERVAL = 0.15
            _WARMUP_CLEAR = 5
            _FLOOR_OK = frozenset({"Floor", "Floor_3", "(null)", ""})
            saw_floor = False
            cons_clean = 0
            saw_new_event_after_first = False
            initial_floor_ts = 0
            for i in range(_WARMUP_MAX):
                try:
                    col = cr.read()
                except Exception as e:
                    rk["termination_reason"] = f"warmup_read_error_{i}:{e}"
                    return AutomaticFlightResult(**rk)
                if col.has_collided:
                    if col.object_name not in _FLOOR_OK:
                        rk["termination_reason"] = f"warmup_non_ground_{i}:{col.object_name}"
                        return AutomaticFlightResult(**rk)
                    # Floor/Floor_3 contact
                    if not saw_floor:
                        # First floor contact — always accept as candidate
                        saw_floor = True
                        initial_floor_ts = col.raw_timestamp
                    elif col.is_new_collision_event and col.raw_timestamp != initial_floor_ts:
                        # Subsequent new collision event (different ts) → reject
                        saw_new_event_after_first = True
                    cons_clean = 0
                else:
                    cons_clean += 1
                if cons_clean >= _WARMUP_CLEAR:
                    break
                time.sleep(_WARMUP_INTERVAL)
            if saw_new_event_after_first:
                rk["termination_reason"] = "warmup_new_collision_event"
                return AutomaticFlightResult(**rk)
            if saw_floor and cons_clean < _WARMUP_CLEAR:
                rk["termination_reason"] = "warmup_floor_persists"
                return AutomaticFlightResult(**rk)
            if saw_floor:
                rk["startup_floor_contact_baseline"] = True
                logger.info("Startup floor contact baseline established (ts=%d).", initial_floor_ts)
                # Pass startup timestamp to session for landing-phase floor latch detection
                self._session.set_startup_floor_baseline(initial_floor_ts)

            # 4. FOV compatibility (NOT hardcoded)
            from perception.perception_config import load_perception_config
            from perception.sensor_fov import load_lidar_fov, validate_sector_fov_coverage

            try:
                pcfg = load_perception_config(self._perception_config_path)
            except Exception as e:
                rk["termination_reason"] = f"preflight_perception_config:{e}"
                return AutomaticFlightResult(**rk)

            try:
                fov = load_lidar_fov(self._session.settings_json, self._vn, self._adapter.lidar_name)
            except Exception as e:
                rk["termination_reason"] = f"preflight_fov_load:{e}"
                return AutomaticFlightResult(**rk)

            fov_results = validate_sector_fov_coverage(pcfg, fov)
            incompatible = [s for s in fov_results if not s.fully_observable]
            if incompatible:
                names = [s.legacy_name for s in incompatible]
                rk["termination_reason"] = f"preflight_fov_incompatible:{names}"
                return AutomaticFlightResult(**rk)

            # 5. Perception config valid (already loaded above)
            sz_cfg = pcfg.sectorization
            pc_cfg = pcfg.pointcloud
            sdefs = list(sz_cfg.sectors)

            # Build FOV observability map from real validation results
            fov_obs = {}
            for sts in fov_results:
                for sd in sdefs:
                    if sd.legacy_name == sts.legacy_name:
                        fov_obs[sd.name] = (sts.fully_observable, 1.0)

            # 6. minimal_flight.yaml already loaded via AutomaticModeParams
            logger.info("Preflight passed — all checks OK.")

            # ────────────────────────────────────────────────
            # TAKEOFF (enableApiControl happens HERE, inside session)
            # ────────────────────────────────────────────────
            self._session.takeoff_and_climb(target_z=self._params.target_z_ned)
            rk["api_control_acquired"] = True
            rk["armed"] = True
            rk["takeoff_completed"] = True
            rk["airborne"] = True
            logger.info("Airborne — starting LiDAR control loop.")

            # ── initialise forward-progress watchdog ──
            self._progress_watchdog.reset(
                time.monotonic(),
                (st0.position_ned_m[0], st0.position_ned_m[1]),
            )

            # ── perception pipeline (post-takeoff) ──
            from perception.pointcloud_filter import filter_pointcloud
            from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
            from control.velocity_controller import VelocityController

            vc = VelocityController(
                self._adapter,
                max_horizontal_speed_mps=self._params.forward_speed_mps,
                # The trajectory tracker supplies a vertical-velocity hold
                # command in NED.  Zero here silently clamped every computed
                # vz to 0 inside VelocityController, so the vehicle drifted
                # away from target_z despite the correct P-control sign.
                max_vertical_speed_mps=self._params.max_vertical_speed_mps,
                command_duration_seconds=self._params.command_duration_s,
            )

            # ── background LiDAR/perception worker (Phase C2 sec 10-14) ──
            # Own an independent read-only AirSim client so the slow
            # getLidarData RPC + filter/sectorisation never block the control
            # loop.  The loop reads a published snapshot instead of calling
            # lidar.read() synchronously.
            def _perceive(_lf):
                """LiDAR frame → (filter_result, sector_result, rays)."""
                if _lf is None or not _lf.frame_valid:
                    return (None, None, None)
                _fr = filter_pointcloud(
                    _lf.point_cloud_sensor,
                    min_range_m=pc_cfg.min_range_m, max_range_m=pc_cfg.max_range_m,
                    self_exclusion={
                        "enabled": pc_cfg.self_exclusion.enabled,
                        "x_min_m": pc_cfg.self_exclusion.x_min_m,
                        "x_max_m": pc_cfg.self_exclusion.x_max_m,
                        "y_min_m": pc_cfg.self_exclusion.y_min_m,
                        "y_max_m": pc_cfg.self_exclusion.y_max_m,
                        "z_min_m": pc_cfg.self_exclusion.z_min_m,
                        "z_max_m": pc_cfg.self_exclusion.z_max_m,
                    },
                    voxel_downsample=pc_cfg.voxel_downsample.enabled,
                    voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m,
                )
                if not _fr.valid:
                    return (_fr, None, None)
                _dd = pointcloud_to_directional_distances(
                    _fr.filtered_points_sensor, sector_defs=sdefs,
                    default_max_range_m=sz_cfg.default_max_range_m,
                    default_min_points=sz_cfg.default_min_points,
                    distance_strategy=sz_cfg.default_distance_strategy,
                    nearest_k=sz_cfg.nearest_k, percentile=sz_cfg.percentile,
                    frame_valid=True, fov_compatible=len(incompatible) == 0,
                    fov_observability=fov_obs,
                )
                if not _dd.frame_valid:
                    return (_fr, _dd, None)
                _rays = _dd.to_legacy_ray_distances()
                return (_fr, _dd, _rays)

            _perc_adapter = self._adapter.clone_readonly()
            _perc_adapter.connect()
            _perc_lidar = LidarReader(_perc_adapter)
            self._perception_worker = PerceptionWorker(
                _perc_lidar, _perceive, poll_hz=self._perception_poll_hz,
            )
            self._perception_worker.start()
            logger.info(
                "perception_worker  state=started  poll_hz=%.1f  "
                "stale_warn_s=%.2f  stale_stop_s=%.2f",
                self._perception_poll_hz,
                self._perception_stale_warn_s,
                self._perception_stale_stop_s,
            )

            # Re-read state for spawn reference
            st0 = sr.read()
            spawn = (st0.position_ned_m[0], st0.position_ned_m[1])

            # ── altitude startup gate (Phase C3-R) ──
            # If climb confirmation failed (drone settled short of target_z),
            # hold horizontal velocity at 0 with an altitude-position command
            # until the drone reaches target_z (within tolerance) or the gate
            # times out.  Only then is navigation cleared to move laterally.
            # This uses the SAME moveByVelocityZBodyFrameAsync altitude-hold
            # (target_z_ned unchanged, Z sign unchanged) as normal dispatch.
            _gate_tolerance_m = 0.3
            _gate_timeout_s = 10.0
            self._navigation_ready = bool(
                getattr(self._session, "altitude_confirmed", False)
            )
            if not self._navigation_ready:
                logger.warning(
                    "altitude_startup_gate  state=enter  target_z=%.2f  "
                    "confirmed=false  tolerance=%.2f  timeout=%.1f",
                    self._params.target_z_ned, _gate_tolerance_m, _gate_timeout_s,
                )
                _gate_start = time.monotonic()
                while time.monotonic() - _gate_start < _gate_timeout_s:
                    try:
                        vc.send_velocity_body_frd_z(
                            0.0, 0.0, self._params.target_z_ned,
                            duration=self._params.command_duration_s,
                            vehicle_name=self._vn,
                        )
                        _gst = sr.read()
                        _gz = _gst.position_ned_m[2]
                    except Exception as _gate_exc:
                        logger.warning("altitude_startup_gate  poll_error: %s", _gate_exc)
                        break
                    if abs(_gz - self._params.target_z_ned) <= _gate_tolerance_m:
                        self._navigation_ready = True
                        logger.info(
                            "altitude_startup_gate  state=reached  current_z=%.2f  "
                            "target_z=%.2f  elapsed=%.2f",
                            _gz, self._params.target_z_ned,
                            time.monotonic() - _gate_start,
                        )
                        break
                    time.sleep(0.05)
                if not self._navigation_ready:
                    logger.warning(
                        "altitude_startup_gate  state=timeout  target_z=%.2f  "
                        "elapsed=%.2f  navigation_ready=false",
                        self._params.target_z_ned, time.monotonic() - _gate_start,
                    )
            logger.info(
                "altitude_startup_gate  state=done  navigation_ready=%s  target_z=%.2f",
                "true" if self._navigation_ready else "false",
                self._params.target_z_ned,
            )

            # ── Mission goal (computed once; never rolls with drone) ──
            # Phase C1 item #1: the real MissionEnd actor (if present in the
            # AirSim world) wins over any test-only fixed goal.  Only when no
            # actor exists do we fall back to the heading-relative fixed goal.
            # Phase C1-R: actor X/Y → navigation endpoint; actor Z is metadata;
            # navigation goal Z = altitude-hold target (cruise altitude).
            _actor_goal = self._discover_mission_goal_actor()
            _actor_xyz = _actor_goal[0] if _actor_goal is not None else None
            _mission_goal_actor = _actor_goal[1] if _actor_goal is not None else None
            _navigation_target_z = float(self._params.target_z_ned)
            _goal_xy_override = None
            if (
                self._cli_overrides is not None
                and self._cli_overrides.get("goal_x") is not None
                and self._cli_overrides.get("goal_y") is not None
            ):
                _goal_xy_override = (
                    float(self._cli_overrides["goal_x"]),
                    float(self._cli_overrides["goal_y"]),
                )
            if _actor_goal is not None and _goal_xy_override is not None:
                logger.warning(
                    "mission_goal_cli_override  actor=%s  actor_xyz=(%.2f,%.2f,%.2f)  "
                    "cli_goal_xy=(%.2f,%.2f)  using=cli_fixed  "
                    "remove --goal-x/--goal-y to use MissionEnd",
                    _mission_goal_actor,
                    _actor_xyz[0], _actor_xyz[1], _actor_xyz[2],
                    _goal_xy_override[0], _goal_xy_override[1],
                )
            _mission_goal, _mission_goal_source, _mission_actor_xyz = _resolve_mission_goal(
                _actor_xyz, _navigation_target_z,
                st0.position_ned_m, st0.yaw_rad, 15.0,
                goal_xy_override=_goal_xy_override,
            )
            logger.info(
                "cbmba_mission_goal  "
                "mission_goal_source=%s  "
                "actor=%s  "
                "mission_actor_xyz=%s  "
                "navigation_goal_xy=(%.2f,%.2f)  "
                "navigation_target_z=%.2f  "
                "use_actor_goal_z=false  "
                "start=(%.2f,%.2f,%.2f)  "
                "goal=(%.2f,%.2f,%.2f)  "
                "distance=%.1f  "
                "heading=%.4f",
                _mission_goal_source,
                _mission_goal_actor or "none",
                "(%s,%s,%s)" % tuple(f"{v:.2f}" for v in _mission_actor_xyz) if _mission_actor_xyz else "none",
                _mission_goal[0], _mission_goal[1],
                _navigation_target_z,
                st0.position_ned_m[0], st0.position_ned_m[1], st0.position_ned_m[2],
                _mission_goal[0], _mission_goal[1], _mission_goal[2],
                math.hypot(_mission_goal[0] - st0.position_ned_m[0],
                           _mission_goal[1] - st0.position_ned_m[1]),
                st0.yaw_rad,
            )

            # Body-frame trajectory primitives are forward-first.  Align the
            # nose with the fixed MissionEnd before starting horizontal
            # navigation, so a goal behind/aside the initial heading does not
            # cause an unintended initial forward leg.
            if self._local_navigation_mode == "trajectory":
                self._align_heading_to_goal(sr, cr, vc, _mission_goal)

            reactive_config = {
                "emergency_distance_m": self._params.emergency_distance_m,
                "front_threshold_m": self._params.front_threshold_m,
                "forward_speed_mps": self._params.forward_speed_mps,
                "side_speed_mps": self._params.side_speed_mps,
            }

            # ── runtime mode banner (sec 27) ──
            logger.info(
                "runtime_mode_banner  local_navigation=%s  planner_mode=%s  "
                "guided_apf_control=%s  command_duration_s=%.2f  "
                "max_flight_duration_s=%.1f  trajectory_config=%s",
                self._local_navigation_mode, self._planner_mode,
                "true" if self._guided_apf_control else "false",
                self._params.command_duration_s,
                self._params.max_flight_duration_s,
                self._traj_config_path,
            )

            # ── trajectory flight validation initialisation (sec 1/19-22) ──
            if self._local_navigation_mode == "trajectory":
                # ── full navigation algorithm banner (Phase C1 sec 4) ──
                logger.info(
                    "navigation_algorithm_banner  "
                    "primary=trajectory  global_planner=CBMBA_A*  "
                    "local_planner=LocalTrajectoryPlanner  "
                    "tracker=TrajectoryTracker  safety=APF_safety_filter  "
                    "recovery=RecoveryCommander  map=persistent_occupancy_grid  "
                    "mission_goal_source=%s  "
                    "navigation_goal_xy=(%.2f,%.2f)  navigation_target_z=%.2f  "
                    "use_actor_goal_z=false  "
                    "planning_hz=%.1f  horizon=%.1fm  adaptive_horizon=%s  "
                    "num_candidates=%d  hard_clearance=%.1fm  "
                    "guided_apf_fallback=%s  command_duration_s=%.2f",
                    _mission_goal_source,
                    _mission_goal[0], _mission_goal[1], _navigation_target_z,
                    self._traj_planning_hz, self._traj_params.horizon_m,
                    "true" if self._traj_params.adaptive_horizon_enabled else "false",
                    self._traj_params.num_candidates,
                    self._traj_params.hard_clearance_m,
                    "true" if self._allow_guided_apf_fallback else "false",
                    self._params.command_duration_s,
                )
                from flight_modes.trajectory_flight_metrics import (
                    TrajectoryFlightMetrics, FlightTraceWriter,
                )
                _PROJECT_ROOT = Path(__file__).resolve().parent.parent
                self._traj_metrics = TrajectoryFlightMetrics(
                    goal_xy=(_mission_goal[0], _mission_goal[1]),
                )
                self._mission_progress_monitor.set_goal(
                    (_mission_goal[0], _mission_goal[1]),
                )
                self._mission_progress_monitor.reset(
                    time.monotonic(), (spawn[0], spawn[1]),
                )
                # In-sim debug drawing + HUD.
                if self._traj_debug_cfg.get("enabled", False):
                    from flight_modes.airsim_debug_draw import AirSimDebugDrawer
                    self._debug_drawer = AirSimDebugDrawer(
                        self._adapter,
                        enabled=True,
                        vehicle_name=self._vn,
                        line_thickness=float(self._traj_debug_cfg.get("line_thickness", 5.0)),
                        point_size=float(self._traj_debug_cfg.get("point_size", 10.0)),
                        goal_marker_size_m=float(self._traj_debug_cfg.get(
                            "goal_marker_size_m", 2.0,
                        )),
                        goal_marker_height_m=float(self._traj_debug_cfg.get(
                            "goal_marker_height_m", 3.0,
                        )),
                        duration_s=float(self._traj_debug_cfg.get("duration_s", 0.3)),
                        async_mode=bool(self._traj_debug_cfg.get("async", True)),
                        queue_size=int(self._traj_debug_cfg.get("queue_size", 1)),
                    )
                # CSV trace (sec 21).
                if self._traj_debug_cfg.get("trace_csv", False):
                    _out_dir = Path(self._traj_debug_cfg.get(
                        "trace_output_dir", "runs",
                    ))
                    if not _out_dir.is_absolute():
                        _out_dir = _PROJECT_ROOT / _out_dir
                    _out_dir.mkdir(parents=True, exist_ok=True)
                    _ts = time.strftime("%Y%m%d_%H%M%S")
                    self._trace_csv_path = str(_out_dir / f"trace_{_ts}.csv")
                    self._trace_writer = FlightTraceWriter(self._trace_csv_path)
                    logger.info("trajectory_trace_csv  path=%s", self._trace_csv_path)

            # ── control-loop warmup (Phase C1-R sec 13): pull sensors + run the
            # perception pipeline once so the first *real* loop iteration no
            # longer carries one-time cold-start cost (first filter, first
            # sectorisation, first distance field build).  The mission clock and
            # loop-dt baseline are reset AFTER this warmup so init is excluded
            # from flight_duration and the first iter_dt.
            try:
                _w_lf = lidar.read()
                _w_st = sr.read()
                _w_col = cr.read()
                if _w_lf.frame_valid:
                    _w_fr = filter_pointcloud(
                        _w_lf.point_cloud_sensor,
                        min_range_m=pc_cfg.min_range_m, max_range_m=pc_cfg.max_range_m,
                        self_exclusion={
                            "enabled": pc_cfg.self_exclusion.enabled,
                            "x_min_m": pc_cfg.self_exclusion.x_min_m,
                            "x_max_m": pc_cfg.self_exclusion.x_max_m,
                            "y_min_m": pc_cfg.self_exclusion.y_min_m,
                            "y_max_m": pc_cfg.self_exclusion.y_max_m,
                            "z_min_m": pc_cfg.self_exclusion.z_min_m,
                            "z_max_m": pc_cfg.self_exclusion.z_max_m,
                        },
                        voxel_downsample=pc_cfg.voxel_downsample.enabled,
                        voxel_size_m=pc_cfg.voxel_downsample.voxel_size_m,
                    )
                    if _w_fr.valid:
                        _ = pointcloud_to_directional_distances(
                            _w_fr.filtered_points_sensor, sector_defs=sdefs,
                            default_max_range_m=sz_cfg.default_max_range_m,
                            default_min_points=sz_cfg.default_min_points,
                            distance_strategy=sz_cfg.default_distance_strategy,
                            nearest_k=sz_cfg.nearest_k, percentile=sz_cfg.percentile,
                            frame_valid=True, fov_compatible=len(incompatible) == 0,
                            fov_observability=fov_obs,
                        )
            except Exception as _warm_exc:
                logger.warning("control_loop_warmup_error: %s", _warm_exc)
            self._loop_last_iter_mono = None
            logger.info(
                "control_loop_timing_baseline_reset  warmup_done=true  "
                "loop_last_iter_mono=null  mission_clock=reset",
            )

            # Phase C3-R: start BOTH planner **processes** before the flight
            # loop so the realtime loop never blocks on a CBMBA A* search or a
            # local trajectory plan (they run in separate OS processes).
            if self._global_planner_worker is not None:
                self._global_planner_worker.start()
                logger.info("global_planner_worker  state=started")
            if self._local_traj_worker is not None:
                self._local_traj_worker.start()
                logger.info("local_traj_worker  state=started")
            if self._mapping_worker is not None:
                self._mapping_worker.start()
                logger.info("mapping_worker  state=started")

            # ── flight loop ──
            from planners.local_recovery import RecoveryDecision as _RecoveryDecision
            t0 = time.monotonic()
            fn = 0
            term = "time_limit"
            _final_xy = None
            # Deadline scheduler: fixed-rate control loop, independent of the
            # AirSim command duration.  next_tick advances by the control period
            # each iteration; on a missed deadline it resets to "now" so it never
            # sleeps to catch up on a burst of late frames.
            _sched_next_tick = time.monotonic()

            self._running = True
            while self._running:
                fn += 1
                # Phase C1-R sec 6/14: per-frame CBMBA duplicate guard + stage
                # timing baselines (perf_counter for intra-loop profiling).
                self._cbmba_searches_this_frame = 0
                _stg_t0 = time.perf_counter()
                _stg_sensor_ms = 0.0
                _stg_perception_ms = 0.0
                _stg_plan_ms = 0.0
                _stg_dispatch_ms = 0.0
                _rpc_lidar_ms = 0.0
                _rpc_state_ms = 0.0
                _rpc_collision_ms = 0.0
                _rpc_velocity_command_ms = 0.0
                _excl_filter_ms = 0.0
                _excl_sectorize_ms = 0.0
                # Phase C3-R: dispatch exclusive per-substep timing (each is a
                # delta, not cumulative) so a single hot substep can be isolated.
                _tracker_ms = 0.0
                _apf_safety_ms = 0.0
                _goal_check_ms = 0.0
                _progress_monitor_ms = 0.0
                _metrics_ms = 0.0
                _csv_enqueue_ms = 0.0
                _log_enqueue_ms = 0.0
                _rpc_velocity_submit_ms = 0.0
                _map_worker_request_ms = 0.0
                _map_worker_poll_ms = 0.0
                _world_transform_metrics_ms = 0.0
                _worker_request_ms = 0.0
                _worker_poll_ms = 0.0
                # Finite mission timeout applies to ALL goal sources (actor and
                # config_fixed).  Phase C1-R: the old actor-skip let a circling
                # planner run forever; trajectory_flight.yaml provides an
                # eight-minute safety timeout, while MissionEnd normally ends
                # the mission earlier.
                if time.monotonic() - t0 >= self._params.max_flight_duration_s:
                    term = "time_limit"
                    break

                # ── control-loop timing (sec 10): real monotonic dt ──
                _iter_mono = time.monotonic()
                _iter_dt_ms = 0.0
                if self._loop_last_iter_mono is not None:
                    _iter_dt_ms = (_iter_mono - self._loop_last_iter_mono) * 1000.0
                    self._loop_dt_sum_ms += _iter_dt_ms
                    self._loop_dt_n += 1
                    self._loop_dt_samples_ms.append(_iter_dt_ms)
                    # Phase C6-R: classify this loop-start interval.
                    if _iter_dt_ms < 25.0:
                        self._loop_tight_lt25_count += 1
                    if _iter_dt_ms < 40.0:
                        self._loop_tight_lt40_count += 1
                        if self._loop_last_was_resync:
                            self._loop_post_resync_tight_count += 1
                    self._loop_last_was_resync = False
                self._loop_last_iter_mono = _iter_mono

                try:
                    _t_rpc = time.perf_counter()
                    _perc_snap = (
                        self._perception_worker.get_latest_snapshot()
                        if self._perception_worker is not None else None
                    )
                    # LiDAR RPC now runs on the worker thread (sec 10-14); the
                    # control loop only reads the published snapshot.
                    _rpc_lidar_ms = 0.0
                    _t_rpc = time.perf_counter()
                    st = sr.read()
                    _rpc_state_ms = (time.perf_counter() - _t_rpc) * 1000.0
                    _t_rpc = time.perf_counter()
                    col = cr.read()
                    _rpc_collision_ms = (time.perf_counter() - _t_rpc) * 1000.0
                except Exception:
                    term = "rpc_error"
                    break
                _stg_sensor_ms = _rpc_lidar_ms + _rpc_state_ms + _rpc_collision_ms

                if col.has_collided:
                    term = f"collision:{col.object_name}"
                    break
                if math.hypot(st.position_ned_m[0] - spawn[0], st.position_ned_m[1] - spawn[1]) > self._params.geofence_radius_m:
                    term = "geofence"
                    break

                # A large altitude error is a hard safety failure. Continuing
                # horizontal navigation while the vehicle is below/above its
                # cruise altitude can turn a valid obstacle plan into a ground
                # or ceiling collision.
                _altitude_error_now = abs(
                    st.position_ned_m[2] - self._params.target_z_ned
                )
                _recovery_climb_active = (
                    self._recovery_sm.state == "RECOVERY_ACTIVE"
                    and self._recovery_sm.mode == "climb"
                )
                _altitude_limit = self._altitude_error_stop_m
                if _recovery_climb_active:
                    # Permit only the small, explicitly bounded climb probe.
                    # Once the probe ends, the normal altitude safety limit is
                    # restored on the next control tick.
                    _altitude_limit = max(
                        _altitude_limit,
                        self._recovery_sm.params.vertical_climb_delta_m + 0.20,
                    )
                if (
                    self._local_navigation_mode == "trajectory"
                    and self._altitude_safety_enabled
                    and _altitude_error_now > _altitude_limit
                ):
                    logger.error(
                        "altitude_safety_stop  current_z=%.2f  target_z=%.2f  "
                        "error=%.2f  limit=%.2f",
                        st.position_ned_m[2], self._params.target_z_ned,
                        _altitude_error_now, _altitude_limit,
                    )
                    try:
                        vc.send_velocity_body_frd(
                            0.0, 0.0, 0.0,
                            duration=min(0.1, self._params.command_duration_s),
                            vehicle_name=self._vn,
                            yaw_rad=st.yaw_rad,
                        )
                    except Exception:
                        pass
                    term = "altitude_error"
                    break

                # ── control-loop watchdog (sec 11/34) + LiDAR health (sec 12) ──
                # Overrun is measured against the control PERIOD (1/target_hz),
                # never the AirSim command duration.  A deliberate sleep to the
                # next period boundary is NOT an overrun.
                _final_xy = (st.position_ned_m[0], st.position_ned_m[1])
                _overrun_ms = max(
                    0.0, _iter_dt_ms - self._control_period_s * 1000.0,
                )
                self._loop_current_overrun_ms = _overrun_ms
                if _overrun_ms > self._control_loop_overrun_stop_ms:
                    self._loop_overrun_stop_count += 1
                    logger.warning(
                        "control_loop_overrun  dt_ms=%.1f  overrun_ms=%.1f  "
                        "stop_ms=%.1f  action=hover",
                        _iter_dt_ms, _overrun_ms, self._control_loop_overrun_stop_ms,
                    )
                elif _overrun_ms > self._control_loop_overrun_warn_ms:
                    logger.warning(
                        "control_loop_overrun  dt_ms=%.1f  overrun_ms=%.1f  "
                        "warn_ms=%.1f",
                        _iter_dt_ms, _overrun_ms, self._control_loop_overrun_warn_ms,
                    )
                if _overrun_ms > self._control_loop_overrun_warn_ms:
                    self._loop_overrun_count += 1
                self._loop_max_overrun_ms = max(self._loop_max_overrun_ms, _overrun_ms)

                # ── perception staleness → hold position (sec 10-14) ──
                # The LiDAR RPC + filter + sectorisation run on the worker; if its
                # latest snapshot is missing or older than stale_stop_s, hold in
                # place (altitude hold in trajectory mode) instead of steering on
                # stale data.
                _perc_age_s = (
                    self._perception_worker.snapshot_age_s()
                    if self._perception_worker is not None else float("inf")
                )
                if math.isfinite(_perc_age_s):
                    self._perception_age_samples_ms.append(_perc_age_s * 1000.0)
                if _perc_snap is None or _perc_age_s > self._perception_stale_stop_s:
                    self._perception_stale_count += 1
                    self._record_perception_stale_hover()
                    # P0-A: the zero-velocity hold is intentional.  Re-anchor the
                    # stuck detector + progress watchdog so the hold period does
                    # NOT later read as "stuck" / "no progress" and trigger a
                    # false recovery once perception recovers.
                    self._reset_stale_hold_accumulators(
                        time.monotonic(),
                        (st.position_ned_m[0], st.position_ned_m[1]),
                    )
                    self._last_frame_stale_hold = True
                    self._log_throttled(
                        "stale_hold_reset", 1.0,
                        "stale_hold_reset  position=(%.3f,%.3f)  "
                        "recovery_window_reset=true  progress_watchdog_reset=true  "
                        "perception_age=%.3f",
                        st.position_ned_m[0], st.position_ned_m[1],
                        _perc_age_s,
                    )
                    self._log_throttled(
                        "perception_stale", 1.0,
                        "perception_stale  age_s=%.2f  stop_s=%.2f  action=hover",
                        _perc_age_s, self._perception_stale_stop_s,
                    )
                    try:
                        _t_rpc = time.perf_counter()
                        if self._local_navigation_mode == "trajectory":
                            self._last_velocity_future = vc.send_velocity_body_frd_z(
                                0.0, 0.0, self._params.target_z_ned,
                                duration=self._params.command_duration_s,
                                vehicle_name=self._vn,
                            )
                        else:
                            self._last_velocity_future = vc.send_velocity_body_frd(
                                0.0, 0.0, 0.0,
                                duration=self._params.command_duration_s,
                                vehicle_name=self._vn,
                            )
                        _rpc_velocity_command_ms = (time.perf_counter() - _t_rpc) * 1000.0
                    except Exception:
                        term = "velocity_send_error"
                        break
                    _sched_next_tick, _sleep_ms, _deadline_late_ms, _missed, _resynced = self._sleep_to_next_period(_sched_next_tick)
                    if _resynced:
                        self._loop_deadline_resync_count += 1
                    self._loop_last_was_resync = _resynced
                    continue

                lf = _perc_snap.lf
                if not lf.frame_valid:
                    if lf.invalid_reason == "empty":
                        # Round 9: FRESH empty point cloud → safety hold, NOT an
                        # immediate abort and NOT "free space".  Only a persistent
                        # run of empty frames terminates (persistent_empty).
                        _now_mono = time.monotonic()
                        term = self._handle_lidar_empty_frame(
                            lf, _now_mono,
                            st.position_ned_m, st.linear_velocity_ned_mps,
                        )
                        if term is not None:
                            break
                        try:
                            _t_rpc = time.perf_counter()
                            if self._local_navigation_mode == "trajectory":
                                self._last_velocity_future = vc.send_velocity_body_frd_z(
                                    0.0, 0.0, self._params.target_z_ned,
                                    duration=self._params.command_duration_s,
                                    vehicle_name=self._vn,
                                )
                            else:
                                self._last_velocity_future = vc.send_velocity_body_frd(
                                    0.0, 0.0, 0.0,
                                    duration=self._params.command_duration_s,
                                    vehicle_name=self._vn,
                                )
                            _rpc_velocity_command_ms = (time.perf_counter() - _t_rpc) * 1000.0
                        except Exception:
                            term = "velocity_send_error"
                            break
                        _sched_next_tick, _sleep_ms, _deadline_late_ms, _missed, _resynced = self._sleep_to_next_period(_sched_next_tick)
                        if _resynced:
                            self._loop_deadline_resync_count += 1
                        self._loop_last_was_resync = _resynced
                        continue
                    # Non-empty invalid reason (rpc_error/malformed/bad_values/
                    # missing_sensor/stale/unknown_error) — hard abort unchanged.
                    self._lidar_invalid_frames += 1
                    term = f"lidar_invalid:{lf.invalid_reason}"
                    break

                # Round 9: valid non-empty frame — bookkeep empty-hold exit so a
                # transient empty burst is fully forgotten on recovery.
                self._lidar_valid_nonempty_frames += 1
                self._lidar_last_nonempty_ts_ns = getattr(lf, "raw_timestamp_ns", None)
                self._lidar_last_valid_point_count = getattr(lf, "point_count", 0)
                self._lidar_prev_ts_ns = getattr(lf, "raw_timestamp_ns", None)
                if self._lidar_consecutive_empty > 0:
                    self._lidar_empty_last_run_frames = self._lidar_consecutive_empty
                    self._lidar_empty_last_run_duration_s = (
                        time.monotonic() - self._lidar_empty_since_mono
                        if self._lidar_empty_since_mono is not None else 0.0
                    )
                self._lidar_consecutive_empty = 0
                self._lidar_empty_since_mono = None
                self._lidar_empty_last_mono = None
                self._lidar_empty_hold_active = False

                # LiDAR staleness is logged independently of the control watchdog
                # so lidar degradation is visible even before a hard break.
                _stale_count = _perc_snap.stale_count
                if (fn == 1 or fn % 10 == 0 or _stale_count > 0):
                    _lidar_age_s = time.monotonic() - lf.received_monotonic_seconds
                    logger.info(
                        "lidar_health  frame_valid=%s  invalid_reason=%s  "
                        "stale_count=%d  received_age_s=%.3f  points=%d",
                        "true" if lf.frame_valid else "false",
                        lf.invalid_reason, _stale_count,
                        _lidar_age_s, lf.point_count,
                    )
                if _stale_count > 0:
                    self._lidar_stale_frames_total += 1

                # Round 9: low-frequency LiDAR-health summary (sec 9).
                _lh_now = time.monotonic()
                if _lh_now - self._lidar_health_log_mono >= 5.0:
                    self._lidar_health_log_mono = _lh_now
                    logger.info(
                        "lidar_health_summary  total_polls=%d  "
                        "valid_nonempty_frames=%d  fresh_empty_frames=%d  "
                        "stale_frames=%d  invalid_frames=%d  "
                        "max_consecutive_empty=%d  last_valid_point_count=%d",
                        self._perception_worker.poll_count
                            if self._perception_worker is not None else 0,
                        self._lidar_valid_nonempty_frames,
                        self._lidar_empty_frames_total,
                        self._lidar_stale_frames_total,
                        self._lidar_invalid_frames,
                        self._lidar_empty_max_consecutive,
                        self._lidar_last_valid_point_count,
                    )

                # Perception (filter + sectorise) already ran on the worker thread;
                # the control loop only unpacks the published results.
                _excl_filter_ms = 0.0
                _excl_sectorize_ms = 0.0
                fr = _perc_snap.fr
                if fr is None or not fr.valid:
                    term = f"filter:{fr.invalid_reason if fr is not None else 'none'}"
                    break
                dd = _perc_snap.dd
                if dd is None or not dd.frame_valid:
                    term = f"dd:{dd.invalid_reason if dd is not None else 'none'}"
                    break
                rays = _perc_snap.rays
                if rays is None:
                    term = "perception:none"
                    break

                dec = choose_reactive_command(
                    rays.get("front", float("inf")),
                    rays.get("left", float("inf")),
                    rays.get("right", float("inf")),
                    dd.minimum_distance_m, reactive_config,
                )
                if dec.should_terminate:
                    term = dec.termination_reason
                    break
                _stg_perception_ms = _excl_filter_ms + _excl_sectorize_ms
                _stg_plan_start = time.perf_counter()

                # ── LocalRecovery shadow detection (compute + log only; no control) ──
                _guidance_result = None  # initialized here; populated by CBMBA guidance below
                recovery_decision = _RecoveryDecision()    # safe default if try fails
                try:
                    # NED → body-frame velocity conversion
                    _yaw = st.yaw_rad
                    _vn, _ve, _vd = st.linear_velocity_ned_mps
                    _vx_body = _vn * math.cos(_yaw) + _ve * math.sin(_yaw)
                    _vy_body = -_vn * math.sin(_yaw) + _ve * math.cos(_yaw)
                    _vz_body = _vd

                    recovery_decision = self._recovery.update(
                        timestamp=time.monotonic(),
                        position=(st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
                        velocity_body=(_vx_body, _vy_body, _vz_body),
                        yaw_rad=_yaw,
                    )
                    # Phase C3-R: verify the recovery detector reports the SAME
                    # position the loop fed it.  The old (0,0,0) sentinel bug made
                    # the detector report a false origin before the window filled.
                    _actual_pos = (st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2])
                    _det_pos = recovery_decision.stuck_latest_position
                    _pos_mismatch_m = math.hypot(
                        _actual_pos[0] - _det_pos[0],
                        _actual_pos[1] - _det_pos[1],
                        _actual_pos[2] - _det_pos[2],
                    )
                    self._log_throttled(
                        "recovery_input_trace", 1.0,
                        "recovery_input_trace  actual_position=(%.3f,%.3f,%.3f)  "
                        "detector_position=(%.3f,%.3f,%.3f)  mismatch_m=%.3f  window=%d",
                        _actual_pos[0], _actual_pos[1], _actual_pos[2],
                        _det_pos[0], _det_pos[1], _det_pos[2],
                        _pos_mismatch_m, recovery_decision.window_size_frames,
                    )
                    if _pos_mismatch_m > 0.5:
                        logger.warning(
                            "RECOVERY_POSITION_SOURCE_MISMATCH  "
                            "actual=(%.3f,%.3f,%.3f)  detector=(%.3f,%.3f,%.3f)  "
                            "mismatch_m=%.3f  window=%d",
                            _actual_pos[0], _actual_pos[1], _actual_pos[2],
                            _det_pos[0], _det_pos[1], _det_pos[2],
                            _pos_mismatch_m, recovery_decision.window_size_frames,
                        )
                    self._log_throttled(
                        "recovery_shadow", 1.0,
                        "recovery_shadow  stuck=%s  oscillating=%s  needs=%s  "
                        "stuck_dur=%.2f  stuck_delta=%.3f  "
                        "position=(%.3f,%.3f,%.3f)  "
                        "oldest_position=(%.3f,%.3f,%.3f)  "
                        "osc_flips=%d  osc_lateral=%.3f  "
                        "candidates=%s  reason=%s  window=%d",
                        recovery_decision.is_stuck,
                        recovery_decision.is_oscillating,
                        recovery_decision.needs_recovery,
                        recovery_decision.stuck_duration_s,
                        recovery_decision.stuck_position_delta_m,
                        recovery_decision.stuck_latest_position[0],
                        recovery_decision.stuck_latest_position[1],
                        recovery_decision.stuck_latest_position[2],
                        recovery_decision.stuck_oldest_position[0],
                        recovery_decision.stuck_oldest_position[1],
                        recovery_decision.stuck_oldest_position[2],
                        recovery_decision.oscillation_vy_sign_flips,
                        recovery_decision.oscillation_lateral_progress_m,
                        recovery_decision.candidate_actions,
                        recovery_decision.reason,
                        recovery_decision.window_size_frames,
                    )
                    if self._last_frame_stale_hold:
                        self._last_frame_stale_hold = False
                        logger.info(
                            "post_stale_recovery  recovery_stuck=%s  "
                            "recovery_needs=%s  window=%d  "
                            "progress_watchdog_fired=%s  position=(%.3f,%.3f,%.3f)",
                            recovery_decision.is_stuck,
                            recovery_decision.needs_recovery,
                            recovery_decision.window_size_frames,
                            "true" if self._progress_watchdog._fired else "false",
                            st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2],
                        )
                    if self._last_frame_empty_hold:
                        self._last_frame_empty_hold = False
                        logger.info(
                            "post_empty_recovery  empty_duration_s=%.3f  "
                            "empty_frames=%d  current_position=(%.3f,%.3f,%.3f)  "
                            "stuck=%s  needs_recovery=%s  watchdog_fired=%s  "
                            "bypass_state=%s  rejoin_state=%s",
                            self._lidar_empty_last_run_duration_s,
                            self._lidar_empty_last_run_frames,
                            st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2],
                            recovery_decision.is_stuck,
                            recovery_decision.needs_recovery,
                            "true" if self._progress_watchdog._fired else "false",
                            "true" if self._bypass.active else "false",
                            "true" if self._rejoin.active else "false",
                        )
                except Exception as _recovery_exc:
                    logger.warning("recovery_compute_error: %s", _recovery_exc)

                # ── Recovery test trigger: one-shot synthetic injection ──
                if (self._recovery_test_trigger is not None
                        and not self._recovery_test_trigger_fired
                        and rk.get("airborne")
                        and fn >= self._recovery_test_trigger_delay_frames):
                    self._recovery_test_trigger_fired = True
                    trigger_type = self._recovery_test_trigger
                    synthetic = _RecoveryDecision(
                        is_stuck=(trigger_type == "stuck"),
                        is_oscillating=(trigger_type == "oscillation"),
                        needs_recovery=True,
                        reason=f"test_trigger:{trigger_type}",
                    )
                    logger.info(
                        "recovery_test_trigger  type=%s  "
                        "stuck=%s  oscillating=%s  needs=%s",
                        trigger_type,
                        synthetic.is_stuck,
                        synthetic.is_oscillating,
                        synthetic.needs_recovery,
                    )
                    # Inject: replace the real recovery_decision for the state machine.
                    # The shadow log above still reflects real AirSim state.
                    recovery_decision = synthetic

                # ── No-feasible fast recovery (trajectory mode) ──
                # The trajectory planner requests recovery directly (without
                # waiting for the stuck detector).  The escape hint biases the
                # recovery side toward the clearer direction.
                _traj_escape_side = None
                if (self._local_navigation_mode == "trajectory"
                        and self._traj_request_recovery):
                    self._traj_request_recovery = False
                    _traj_escape_side = int((self._traj_escape_hint or {}).get("side", 0) or 0)
                    synthetic = _RecoveryDecision(
                        is_stuck=True,
                        is_oscillating=False,
                        needs_recovery=True,
                        reason="trajectory_no_feasible",
                    )
                    logger.warning(
                        "trajectory_no_feasible_recovery  inject=true  escape_side=%s",
                        { -1: "LEFT", 1: "RIGHT", 0: "NONE" }[_traj_escape_side],
                    )
                    recovery_decision = synthetic

                # ── Recovery takeover state machine ──
                _guidance_dir_for_recovery = None
                if _guidance_result is not None and _guidance_result.valid:
                    _guidance_dir_for_recovery = _guidance_result.direction_body_xy
                recovery_result = self._recovery_sm.tick(
                    time.monotonic(), recovery_decision, rays,
                    current_position=(
                        st.position_ned_m[0],
                        st.position_ned_m[1],
                        st.position_ned_m[2],
                    ),
                    guidance_dir=_guidance_dir_for_recovery,
                    bypass_side=(
                        _traj_escape_side if _traj_escape_side
                        else (self._bypass.side if self._bypass.active else None)
                    ),
                )
                if recovery_result.event == "enter":
                    logger.info(
                        "recovery_enter  reason=%s  action=(%.3f,%.3f,%.3f)  "
                        "cmd=(%.3f,%.3f,%.3f)  committed_side=%s",
                        recovery_decision.reason,
                        recovery_result.vx_body,
                        recovery_result.vy_body,
                        recovery_result.vz_body,
                        recovery_result.vx_body,
                        recovery_result.vy_body,
                        recovery_result.vz_body,
                        self._side_label(recovery_result.committed_side),
                    )
                elif recovery_result.event == "active":
                    logger.info(
                        "recovery_active  elapsed=%.2f",
                        recovery_result.elapsed_s,
                    )
                elif recovery_result.event in (
                    "exit_timeout", "exit_progress", "exit_climb",
                ):
                    logger.info(
                        "recovery_exit  reason=%s  elapsed=%.2f  committed_side=%s",
                        recovery_result.event,
                        recovery_result.elapsed_s,
                        self._side_label(recovery_result.committed_side),
                    )
                    logger.info("handoff_to_apf")
                    logger.info(
                        "recovery_cooldown  remaining=%.2f",
                        recovery_result.cooldown_remaining_s,
                    )
                    # Inherit committed_side into bypass if recovery chose a side
                    if (recovery_result.committed_side is not None
                            and recovery_result.committed_side != 0
                            and not self._bypass.active):
                        # Freeze the reference BEFORE the drone deviates.  The
                        # latest CBMBA plan is used here (cbmba_result for THIS
                        # frame is not computed until later in the loop), so the
                        # snapshot is the last valid path — not a live, re-seeded
                        # path whose [0] would equal the current position.
                        _byp_pos_xy = (st.position_ned_m[0], st.position_ned_m[1])
                        _cached_path = (
                            self._cbmba_cached_result.path_world
                            if self._cbmba_cached_result is not None else None
                        )
                        _byp_ref = self._freeze_reference_xy(_byp_pos_xy, _cached_path)
                        self._bypass = BypassEpisode(
                            active=True,
                            side=recovery_result.committed_side,
                            start_time=time.monotonic(),
                            reason=f"inherited_from_recovery(side={self._side_label(recovery_result.committed_side)})",
                            min_duration_s=self._bypass_min_duration_s,
                            entry_clearance_side_m=(
                                rays.get("right", 0.0) if recovery_result.committed_side == 1
                                else rays.get("left", 0.0)
                            ) or 0.0,
                            reference_path_xy=_byp_ref[0],
                            reference_source=_byp_ref[1],
                            reference_generation_id=_byp_ref[2],
                            reference_first_xy=_byp_ref[3],
                            reference_frozen_position_xy=_byp_pos_xy,
                            trajectory_dead_end=(self._local_navigation_mode == "trajectory"),
                        )
                        self._bypass_unsafe_start = None
                        _byp_last = _byp_ref[0][-1] if _byp_ref[0] else None
                        _byp_fp = self._reference_fingerprint(_byp_ref[0])
                        logger.info(
                            "bypass_inherited_from_recovery  side=%s  "
                            "min_duration=%.2f  entry_clearance=%.2f  "
                            "reference_source=%s  reference_generation=%s  "
                            "reference_first=(%.2f,%.2f)  "
                            "reference_last=(%.2f,%.2f)  ref_len=%d  "
                            "fingerprint=%s",
                            self._side_label(recovery_result.committed_side),
                            self._bypass_min_duration_s,
                            self._bypass.entry_clearance_side_m,
                            _byp_ref[1] or "none",
                            self._bypass.reference_generation_id,
                            _byp_ref[3][0] if _byp_ref[3] else float("nan"),
                            _byp_ref[3][1] if _byp_ref[3] else float("nan"),
                            _byp_last[0] if _byp_last else float("nan"),
                            _byp_last[1] if _byp_last else float("nan"),
                            len(_byp_ref[0]),
                            _byp_fp,
                        )
                elif recovery_result.event == "cooldown_expired":
                    logger.info("recovery_cooldown  expired")
                elif recovery_result.event and recovery_result.event.startswith("exit_safety"):
                    logger.info(
                        "recovery_exit  reason=%s  elapsed=%.2f",
                        recovery_result.event, recovery_result.elapsed_s,
                    )
                    logger.info("handoff_to_apf")
                    logger.info(
                        "recovery_cooldown  remaining=%.2f",
                        recovery_result.cooldown_remaining_s,
                    )

                # ── Trajectory recovery-exit reset (forget stale memory) ──
                if self._local_navigation_mode == "trajectory":
                    if recovery_result.should_override:
                        self._traj_recovery_was_active = True
                    elif self._traj_recovery_was_active:
                        # Recovery just handed control back → the stale trajectory
                        # memory must not pull the drone back toward the dead end.
                        self._traj_recovery_was_active = False
                        # Phase C3-R: memory now lives in the planner PROCESS, so
                        # forward the reset across the boundary (the main loop's
                        # copy is no longer the one the planner reads).
                        if self._local_traj_worker is not None:
                            self._local_traj_worker.reset_memory()
                        self._traj_memory.reset()
                        self._traj_force_replan = True
                        self._traj_global_replan_requested = True
                        logger.info("trajectory_memory_reset  reason=recovery_exit")
                        logger.info("trajectory_global_replan  reason=recovery_exit")

                # ── APF computation (apf_shadow: log only; apf: control) ──
                apf_output = None
                apf_label = "apf_shadow" if self._planner_mode == "apf_shadow" else "apf_control"
                if self._planner_mode in ("apf_shadow", "apf"):
                    try:
                        from planners.improved_potential_field import ImprovedPotentialField
                        apf_output = self._apf.update(
                            sector_distances=rays,
                            sector_point_counts=None,
                            goal_body=(1.0, 0.0, 0.0),
                            current_velocity_body=(dec.vx_body_mps, dec.vy_body_mps, 0.0),
                            minimum_distance_m=dd.minimum_distance_m,
                        )
                        logger.info(
                            "%s  front=%.2f left=%.2f right=%.2f minD=%.2f  "
                            "reactive=(%.3f,%.3f,%.3f)  "
                            "attractive=(%.3f,%.3f,%.3f)  "
                            "repulsive=(%.3f,%.3f,%.3f)  "
                            "force_mag=%.3f  "
                            "apf_cmd=(%.3f,%.3f,%.3f)  cmd_mag=%.3f  "
                            "valid=%s  reason=%s  nan=%s  inf=%s  sat=%s",
                            apf_label,
                            rays.get("front", float("inf")),
                            rays.get("left", float("inf")),
                            rays.get("right", float("inf")),
                            dd.minimum_distance_m,
                            dec.vx_body_mps, dec.vy_body_mps, 0.0,
                            apf_output.attractive_force[0],
                            apf_output.attractive_force[1],
                            apf_output.attractive_force[2],
                            apf_output.repulsive_force[0],
                            apf_output.repulsive_force[1],
                            apf_output.repulsive_force[2],
                            apf_output.force_magnitude,
                            apf_output.desired_vx_body,
                            apf_output.desired_vy_body,
                            apf_output.desired_vz_body,
                            apf_output.command_magnitude,
                            apf_output.valid, apf_output.reason,
                            apf_output.nan_detected, apf_output.inf_detected,
                            apf_output.saturated,
                        )
                        # ── per-sector repulsive contributions (diagnostic) ──
                        if apf_output.per_sector_contributions:
                            for sc in apf_output.per_sector_contributions:
                                logger.info(
                                    "apf_sector  name=%-10s  dist=%.2f  "
                                    "dir=(%+.3f,%+.3f,%+.3f)  "
                                    "rep=(%+.4f,%+.4f,%+.4f)  "
                                    "used_for_control=%s",
                                    sc["name"], sc["distance"],
                                    sc["dir_x"], sc["dir_y"], sc["dir_z"],
                                    sc["rep_x"], sc["rep_y"], sc["rep_z"],
                                    sc.get("used_for_control", True),
                                )
                    except Exception as e:
                        logger.warning("apf_compute_error: %s", e)

                # ── CBMBA A* shadow (compute + log only; never dispatches) ──
                cbmba_result = None
                cbmba_obstacles: list = []
                if self._cbmba_enabled:
                    try:
                        cbmba_start = [
                            st.position_ned_m[0],
                            st.position_ned_m[1],
                            st.position_ned_m[2],
                        ]
                        # Fixed mission goal (computed once at airborne; never rolls)
                        cbmba_goal = [
                            _mission_goal[0],
                            _mission_goal[1],
                            _mission_goal[2],
                        ]
                        # Build synthetic obstacles from LiDAR sector distances
                        cbmba_obstacles = _sector_distances_to_obstacles(
                            rays, st.position_ned_m, _yaw,
                        )
                        # ── diagnostic: obstacle samples (count change or ≤1 Hz) ──
                        _now_abs = time.monotonic()
                        _obs_count = len(cbmba_obstacles)
                        _obs_changed = _obs_count != self._diag_last_obstacle_count
                        _obs_stale = (_now_abs - self._diag_last_obstacle_log_time) >= 1.0
                        if cbmba_obstacles and (_obs_changed or _obs_stale or fn == 1):
                            _sample_parts = []
                            for _obs in cbmba_obstacles[:12]:  # cap per log line
                                _sec = _obs.get("_diag_sector", "?")
                                _dist = _obs.get("_diag_distance", -1.0)
                                _bxy = _obs.get("_diag_body_xy", (0.0, 0.0))
                                _wpos = _obs["position"]
                                _sample_parts.append(
                                    f"{{sector={_sec} dist={_dist:.2f} "
                                    f"body=({_bxy[0]:.2f},{_bxy[1]:.2f}) "
                                    f"world=({_wpos[0]:.2f},{_wpos[1]:.2f}) "
                                    f"footprint={_obs['footprint_half_extents']}}}"
                                )
                            logger.info(
                                "cbmba_obstacles  "
                                "count=%d  "
                                "samples=[%s]  "
                                "planner_inflation=%.1f  "
                                "effective_extent=%.1f",
                                _obs_count,
                                "  ".join(_sample_parts) if _sample_parts else "none",
                                self._cbmba.params.inflation_radius,
                                self._cbmba.params.inflation_radius,
                            )
                            self._diag_last_obstacle_count = _obs_count
                            self._diag_last_obstacle_log_time = _now_abs
                        elif not cbmba_obstacles and _obs_changed:
                            logger.info(
                                "cbmba_obstacles  count=0  samples=[]  "
                                "planner_inflation=%.1f  effective_extent=%.1f",
                                self._cbmba.params.inflation_radius,
                                self._cbmba.params.inflation_radius,
                            )
                            self._diag_last_obstacle_count = 0
                            self._diag_last_obstacle_log_time = _now_abs

                        # Phase C1-R sec 6: in trajectory mode the global CBMBA A*
                        # is owned by the background worker (see _tick_global_replan),
                        # so this legacy shadow search must NOT run — it doubled the
                        # A* cost and blocked the realtime loop.
                        # Non-trajectory modes replan at a fixed cadence (2 Hz) and
                        # reuse the latest valid path in between, so the pure-Python
                        # A* never holds the GIL long enough to starve the
                        # PerceptionWorker → LiDAR stale.
                        _using_cached = False
                        if self._local_navigation_mode != "trajectory":
                            _cbmba_now = time.monotonic()
                            _replan_due = (
                                self._cbmba_cached_result is None
                                or _cbmba_now - self._cbmba_last_replan_time
                                    >= self._cbmba_replan_interval_s
                            )
                            if _replan_due:
                                self._record_cbmba_search("cbmba_shadow")
                                _fresh = self._cbmba.plan_with_result(
                                    cbmba_obstacles, cbmba_start, cbmba_goal,
                                )
                                if (_fresh is not None and _fresh.success
                                        and len(_fresh.path_world) >= 2):
                                    self._cbmba_cached_result = _fresh
                                    self._cbmba_last_replan_time = _cbmba_now
                                    self._cbmba_path_generation += 1
                                    cbmba_result = _fresh
                                else:
                                    # Budget-exceeded / no path → reuse previous valid path.
                                    cbmba_result = self._cbmba_cached_result
                                    _using_cached = True
                            else:
                                cbmba_result = self._cbmba_cached_result
                                _using_cached = True

                        # ── diagnostic: CBMBA path XY (material change or ≤1 Hz) ──
                        if cbmba_result is not None and cbmba_result.success and len(cbmba_result.path_world) >= 2:
                            _path_xy = tuple(
                                (round(p[0], 2), round(p[1], 2))
                                for p in cbmba_result.path_world
                            )
                            _path_changed = _path_xy != self._diag_last_path_points
                            _path_stale = (_now_abs - self._diag_last_path_log_time) >= 1.0
                            if _path_changed or _path_stale:
                                _xs = [p[0] for p in cbmba_result.path_world]
                                _ys = [p[1] for p in cbmba_result.path_world]
                                _pt_strs = [f"({p[0]:.2f},{p[1]:.2f})" for p in cbmba_result.path_world]
                                logger.info(
                                    "cbmba_path_xy  "
                                    "points=[%s]  "
                                    "min_x=%.2f  max_x=%.2f  "
                                    "min_y=%.2f  max_y=%.2f",
                                    " ".join(_pt_strs),
                                    min(_xs), max(_xs),
                                    min(_ys), max(_ys),
                                )
                                self._diag_last_path_points = _path_xy
                                self._diag_last_path_log_time = _now_abs
                        if cbmba_result is not None:
                            logger.info(
                                "cbmba_shadow  success=%s  using_cached_path=%s  "
                                "nodes=%d  path_len=%d  "
                                "grid_size=%d  time_ms=%.2f  "
                                "start=(%.2f,%.2f,%.2f)  goal=(%.2f,%.2f,%.2f)  "
                                "num_obstacles=%d  fixed=true",
                                cbmba_result.success,
                                "true" if _using_cached else "false",
                                cbmba_result.nodes_expanded,
                                len(cbmba_result.path_world),
                                cbmba_result.grid_size,
                                cbmba_result.planning_time_ms,
                                cbmba_start[0], cbmba_start[1], cbmba_start[2],
                                cbmba_goal[0], cbmba_goal[1], cbmba_goal[2],
                                len(cbmba_obstacles),
                            )
                        # Log path shape for diagnostics
                        if cbmba_result is not None and cbmba_result.success and len(cbmba_result.path_world) >= 2:
                            wp_first = cbmba_result.path_world[0]
                            wp_last = cbmba_result.path_world[-1]
                            # ── next = first waypoint meaningfully different from start ──
                            _eps = 0.05
                            wp_next = wp_first
                            for _pt in cbmba_result.path_world[1:]:
                                if (abs(_pt[0] - wp_first[0]) > _eps
                                        or abs(_pt[1] - wp_first[1]) > _eps
                                        or abs(_pt[2] - wp_first[2]) > _eps):
                                    wp_next = _pt
                                    break
                            # ── max_lateral_dev = max perpendicular distance from start→goal XY line ──
                            _sx, _sy = cbmba_start[0], cbmba_start[1]
                            _gx, _gy = cbmba_goal[0], cbmba_goal[1]
                            _seg_dx = _gx - _sx
                            _seg_dy = _gy - _sy
                            _seg_len = math.hypot(_seg_dx, _seg_dy)
                            _max_dev = 0.0
                            if _seg_len > 1e-6:
                                _max_dev = max(
                                    abs((_pt[0] - _sx) * _seg_dy - (_pt[1] - _sy) * _seg_dx) / _seg_len
                                    for _pt in cbmba_result.path_world
                                )
                            logger.info(
                                "cbmba_path  waypoints=%d  "
                                "first=(%.2f,%.2f,%.2f)  "
                                "next=(%.2f,%.2f,%.2f)  "
                                "last=(%.2f,%.2f,%.2f)  "
                                "max_lateral_dev=%.3f",
                                len(cbmba_result.path_world),
                                wp_first[0], wp_first[1], wp_first[2],
                                wp_next[0], wp_next[1], wp_next[2],
                                wp_last[0], wp_last[1], wp_last[2],
                                _max_dev,
                            )
                    except Exception as _cbmba_exc:
                        logger.warning("cbmba_compute_error: %s", _cbmba_exc)

                # ── Path validity gate (Failures A & B) ──
                # Validate that the CBMBA path is acceptable before using it
                # for guidance.  An invalid path means guidance falls back to
                # pure-forward, bypassing the guided APF takeover.
                self._path_valid = True
                self._path_fail_reason = ""
                if self._cbmba_enabled and cbmba_result is not None:
                    _pv_path = cbmba_result.path_world
                    # Check 1: planning success
                    if not cbmba_result.success:
                        self._path_valid = False
                        self._path_fail_reason = "planning_failed"
                    # Check 2: path within planning bounds
                    elif not self._cbmba.is_path_in_bounds(_pv_path):
                        self._path_valid = False
                        self._path_fail_reason = (
                            f"out_of_bounds(max_dev={cbmba_result.max_lateral_deviation_m:.2f}m"
                            f" > {self._cbmba.params.planning_bounds_xy_m:.1f}m)"
                        )
                    # Check 3: path not blocked by current obstacles
                    elif self._cbmba._is_path_blocked(
                        cbmba_obstacles, _pv_path,
                        self._cbmba.params.inflation_radius,
                        self._cbmba.params,
                    ):
                        self._path_valid = False
                        self._path_fail_reason = "path_blocked_by_obstacles"

                if not self._path_valid:
                    self._consecutive_invalid_paths += 1
                    logger.warning(
                        "path_validity_gate  valid=false  reason=%s  "
                        "consecutive=%d/%d  max_lateral_dev=%.2f",
                        self._path_fail_reason,
                        self._consecutive_invalid_paths,
                        self._max_consecutive_invalid_paths,
                        cbmba_result.max_lateral_deviation_m if cbmba_result is not None else 0.0,
                    )
                else:
                    self._consecutive_invalid_paths = 0

                # ── Forward-progress watchdog (Failures A & B) ──
                if self._planner_mode in ("apf", "apf_shadow"):
                    _wp = self._progress_watchdog.update(
                        time.monotonic(),
                        (st.position_ned_m[0], st.position_ned_m[1]),
                    )
                    if _wp:
                        logger.warning(
                            "progress_watchdog  fired=true  "
                            "fired_count=%d  position=(%.2f,%.2f)",
                            self._progress_watchdog._fired_count,
                            st.position_ned_m[0], st.position_ned_m[1],
                        )
                        # Force bypass release on watchdog fire (stale bypass)
                        if self._bypass.active:
                            logger.warning(
                                "bypass_veto  reason=progress_watchdog  "
                                "side=%s  elapsed=%.2f",
                                self._side_label(self._bypass.side),
                                time.monotonic() - self._bypass.start_time,
                            )
                            self._bypass = BypassEpisode()

                # ── CBMBA guidance shadow (segment-crossing; never dispatches) ──
                _guidance_result = None  # saved for guided APF shadow below
                if self._cbmba_enabled and self._cbmba_guidance_enabled:
                    try:
                        _cbmba_path = getattr(self._cbmba, "last_path", None)
                        if _cbmba_path and len(_cbmba_path) >= 2:
                            _guidance_result = self._cbmba_guidance.select_waypoint(
                                (st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
                                _yaw,
                                _cbmba_path,
                            )
                            if _guidance_result.valid:
                                _seg = _guidance_result.source_segment
                                _seg_str = f"({_seg[0]},{_seg[1]})" if _seg else "none"
                                logger.info(
                                    "cbmba_guidance_shadow  valid=true  "
                                    "source_segment=%s  "
                                    "interpolated=%s  "
                                    "body_target=(%.2f,%.2f)  "
                                    "forward_progress=%.2f  "
                                    "lateral_offset=%.2f  "
                                    "direction=(%.3f,%.3f)  "
                                    "reason=%s",
                                    _seg_str,
                                    "true" if _guidance_result.interpolated else "false",
                                    _guidance_result.target_body_xy[0],
                                    _guidance_result.target_body_xy[1],
                                    _guidance_result.forward_progress_m,
                                    _guidance_result.lateral_offset_m,
                                    _guidance_result.direction_body_xy[0],
                                    _guidance_result.direction_body_xy[1],
                                    _guidance_result.reason,
                                )
                            else:
                                logger.info(
                                    "cbmba_guidance_shadow  valid=false  reason=%s",
                                    _guidance_result.reason,
                                )
                    except Exception as _guid_exc:
                        logger.warning("cbmba_guidance_error: %s", _guid_exc)

                # ── Guided APF lateral shadow (CBMBA → lateral attractive bias) ──
                _guided_output = None
                _g_cmd = (0.0, 0.0)
                _n_cmd = (0.0, 0.0)
                _guide_valid_flag = False
                _fallback_reason = ""
                if apf_output is not None and apf_output.valid:
                    try:
                        _lateral_bias = 0.0
                        _guide_valid_flag = False
                        _guide_dir_x, _guide_dir_y = 1.0, 0.0
                        if _guidance_result is not None and _guidance_result.valid:
                            _gdx, _gdy = _guidance_result.direction_body_xy
                            if (math.isfinite(_gdx) and math.isfinite(_gdy)
                                    and (_gdx != 0.0 or _gdy != 0.0)):
                                _guide_dir_x = _gdx
                                _guide_dir_y = _gdy
                                # Lateral bias bounded by ±attractive_gain
                                # because |guidance_direction_y| ≤ 1 (unit vector)
                                _lateral_bias = self._apf._params.attractive_gain * _gdy
                                _guide_valid_flag = True

                        _guided_output = self._apf.update(
                            sector_distances=rays,
                            sector_point_counts=None,
                            goal_body=(1.0, 0.0, 0.0),
                            current_velocity_body=(dec.vx_body_mps, dec.vy_body_mps, 0.0),
                            minimum_distance_m=dd.minimum_distance_m,
                            lateral_guidance_bias=_lateral_bias,
                        )

                        _n_att = apf_output.attractive_force
                        _g_att = _guided_output.attractive_force
                        _rep = apf_output.repulsive_force
                        _n_cmd = (apf_output.desired_vx_body, apf_output.desired_vy_body)
                        _g_cmd = (_guided_output.desired_vx_body, _guided_output.desired_vy_body)
                        _delta_x = _g_cmd[0] - _n_cmd[0]
                        _delta_y = _g_cmd[1] - _n_cmd[1]
                        _forward_preserved = abs(_g_att[0] - _n_att[0]) < 1e-9

                        self._log_throttled(
                            "guided_apf_lateral_shadow", 1.0,
                            "guided_apf_lateral_shadow  "
                            "guidance_valid=%s  "
                            "guidance_direction=(%.3f,%.3f)  "
                            "normal_attractive=(%.3f,%.3f)  "
                            "guided_attractive=(%.3f,%.3f)  "
                            "repulsive=(%.3f,%.3f)  "
                            "normal_cmd=(%.3f,%.3f)  "
                            "guided_cmd=(%.3f,%.3f)  "
                            "cmd_delta=(%.3f,%.3f)  "
                            "forward_preserved=%s  "
                            "guided_cmd_mag=%.3f  "
                            "valid=%s  reason=%s",
                            "true" if _guide_valid_flag else "false",
                            _guide_dir_x, _guide_dir_y,
                            _n_att[0], _n_att[1],
                            _g_att[0], _g_att[1],
                            _rep[0], _rep[1],
                            _n_cmd[0], _n_cmd[1],
                            _g_cmd[0], _g_cmd[1],
                            _delta_x, _delta_y,
                            "true" if _forward_preserved else "false",
                            _guided_output.command_magnitude,
                            _guided_output.valid,
                            _guided_output.reason,
                        )

                        # ── guided sign trace (Failure A Y-sign audit) ──
                        # Full chain: guidance_y → attractive_y → repulsive_y →
                        # resultant_y → mapped_vy → final_vy.  ``transform``
                        # records the coordinate transform applied (body FRD with
                        # NO vy=-vy flip and no world→body double conversion).
                        self._log_throttled(
                            "guided_sign_trace", 1.0,
                            "guided_sign_trace  "
                            "guidance_y=%.4f  attractive_y=%.4f  "
                            "repulsive_y=%.4f  resultant_y=%.4f  "
                            "mapped_vy=%.4f  final_vy=%.4f  "
                            "transform=%s",
                            _guide_dir_y,
                            _guided_output.attractive_force[1],
                            _guided_output.repulsive_force[1],
                            _guided_output.resultant_force[1],
                            _guided_output.desired_vy_body,
                            _g_cmd[1],
                            "body_frd_no_flip",
                        )
                    except Exception as _gapf_exc:
                        logger.warning("guided_apf_lateral_shadow_error: %s", _gapf_exc)

                # ── Trajectory-centric local planning (local_navigation_mode == "trajectory") ──
                # Phase C3-R: BOTH the global CBMBA replan and the local
                # trajectory planning run on separate OS processes.  This frame
                # only (a) requests replans/plans at cadence and (b) reads the
                # latest finished result non-blocking; the tracker re-derives the
                # command at the control rate from the cached trajectory, so the
                # slow planners and the fast control loop stay fully decoupled.
                traj_result = None
                if self._local_navigation_mode == "trajectory":
                    _now = time.monotonic()

                    # Global reference-path replan tick (non-blocking; the CBMBA
                    # A* runs in a separate process).  Runs every frame and is
                    # internally rate-limited to global_replan_hz.
                    _worker_t0 = time.perf_counter()
                    try:
                        self._tick_global_replan(st, _mission_goal, cbmba_obstacles, _now)
                    except Exception as _gre_exc:
                        logger.warning("trajectory_global_replan_tick_error: %s", _gre_exc)

                    # Local trajectory planning: request a fresh snapshot at
                    # ~planning_hz (latest-request-wins — at most one running +
                    # one pending slot in the worker), then poll the finished
                    # result every frame without ever blocking.
                    _plan_interval = 1.0 / max(0.1, self._traj_planning_hz)
                    _should_plan = (
                        not self._runtime_heading_alignment_active
                        and (
                            self._traj_force_replan
                            or not self._traj_cached_points
                            or _now - self._traj_last_plan_time >= _plan_interval
                        )
                    )
                    if _should_plan:
                        try:
                            self._request_local_plan(
                                st, _mission_goal, fr.filtered_points_sensor,
                            )
                            self._traj_last_plan_time = _now
                        except Exception as _traj_exc:
                            logger.warning("trajectory_plan_request_error: %s", _traj_exc)
                    _worker_request_ms = (time.perf_counter() - _worker_t0) * 1000.0

                    _worker_t0 = time.perf_counter()
                    try:
                        traj_result = self._poll_local_plan()
                    except Exception as _traj_poll_exc:
                        logger.warning("trajectory_plan_poll_error: %s", _traj_poll_exc)
                        traj_result = None
                    _worker_poll_ms = (time.perf_counter() - _worker_t0) * 1000.0

                    if traj_result is not None:
                        self._traj_last_apply_time = _now
                        _sel = traj_result.selected
                        if _sel is not None:
                            # Planning budget: log an overshoot but still
                            # PUBLISH the valid plan.  A plan that took 45 ms
                            # is still a valid plan; discarding it produced
                            # trajectory_no_feasible → dispatch(0,0,0) every
                            # frame (Phase C1-R sec 11).
                            if traj_result.compute_ms > self._traj_params.max_compute_ms:
                                logger.warning(
                                    "trajectory_planning_budget_exceeded  "
                                    "compute_ms=%.2f  max_ms=%.2f  fallback=none  "
                                    "publish=true",
                                    traj_result.compute_ms,
                                    self._traj_params.max_compute_ms,
                                )
                            self._traj_cached_points = list(_sel.points)
                            self._traj_cached_family = _sel.family
                            self._traj_narrow_passage_active = bool(
                                getattr(traj_result, "narrow_passage_active", False)
                            )
                            self._traj_force_replan = False
                            self._traj_no_feasible_count = 0
                            self._traj_no_feasible_start = None
                            logger.info(
                                "trajectory_plan  family=%s  score=%.3f  "
                                "clearance_min=%.2f  clear_mean=%.2f  "
                                "goal_progress=%.2f  goal_heading=%.2f  "
                                "path_align=%.2f  "
                                "smooth=%.2f  consist=%.2f  curv_pen=%.2f  "
                                "rev_pen=%.2f  unknown_pen=%.2f  "
                                "deviation=%.2f  cmd=(%.3f,%.3f)  "
                                "valid=%d/%d  best_clear=%.2f  "
                                "compute_ms=%.2f  switch=%s",
                                _sel.family, _sel.total_score,
                                _sel.min_clearance_m, _sel.mean_clearance_m,
                                _sel.goal_progress, _sel.goal_heading_alignment,
                                _sel.global_path_alignment,
                                _sel.smoothness, _sel.consistency,
                                _sel.curvature_penalty, _sel.reverse_penalty,
                                _sel.unknown_penalty, _sel.path_deviation_m,
                                traj_result.command_vx, traj_result.command_vy,
                                traj_result.valid_count, traj_result.generated,
                                traj_result.best_clearance_m,
                                traj_result.compute_ms,
                                traj_result.family_switch,
                            )
                        else:
                            # No feasible trajectory means the previously
                            # cached trajectory is no longer safe.  Invalidate
                            # it immediately; continuing to track the old plan
                            # here can drive straight into the obstacle while
                            # the recovery request is being accumulated.
                            self._traj_cached_points = []
                            self._traj_cached_family = None
                            self._traj_narrow_passage_active = False
                            self._traj_force_replan = True

                            # Accumulate the streak toward a fast Recovery
                            # request.  The dispatcher will hold position on
                            # this frame (or use the explicitly safe recovery
                            # command once the recovery state machine enters).
                            if self._traj_no_feasible_start is None:
                                self._traj_no_feasible_start = _now
                            self._traj_no_feasible_count += 1
                            _nfc = self._traj_no_feasible_count
                            _nfd = _now - self._traj_no_feasible_start
                            self._traj_escape_hint = traj_result.escape_hint
                            logger.warning(
                                "trajectory_plan  no_feasible  valid=%d/%d  "
                                "best_clear=%.2f  streak=%d/%d  dur=%.2f/%.2f  "
                                "escape_side=%s",
                                traj_result.valid_count, traj_result.generated,
                                traj_result.best_clearance_m, _nfc,
                                self._traj_no_feasible_recovery_count, _nfd,
                                self._traj_no_feasible_recovery_duration_s,
                                (self._traj_escape_hint or {}).get("side_label", "NONE"),
                            )
                            if (_nfc >= self._traj_no_feasible_recovery_count
                                    and _nfd >= self._traj_no_feasible_recovery_duration_s):
                                logger.warning(
                                    "trajectory_no_feasible_recovery  request=true  "
                                    "escape_side=%s",
                                    (self._traj_escape_hint or {}).get("side_label", "NONE"),
                                )
                                self._traj_request_recovery = True
                                self._traj_no_feasible_count = 0
                                self._traj_no_feasible_start = None

                # ── Goal termination (mission_complete → success) ──
                if self._local_navigation_mode == "trajectory":
                    _goal_t0 = time.perf_counter()
                    _goal_res = self._goal_term.update(
                        (st.position_ned_m[0], st.position_ned_m[1], st.position_ned_m[2]),
                        math.hypot(st.linear_velocity_ned_mps[0], st.linear_velocity_ned_mps[1]),
                        _mission_goal,
                        time.monotonic(),
                        velocity_ned_mps=st.linear_velocity_ned_mps,
                    )
                    _goal_check_ms = (time.perf_counter() - _goal_t0) * 1000.0
                    _goal_alt_err = abs(st.position_ned_m[2] - _navigation_target_z)
                    if _goal_res.reached or (fn == 1 or fn % 10 == 0):
                        logger.info(
                            "goal_termination  reached=%s  distance_xy=%.2f  "
                            "current_z=%.2f  target_z=%.2f  actor_z=%.2f  "
                            "altitude_error=%.2f  speed=%.2f  dwell=%.2f  "
                            "within_alt=%s  speed_low=%s",
                            "true" if _goal_res.reached else "false",
                            _goal_res.distance_to_goal_m,
                            st.position_ned_m[2], _navigation_target_z,
                            _mission_actor_xyz[2] if _mission_actor_xyz is not None else float("nan"),
                            _goal_alt_err,
                            math.hypot(st.linear_velocity_ned_mps[0], st.linear_velocity_ned_mps[1]),
                            _goal_res.dwell_elapsed_s,
                            "true" if _goal_res.within_altitude else "false",
                            "true" if _goal_res.speed_low else "false",
                        )
                    if _goal_res.reached:
                        term = "mission_complete"
                        break

                # Phase C1-R sec 14: planning-stage wall time ends here (all
                # planning/guidance done; only dispatch + logging remain).
                # Exclusive plan duration (not cumulative from loop start).
                _stg_plan_ms = (time.perf_counter() - _stg_plan_start) * 1000.0
                _stg_dispatch_start = time.perf_counter()

                # ── Command dispatch ──
                # Priority (highest first):
                #   1. Safety: collision / geofence / emergency → break (already handled above)
                #   2. Recovery: if state machine says should_override
                #   3. Trajectory (opt-in: local_navigation_mode == "trajectory")
                #   4. Guided APF (opt-in + conditions met, including path validity)
                #   5. Normal APF fallback
                #   6. Safe hold (no valid path + APF unavailable)
                #   7. Reactive

                # ── Bypass / rejoin episode lifecycle (Failure A) ──
                # Manage the state machine BEFORE dispatch so side enforcement
                # is available to all command sources.  Transition graph:
                #   NORMAL → BYPASS → REJOIN → NORMAL.
                # BYPASS commits a side to clear an obstacle; on
                # "obstacle_passed" it hands off to REJOIN (no side
                # enforcement — CBMBA guidance + guided APF re-aim at the goal)
                # rather than snapping straight back to NORMAL.
                _bypass_now = time.monotonic()
                _ref_path = (
                    cbmba_result.path_world
                    if (cbmba_result is not None and cbmba_result.success)
                    else None
                )
                if self._rejoin.active:
                    # ── REJOIN: no side commitment; evaluate exit → NORMAL ──
                    _rej_pos = (st.position_ned_m[0], st.position_ned_m[1])
                    _rej_ep = self._rejoin
                    _rej_ref_valid = bool(_rej_ep.reference_path_xy)
                    _rej_path_err = self._rejoin_path_error(_rej_pos, _rej_ep.reference_path_xy)
                    _rej_heading_err = self._rejoin_heading_error(st, _mission_goal)
                    _rej_elapsed = _bypass_now - _rej_ep.start_time
                    # Is the LIVE CBMBA path anchored to the current position?
                    # (If distance_to_latest_cbmba_first ≈ 0 while path_error > 0,
                    #  REJOIN is correctly NOT self-referencing.)
                    _latest_first = _ref_path[0] if _ref_path else None
                    if _latest_first is not None and len(_latest_first) >= 2:
                        _dist_latest_first = math.hypot(
                            _latest_first[0] - _rej_pos[0],
                            _latest_first[1] - _rej_pos[1],
                        )
                    else:
                        _dist_latest_first = float("inf")
                    logger.info(
                        "rejoin_status  elapsed=%.2f  "
                        "reference_valid=%s  reference_source=%s  "
                        "reference_generation_id=%s  "
                        "start_path_error=%.3f  path_error=%.3f  "
                        "path_error_reduction=%.3f  heading_error=%.3f  "
                        "path_threshold=%.3f  min_duration=%.2f  "
                        "latest_cbmba_first=(%.2f,%.2f)  "
                        "distance_to_latest_cbmba_first=%.3f  "
                        "exit_candidate=%s",
                        _rej_elapsed,
                        "true" if _rej_ref_valid else "false",
                        _rej_ep.reference_source or "none",
                        _rej_ep.reference_generation_id,
                        _rej_ep.start_path_error,
                        _rej_path_err,
                        _rej_ep.start_path_error - _rej_path_err,
                        _rej_heading_err,
                        _rej_ep.exit_path_error_m,
                        _rej_ep.min_duration_s,
                        _latest_first[0] if _latest_first is not None and len(_latest_first) >= 2 else float("nan"),
                        _latest_first[1] if _latest_first is not None and len(_latest_first) >= 2 else float("nan"),
                        _dist_latest_first,
                        "true" if _rej_path_err < _rej_ep.exit_path_error_m else "false",
                    )
                    if not _rej_ref_valid:
                        # Fail-closed: an empty frozen reference must NEVER be
                        # silently swapped for a live path — but it must be loud
                        # so the drone doesn't look "stuck in REJOIN for no reason".
                        logger.warning(
                            "rejoin_hold  reason=empty_frozen_reference  "
                            "path_error=inf  reference_source=\"%s\"  "
                            "reference_generation_id=%s  elapsed=%.2f",
                            _rej_ep.reference_source or "none",
                            _rej_ep.reference_generation_id,
                            _rej_elapsed,
                        )
                    _should_exit, _exit_reason = self._should_exit_rejoin(
                        st, _mission_goal, _ref_path, _bypass_now,
                    )
                    if _should_exit:
                        logger.info(
                            "rejoin_exit  reason=%s  elapsed=%.2f  "
                            "start_path_error=%.3f  final_path_error=%.3f  "
                            "path_error_reduction=%.3f",
                            _exit_reason,
                            _rej_elapsed,
                            self._rejoin.start_path_error,
                            _rej_path_err,
                            self._rejoin.start_path_error - _rej_path_err,
                        )
                        logger.info(
                            "normal_navigation  reason=rejoin_exit:%s  "
                            "rejoin_elapsed=%.2f",
                            _exit_reason,
                            _rej_elapsed,
                        )
                        self._rejoin = RejoinEpisode()
                        self._bypass = BypassEpisode()
                        self._bypass_unsafe_start = None
                elif self._bypass.active:
                    # Record the peak cross-track excursion against the frozen
                    # reference BEFORE deciding release, so the release
                    # destination (REJOIN vs NORMAL) reflects the whole episode.
                    self._track_bypass_excursion(
                        (st.position_ned_m[0], st.position_ned_m[1]),
                    )
                    # Check release conditions
                    _rel_side = self._side_label(self._bypass.side)
                    _should_rel, _rel_reason = self._should_release_bypass(rays, _bypass_now)
                    _byp_elapsed = _bypass_now - self._bypass.start_time
                    _byp_ref = self._bypass.reference_path_xy
                    _byp_ref_first = self._bypass.reference_first_xy
                    _byp_frozen = self._bypass.reference_frozen_position_xy
                    _byp_ref_last = _byp_ref[-1] if _byp_ref else None
                    _byp_fp = self._reference_fingerprint(_byp_ref)
                    logger.info(
                        "bypass_status  side=%s  elapsed=%.2f  "
                        "reference_source=%s  reference_generation=%s  "
                        "reference_points=%d  reference_first=(%.2f,%.2f)  "
                        "reference_last=(%.2f,%.2f)  "
                        "reference_frozen_position=(%.2f,%.2f)  "
                        "max_path_error=%.3f  rejoin_excursion_threshold=%.3f  "
                        "release_candidate=%s  fingerprint=%s",
                        _rel_side,
                        _byp_elapsed,
                        self._bypass.reference_source or "none",
                        self._bypass.reference_generation_id,
                        len(_byp_ref),
                        _byp_ref_first[0] if _byp_ref_first is not None else float("nan"),
                        _byp_ref_first[1] if _byp_ref_first is not None else float("nan"),
                        _byp_ref_last[0] if _byp_ref_last is not None else float("nan"),
                        _byp_ref_last[1] if _byp_ref_last is not None else float("nan"),
                        _byp_frozen[0] if _byp_frozen is not None else float("nan"),
                        _byp_frozen[1] if _byp_frozen is not None else float("nan"),
                        self._bypass.max_path_error_m,
                        self._rejoin_excursion_m,
                        "true" if _should_rel else "false",
                        _byp_fp,
                    )
                    if _should_rel:
                        _elapsed = _bypass_now - self._bypass.start_time
                        _byp_peak_err = self._bypass.max_path_error_m
                        _dest = self._bypass_release_destination(_rel_reason)
                        logger.info(
                            "bypass_release  bypass_state=bypass  "
                            "bypass_side=%s  bypass_elapsed=%.2f  "
                            "bypass_release_candidate=true  "
                            "bypass_release_reason=%s  release_destination=%s",
                            _rel_side, _elapsed, _rel_reason, _dest,
                        )
                        self._bypass_unsafe_start = None
                        if _dest == "rejoin":
                            _rej_pos = (st.position_ned_m[0], st.position_ned_m[1])
                            # Transfer the reference snapshot from the BYPASS
                            # episode.  It was frozen at bypass_enter (BEFORE the
                            # drone deviated), so reference[0] is NOT the current
                            # position — REJOIN's path_error stays meaningful.
                            # Never fall back to the live path here.
                            _rej_heading_err = self._rejoin_heading_error(st, _mission_goal)
                            self._rejoin = self._build_rejoin_from_bypass(
                                _rej_pos, _bypass_now, _rel_reason,
                            )
                            _ref_xy = self._rejoin.reference_path_xy
                            _ref_source = self._rejoin.reference_source
                            _ref_gen = self._rejoin.reference_generation_id
                            _ref_first = self._rejoin.reference_first_xy
                            _frozen_pos = self._rejoin.reference_frozen_position_xy
                            _rej_path_err = self._rejoin.start_path_error
                            _ref_last = _ref_xy[-1] if _ref_xy else None
                            _ref_fp = self._reference_fingerprint(_ref_xy)
                            _dist_ref_first = (
                                math.hypot(_ref_first[0] - _rej_pos[0], _ref_first[1] - _rej_pos[1])
                                if _ref_first is not None else float("inf")
                            )
                            _dist_frozen_pos = (
                                math.hypot(_frozen_pos[0] - _rej_pos[0], _frozen_pos[1] - _rej_pos[1])
                                if _frozen_pos is not None else float("inf")
                            )
                            self._bypass = BypassEpisode()
                            logger.info(
                                "rejoin_reference_inherit  same_as_bypass=true  "
                                "reference_generation=%s  "
                                "reference_first=(%.2f,%.2f)  "
                                "reference_last=(%.2f,%.2f)  "
                                "reference_frozen_position=(%.2f,%.2f)  "
                                "fingerprint=%s",
                                _ref_gen,
                                _ref_first[0] if _ref_first is not None else float("nan"),
                                _ref_first[1] if _ref_first is not None else float("nan"),
                                _ref_last[0] if _ref_last is not None else float("nan"),
                                _ref_last[1] if _ref_last is not None else float("nan"),
                                _frozen_pos[0] if _frozen_pos is not None else float("nan"),
                                _frozen_pos[1] if _frozen_pos is not None else float("nan"),
                                _ref_fp,
                            )
                            logger.info(
                                "rejoin_enter  reason=%s  side=%s  elapsed=%.2f  "
                                "position=(%.2f,%.2f)  "
                                "reference_source=%s  reference_generation_id=%s  "
                                "reference_points=%d  reference_first=(%.2f,%.2f)  "
                                "reference_last=(%.2f,%.2f)  "
                                "reference_frozen_position=(%.2f,%.2f)  "
                                "distance_to_reference_first=%.3f  "
                                "distance_to_frozen_position=%.3f  "
                                "start_path_error=%.3f  heading_error=%.3f  "
                                "fingerprint=%s",
                                _rel_reason, _rel_side, _elapsed,
                                _rej_pos[0], _rej_pos[1],
                                _ref_source,
                                _ref_gen,
                                len(_ref_xy),
                                _ref_first[0] if _ref_first is not None else float("nan"),
                                _ref_first[1] if _ref_first is not None else float("nan"),
                                _ref_last[0] if _ref_last is not None else float("nan"),
                                _ref_last[1] if _ref_last is not None else float("nan"),
                                _frozen_pos[0] if _frozen_pos is not None else float("nan"),
                                _frozen_pos[1] if _frozen_pos is not None else float("nan"),
                                _dist_ref_first,
                                _dist_frozen_pos,
                                _rej_path_err, _rej_heading_err,
                                _ref_fp,
                            )
                        else:
                            if _rel_reason.startswith("obstacle_passed"):
                                logger.info(
                                    "bypass_release  reason=no_excursion  "
                                    "peak_path_error=%.3f < excursion=%.3f  "
                                    "destination=normal",
                                    _byp_peak_err, self._rejoin_excursion_m,
                                )
                            self._bypass = BypassEpisode()
                else:
                    # Check entry conditions (only when guided APF is available)
                    if self._guided_apf_control and _guidance_result is not None:
                        _should_enter, _enter_reason = self._should_enter_bypass(
                            rays,
                            _guidance_result.direction_body_xy if _guidance_result.valid else None,
                        )
                        if _should_enter:
                            _chosen_side = self._choose_bypass_side(
                                rays,
                                _guidance_result.direction_body_xy if _guidance_result.valid else None,
                            )
                            # Freeze the reference at bypass_enter, BEFORE the
                            # drone deviates, so REJOIN can later measure a
                            # meaningful cross-track error against it (not a live
                            # path re-seeded from the current position).
                            _byp_enter_pos_xy = (st.position_ned_m[0], st.position_ned_m[1])
                            _byp_enter_ref = self._freeze_reference_xy(
                                _byp_enter_pos_xy, _ref_path,
                            )
                            self._bypass = BypassEpisode(
                                active=True,
                                side=_chosen_side,
                                start_time=_bypass_now,
                                reason=_enter_reason,
                                min_duration_s=self._bypass_min_duration_s,
                                entry_clearance_side_m=(
                                    rays.get("right", 0.0) if _chosen_side == 1
                                    else rays.get("left", 0.0)
                                ) or 0.0,
                                reference_path_xy=_byp_enter_ref[0],
                                reference_source=_byp_enter_ref[1],
                                reference_generation_id=_byp_enter_ref[2],
                                reference_first_xy=_byp_enter_ref[3],
                                reference_frozen_position_xy=_byp_enter_pos_xy,
                            )
                            self._bypass_unsafe_start = None
                            _byp_enter_last = (
                                _byp_enter_ref[0][-1] if _byp_enter_ref[0] else None
                            )
                            _byp_enter_fp = self._reference_fingerprint(_byp_enter_ref[0])
                            logger.info(
                                "bypass_enter  side=%s  reason=%s  "
                                "entry_clearance=%.2f  min_duration=%.2f  "
                                "reference_source=%s  reference_generation=%s  "
                                "reference_first=(%.2f,%.2f)  "
                                "reference_last=(%.2f,%.2f)  ref_len=%d  "
                                "fingerprint=%s",
                                self._side_label(_chosen_side),
                                _enter_reason,
                                self._bypass.entry_clearance_side_m,
                                self._bypass_min_duration_s,
                                _byp_enter_ref[1] or "none",
                                self._bypass.reference_generation_id,
                                _byp_enter_ref[3][0] if _byp_enter_ref[3] else float("nan"),
                                _byp_enter_ref[3][1] if _byp_enter_ref[3] else float("nan"),
                                _byp_enter_last[0] if _byp_enter_last else float("nan"),
                                _byp_enter_last[1] if _byp_enter_last else float("nan"),
                                len(_byp_enter_ref[0]),
                                _byp_enter_fp,
                            )

                _family = "STRAIGHT"  # default; trajectory branch overrides
                _selected_yaw_rate: Optional[float] = None
                _selected_command_duration = self._params.command_duration_s
                if recovery_result.should_override:
                    selected_vx = recovery_result.vx_body
                    selected_vy = recovery_result.vy_body
                    selected_vz = recovery_result.vz_body
                    selected_vx, selected_vy = self._recovery_directional_guard(
                        selected_vx, selected_vy, rays,
                    )
                    command_source = "recovery"
                    # Update bypass side from recovery (sync)
                    if (recovery_result.committed_side is not None
                            and recovery_result.committed_side != 0):
                        if not self._bypass.active:
                            # P1-B: recovery may only persist a committed side
                            # into a BYPASS episode when the formal entry gate
                            # passes (a real corridor constraint).  When both
                            # sides are open the recovery was a false trigger
                            # (e.g. a stale-hover false stuck) and there is no
                            # side to persist — skip the bypass so the drone goes
                            # straight back to NORMAL rather than a spurious
                            # BYPASS→REJOIN round-trip.
                            _formal_ok, _formal_reason = self._inheritance_formal_entry(
                                rays,
                                (
                                    _guidance_result.direction_body_xy
                                    if _guidance_result is not None and _guidance_result.valid
                                    else None
                                ),
                            )
                            logger.info(
                                "bypass_recovery_inherit  source=recovery_inherit  "
                                "side=%s  left=%.2f  right=%.2f  front=%.2f  "
                                "formal_entry_allowed=%s  formal_entry_reason=%s  "
                                "inherited=%s",
                                self._side_label(recovery_result.committed_side),
                                rays.get("left", float("inf")) or float("inf"),
                                rays.get("right", float("inf")) or float("inf"),
                                rays.get("front", float("inf")) or float("inf"),
                                "true" if _formal_ok else "false",
                                _formal_reason,
                                "true" if _formal_ok else "false",
                            )
                            if _formal_ok:
                                # Freeze the reference at recovery-inherited bypass
                                # creation, BEFORE the drone deviates, using this
                                # frame's CBMBA path (in scope as _ref_path).
                                _inh_pos_xy = (st.position_ned_m[0], st.position_ned_m[1])
                                _inh_ref = self._freeze_reference_xy(_inh_pos_xy, _ref_path)
                                self._bypass = BypassEpisode(
                                    active=True,
                                    side=recovery_result.committed_side,
                                    start_time=_bypass_now,
                                    reason="inherited_from_recovery",
                                    min_duration_s=self._bypass_min_duration_s,
                                    entry_clearance_side_m=0.0,
                                    reference_path_xy=_inh_ref[0],
                                    reference_source=_inh_ref[1],
                                    reference_generation_id=_inh_ref[2],
                                    reference_first_xy=_inh_ref[3],
                                    reference_frozen_position_xy=_inh_pos_xy,
                                )
                                self._bypass_unsafe_start = None
                        elif self._bypass.side != recovery_result.committed_side:
                            # Recovery changed side — update bypass
                            self._bypass.side = recovery_result.committed_side
                            self._bypass_unsafe_start = None
                elif self._local_navigation_mode == "trajectory":
                    # ── Trajectory-centric dispatch (recovery already handled above) ──
                    # Track the cached trajectory at control rate via pure pursuit
                    # (NOT a cached velocity command), then apply the APF safety
                    # filter — never the old Bypass hard-enforce.
                    _family = self._traj_cached_family or "STRAIGHT"
                    # Phase C3-R: staleness is measured from the last APPLY time
                    # (not the last request), so worker latency (~100-250 ms)
                    # doesn't trip the stale warning spuriously.
                    _traj_age_s = time.monotonic() - self._traj_last_apply_time

                    # ── cached-trajectory staleness (sec 9) ──
                    _stale = False
                    if self._traj_cached_points:
                        if _traj_age_s > self._traj_stale_stop_s:
                            _stale = True
                            logger.warning(
                                "trajectory_stale  age=%.3f > stop=%.3f  action=hover",
                                _traj_age_s, self._traj_stale_stop_s,
                            )
                        elif _traj_age_s > self._traj_stale_warn_s:
                            logger.warning(
                                "trajectory_stale  age=%.3f > warn=%.3f",
                                _traj_age_s, self._traj_stale_warn_s,
                            )

                    if _stale:
                        # Do not track a trajectory older than the stop limit.
                        selected_vx = selected_vy = selected_vz = 0.0
                        command_source = "trajectory_stale"
                    elif self._traj_cached_points:
                        _track_t0 = time.perf_counter()
                        _track = self._traj_tracker.compute_command(
                            self._traj_cached_points, st.position_ned_m, st.yaw_rad,
                            is_reverse=_family.startswith("REVERSE_"),
                            goal_xy=(_mission_goal[0], _mission_goal[1]),
                            goal_ned=(_mission_goal[0], _mission_goal[1], _mission_goal[2]),
                        )
                        _tracker_ms = (time.perf_counter() - _track_t0) * 1000.0
                        selected_vx = _track.vx
                        selected_vy = _track.vy
                        selected_vz = _track.vz
                        command_source = "trajectory"
                        _cmd_body_vy = selected_vy
                        if self._traj_narrow_passage_active and _family == "STRAIGHT":
                            # Do not let pure-pursuit or a stale lateral error
                            # create a last-second side step inside a gap.
                            selected_vy = 0.0
                        if self._trajectory_yaw_enabled and not _family.startswith("REVERSE_"):
                            _selected_yaw_rate = _track.yaw_rate_radps

                        # APF safety filter: limited nudge + forward-speed scaling.
                        _apf_t0 = time.perf_counter()
                        selected_vx, selected_vy = self._apf_safety_filter(
                            selected_vx, selected_vy, rays, dd.minimum_distance_m,
                            fr.filtered_points_sensor,
                            preserve_centerline=self._traj_narrow_passage_active,
                        )
                        if dd.minimum_distance_m < self._params.emergency_distance_m:
                            _selected_yaw_rate = 0.0
                        _apf_safety_ms = (time.perf_counter() - _apf_t0) * 1000.0
                        _filtered_body_vy = selected_vy

                        # Recovery may have committed to one wall of a large
                        # U-shaped dead end. Apply that commitment after the
                        # tracker and safety filter so a fresh goal-directed
                        # trajectory cannot pull the vehicle back inside.
                        if self._bypass.active and self._bypass.trajectory_dead_end:
                            _pre_vx, _pre_vy = selected_vx, selected_vy
                            selected_vx, selected_vy = self._enforce_trajectory_dead_end_bypass(
                                selected_vx, selected_vy, rays,
                            )
                            _selected_yaw_rate = 0.0
                            if (
                                abs(selected_vx - _pre_vx) > 1e-9
                                or abs(selected_vy - _pre_vy) > 1e-9
                            ):
                                self._log_throttled(
                                    "dead_end_bypass_enforce", 1.0,
                                    "dead_end_bypass_enforce  side=%s  "
                                    "pre=(%.3f,%.3f)  post=(%.3f,%.3f)",
                                    self._side_label(self._bypass.side),
                                    _pre_vx, _pre_vy, selected_vx, selected_vy,
                                )

                        # ── sign trace (LEFT → negative body vy contract) ──
                        _gen_end_y = self._traj_end_body_y(self._traj_cached_points, st)
                        _world_end_y = self._traj_cached_points[-1][1]
                        self._log_throttled(
                            "trajectory_sign_trace", 1.0,
                            "trajectory_sign_trace  family=%s  generator_end_y=%.3f  "
                            "world_end_y=%.3f  command_body_vy=%.3f  "
                            "filtered_body_vy=%.3f  final_body_vy=%.3f",
                            _family, _gen_end_y, _world_end_y,
                            _cmd_body_vy, _filtered_body_vy, selected_vy,
                        )
                        from planners.local_trajectory_planner import family_side as _family_side
                        if (_family_side(_family) == -1
                                and selected_vy > 0.0
                                and command_source == "trajectory"):
                            logger.warning(
                                "TRAJECTORY_DIRECTION_VIOLATION  family=%s  "
                                "final_body_vy=%.3f  (LEFT must be ≤ 0)",
                                _family, selected_vy,
                            )

                        # ── tracking error → force replan when too large ──
                        self._update_trajectory_tracking(st, selected_vx, selected_vy)
                    else:
                        # No feasible trajectory cached → hold in place.  Fast
                        # recovery is requested by the planning block above.
                        selected_vx = selected_vy = selected_vz = 0.0
                        command_source = "trajectory_no_feasible"
                elif self._planner_mode == "apf":
                    # ── Guided APF takeover evaluation ──
                    _takeover = False
                    if self._guided_apf_control:
                        # Path validity gate: reject guided takeover when CBMBA
                        # path is invalid (Failures A & B).
                        if not self._path_valid:
                            _fallback_reason = f"path_invalid:{self._path_fail_reason}"
                        elif (_guide_valid_flag
                                and _guided_output is not None
                                and _guided_output.valid
                                and math.isfinite(_g_cmd[0])
                                and math.isfinite(_g_cmd[1])):
                            # forward_sign_guard: if normal pushes forward but
                            # guided pushes backward, refuse takeover
                            if _n_cmd[0] > 0.0 and _g_cmd[0] < 0.0:
                                _fallback_reason = "forward_sign_guard"
                            else:
                                _takeover = True
                        else:
                            if not _guide_valid_flag:
                                _fallback_reason = "guidance_invalid"
                            elif _guided_output is None:
                                _fallback_reason = "guided_unavailable"
                            elif not _guided_output.valid:
                                _fallback_reason = "guided_invalid"
                            else:
                                _fallback_reason = "guided_nan_inf"

                    if _takeover:
                        selected_vx = _g_cmd[0]
                        selected_vy = _g_cmd[1]
                        selected_vz = 0.0
                        command_source = "guided_apf"
                        # ── Bypass enforcement: clamp vy when bypass active ──
                        if self._bypass.active:
                            _pre_vx, _pre_vy = selected_vx, selected_vy
                            selected_vx, selected_vy = self._enforce_bypass_side(
                                selected_vx, selected_vy, self._bypass.side,
                            )
                            if abs(selected_vy - _pre_vy) > 1e-9:
                                logger.info(
                                    "bypass_enforce  side=%s  "
                                    "pre=(%.3f,%.3f)  post=(%.3f,%.3f)",
                                    self._side_label(self._bypass.side),
                                    _pre_vx, _pre_vy,
                                    selected_vx, selected_vy,
                                )
                    elif apf_output is not None and apf_output.valid:
                        selected_vx = apf_output.desired_vx_body
                        selected_vy = apf_output.desired_vy_body
                        selected_vz = apf_output.desired_vz_body
                        command_source = "apf"
                        # ── Bypass enforcement for normal APF ──
                        if self._bypass.active:
                            _pre_vx, _pre_vy = selected_vx, selected_vy
                            selected_vx, selected_vy = self._enforce_bypass_side(
                                selected_vx, selected_vy, self._bypass.side,
                            )
                            if abs(selected_vy - _pre_vy) > 1e-9:
                                logger.info(
                                    "bypass_enforce  side=%s  "
                                    "pre=(%.3f,%.3f)  post=(%.3f,%.3f)",
                                    self._side_label(self._bypass.side),
                                    _pre_vx, _pre_vy,
                                    selected_vx, selected_vy,
                                )
                    else:
                        # ── Safe hold: no valid APF output ──
                        # If the path is persistently invalid, hold position
                        # rather than fall through to reactive (which may drive
                        # the drone into danger).
                        if (self._consecutive_invalid_paths
                                >= self._max_consecutive_invalid_paths):
                            selected_vx = selected_vy = selected_vz = 0.0
                            command_source = "safe_hold"
                            logger.warning(
                                "safe_hold  consecutive_invalid_paths=%d  "
                                "reason=%s",
                                self._consecutive_invalid_paths,
                                self._path_fail_reason,
                            )
                        else:
                            selected_vx = selected_vy = selected_vz = 0.0
                            command_source = "apf_invalid_hold"

                    # ── Guided APF takeover log (only when feature enabled) ──
                    if self._guided_apf_control:
                        self._log_throttled(
                            "guided_apf_takeover", 1.0,
                            "guided_apf_takeover  "
                            "enabled=true  "
                            "guidance_valid=%s  "
                            "guided_valid=%s  "
                            "normal_cmd=(%.4f,%.4f)  "
                            "guided_cmd=(%.4f,%.4f)  "
                            "source=%s  "
                            "fallback_reason=%s  "
                            "path_valid=%s  "
                            "bypass_active=%s",
                            "true" if _guide_valid_flag else "false",
                            "true" if (_guided_output is not None and _guided_output.valid) else "false",
                            _n_cmd[0], _n_cmd[1],
                            _g_cmd[0], _g_cmd[1],
                            command_source,
                            _fallback_reason if not _takeover else "",
                            "true" if self._path_valid else "false",
                            "true" if self._bypass.active else "false",
                        )
                else:
                    # reactive or apf_shadow: reactive commands the drone
                    selected_vx = dec.vx_body_mps
                    selected_vy = dec.vy_body_mps
                    selected_vz = 0.0
                    command_source = "reactive"
                    # ── Bypass enforcement for reactive mode ──
                    if self._bypass.active:
                        _pre_vx, _pre_vy = selected_vx, selected_vy
                        selected_vx, selected_vy = self._enforce_bypass_side(
                            selected_vx, selected_vy, self._bypass.side,
                        )
                        if abs(selected_vy - _pre_vy) > 1e-9:
                            logger.info(
                                "bypass_enforce  side=%s  "
                                "pre=(%.3f,%.3f)  post=(%.3f,%.3f)",
                                self._side_label(self._bypass.side),
                                _pre_vx, _pre_vy,
                                selected_vx, selected_vy,
                            )

                # ── trajectory-mode runtime invariants (defensive checks) ──
                # Runtime goal-direction guard.  This must run after the
                # trajectory/recovery selection, otherwise a stale STRAIGHT
                # command can continue after the goal has moved behind the
                # vehicle.
                if (
                    self._local_navigation_mode == "trajectory"
                    and not recovery_result.should_override
                    and not self._bypass.active
                ):
                    (
                        _runtime_align_active,
                        _runtime_align_completed,
                        _runtime_yaw_rate,
                    ) = self._runtime_heading_alignment_command(
                        st, _mission_goal, time.monotonic(),
                    )
                    if _runtime_align_active:
                        selected_vx = selected_vy = 0.0
                        command_source = "runtime_heading_alignment"
                        _selected_yaw_rate = _runtime_yaw_rate
                        _selected_command_duration = min(
                            self._params.command_duration_s,
                            self._runtime_heading_alignment_command_duration_s,
                        )
                    elif _runtime_align_completed:
                        selected_vx = selected_vy = 0.0
                        command_source = "runtime_heading_alignment_settle"
                        _selected_yaw_rate = 0.0
                        _selected_command_duration = min(
                            self._params.command_duration_s,
                            self._runtime_heading_alignment_command_duration_s,
                        )

                if self._local_navigation_mode == "trajectory":
                    if command_source in ("guided_apf", "apf", "reactive"):
                        logger.warning(
                            "TRAJECTORY_MODE_UNEXPECTED_FALLBACK  source=%s  "
                            "planner_mode=%s",
                            command_source, self._planner_mode,
                        )
                    if (not recovery_result.should_override
                            and self._traj_cached_points
                            and command_source not in (
                                "trajectory",
                                "runtime_heading_alignment",
                                "runtime_heading_alignment_settle",
                            )):
                        logger.warning(
                            "TRAJECTORY_MODE_INVARIANT_VIOLATION  "
                            "cached_trajectory=true  source=%s",
                            command_source,
                        )

                # ── control-loop watchdog hover (sec 11): a severe stall must
                #    not let the drone blindly continue a stale command. ──
                if self._loop_current_overrun_ms > self._control_loop_overrun_stop_ms:
                    selected_vx = selected_vy = selected_vz = 0.0
                    command_source = "control_loop_hover"

                # ── trajectory flight validation (sec 1-3, 23-25) + trace ──
                if self._local_navigation_mode == "trajectory":
                    _px = st.position_ned_m[0]
                    _py = st.position_ned_m[1]
                    _pos_xy = (_px, _py)

                    # ── low-frequency flight_state log (Phase C1 sec 5) ──
                    if fn == 1 or fn % 10 == 0:
                        _goal_dist = math.hypot(
                            _px - _mission_goal[0], _py - _mission_goal[1],
                        )
                        _path_error = float("inf")
                        if self._traj_cached_points:
                            _path_error = min(
                                math.hypot(wp[0] - _px, wp[1] - _py)
                                for wp in self._traj_cached_points
                                if wp is not None and len(wp) >= 2
                            )
                        logger.info(
                            "flight_state  pos=(%.2f,%.2f,%.2f)  "
                            "vel=(%.2f,%.2f,%.2f)  goal_distance=%.2f  "
                            "path_error=%.3f  family=%s  "
                            "mission_goal_source=%s  min_clearance=%.2f  "
                            "command_source=%s",
                            st.position_ned_m[0], st.position_ned_m[1],
                            st.position_ned_m[2],
                            st.linear_velocity_ned_mps[0],
                            st.linear_velocity_ned_mps[1],
                            st.linear_velocity_ned_mps[2],
                            _goal_dist, _path_error, _family,
                            _mission_goal_source, dd.minimum_distance_m,
                            command_source,
                        )

                    # nearest current-LiDAR obstacle world XY (for persistence).
                    _nearest_obs_xy = None
                    if (fr.filtered_points_sensor is not None
                            and getattr(fr.filtered_points_sensor, "size", 0)):
                        _wt_t0 = time.perf_counter()
                        _wxy = _sensor_points_to_world_xy(
                            fr.filtered_points_sensor, st.position_ned_m, st.yaw_rad,
                            max_range=self._occ_grid_params.max_range_m,
                            horizontal_band=self._occ_grid_params.horizontal_band_half_height_m,
                        )
                        _world_transform_metrics_ms = (time.perf_counter() - _wt_t0) * 1000.0
                        if _wxy:
                            _nearest_obs_xy = min(
                                _wxy, key=lambda p: (p[0] - _px) ** 2 + (p[1] - _py) ** 2,
                            )

                    # (1) metrics frame
                    _metrics_t0 = time.perf_counter()
                    self._traj_metrics.record_frame(
                        _pos_xy, dd.minimum_distance_m, selected_vx, selected_vy,
                    )
                    self._traj_metrics.total_frames = fn

                    # (2) avoidance episode (observation-only)
                    _fam_side = 0
                    if command_source == "trajectory":
                        from planners.local_trajectory_planner import family_side as _fs2
                        _fam_side = _fs2(_family)
                    _ep = self._traj_episode_tracker.update(
                        fn, time.monotonic(), dd.minimum_distance_m,
                        _fam_side, _pos_xy, _nearest_obs_xy,
                    )
                    if _ep is not None:
                        self._traj_metrics.num_avoidance_episodes += 1
                        self._traj_metrics.total_avoidance_duration_s += _ep.duration_s
                        if _ep.success:
                            self._traj_metrics.avoidance_successes += 1
                        logger.info(
                            "avoidance_episode  id=%d  side=%s  dur=%.2f  "
                            "approach=%.2f  closest=%.2f  final=%.2f  success=%s",
                            _ep.episode_id, _ep.side, _ep.duration_s,
                            _ep.approach_distance_m, _ep.closest_distance_m,
                            _ep.final_distance_m, "true" if _ep.success else "false",
                        )

                    # (25) single-obstacle persistence
                    self._obstacle_behavior_monitor.update(_pos_xy, _nearest_obs_xy)
                    _metrics_ms = (time.perf_counter() - _metrics_t0) * 1000.0

                    # (23/24) mission progress monitor
                    _progress_t0 = time.perf_counter()
                    _mp_status = self._mission_progress_monitor.update(
                        time.monotonic(), _pos_xy,
                    )
                    _progress_monitor_ms = (time.perf_counter() - _progress_t0) * 1000.0
                    if _mp_status is not None:
                        if _mp_status.is_stuck:
                            self._traj_metrics.mission_stuck_count += 1
                            logger.warning(
                                "mission_progress  status=stuck  path_len=%.2f  window=%.1f",
                                _mp_status.path_length_m, _mp_status.window_s,
                            )
                        elif _mp_status.is_mission_stalled:
                            self._traj_metrics.mission_stalled_count += 1
                            logger.warning(
                                "mission_progress  status=mission_stalled  "
                                "progress=%.2f  path_len=%.2f  window=%.1f",
                                _mp_status.progress_m, _mp_status.path_length_m,
                                _mp_status.window_s,
                            )

                    # (3) family transition log (from this tick's plan, if any)
                    if traj_result is not None and traj_result.family_switch is not None:
                        _from, _to, _reason = traj_result.family_switch
                        from planners.local_trajectory_planner import family_side as _fs3
                        _direct = (_fs3(_from) != 0 and _fs3(_to) == -_fs3(_from))
                        self._traj_family_log.record(
                            _from, _to, _reason, fn, time.monotonic(), _direct,
                        )
                        self._traj_metrics.num_family_switches += 1
                        if _direct:
                            self._traj_metrics.num_direct_opposite_switches += 1
                        logger.info(
                            "family_transition  from=%s  to=%s  reason=%s  "
                            "direct_opposite=%s",
                            _from, _to, _reason, "true" if _direct else "false",
                        )
                    if traj_result is not None and traj_result.selected is None:
                        self._traj_metrics.num_no_feasible_events += 1
                    if recovery_result.should_override:
                        self._traj_metrics.num_recovery_events += 1

                    # (19/20) debug drawing + HUD
                    # simPlot* and simPrintLogMessage are synchronous RPCs. A
                    # full draw set every 20 Hz can block the control loop for
                    # hundreds of milliseconds. Keep the diagnostics visible,
                    # but update them at a low bounded rate.
                    _log_t0 = time.perf_counter()
                    _draw_now = time.monotonic()
                    _draw_due = (
                        self._debug_drawer is not None
                        and _draw_now - self._debug_draw_last_mono
                        >= self._debug_draw_period_s
                    )
                    if _draw_due:
                        self._debug_draw_last_mono = _draw_now
                        _z = st.position_ned_m[2]
                        _draw_global_path = None
                        _draw_selected = None
                        _draw_obstacles = None
                        _draw_goal = None
                        _draw_goal_line = None
                        if self._traj_debug_cfg.get("draw_global_path", True):
                            _draw_global_path = list(self._traj_global_path)
                        if (self._traj_debug_cfg.get("draw_selected_trajectory", True)
                                and self._traj_cached_points):
                            _draw_selected = list(self._traj_cached_points)
                        if self._traj_debug_cfg.get("draw_obstacles", True):
                            _r2 = self._traj_dfield_radius_m ** 2
                            _occupied_near = [
                                (x, y) for (x, y) in
                                self._map_snapshot.get("occupied_points", [])
                                if (x - _px) ** 2 + (y - _py) ** 2 <= _r2
                            ]
                            _max_draw_obs = max(
                                1, int(self._traj_debug_cfg.get(
                                    "max_obstacle_points", 200,
                                ))
                            )
                            if len(_occupied_near) > _max_draw_obs:
                                _step = max(1, len(_occupied_near) // _max_draw_obs)
                                _occupied_near = _occupied_near[::_step][:_max_draw_obs]
                            _draw_obstacles = _occupied_near
                        if self._traj_debug_cfg.get("draw_mission_goal", True):
                            _draw_goal = (_mission_goal[0], _mission_goal[1])
                        if self._traj_debug_cfg.get("draw_goal_line", True):
                            _draw_goal_line = (
                                (_px, _py), (_mission_goal[0], _mission_goal[1]),
                            )
                        _draw_hud = None
                        if self._traj_debug_cfg.get("hud_status", True):
                            _draw_hud = (
                                f"{command_source} {_family} "
                                f"minD={dd.minimum_distance_m:.1f}m"
                            )
                        # This is a non-blocking queue operation.  The actual
                        # simPlot*/HUD RPCs run on the drawer's separate thread
                        # and separate AirSim connection.
                        self._debug_drawer.submit_frame(
                            global_path=_draw_global_path,
                            selected_trajectory=_draw_selected,
                            obstacles=_draw_obstacles,
                            mission_goal=_draw_goal,
                            goal_line=_draw_goal_line,
                            z=_z,
                            goal_z=(
                                _z - float(self._traj_debug_cfg.get(
                                    "goal_z_offset_m", 2.0,
                                ))
                            ),
                            hud_message=_draw_hud,
                        )
                    _log_enqueue_ms = (time.perf_counter() - _log_t0) * 1000.0

                    # (21) CSV trace
                    if self._trace_writer is not None:
                        _csv_t0 = time.perf_counter()
                        _dist_to_goal = math.hypot(
                            _px - _mission_goal[0], _py - _mission_goal[1],
                        )
                        _goal_progress = (
                            self._traj_metrics.initial_distance_to_goal_m - _dist_to_goal
                        )
                        self._trace_writer.write_row([
                            fn, time.monotonic(),
                            _px, _py, st.position_ned_m[2], st.yaw_rad,
                            st.linear_velocity_ned_mps[0],
                            st.linear_velocity_ned_mps[1],
                            st.linear_velocity_ned_mps[2],
                            selected_vx, selected_vy,
                            command_source, _family, dd.minimum_distance_m,
                            _goal_progress,
                            self._traj_metrics.max_lateral_deviation_m,
                        ])
                        _csv_enqueue_ms = (time.perf_counter() - _csv_t0) * 1000.0

                # ── command dispatch ──
                # SimpleFlight 不支持 moveByVelocityZBodyFrameAsync（altitude
                # position hold），命令被忽略导致无人机不动。改用纯速度控制
                # (moveByVelocityBodyFrameAsync) + 手动高度 P 控制器。
                _use_altitude_hold = False

                # trajectory 模式加高度 P 控制器（其他模式保持原有 vz）
                if self._local_navigation_mode == "trajectory" and abs(selected_vz) < 0.01:
                    selected_vz = self._altitude_hold_velocity(
                        st.position_ned_m[2], self._params.target_z_ned,
                        self._params.max_vertical_speed_mps,
                    )
                # Altitude P-control must not reintroduce vz after a severe
                # control-loop stall has requested a hover.
                if self._loop_current_overrun_ms > self._control_loop_overrun_stop_ms:
                    selected_vx = selected_vy = selected_vz = 0.0
                    command_source = "control_loop_hover"
                    _selected_yaw_rate = 0.0
                self._record_dispatch_source(command_source)
                _alt_err = abs(st.position_ned_m[2] - self._params.target_z_ned)
                self._max_altitude_error_m = max(self._max_altitude_error_m, _alt_err)
                if _use_altitude_hold:
                    self._log_throttled(
                        "control_dispatch", 1.0,
                        "control_dispatch  planner_mode=%s  source=%s  "
                        "cmd_xy=(%.4f,%.4f)  target_z=%.4f  "
                        "yaw_rate=%s  duration=%.3f  "
                        "api=moveByVelocityZBodyFrameAsync",
                        self._planner_mode, command_source,
                        selected_vx, selected_vy, self._params.target_z_ned,
                        _selected_yaw_rate, _selected_command_duration,
                    )
                else:
                    self._log_throttled(
                        "control_dispatch", 1.0,
                        "control_dispatch  planner_mode=%s  source=%s  "
                        "cmd=(%.4f,%.4f,%.4f)  yaw_rate=%s  duration=%.3f  "
                        "api=moveByVelocityBodyFrameAsync",
                        self._planner_mode, command_source,
                        selected_vx, selected_vy, selected_vz,
                        _selected_yaw_rate, _selected_command_duration,
                    )

                try:
                    _t_rpc = time.perf_counter()
                    if _use_altitude_hold:
                        self._last_velocity_future = vc.send_velocity_body_frd_z(
                            selected_vx, selected_vy,
                            self._params.target_z_ned,
                            duration=_selected_command_duration,
                            vehicle_name=self._vn,
                            yaw_rate=_selected_yaw_rate,
                        )
                    else:
                        self._last_velocity_future = vc.send_velocity_body_frd(
                            selected_vx, selected_vy, selected_vz,
                            duration=_selected_command_duration,
                            vehicle_name=self._vn,
                            yaw_rate=_selected_yaw_rate,
                            yaw_rad=st.yaw_rad,
                        )
                    _rpc_velocity_command_ms = (time.perf_counter() - _t_rpc) * 1000.0
                    _rpc_velocity_submit_ms = _rpc_velocity_command_ms
                except Exception:
                    term = "velocity_send_error"
                    break

                # ── write current LiDAR into persistent map (read-before-write) ──
                # Phase C4-R: the ray-casting ``update()`` now runs in the
                # mapping **process**; the loop only requests + polls non-blocking.
                if self._local_navigation_mode == "trajectory":
                    try:
                        _mw_t0 = time.perf_counter()
                        self._request_map_update(lf, st, fr)
                        _map_worker_request_ms = (time.perf_counter() - _mw_t0) * 1000.0
                        _mw_t0 = time.perf_counter()
                        self._poll_map_update()
                        _map_worker_poll_ms = (time.perf_counter() - _mw_t0) * 1000.0
                    except Exception as _map_exc:
                        logger.warning("mapping_worker_error: %s", _map_exc)

                # Phase C2: exclusive stage breakdown (each stage is a delta, not
                # cumulative-from-loop-start) + individual RPC timings.  The old
                # cumulative numbers (168+264+321+233 > 555) over-stated each stage.
                _stg_total_ms = (time.perf_counter() - _stg_t0) * 1000.0
                _stg_dispatch_ms = (time.perf_counter() - _stg_dispatch_start) * 1000.0
                # Phase C5-R: sum of top-level *exclusive* stages (sequential,
                # non-overlapping deltas).  The cbmba_*/trajectory_*/map_* fields
                # are drill-downs nested inside worker_request_ms /
                # map_worker_request_ms and are intentionally NOT re-summed here.
                _exclusive_sum_ms = (
                    _rpc_lidar_ms + _rpc_state_ms + _rpc_collision_ms
                    + _excl_filter_ms + _excl_sectorize_ms
                    + _map_worker_request_ms + _map_worker_poll_ms
                    + _world_transform_metrics_ms
                    + _worker_request_ms + _worker_poll_ms
                    + _rpc_velocity_command_ms
                    + _tracker_ms + _apf_safety_ms + _goal_check_ms + _progress_monitor_ms
                    + _metrics_ms + _csv_enqueue_ms + _log_enqueue_ms
                )
                _other_ms = max(0.0, _stg_total_ms - _exclusive_sum_ms)
                _stg_now = time.monotonic()
                if (_stg_now - getattr(self, "_loop_stage_last_log", 0.0)) >= 1.0:
                    self._loop_stage_last_log = _stg_now
                    logger.info(
                        "loop_exclusive_timing  frame=%d  "
                        "rpc_lidar_ms=%.2f  rpc_state_ms=%.2f  rpc_collision_ms=%.2f  "
                        "lidar_filter_ms=%.2f  sectorize_ms=%.2f  "
                        "map_worker_request_ms=%.2f  map_worker_poll_ms=%.2f  "
                        "world_transform_metrics_ms=%.2f  "
                        "worker_request_ms=%.2f  worker_poll_ms=%.2f  "
                        "cbmba_request_prepare_ms=%.2f  cbmba_request_put_ms=%.2f  "
                        "cbmba_obstacle_snapshot_ms=%.2f  "
                        "trajectory_request_prepare_ms=%.2f  trajectory_request_put_ms=%.2f  "
                        "trajectory_path_copy_ms=%.2f  trajectory_snapshot_build_ms=%.2f  "
                        "map_request_prepare_ms=%.2f  map_request_put_ms=%.2f  "
                        "map_snapshot_build_ms=%.2f  "
                        "plan_phase_total_ms=%.2f  dispatch_phase_total_ms=%.2f  "
                        "rpc_velocity_command_ms=%.2f  "
                        "tracker_ms=%.2f  apf_safety_ms=%.2f  "
                        "goal_check_ms=%.2f  progress_monitor_ms=%.2f  "
                        "metrics_ms=%.2f  csv_enqueue_ms=%.2f  "
                        "log_enqueue_ms=%.2f  rpc_velocity_submit_ms=%.2f  "
                        "exclusive_sum_ms=%.2f  other_ms=%.2f  "
                        "work_total_ms=%.2f  cbmba_searches_frame=%d",
                        fn,
                        _rpc_lidar_ms, _rpc_state_ms, _rpc_collision_ms,
                        _excl_filter_ms, _excl_sectorize_ms,
                        _map_worker_request_ms, _map_worker_poll_ms,
                        _world_transform_metrics_ms,
                        _worker_request_ms, _worker_poll_ms,
                        self._cbmba_request_prepare_ms, self._cbmba_request_put_ms,
                        self._cbmba_obstacle_snapshot_ms,
                        self._traj_request_prepare_ms, self._traj_request_put_ms,
                        self._traj_path_copy_ms, self._traj_snapshot_build_ms,
                        self._map_request_prepare_ms, self._map_request_put_ms,
                        self._map_snapshot_build_ms,
                        _stg_plan_ms, _stg_dispatch_ms,
                        _rpc_velocity_command_ms,
                        _tracker_ms, _apf_safety_ms,
                        _goal_check_ms, _progress_monitor_ms,
                        _metrics_ms, _csv_enqueue_ms,
                        _log_enqueue_ms, _rpc_velocity_submit_ms,
                        _exclusive_sum_ms, _other_ms,
                        _stg_total_ms, self._cbmba_searches_this_frame,
                    )

                # ── deadline scheduler (sec 4/34): sleep to the next period
                # boundary, NOT a fixed command_duration_s. ──
                _work_ms = (time.perf_counter() - _stg_t0) * 1000.0
                _sched_next_tick, _sleep_ms, _deadline_late_ms, _missed_periods, _resynced = self._sleep_to_next_period(_sched_next_tick)
                if _resynced:
                    self._loop_deadline_resync_count += 1
                self._loop_last_was_resync = _resynced
                _actual_dt_ms = (time.monotonic() - _iter_mono) * 1000.0
                # Phase C3-R GIL verification: report whether a CBMBA search is
                # executing in the planner process at this instant.  If the loop
                # keeps hitting ~50 ms frames while ``cbmba_running=true``, the
                # search is NOT starving the loop.
                _cbmba_running = (
                    self._global_planner_worker.is_running()
                    if self._global_planner_worker is not None else False
                )
                _traj_worker_running = (
                    self._local_traj_worker.is_running()
                    if self._local_traj_worker is not None else False
                )
                _map_worker_running = (
                    self._mapping_worker.is_running()
                    if self._mapping_worker is not None else False
                )
                if (fn == 1 or fn % 10 == 0 or _deadline_late_ms > 0):
                    logger.info(
                        "control_scheduler  target_hz=%.1f  period_ms=%.1f  "
                        "iteration_work_ms=%.2f  scheduler_sleep_ms=%.2f  "
                        "deadline_lateness_ms=%.2f  iteration_duration_ms=%.2f  "
                        "loop_start_interval_ms=%.2f  "
                        "missed_periods=%d  deadline_resync=%s  "
                        "cbmba_running=%s  trajectory_worker_running=%s  "
                        "map_worker_running=%s",
                        self._control_loop_target_hz,
                        self._control_period_s * 1000.0,
                        _work_ms, _sleep_ms, _deadline_late_ms, _actual_dt_ms,
                        _iter_dt_ms,
                        _missed_periods, "true" if _resynced else "false",
                        "true" if _cbmba_running else "false",
                        "true" if _traj_worker_running else "false",
                        "true" if _map_worker_running else "false",
                    )

            # ── loop exited — stop producing commands, join last future ──
            if self._debug_drawer is not None:
                self._debug_drawer.close()
                self._debug_drawer = None
            if self._global_planner_worker is not None:
                self._global_planner_worker.shutdown()
                logger.info(
                    "global_planner_worker  state=stopped  searches=%d",
                    self._global_planner_worker.search_count,
                )
            if self._local_traj_worker is not None:
                self._local_traj_worker.shutdown()
                logger.info(
                    "local_traj_worker  state=stopped  plans=%d",
                    self._local_traj_worker.plan_count,
                )
            if self._mapping_worker is not None:
                self._mapping_worker.shutdown()
                logger.info(
                    "mapping_worker  state=stopped  updates=%d",
                    self._mapping_worker.update_count,
                )
            if self._perception_worker is not None:
                self._perception_worker.shutdown()
                logger.info(
                    "perception_worker  state=stopped  polls=%d  errors=%d",
                    self._perception_worker.poll_count,
                    self._perception_worker.error_count,
                )
            logger.info("last_velocity_future_wait_started")
            if self._last_velocity_future is not None:
                try:
                    self._last_velocity_future.join()
                    logger.info("last_velocity_future_wait_completed")
                except Exception as e:
                    logger.warning("last_velocity_future_join_error: %s", e)
                self._last_velocity_future = None

            rk.update(termination_reason=term, frames_completed=fn,
                       flight_duration_s=time.monotonic() - t0)

            # ── trajectory flight summary + navigation benchmark (sec 1/31) ──
            if self._local_navigation_mode == "trajectory" and self._traj_metrics is not None:
                try:
                    from dataclasses import asdict
                    _success = term == "mission_complete"
                    self._traj_metrics.flight_duration_s = time.monotonic() - t0
                    self._traj_metrics.frames_completed = fn
                    self._traj_metrics.control_loop_overruns = self._loop_overrun_count
                    self._traj_metrics.control_loop_max_overrun_ms = self._loop_max_overrun_ms
                    if self._loop_dt_n > 0:
                        self._traj_metrics.control_loop_avg_dt_ms = (
                            self._loop_dt_sum_ms / self._loop_dt_n
                        )
                    self._traj_metrics.lidar_stale_frames = self._lidar_stale_frames_total
                    self._traj_metrics.finalize(term, _success, _final_xy)
                    logger.info(self._traj_metrics.to_log_string())

                    _bench = self._traj_metrics.summary()
                    _bench["family_transition_breakdown"] = {
                        f"{f}->{t}": n for (f, t), n in
                        self._traj_family_log.breakdown.items()
                    }
                    _bench["obstacle_behavior_verdict"] = asdict(
                        self._obstacle_behavior_monitor.finalize(),
                    )
                    _bench["mission_goal_xy"] = [round(_mission_goal[0], 3),
                                                 round(_mission_goal[1], 3)]
                    _bench["mission_goal_source"] = _mission_goal_source
                    _bench["mission_goal_actor"] = _mission_goal_actor
                    _bench["start_xy"] = [round(spawn[0], 3), round(spawn[1], 3)]
                    _bench["local_navigation_mode"] = self._local_navigation_mode
                    _bench["planner_mode"] = self._planner_mode
                    _bench["trace_csv"] = self._trace_csv_path

                    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
                    _out_dir = Path(self._traj_debug_cfg.get(
                        "trace_output_dir", "runs",
                    ))
                    if not _out_dir.is_absolute():
                        _out_dir = _PROJECT_ROOT / _out_dir
                    _out_dir.mkdir(parents=True, exist_ok=True)
                    _nb_path = _out_dir / "navigation_benchmark.json"
                    _nb_path.write_text(
                        json.dumps(_bench, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    logger.info("navigation_benchmark  path=%s", _nb_path)
                except Exception as _sum_exc:  # noqa: BLE001
                    logger.warning("trajectory_flight_summary_error: %s", _sum_exc)

                # Phase C4: realtime architecture summary (P0 acceptance).
                try:
                    _dts = sorted(self._loop_dt_samples_ms)
                    _dt_median = _dts[len(_dts) // 2] if _dts else 0.0
                    _dt_p95 = _dts[min(len(_dts) - 1, int(len(_dts) * 0.95))] if _dts else 0.0
                    _dt_max = _dts[-1] if _dts else 0.0
                    _ages = sorted(self._perception_age_samples_ms)
                    _age_mean = (sum(_ages) / len(_ages)) if _ages else 0.0
                    _age_p95 = _ages[min(len(_ages) - 1, int(len(_ages) * 0.95))] if _ages else 0.0
                    _dur_s = time.monotonic() - t0
                    _cbmba_n = self._global_planner_worker.search_count if self._global_planner_worker is not None else 0
                    _cbmba_mean = (self._global_planner_worker.time_ms_sum / _cbmba_n) if _cbmba_n else 0.0
                    _cbmba_max = self._global_planner_worker.time_ms_max if self._global_planner_worker is not None else 0.0
                    _traj_n = self._local_traj_worker.plan_count if self._local_traj_worker is not None else 0
                    _traj_mean = (self._local_traj_worker.plan_time_ms_sum / _traj_n) if _traj_n else 0.0
                    _traj_max = self._local_traj_worker.plan_time_ms_max if self._local_traj_worker is not None else 0.0
                    _map_n = self._mapping_worker.update_count if self._mapping_worker is not None else 0
                    _map_samples = sorted(
                        self._mapping_worker.compute_ms_samples
                        if self._mapping_worker is not None else []
                    )
                    _map_mean = (sum(_map_samples) / len(_map_samples)) if _map_samples else 0.0
                    _map_p95 = _map_samples[min(len(_map_samples) - 1, int(len(_map_samples) * 0.95))] if _map_samples else 0.0
                    _map_max = _map_samples[-1] if _map_samples else 0.0
                    _map_put_mean = (
                        self._mapping_worker._ipc_put_ms_sum / self._mapping_worker._ipc_put_ms_n
                        if (self._mapping_worker is not None
                            and self._mapping_worker._ipc_put_ms_n > 0) else 0.0
                    )
                    _map_poll_mean = (
                        self._mapping_worker._ipc_poll_ms_sum / self._mapping_worker._ipc_poll_ms_n
                        if (self._mapping_worker is not None
                            and self._mapping_worker._ipc_poll_ms_n > 0) else 0.0
                    )
                    logger.info(
                        "realtime_architecture_summary  duration_s=%.1f  "
                        "control_frames=%d  control_hz_mean=%.2f  "
                        "dt_median_ms=%.2f  dt_p95_ms=%.2f  dt_max_ms=%.2f  "
                        "control_overrun_warn_count=%d  control_overrun_stop_count=%d  "
                        "cbmba_completed=%d  cbmba_mean_ms=%.2f  cbmba_max_ms=%.2f  "
                        "trajectory_plans_completed=%d  trajectory_mean_ms=%.2f  "
                        "trajectory_max_ms=%.2f  perception_snapshot_age_mean_ms=%.2f  "
                        "perception_snapshot_age_p95_ms=%.2f  perception_stale_count=%d  "
                        "trajectory_dispatch_count=%d  hover_dispatch_count=%d  "
                        "hover_dispatch_total=%d  hover_due_perception_stale=%d  "
                        "hover_due_control_overrun=%d  hover_due_trajectory_stale=%d  "
                        "hover_due_no_feasible=%d  hover_due_other=%d  "
                        "map_updates_completed=%d  map_mean_ms=%.2f  map_p95_ms=%.2f  "
                        "map_max_ms=%.2f  map_requests_submitted=%d  "
                        "map_requests_coalesced=%d  map_ipc_put_mean_ms=%.3f  "
                        "map_ipc_poll_mean_ms=%.3f  loop_start_interval_median_ms=%.2f  "
                        "loop_start_interval_p95_ms=%.2f  tight_loop_lt25ms_count=%d  "
                        "tight_loop_lt40ms_count=%d  deadline_resync_count=%d  "
                        "post_resync_tight_loop_count=%d  max_altitude_error_m=%.3f",
                        _dur_s, fn, (fn / _dur_s if _dur_s > 0 else 0.0),
                        _dt_median, _dt_p95, _dt_max,
                        self._loop_overrun_count, self._loop_overrun_stop_count,
                        _cbmba_n, _cbmba_mean, _cbmba_max,
                        _traj_n, _traj_mean, _traj_max,
                        _age_mean, _age_p95, self._perception_stale_count,
                        self._trajectory_dispatch_count,
                        self._hover_dispatch_count,
                        self._hover_dispatch_count,
                        self._hover_due_perception_stale,
                        self._hover_due_control_overrun,
                        self._hover_due_trajectory_stale,
                        self._hover_due_no_feasible,
                        self._hover_due_other,
                        _map_n, _map_mean, _map_p95, _map_max,
                        self._mapping_worker.submitted_count if self._mapping_worker is not None else 0,
                        self._mapping_worker.coalesced_count if self._mapping_worker is not None else 0,
                        _map_put_mean, _map_poll_mean,
                        _dt_median, _dt_p95,
                        self._loop_tight_lt25_count, self._loop_tight_lt40_count,
                        self._loop_deadline_resync_count,
                        self._loop_post_resync_tight_count,
                        self._max_altitude_error_m,
                    )
                except Exception as _rts_exc:  # noqa: BLE001
                    logger.warning("realtime_architecture_summary_error: %s", _rts_exc)

            if self._trace_writer is not None:
                try:
                    self._trace_writer.close()
                    logger.info("trajectory_trace_csv_closed  path=%s", self._trace_csv_path)
                except Exception as _tw_exc:  # noqa: BLE001
                    logger.warning("trajectory_trace_csv_close_error: %s", _tw_exc)

        except KeyboardInterrupt:
            rk["termination_reason"] = "ctrl_c"
            logger.info("Ctrl+C.")
        except Exception as e:
            rk["termination_reason"] = f"exception:{e}"
            logger.exception("Unhandled exception.")
        finally:
            self._running = False
            # Also cover setup/warmup exceptions that happen before the normal
            # loop-exit cleanup block.  The drawer owns a daemon renderer, but
            # closing it here prevents an extra AirSim connection from
            # lingering while SharedFlightSession performs its cleanup.
            if self._debug_drawer is not None:
                try:
                    self._debug_drawer.close()
                except Exception:
                    pass
                self._debug_drawer = None
            # Cleanup is owned by CLI's finally → session.cleanup() → land_and_disarm()
            # automatic_mode only reports termination_reason, no independent landing.
            rk["success"] = (rk["termination_reason"] == "mission_complete")

        return AutomaticFlightResult(**rk)

    def stop(self) -> None:
        self._running = False

"""
Local Recovery — stuck/oscillation detection module.

Pure calculation module.  Does NOT call any AirSim API.

Detects two stuck conditions from a sliding window of position + velocity frames:

1. **Stuck**: horizontal (XY) position unchanged (delta < epsilon) for >=
   ``stuck_time_window_s``.  Captures the case where the drone is physically
   blocked and cannot move despite the planner issuing non-zero velocity
   commands.

2. **Oscillation**: body-frame vy sign flips repeatedly without meaningful
   lateral progress.  Captures the "local minimum" case where the reactive
   planner alternates left / right without escaping.

Outputs a ``RecoveryDecision`` dataclass with detection flags and candidate
recovery actions (designed but NEVER sent to AirSim by this module).

FPS-independent design
----------------------
The sliding window is pruned by ``history_window_s`` (default 4.0 s), which
is intentionally larger than ``stuck_time_window_s`` (default 2.5 s).  This
ensures the stuck detector can always find a frame pair whose timestamps
span the full detection window, regardless of frame rate (5 / 10 / 20 Hz).

Coordinate convention
---------------------
- ``position`` is expected in any consistent Cartesian frame (e.g. NED).
  Only *deltas* are computed, so the absolute origin does not matter.
- ``velocity_body`` is expected in **body FRD**: +X forward, +Y right, +Z down.
  Only ``vy`` (lateral) is used for oscillation sign-flip detection.

Usage::

    recovery = LocalRecovery(params)
    for each frame:
        decision = recovery.update(
            timestamp=time.monotonic(),
            position=(x, y, z),
            velocity_body=(vx, vy, vz),
        )
        if decision.needs_recovery:
            logger.info("recovery_shadow  stuck=%s  oscillating=%s  ...",
                        decision.is_stuck, decision.is_oscillating)

References
----------
Adapted from the old JS project's stuck detection in ``physicsworker.js``
(``updateNavigationProgressWatchdog`` / ``computeStuckEscapeVelocity``)
and the ``AvoidanceSupervisor`` RECOVERY mode.  Key differences:

* Old: goal-distance-progress + speed + obstacle proximity → stuck timer.
* New: pure position-delta-in-window → stuck (simpler, no goal dependency).
* Old: no oscillation detection.
* New: vy-sign-flip counter → oscillation (new capability).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


# ── data classes ──


@dataclass
class RecoveryDecision:
    """Output of one LocalRecovery detection step.

    All fields are read-only diagnostics.  This module NEVER sends
    commands to AirSim — the ``candidate_actions`` list is for logging only.
    """

    # Detection flags
    is_stuck: bool = False
    is_oscillating: bool = False
    needs_recovery: bool = False  # OR of is_stuck and is_oscillating

    # Stuck diagnostics
    stuck_duration_s: float = 0.0
    stuck_position_delta_m: float = 0.0           # horizontal XY delta
    stuck_latest_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stuck_oldest_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Oscillation diagnostics
    oscillation_vy_sign_flips: int = 0
    oscillation_lateral_progress_m: float = 0.0

    # Candidate recovery actions (strings only — designed but NOT sent)
    candidate_actions: List[str] = field(default_factory=list)

    # General
    reason: str = ""
    window_size_frames: int = 0


@dataclass
class RecoveryParams:
    """Configurable recovery detection parameters.

    Attributes:
        history_window_s:
            How long to retain frames.  Must be **larger** than
            ``stuck_time_window_s`` so that a full-duration frame pair
            is always available regardless of frame alignment / rate.
        stuck_time_window_s:
            Horizontal position must be unchanged for at least this
            duration before ``is_stuck`` flips to True.
        stuck_position_epsilon_m:
            Max horizontal (XY) delta considered "unchanged".
        stuck_min_frames:
            Minimum number of frames required in the window before
            stuck detection is eligible (avoids false positives on startup).
        oscillation_time_window_s:
            Lookback window for vy sign-flip counting.
        oscillation_min_sign_flips:
            Number of vy sign changes required to declare oscillation.
        oscillation_lateral_epsilon_m:
            Lateral (horizontal) progress below this threshold is
            considered "no progress" for oscillation confirmation.
    """

    history_window_s: float = 4.0

    stuck_time_window_s: float = 2.5
    stuck_position_epsilon_m: float = 0.15
    stuck_min_frames: int = 10

    oscillation_time_window_s: float = 2.0
    oscillation_min_sign_flips: int = 3
    oscillation_lateral_epsilon_m: float = 0.2

    # Candidate recovery action labels (for diagnostics only)
    candidate_actions: Tuple[str, ...] = (
        "escape_maneuver",
        "vertical_climb",
        "lateral_sidestep",
    )


# ── internal frame record ──


class _RecoveryFrame:
    """Single-frame snapshot stored in the sliding window."""

    __slots__ = ("timestamp", "px", "py", "pz", "vx_body", "vy_body", "vz_body", "yaw_rad")

    def __init__(
        self,
        timestamp: float,
        position: Tuple[float, float, float],
        velocity_body: Tuple[float, float, float],
        yaw_rad: float = 0.0,
    ) -> None:
        self.timestamp = float(timestamp)
        self.px, self.py, self.pz = (
            float(position[0]), float(position[1]), float(position[2]),
        )
        self.vx_body = float(velocity_body[0])
        self.vy_body = float(velocity_body[1])
        self.vz_body = float(velocity_body[2])
        self.yaw_rad = float(yaw_rad)


# ── main class ──


class LocalRecovery:
    """Sliding-window stuck and oscillation detector.

    Maintains a ``deque`` of recent ``_RecoveryFrame`` entries and runs
    two independent detectors on each ``update()`` call.

    The window is pruned by ``history_window_s`` (not ``stuck_time_window_s``),
    guaranteeing that a frame pair spanning the full stuck detection window
    is always available regardless of frame rate.

    This is a **pure calculation** module — it does NOT call any AirSim
    API and does NOT send commands to the drone.
    """

    def __init__(self, params: Optional[RecoveryParams] = None) -> None:
        self._params = params or RecoveryParams()
        self._window: Deque[_RecoveryFrame] = deque()
        # Rolling counter to avoid re-scanning vy signs on every frame.
        # Stores (vy_sign, timestamp) for quick flip counting.
        self._vy_history: Deque[Tuple[int, float]] = deque()

    # ── public API ──

    def update(
        self,
        timestamp: float,
        position: Tuple[float, float, float],
        velocity_body: Tuple[float, float, float],
        yaw_rad: float = 0.0,
    ) -> RecoveryDecision:
        """Ingest one frame and return the current detection state.

        Args:
            timestamp:
                Monotonic seconds (e.g. ``time.monotonic()``).  Must be
                non-decreasing across calls.
            position:
                Drone position as ``(x, y, z)`` in any consistent Cartesian
                frame (NED recommended).  Only deltas are computed.
            velocity_body:
                Drone velocity in **body FRD**: ``(vx_forward, vy_right, vz_down)``.
                Only ``vy`` is used by oscillation detection.
            yaw_rad:
                Yaw angle in radians (NED convention: 0=North, π/2=East).
                Used to project NED position deltas onto the body-lateral
                axis for oscillation lateral-progress computation.

        Returns:
            RecoveryDecision with current detection flags and diagnostics.
        """
        p = self._params

        # ── guard: NaN / Inf ──
        for label, val in [
            ("px", position[0]), ("py", position[1]), ("pz", position[2]),
            ("vx", velocity_body[0]), ("vy", velocity_body[1]), ("vz", velocity_body[2]),
        ]:
            if math.isnan(val) or math.isinf(val):
                return RecoveryDecision(
                    needs_recovery=False,
                    reason=f"invalid_input:{label}",
                    window_size_frames=len(self._window),
                )

        frame = _RecoveryFrame(timestamp, position, velocity_body, yaw_rad=yaw_rad)
        self._window.append(frame)
        self._vy_history.append((_vy_sign(frame.vy_body), frame.timestamp))

        # ── prune old frames by history_window_s (larger than stuck window) ──
        # This guarantees enough history for the stuck detector to find a
        # frame pair spanning the full stuck_time_window_s, regardless of
        # frame-rate jitter or alignment.
        self._prune_before(timestamp - p.history_window_s)

        # ── run detectors ──
        stuck_result = self._detect_stuck(p, timestamp)
        osc_result = self._detect_oscillation(p, timestamp)

        # ── assemble decision ──
        is_stuck = stuck_result[0]
        is_oscillating = osc_result[0]
        needs = is_stuck or is_oscillating

        reason_parts: List[str] = []
        if is_stuck:
            reason_parts.append("stuck")
        if is_oscillating:
            reason_parts.append("oscillation")
        if not reason_parts:
            reason_parts.append("none")

        candidate: List[str] = []
        if needs:
            candidate = list(p.candidate_actions)

        # Latest/oldest positions for diagnostics (from the stuck detector's
        # frame pair; fall back to window ends).
        _oldest_pos = (stuck_result[3], stuck_result[4], stuck_result[5])
        _latest_pos = (stuck_result[6], stuck_result[7], stuck_result[8])

        return RecoveryDecision(
            is_stuck=is_stuck,
            is_oscillating=is_oscillating,
            needs_recovery=needs,
            stuck_duration_s=stuck_result[1],
            stuck_position_delta_m=stuck_result[2],
            stuck_latest_position=_latest_pos,
            stuck_oldest_position=_oldest_pos,
            oscillation_vy_sign_flips=osc_result[1],
            oscillation_lateral_progress_m=osc_result[2],
            candidate_actions=candidate,
            reason="+".join(reason_parts),
            window_size_frames=len(self._window),
        )

    def reset(self) -> None:
        """Clear the sliding window (e.g. on mode transition)."""
        self._window.clear()
        self._vy_history.clear()

    # ── detectors ──

    def _detect_stuck(
        self, p: RecoveryParams, now: float,
    ) -> Tuple[bool, float, float, float, float, float, float, float, float]:
        """Check whether horizontal position has been stationary for >= time window.

        Always computes and returns the actual XY position delta, regardless
        of whether the stuck condition is met.  This allows the diagnostic
        log to report the true delta even when the drone is moving normally.

        Returns:
            (is_stuck,
             duration_s,
             xy_delta_m,
             oldest_px, oldest_py, oldest_pz,
             latest_px, latest_py, latest_pz)
        """
        _zero = (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        if len(self._window) < p.stuck_min_frames:
            return _zero

        # Find the last frame at or before the cutoff so that
        #   ``last.timestamp - first.timestamp >= stuck_time_window_s``
        # is guaranteed regardless of frame-rate alignment.  Using the
        # first frame *after* the cutoff loses up to one interval at low
        # rates (5 Hz → 0.2 s lost), making the duration check fail.
        cutoff = now - p.stuck_time_window_s
        first_idx: Optional[int] = None
        for i, f in enumerate(self._window):
            if f.timestamp <= cutoff:
                first_idx = i    # keep the last frame at or before cutoff
            else:
                break            # frames are chronological

        last = self._window[-1]

        if first_idx is None:
            # Window doesn't reach back to the cutoff yet — compute
            # diagnostics from the earliest available frame but never
            # declare stuck (not enough history).
            first = self._window[0]
            dx = last.px - first.px
            dy = last.py - first.py
            xy_delta = math.sqrt(dx * dx + dy * dy)
            return (False, last.timestamp - first.timestamp, xy_delta,
                    first.px, first.py, first.pz,
                    last.px, last.py, last.pz)

        first = self._window[first_idx]

        # ── always compute horizontal (XY) delta for diagnostics ──
        dx = last.px - first.px
        dy = last.py - first.py
        xy_delta = math.sqrt(dx * dx + dy * dy)

        oldest_pos = (first.px, first.py, first.pz)
        latest_pos = (last.px, last.py, last.pz)

        duration = last.timestamp - first.timestamp

        is_stuck = xy_delta < p.stuck_position_epsilon_m
        return (is_stuck, duration, xy_delta,
                oldest_pos[0], oldest_pos[1], oldest_pos[2],
                latest_pos[0], latest_pos[1], latest_pos[2])

    def _detect_oscillation(
        self, p: RecoveryParams, now: float,
    ) -> Tuple[bool, int, float]:
        """Count vy sign flips in the oscillation window.

        Oscillation is declared when:
        - vy sign flips >= min_sign_flips, AND
        - lateral (horizontal) position progress < lateral_epsilon.

        Returns:
            (is_oscillating, sign_flip_count, lateral_progress_m)
        """
        cutoff = now - p.oscillation_time_window_s

        # ── find frames in oscillation window ──
        osc_frames: List[_RecoveryFrame] = []
        for f in self._window:
            if f.timestamp >= cutoff:
                osc_frames.append(f)

        if len(osc_frames) < 2:
            return (False, 0, 0.0)

        # ── count vy sign flips ──
        flips = 0
        prev_sign = _vy_sign(osc_frames[0].vy_body)
        for f in osc_frames[1:]:
            cur_sign = _vy_sign(f.vy_body)
            if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
                flips += 1
            if cur_sign != 0:
                prev_sign = cur_sign  # only update on non-zero (ignore dead zone)

        # ── lateral progress in body frame ──
        # Project the NED position delta onto the body-lateral axis
        # (-sin(ψ), cos(ψ)) using the reference yaw from the first frame.
        # This isolates true lateral (Y-body) progress, ignoring forward
        # (X-body) displacement that would otherwise mask oscillation
        # during pure forward flight.
        first = osc_frames[0]
        last = osc_frames[-1]
        dx = last.px - first.px
        dy = last.py - first.py
        ref_yaw = first.yaw_rad
        # body-right direction in NED: R = (-sin(ψ), cos(ψ))
        # lateral displacement = dx * R_x + dy * R_y
        lateral = abs(-dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw))

        is_osc = (
            flips >= p.oscillation_min_sign_flips
            and lateral < p.oscillation_lateral_epsilon_m
        )
        return (is_osc, flips, lateral)

    # ── helpers ──

    def _prune_before(self, cutoff: float) -> None:
        """Remove frames with timestamp < cutoff from both buffers."""
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()
        while self._vy_history and self._vy_history[0][1] < cutoff:
            self._vy_history.popleft()


def _vy_sign(vy: float) -> int:
    """Return -1, 0, or +1 for vy, with a small dead-zone around zero."""
    if vy > 0.02:
        return 1
    if vy < -0.02:
        return -1
    return 0

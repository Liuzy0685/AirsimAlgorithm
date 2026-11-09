"""
Recovery Commander — recovery action selection and state machine.

Produces conservative recovery velocity commands when the
LocalRecovery detector flags ``needs_recovery``.  All commands are
body-FRD and strictly rate-limited.

State machine
-------------
::

    APF_ACTIVE ──(needs_recovery)──▶ RECOVERY_ACTIVE ──(max_active elapses)──▶ RECOVERY_COOLDOWN
         ▲                                  │                                  │
         │                                  │ (safety override)                │ (cooldown expires)
         │                                  ▼                                  │
         ◀────────────────────────────────────────────────────────────────────┘

Safety conditions (collision, geofence, emergency) always preempt
recovery — the loop breaks before any recovery command is sent.

Enhancements over the legacy version
------------------------------------
- **committed_side**: Side choice is persisted across active ticks so the
  drone doesn't oscillate between left/right during recovery.
- **Guidance-aware side choice**: When CBMBA guidance direction is
  available, it biases the side choice (LiDAR still gates safety).
- **Bypass inheritance**: Recovery inherits the bypass side when a
  bypass episode is active, maintaining consistent direction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from planners.local_recovery import RecoveryDecision

# ── compatibility aliases (for backward compat with planners/__init__.py) ──

MAX_HORIZONTAL_SPEED_MPS = 0.35
STUCK_BACKWARD_VX = -0.35
OSCILLATION_LATERAL_VY = 0.35
RECOVERY_MAX_ACTIVE_S = 1.0
RECOVERY_COOLDOWN_S = 2.5


# ── configurable parameters ──


@dataclass
class RecoveryCommanderParams:
    """Configurable recovery parameters.

    Defaults match the legacy hardcoded constants for backward
    compatibility.  All speeds are body-FRD, all durations in seconds.
    """

    reverse_speed: float = 0.12       # vx during stuck recovery (negative = backward)
    lateral_speed: float = 0.12       # |vy| during oscillation recovery
    min_duration_s: float = 0.0       # minimum active time before exit
    max_duration_s: float = 1.0       # maximum active time before forced exit
    cooldown_s: float = 2.5           # cooldown before re-entry allowed
    required_progress_m: float = 0.5  # min XY progress for early exit
    clear_distance_m: float = 2.0     # min front/left/right distance for clear exit
    dead_end_escape_enabled: bool = True
    dead_end_front_trigger_m: float = 2.5
    dead_end_side_trigger_m: float = 2.0
    vertical_climb_enabled: bool = True
    vertical_clearance_m: float = 2.5
    vertical_climb_speed_mps: float = 0.20
    vertical_climb_duration_s: float = 1.2
    vertical_climb_delta_m: float = 0.40
    wall_follow_forward_speed_mps: float = 0.15
    wall_follow_duration_s: float = 4.0
    # U-shaped obstacles need a persistent wall side. Re-evaluating the side
    # whenever that wall is close causes left/right oscillation in the opening.
    wall_follow_side_lock_enabled: bool = True


# ── state enum ──


class RecoveryState:
    APF_ACTIVE = "APF_ACTIVE"
    RECOVERY_ACTIVE = "RECOVERY_ACTIVE"
    RECOVERY_COOLDOWN = "RECOVERY_COOLDOWN"


# ── commander ──


def compute_recovery_command(
    decision: RecoveryDecision,
    lidar_rays: Dict[str, float],
    committed_side: Optional[int] = None,
    guidance_dir: Optional[Tuple[float, float]] = None,
    params: Optional[RecoveryCommanderParams] = None,
    elapsed_s: float = 0.0,
    forced_mode: Optional[str] = None,
) -> Tuple[float, float, float, int]:
    """Return body-FRD recovery velocity ``(vx, vy, vz, committed_side)``.

    The command is always conservative (max horizontal speed ≤ params).
    ``vz`` is normally 0; the dead-end climb probe uses negative ``vz``
    because AirSim is NED.  ``committed_side`` is +1 (right), -1 (left),
    or 0 (none/not applicable).

    Priority:
    1. **Stuck** → backward (vx=-reverse_speed) + lateral toward clearer
       side or inherited committed_side.
    2. **Oscillation** → lateral sidestep toward clearer side.
    3. Neither → (0, 0, 0, 0).

    Args:
        decision: RecoveryDecision from LocalRecovery.
        lidar_rays: Legacy ray distances (keys: front, left, right).
        committed_side: Previously committed side, persisted across ticks.
        guidance_dir: Optional CBMBA guidance (dir_x, dir_y) in body frame
                      for side-choice biasing.
        params: Recovery parameters (uses defaults if None).
    """
    p = params or RecoveryCommanderParams()

    dead_end_mode = forced_mode or _dead_end_mode(lidar_rays, p)
    if not decision.needs_recovery and dead_end_mode is None:
        return (0.0, 0.0, 0.0, 0)

    # ── side choice ──
    if (
        p.wall_follow_side_lock_enabled
        and forced_mode in ("wall", "trajectory_no_feasible")
        and committed_side is not None
        and committed_side != 0
    ):
        # Keep the selected wall even when its local ray becomes short. The
        # dispatch directional guard suppresses unsafe lateral motion; changing
        # sides here is what creates the U-opening oscillation.
        side = int(committed_side)
    else:
        side = _choose_side(lidar_rays, committed_side, guidance_dir)

    if (
        dead_end_mode == "climb"
        and p.vertical_climb_enabled
        and elapsed_s < p.vertical_climb_duration_s
        and _upward_clearance(lidar_rays) >= p.vertical_clearance_m
    ):
        return (0.0, 0.0, -abs(p.vertical_climb_speed_mps), side)

    if dead_end_mode == "wall":
        front = _safe_ray(lidar_rays.get("front"))
        vx = (
            -abs(p.reverse_speed)
            if front < p.dead_end_front_trigger_m
            else abs(p.wall_follow_forward_speed_mps)
        )
        vy = side * abs(p.lateral_speed) if side != 0 else 0.0
        return (vx, vy, 0.0, side)

    # A trajectory planner local minimum is not necessarily a dead end.  If
    # the forward sector is still open, advance while sidestepping around the
    # obstacle; unconditional reverse here made Forest flights drift away
    # from the goal and repeatedly re-enter recovery.
    if forced_mode == "trajectory_no_feasible":
        front = _safe_ray(lidar_rays.get("front"))
        if front >= p.dead_end_front_trigger_m:
            return (
                abs(p.wall_follow_forward_speed_mps),
                side * abs(p.lateral_speed) if side != 0 else 0.0,
                0.0,
                side,
            )

    if decision.is_stuck:
        vy = side * p.lateral_speed if side != 0 else 0.0
        return (-p.reverse_speed, vy, 0.0, side)

    if decision.is_oscillating:
        vy = side * p.lateral_speed if side != 0 else p.lateral_speed
        return (0.0, vy, 0.0, side)

    return (0.0, 0.0, 0.0, 0)


def _upward_clearance(lidar_rays: Dict[str, float]) -> float:
    """Return the most restrictive clearance in the upward-facing sectors.

    AirSim uses NED coordinates, but these are distances rather than signed
    coordinates.  Missing vertical sectors are treated as zero clearance so
    a climb can never be authorized by incomplete sensing.
    """
    names = ("up", "frontUp", "leftUp", "rightUp")
    return min(_safe_ray(lidar_rays.get(name)) for name in names)


def _dead_end_mode(
    lidar_rays: Dict[str, float],
    params: RecoveryCommanderParams,
) -> Optional[str]:
    """Classify a likely U-shaped dead end as ``climb`` or ``wall``.

    This gate is deliberately strict: front and both horizontal sides must be
    constrained before the special escape is activated.  A single blocked
    forward ray therefore does not make the vehicle climb or wall-follow.
    """
    if not params.dead_end_escape_enabled:
        return None

    front = _safe_ray(lidar_rays.get("front"))
    left = _safe_ray(lidar_rays.get("left"))
    right = _safe_ray(lidar_rays.get("right"))
    trapped = (
        front < params.dead_end_front_trigger_m
        and left < params.dead_end_side_trigger_m
        and right < params.dead_end_side_trigger_m
    )
    if not trapped:
        return None

    if (
        params.vertical_climb_enabled
        and _upward_clearance(lidar_rays) >= params.vertical_clearance_m
    ):
        return "climb"
    return "wall"


def _choose_side(
    lidar_rays: Dict[str, float],
    committed_side: Optional[int] = None,
    guidance_dir: Optional[Tuple[float, float]] = None,
) -> int:
    """Choose recovery lateral direction: +1 (right), -1 (left), or 0 (hold).

    Decision logic (in order):
    1. If committed_side is set and both sides have adequate clearance,
       persist the committed side (prevents oscillation).
    2. If guidance_dir is available and LiDAR confirms the guidance side
       has at least minimum clearance, follow guidance.
    3. Otherwise pick the side with more LiDAR clearance.
    """
    left = _safe_ray(lidar_rays.get("left"))
    right = _safe_ray(lidar_rays.get("right"))

    # ── 1. Persist committed side if still safe ──
    if committed_side is not None and committed_side != 0:
        min_clearance = 1.5  # minimum meters required to persist a side
        if committed_side == 1 and right >= min_clearance:
            return 1
        if committed_side == -1 and left >= min_clearance:
            return -1
        # Committed side is no longer safe — fall through to re-evaluate.

    # ── 2. Guidance-aware choice ──
    if guidance_dir is not None:
        gx, gy = guidance_dir
        min_guidance_clearance = 1.0
        if abs(gy) > 0.05:  # meaningful lateral component
            guidance_side = 1 if gy > 0 else -1
            if guidance_side == 1 and right >= min_guidance_clearance:
                return 1
            if guidance_side == -1 and left >= min_guidance_clearance:
                return -1
        # Strong forward guidance: prefer more open side
        if gx > 0.7 and abs(gy) <= 0.05:
            # Guidance says "go straight" — pick safer side
            if right > left and right >= min_guidance_clearance:
                return 1
            if left > right and left >= min_guidance_clearance:
                return -1

    # ── 3. Pure LiDAR choice ──
    if right > left:
        return 1
    if left > right:
        return -1
    # Equal: default left (conservative — most drones have better left visibility)
    return -1


def _safe_ray(value) -> float:
    """Coerce None / non-numeric LiDAR values to 0.0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── state machine ──


@dataclass
class RecoveryStateResult:
    """Output of one state-machine tick."""

    state: str = RecoveryState.APF_ACTIVE
    should_override: bool = False
    vx_body: float = 0.0
    vy_body: float = 0.0
    vz_body: float = 0.0
    event: Optional[str] = None          # "enter", "active", "exit_climb", "exit_timeout", "exit_safety", "exit_progress", "cooldown_expired"
    elapsed_s: float = 0.0
    cooldown_remaining_s: float = 0.0
    committed_side: Optional[int] = None  # +1 right, -1 left, 0 none, None not set
    recovery_progress_m: float = 0.0      # XY progress made during active recovery
    needs_stuck_reset: bool = False       # Signal to reset LocalRecovery stuck detector


class RecoveryStateMachine:
    """Manages recovery state transitions.

    Enforces committed-side persistence across ticks to prevent
    left/right oscillation during recovery.

    Call ``tick(now, decision, lidar_rays, ...)`` once per flight-loop
    iteration.  The returned ``RecoveryStateResult.should_override``
    tells the dispatcher whether to use the recovery command instead
    of the normal APF / reactive command.
    """

    def __init__(self, params: Optional[RecoveryCommanderParams] = None) -> None:
        self._params = params or RecoveryCommanderParams()
        self._state = RecoveryState.APF_ACTIVE
        self._active_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._command: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._committed_side: Optional[int] = None
        self._entry_position: Optional[Tuple[float, float, float]] = None
        self._mode: str = "normal"

    # ── public API ──

    def tick(
        self,
        now: float,
        decision: RecoveryDecision,
        lidar_rays: Dict[str, float],
        current_position: Optional[Tuple[float, float, float]] = None,
        guidance_dir: Optional[Tuple[float, float]] = None,
        bypass_side: Optional[int] = None,
    ) -> RecoveryStateResult:
        """Advance the state machine one step.

        Args:
            now: ``time.monotonic()``.
            decision: Latest ``RecoveryDecision`` from ``LocalRecovery``.
            lidar_rays: Legacy ray distances dict (keys: front, left, right, …).
            current_position: Current NED position (x, y, z) in meters for
                              progress-based exit tracking.
            guidance_dir: Optional CBMBA guidance (dir_x, dir_y) in body frame.
            bypass_side: Optional bypass side (+1=right, -1=left) for inheritance.

        Returns:
            ``RecoveryStateResult`` with current state and (if active)
            the recovery velocity command.
        """
        # 1. Cooldown expiry → APF_ACTIVE (fall through to entry check)
        had_cooldown = self._state == RecoveryState.RECOVERY_COOLDOWN
        if had_cooldown and now >= self._cooldown_until:
            self._state = RecoveryState.APF_ACTIVE
            self._committed_side = None
            self._entry_position = None
            return RecoveryStateResult(
                state=self._state,
                event="cooldown_expired",
            )
        elif had_cooldown:
            return RecoveryStateResult(
                state=self._state,
                cooldown_remaining_s=max(0.0, self._cooldown_until - now),
            )

        # 2. RECOVERY_ACTIVE: check exit conditions
        if self._state == RecoveryState.RECOVERY_ACTIVE:
            elapsed = now - self._active_start
            p = self._params

            # 2a. Progress-based early exit
            if current_position is not None and self._entry_position is not None:
                dx = current_position[0] - self._entry_position[0]
                dy = current_position[1] - self._entry_position[1]
                progress = math.hypot(dx, dy)
                front = _safe_ray(lidar_rays.get("front"))
                left = _safe_ray(lidar_rays.get("left"))
                right = _safe_ray(lidar_rays.get("right"))
                min_clearance = min(front, left, right)
                # A trajectory-local minimum is often a narrow passage: the
                # side surfaces can remain close even after the forward exit
                # has opened. Requiring both side rays to clear here turns a
                # short escape into the full recovery timeout. For this mode,
                # progress plus a clear front is sufficient; genuine dead-end
                # climb/wall recovery keeps the stricter three-ray condition.
                clear_for_exit = (
                    front >= p.clear_distance_m
                    if self._mode == "trajectory_no_feasible"
                    else min_clearance >= p.clear_distance_m
                )
                if (elapsed >= p.min_duration_s
                        and progress >= p.required_progress_m
                        and clear_for_exit):
                    self._state = RecoveryState.RECOVERY_COOLDOWN
                    self._cooldown_until = now + p.cooldown_s
                    return RecoveryStateResult(
                        state=self._state,
                        event="exit_progress",
                        elapsed_s=elapsed,
                        committed_side=self._committed_side,
                        recovery_progress_m=progress,
                        needs_stuck_reset=True,
                        cooldown_remaining_s=p.cooldown_s,
                    )
            else:
                progress = 0.0

            # A climb is a bounded probe, not a new indefinite flight mode.
            # If the ceiling becomes constrained, immediately switch to the
            # already-committed wall side.  If the probe reaches its time or
            # height budget while remaining clear, hand control back to the
            # planner so it can replan at the new altitude.
            if self._mode == "climb":
                climbed_m = 0.0
                if current_position is not None and self._entry_position is not None:
                    # NED Z decreases when the vehicle climbs.
                    climbed_m = max(0.0, self._entry_position[2] - current_position[2])
                upward_clear = _upward_clearance(lidar_rays)
                if upward_clear < p.vertical_clearance_m:
                    self._mode = "wall"
                elif (
                    elapsed >= p.vertical_climb_duration_s
                    or climbed_m >= p.vertical_climb_delta_m
                ):
                    self._state = RecoveryState.RECOVERY_COOLDOWN
                    self._cooldown_until = now + p.cooldown_s
                    return RecoveryStateResult(
                        state=self._state,
                        event="exit_climb",
                        elapsed_s=elapsed,
                        committed_side=self._committed_side,
                        recovery_progress_m=progress,
                        needs_stuck_reset=True,
                        cooldown_remaining_s=p.cooldown_s,
                    )

            # 2b. Timeout exit
            active_limit = p.max_duration_s
            if self._mode == "climb":
                active_limit = max(active_limit, p.vertical_climb_duration_s)
            elif self._mode == "wall":
                active_limit = max(active_limit, p.wall_follow_duration_s)
            if elapsed >= active_limit:
                self._state = RecoveryState.RECOVERY_COOLDOWN
                self._cooldown_until = now + p.cooldown_s
                return RecoveryStateResult(
                    state=self._state,
                    event="exit_timeout",
                    elapsed_s=elapsed,
                    committed_side=self._committed_side,
                    cooldown_remaining_s=p.cooldown_s,
                )

            # 2c. Stay active — re-compute command to refresh committed_side
            self._command, self._committed_side = self._compute_cmd(
                decision, lidar_rays, guidance_dir, bypass_side, elapsed,
            )
            return RecoveryStateResult(
                state=self._state,
                should_override=True,
                vx_body=self._command[0],
                vy_body=self._command[1],
                vz_body=self._command[2],
                event="active",
                elapsed_s=elapsed,
                committed_side=self._committed_side,
                recovery_progress_m=progress,
            )

        # 3. APF_ACTIVE: check entry condition
        if self._state == RecoveryState.APF_ACTIVE:
            entry_mode = _dead_end_mode(lidar_rays, self._params)
            if decision.needs_recovery or entry_mode is not None:
                self._state = RecoveryState.RECOVERY_ACTIVE
                self._active_start = now
                self._entry_position = current_position
                self._mode = entry_mode or (
                    "trajectory_no_feasible"
                    if decision.reason == "trajectory_no_feasible"
                    else "normal"
                )
                self._command, self._committed_side = self._compute_cmd(
                    decision, lidar_rays, guidance_dir, bypass_side, 0.0,
                )
                return RecoveryStateResult(
                    state=self._state,
                    should_override=True,
                    vx_body=self._command[0],
                    vy_body=self._command[1],
                    vz_body=self._command[2],
                    event="enter",
                    committed_side=self._committed_side,
                )

        return RecoveryStateResult(state=self._state)

    def force_exit(self, reason: str, now: float) -> RecoveryStateResult:
        """Safety override: immediately exit recovery (if active) into cooldown."""
        if self._state == RecoveryState.RECOVERY_ACTIVE:
            elapsed = now - self._active_start
            self._state = RecoveryState.RECOVERY_COOLDOWN
            self._cooldown_until = now + self._params.cooldown_s
            return RecoveryStateResult(
                state=self._state,
                event=f"exit_safety:{reason}",
                elapsed_s=elapsed,
                committed_side=self._committed_side,
                cooldown_remaining_s=self._params.cooldown_s,
            )
        return RecoveryStateResult(state=self._state)

    def reset(self) -> None:
        """Full reset (e.g. on mode exit)."""
        self._state = RecoveryState.APF_ACTIVE
        self._active_start = 0.0
        self._cooldown_until = 0.0
        self._command = (0.0, 0.0, 0.0)
        self._committed_side = None
        self._entry_position = None
        self._mode = "normal"

    # ── private ──

    def _compute_cmd(
        self,
        decision: RecoveryDecision,
        lidar_rays: Dict[str, float],
        guidance_dir: Optional[Tuple[float, float]],
        bypass_side: Optional[int],
        elapsed_s: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], int]:
        """Compute recovery command, inheriting bypass_side when active."""
        # Bypass inheritance: if a bypass episode is active, use its side
        effective_committed = self._committed_side
        if bypass_side is not None and bypass_side != 0:
            effective_committed = bypass_side

        cmd = compute_recovery_command(
            decision, lidar_rays,
            committed_side=effective_committed,
            guidance_dir=guidance_dir,
            params=self._params,
            elapsed_s=elapsed_s,
            forced_mode=self._mode if self._mode != "normal" else None,
        )
        return (cmd[0], cmd[1], cmd[2]), cmd[3]

    @property
    def state(self) -> str:
        return self._state

    @property
    def mode(self) -> str:
        """Current recovery strategy: ``normal``, ``climb`` or ``wall``."""
        return self._mode

    @property
    def params(self) -> RecoveryCommanderParams:
        """Configured parameters for safety integration and diagnostics."""
        return self._params

"""
Recovery Commander — recovery action selection and state machine.

Produces conservative recovery velocity commands when the
LocalRecovery detector flags ``needs_recovery``.  All commands are
body-FRD and strictly rate-limited.

State machine
-------------
::

    APF_ACTIVE ──(needs_recovery)──▶ RECOVERY_ACTIVE ──(1.0s timeout)──▶ RECOVERY_COOLDOWN
         ▲                                  │                                  │
         │                                  │ (safety override)                │ (2.5s expires)
         │                                  ▼                                  │
         ◀────────────────────────────────────────────────────────────────────┘

Safety conditions (collision, geofence, emergency) always preempt
recovery — the loop breaks before any recovery command is sent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from planners.local_recovery import RecoveryDecision

# ── speed limits ──

MAX_HORIZONTAL_SPEED_MPS = 0.12
STUCK_BACKWARD_VX = -0.12
OSCILLATION_LATERAL_VY = 0.12

# ── timing ──

RECOVERY_MAX_ACTIVE_S = 1.0
RECOVERY_COOLDOWN_S = 2.5


# ── state enum ──

class RecoveryState:
    APF_ACTIVE = "APF_ACTIVE"
    RECOVERY_ACTIVE = "RECOVERY_ACTIVE"
    RECOVERY_COOLDOWN = "RECOVERY_COOLDOWN"


# ── commander ──


def compute_recovery_command(
    decision: RecoveryDecision,
    lidar_rays: Dict[str, float],
) -> Tuple[float, float, float]:
    """Return a body-FRD recovery velocity ``(vx, vy, vz)``.

    The command is always conservative: max horizontal speed ≤ 0.12 m/s
    and ``vz = 0``.

    Stuck → short backward movement (``vx = -0.12``).
    Oscillation → lateral sidestep toward the more open side (±0.12).

    If both stuck and oscillating, stuck takes priority (backward
    movement breaks the deadlock first).
    """
    if decision.is_stuck:
        return (STUCK_BACKWARD_VX, 0.0, 0.0)

    if decision.is_oscillating:
        left = lidar_rays.get("left", 0.0) or 0.0
        right = lidar_rays.get("right", 0.0) or 0.0
        vy = OSCILLATION_LATERAL_VY if right > left else -OSCILLATION_LATERAL_VY
        return (0.0, vy, 0.0)

    return (0.0, 0.0, 0.0)


# ── state machine ──


@dataclass
class RecoveryStateResult:
    """Output of one state-machine tick."""

    state: str = RecoveryState.APF_ACTIVE
    should_override: bool = False
    vx_body: float = 0.0
    vy_body: float = 0.0
    vz_body: float = 0.0
    event: Optional[str] = None          # "enter", "active", "exit_timeout", "exit_safety"
    elapsed_s: float = 0.0
    cooldown_remaining_s: float = 0.0


class RecoveryStateMachine:
    """Manages recovery state transitions.

    Call ``tick(now, decision, lidar_rays)`` once per flight-loop
    iteration.  The returned ``RecoveryStateResult.should_override``
    tells the dispatcher whether to use the recovery command instead
    of the normal APF / reactive command.
    """

    def __init__(self) -> None:
        self._state = RecoveryState.APF_ACTIVE
        self._active_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._command: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── public API ──

    def tick(
        self,
        now: float,
        decision: RecoveryDecision,
        lidar_rays: Dict[str, float],
    ) -> RecoveryStateResult:
        """Advance the state machine one step.

        Args:
            now: ``time.monotonic()``.
            decision: Latest ``RecoveryDecision`` from ``LocalRecovery``.
            lidar_rays: Legacy ray distances dict (keys: front, left, right, …).

        Returns:
            ``RecoveryStateResult`` with current state and (if active)
            the recovery velocity command.
        """
        # 1. Cooldown expiry → APF_ACTIVE (fall through to entry check)
        had_cooldown = self._state == RecoveryState.RECOVERY_COOLDOWN
        if had_cooldown and now >= self._cooldown_until:
            self._state = RecoveryState.APF_ACTIVE
            # Fall through to 3 — check entry condition on same tick.
        elif had_cooldown:
            return RecoveryStateResult(
                state=self._state,
                cooldown_remaining_s=max(0.0, self._cooldown_until - now),
            )

        # 2. RECOVERY_ACTIVE: check timeout
        if self._state == RecoveryState.RECOVERY_ACTIVE:
            elapsed = now - self._active_start
            if elapsed >= RECOVERY_MAX_ACTIVE_S:
                self._state = RecoveryState.RECOVERY_COOLDOWN
                self._cooldown_until = now + RECOVERY_COOLDOWN_S
                return RecoveryStateResult(
                    state=self._state,
                    event="exit_timeout",
                    elapsed_s=elapsed,
                    cooldown_remaining_s=RECOVERY_COOLDOWN_S,
                )
            return RecoveryStateResult(
                state=self._state,
                should_override=True,
                vx_body=self._command[0],
                vy_body=self._command[1],
                vz_body=self._command[2],
                event="active",
                elapsed_s=elapsed,
            )

        # 3. APF_ACTIVE: check entry condition
        if self._state == RecoveryState.APF_ACTIVE:
            if decision.needs_recovery:
                self._state = RecoveryState.RECOVERY_ACTIVE
                self._active_start = now
                self._command = compute_recovery_command(decision, lidar_rays)
                return RecoveryStateResult(
                    state=self._state,
                    should_override=True,
                    vx_body=self._command[0],
                    vy_body=self._command[1],
                    vz_body=self._command[2],
                    event="enter",
                )

        return RecoveryStateResult(state=self._state)

    def force_exit(self, reason: str, now: float) -> RecoveryStateResult:
        """Safety override: immediately exit recovery (if active) into cooldown."""
        if self._state == RecoveryState.RECOVERY_ACTIVE:
            elapsed = now - self._active_start
            self._state = RecoveryState.RECOVERY_COOLDOWN
            self._cooldown_until = now + RECOVERY_COOLDOWN_S
            return RecoveryStateResult(
                state=self._state,
                event=f"exit_safety:{reason}",
                elapsed_s=elapsed,
                cooldown_remaining_s=RECOVERY_COOLDOWN_S,
            )
        return RecoveryStateResult(state=self._state)

    def reset(self) -> None:
        """Full reset (e.g. on mode exit)."""
        self._state = RecoveryState.APF_ACTIVE
        self._active_start = 0.0
        self._cooldown_until = 0.0
        self._command = (0.0, 0.0, 0.0)

    @property
    def state(self) -> str:
        return self._state

"""Tests for planners/recovery_commander.py — recovery takeover and state machine."""

import math
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.local_recovery import RecoveryDecision
from planners.recovery_commander import (
    RecoveryStateMachine,
    RecoveryState,
    RecoveryStateResult,
    compute_recovery_command,
    MAX_HORIZONTAL_SPEED_MPS,
    RECOVERY_MAX_ACTIVE_S,
    RECOVERY_COOLDOWN_S,
    STUCK_BACKWARD_VX,
    OSCILLATION_LATERAL_VY,
)


# ── helpers ──


def _stuck_decision() -> RecoveryDecision:
    return RecoveryDecision(
        is_stuck=True,
        is_oscillating=False,
        needs_recovery=True,
        reason="stuck",
    )


def _osc_decision() -> RecoveryDecision:
    return RecoveryDecision(
        is_stuck=False,
        is_oscillating=True,
        needs_recovery=True,
        reason="oscillation",
    )


def _both_decision() -> RecoveryDecision:
    return RecoveryDecision(
        is_stuck=True,
        is_oscillating=True,
        needs_recovery=True,
        reason="stuck+oscillation",
    )


def _none_decision() -> RecoveryDecision:
    return RecoveryDecision()


def _rays(**overrides):
    defaults = {"front": 10.0, "left": 8.0, "right": 5.0}
    defaults.update(overrides)
    return defaults


# ── compute_recovery_command ──


class TestComputeRecoveryCommand:
    def test_stuck_produces_backward(self):
        cmd = compute_recovery_command(_stuck_decision(), _rays())
        assert cmd == (-0.12, 0.0, 0.0)

    def test_oscillation_left_more_open(self):
        """Left clearance > right → go left (vy = -0.12)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=10.0, right=2.0))
        assert cmd[0] == 0.0
        assert cmd[1] == -0.12   # left side more open → go left
        assert cmd[2] == 0.0

    def test_oscillation_right_more_open(self):
        """Right clearance > left → go right (vy = +0.12)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=2.0, right=10.0))
        assert cmd[0] == 0.0
        assert cmd[1] == 0.12    # right side more open → go right
        assert cmd[2] == 0.0

    def test_oscillation_equal_goes_left(self):
        """Equal clearance → right > left is False → go left (vy=-0.12)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=5.0, right=5.0))
        assert cmd[0] == 0.0
        assert cmd[1] == -0.12
        assert cmd[2] == 0.0

    def test_stuck_overrides_oscillation(self):
        """When both stuck and oscillating, stuck (backward) takes priority."""
        cmd = compute_recovery_command(_both_decision(), _rays(left=10.0, right=2.0))
        assert cmd == (-0.12, 0.0, 0.0)

    def test_no_recovery_returns_zero(self):
        cmd = compute_recovery_command(_none_decision(), _rays())
        assert cmd == (0.0, 0.0, 0.0)

    def test_vz_always_zero(self):
        """All recovery commands must have vz=0."""
        for decision in [_stuck_decision(), _osc_decision(), _none_decision()]:
            cmd = compute_recovery_command(decision, _rays())
            assert cmd[2] == 0.0, f"vz must be 0 for {decision.reason}"

    def test_max_speed_limit(self):
        """All recovery commands must respect MAX_HORIZONTAL_SPEED_MPS (0.12)."""
        for decision in [_stuck_decision(), _osc_decision(), _both_decision()]:
            cmd = compute_recovery_command(decision, _rays())
            assert abs(cmd[0]) <= MAX_HORIZONTAL_SPEED_MPS
            assert abs(cmd[1]) <= MAX_HORIZONTAL_SPEED_MPS
            assert abs(cmd[2]) == 0.0

    def test_none_lidar_values(self):
        """None or missing LiDAR values default to 0."""
        cmd = compute_recovery_command(_osc_decision(), {})
        assert cmd[1] == -0.12  # both 0 → right > left is False → go left
        cmd2 = compute_recovery_command(_osc_decision(),
                                         {"left": None, "right": None})
        assert cmd2[1] == -0.12


# ── RecoveryStateMachine ──


class TestStateMachineEntry:
    def test_initial_state_is_apf_active(self):
        sm = RecoveryStateMachine()
        assert sm.state == RecoveryState.APF_ACTIVE

    def test_stuck_triggers_recovery_active(self):
        sm = RecoveryStateMachine()
        result = sm.tick(1000.0, _stuck_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "enter"
        assert result.vx_body == -0.12
        assert result.vy_body == 0.0
        assert result.vz_body == 0.0

    def test_oscillation_triggers_recovery_active(self):
        sm = RecoveryStateMachine()
        # left=10, right=2 → left side more open → go left (vy = -0.12)
        result = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=2.0))
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "enter"
        assert result.vx_body == 0.0
        assert result.vy_body == -0.12

    def test_no_recovery_no_transition(self):
        sm = RecoveryStateMachine()
        result = sm.tick(1000.0, _none_decision(), _rays())
        assert result.state == RecoveryState.APF_ACTIVE
        assert not result.should_override

    def test_second_tick_stays_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())  # enter
        result = sm.tick(1000.5, _stuck_decision(), _rays())  # still active
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "active"
        assert result.elapsed_s == pytest.approx(0.5)

    def test_commands_persist_across_ticks(self):
        """Once entered, the command is held for the full active period."""
        sm = RecoveryStateMachine()
        r1 = sm.tick(1000.0, _stuck_decision(), _rays())
        # Now feed oscillation decision — command should still be stuck-backward
        r2 = sm.tick(1000.3, _osc_decision(), _rays(left=10.0, right=2.0))
        assert r2.should_override
        assert r2.vx_body == -0.12  # stuck command persisted
        assert r2.vy_body == 0.0


class TestStateMachineTimeout:
    def test_timeout_returns_to_cooldown(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())  # enter
        # Tick exactly at 1.0s — should time out
        result = sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S, _stuck_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override
        assert result.event == "exit_timeout"
        assert result.elapsed_s == pytest.approx(RECOVERY_MAX_ACTIVE_S)

    def test_timeout_beyond_1s(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        result = sm.tick(1002.0, _stuck_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override

    def test_cooldown_prevents_reentry(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())             # enter
        sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S, _stuck_decision(), _rays())  # timeout → cooldown
        # Try to re-enter during cooldown
        result = sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S + 0.5, _stuck_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override

    def test_cooldown_expires_allows_reentry(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())             # enter
        now = 1000.0 + RECOVERY_MAX_ACTIVE_S                     # timeout
        sm.tick(now, _stuck_decision(), _rays())                 # → cooldown
        # Advance past cooldown
        now += RECOVERY_COOLDOWN_S
        result = sm.tick(now, _stuck_decision(), _rays())
        # Cooldown expired → falls through to entry check → RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "enter"

    def test_cooldown_state_reports_remaining(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        sm.tick(1001.0, _stuck_decision(), _rays())  # timeout → cooldown
        now = 1001.0 + 0.3
        result = sm.tick(now, _none_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert result.cooldown_remaining_s == pytest.approx(RECOVERY_COOLDOWN_S - 0.3, abs=0.01)


class TestStateMachineSafety:
    def test_force_exit_from_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        result = sm.force_exit("collision", 1000.5)
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert result.event == "exit_safety:collision"
        assert result.elapsed_s == pytest.approx(0.5)
        assert result.cooldown_remaining_s == RECOVERY_COOLDOWN_S

    def test_force_exit_from_apf_does_nothing(self):
        sm = RecoveryStateMachine()
        result = sm.force_exit("collision", 1000.0)
        assert result.state == RecoveryState.APF_ACTIVE
        assert not result.should_override

    def test_force_exit_from_cooldown_does_nothing(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        sm.tick(1001.0, _stuck_decision(), _rays())  # timeout → cooldown
        result = sm.force_exit("geofence", 1001.5)
        assert result.state == RecoveryState.RECOVERY_COOLDOWN

    def test_after_force_exit_cooldown_applies(self):
        """After safety force-exits, cooldown still prevents immediate re-entry."""
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        sm.force_exit("collision", 1000.3)
        # Try to re-enter immediately
        result = sm.tick(1000.4, _stuck_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override


class TestStateMachineReset:
    def test_reset_from_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        sm.reset()
        assert sm.state == RecoveryState.APF_ACTIVE
        result = sm.tick(1001.0, _none_decision(), _rays())
        assert not result.should_override

    def test_reset_from_cooldown(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays())
        sm.tick(1001.0, _stuck_decision(), _rays())
        assert sm.state == RecoveryState.RECOVERY_COOLDOWN
        sm.reset()
        assert sm.state == RecoveryState.APF_ACTIVE


class TestStateMachineEdgeCases:
    def test_decision_not_needing_recovery_does_not_transition(self):
        """needs_recovery=False should never trigger entry."""
        sm = RecoveryStateMachine()
        d = RecoveryDecision(is_stuck=True, needs_recovery=False, reason="stuck")
        result = sm.tick(1000.0, d, _rays())
        assert not result.should_override
        assert result.state == RecoveryState.APF_ACTIVE

    def test_multiple_complete_cycles(self):
        """Full cycle: APF → Recovery → Cooldown → APF → Recovery."""
        sm = RecoveryStateMachine()
        t = 1000.0

        # Cycle 1
        r1 = sm.tick(t, _stuck_decision(), _rays())
        assert r1.event == "enter"
        t += RECOVERY_MAX_ACTIVE_S
        r2 = sm.tick(t, _stuck_decision(), _rays())
        assert r2.event == "exit_timeout"
        t += RECOVERY_COOLDOWN_S
        r3 = sm.tick(t, _stuck_decision(), _rays())
        # Cooldown expired this tick → falls through → entry on same tick
        assert r3.event == "enter"
        assert r3.should_override

    def test_vz_always_zero_from_state_machine(self):
        sm = RecoveryStateMachine()
        # Stuck
        r = sm.tick(1000.0, _stuck_decision(), _rays())
        assert r.vz_body == 0.0
        sm.reset()
        # Oscillation
        r = sm.tick(1000.0, _osc_decision(), _rays())
        assert r.vz_body == 0.0

    def test_oscillation_picks_right_when_left_greater(self):
        """left=10, right=2 → left more open → go left (vy=-0.12)."""
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=2.0))
        assert r.vy_body == -0.12

    def test_oscillation_picks_left_when_right_greater(self):
        """right=10, left=2 → right more open → go right (vy=+0.12)."""
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _osc_decision(), _rays(left=2.0, right=10.0))
        assert r.vy_body == 0.12

    def test_invalid_decision_not_overriding(self):
        """A decision with invalid_input reason should not trigger takeover."""
        sm = RecoveryStateMachine()
        d = RecoveryDecision(needs_recovery=False, reason="invalid_input:px")
        r = sm.tick(1000.0, d, _rays())
        assert not r.should_override


# ── Constants ──

class TestConstants:
    def test_max_speed_limits(self):
        assert MAX_HORIZONTAL_SPEED_MPS == 0.12
        assert abs(STUCK_BACKWARD_VX) <= MAX_HORIZONTAL_SPEED_MPS
        assert abs(OSCILLATION_LATERAL_VY) <= MAX_HORIZONTAL_SPEED_MPS

    def test_timing_bounds(self):
        assert RECOVERY_MAX_ACTIVE_S == 1.0
        assert RECOVERY_COOLDOWN_S == 2.5
        assert RECOVERY_COOLDOWN_S > RECOVERY_MAX_ACTIVE_S

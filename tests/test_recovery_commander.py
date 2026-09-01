"""Tests for planners/recovery_commander.py — recovery takeover and state machine.

Updated for the enhanced API: 4-tuple return from compute_recovery_command,
committed_side persistence, guidance-aware side choice, RecoveryCommanderParams.
"""

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
    RecoveryCommanderParams,
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


def _dead_end_rays(**overrides):
    defaults = {
        "front": 1.0, "left": 1.0, "right": 1.0,
        "up": 10.0, "frontUp": 10.0, "leftUp": 10.0, "rightUp": 10.0,
    }
    defaults.update(overrides)
    return defaults


# ── compute_recovery_command ──


class TestComputeRecoveryCommand:
    def test_stuck_produces_backward_and_lateral(self):
        """Stuck → backward + lateral toward clearer side (left > right → go left)."""
        cmd = compute_recovery_command(_stuck_decision(), _rays(left=8.0, right=3.0))
        assert cmd[0] < 0  # backward
        assert cmd[1] < 0  # left side more open
        assert cmd[2] == 0.0
        assert cmd[3] == -1  # committed side = left

    def test_stuck_lateral_toward_right(self):
        """Right clearance > left → go right."""
        cmd = compute_recovery_command(_stuck_decision(), _rays(left=3.0, right=8.0))
        assert cmd[0] < 0  # backward
        assert cmd[1] > 0  # right
        assert cmd[3] == 1

    def test_no_feasible_advances_when_front_is_open(self):
        """Trajectory local minima should sidestep forward, not always reverse."""
        decision = RecoveryDecision(
            is_stuck=True,
            needs_recovery=True,
            reason="trajectory_no_feasible",
        )
        cmd = compute_recovery_command(
            decision,
            _rays(front=3.2, left=1.8, right=4.0),
            forced_mode="trajectory_no_feasible",
        )
        assert cmd[0] > 0
        assert cmd[1] > 0

    def test_oscillation_left_more_open(self):
        """Left clearance > right → go left (vy = -0.35)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=10.0, right=2.0))
        assert cmd[0] == 0.0
        assert cmd[1] < 0    # left side more open → go left
        assert cmd[2] == 0.0
        assert cmd[3] == -1  # committed side = left

    def test_oscillation_right_more_open(self):
        """Right clearance > left → go right (vy = +0.35)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=2.0, right=10.0))
        assert cmd[0] == 0.0
        assert cmd[1] > 0     # right side more open → go right
        assert cmd[2] == 0.0
        assert cmd[3] == 1    # committed side = right

    def test_oscillation_equal_goes_left(self):
        """Equal clearance → right > left is False → go left (vy=-0.35)."""
        cmd = compute_recovery_command(_osc_decision(), _rays(left=5.0, right=5.0))
        assert cmd[0] == 0.0
        assert cmd[1] < 0    # go left
        assert cmd[3] == -1

    def test_stuck_overrides_oscillation(self):
        """When both stuck and oscillating, stuck (backward) takes priority."""
        cmd = compute_recovery_command(_both_decision(), _rays(left=10.0, right=2.0))
        # Stuck → backward, not forward=0
        assert cmd[0] < 0  # backward

    def test_no_recovery_returns_zero(self):
        cmd = compute_recovery_command(_none_decision(), _rays())
        assert cmd == (0.0, 0.0, 0.0, 0)

    def test_vz_always_zero(self):
        """All recovery commands must have vz=0."""
        for decision in [_stuck_decision(), _osc_decision(), _none_decision()]:
            cmd = compute_recovery_command(decision, _rays())
            assert cmd[2] == 0.0, f"vz must be 0 for {decision.reason}"

    def test_max_speed_limit(self):
        """All recovery commands must respect MAX_HORIZONTAL_SPEED_MPS."""
        for decision in [_stuck_decision(), _osc_decision(), _both_decision()]:
            cmd = compute_recovery_command(decision, _rays())
            assert abs(cmd[0]) <= MAX_HORIZONTAL_SPEED_MPS
            assert abs(cmd[1]) <= MAX_HORIZONTAL_SPEED_MPS
            assert abs(cmd[2]) == 0.0

    def test_none_lidar_values(self):
        """None or missing LiDAR values default to 0 (not inf)."""
        cmd = compute_recovery_command(_osc_decision(), {})
        assert cmd[1] < 0  # both 0 → right > left is False → go left
        cmd2 = compute_recovery_command(_osc_decision(),
                                         {"left": None, "right": None})
        assert cmd2[1] < 0

    def test_committed_side_persists(self):
        """If committed_side is provided and safe, it persists."""
        cmd = compute_recovery_command(
            _stuck_decision(), _rays(left=8.0, right=2.0),
            committed_side=1,  # force right
        )
        assert cmd[1] > 0  # right, despite left being more open
        assert cmd[3] == 1

    def test_guidance_overrides_lidar(self):
        """CBMBA guidance direction biases side choice."""
        cmd = compute_recovery_command(
            _stuck_decision(), _rays(left=8.0, right=2.0),
            guidance_dir=(0.5, -0.8),  # strong left guidance
        )
        assert cmd[1] < 0  # should go left despite right being less open
        assert cmd[3] == -1

    def test_dead_end_climbs_when_all_upward_sectors_are_clear(self):
        p = RecoveryCommanderParams(vertical_clearance_m=2.0)
        cmd = compute_recovery_command(
            _none_decision(), _dead_end_rays(), params=p,
        )
        assert cmd[0] == 0.0
        assert cmd[1] == 0.0  # climb first; no diagonal escape command
        assert cmd[2] < 0.0  # NED: negative vz is upward

    def test_dead_end_uses_wall_follow_when_upward_is_blocked(self):
        p = RecoveryCommanderParams(vertical_clearance_m=2.0)
        cmd = compute_recovery_command(
            _none_decision(), _dead_end_rays(up=1.0), params=p,
        )
        assert cmd[2] == 0.0
        assert cmd[0] < 0.0
        assert cmd[1] != 0.0

    def test_wall_follow_keeps_committed_side_when_wall_is_close(self):
        # The selected wall can be close by design. Do not switch to the other
        # wall just because its instantaneous ray is shorter than 1.5 m.
        p = RecoveryCommanderParams(wall_follow_side_lock_enabled=True)
        cmd = compute_recovery_command(
            _none_decision(),
            _dead_end_rays(front=1.0, left=0.8, right=1.8, up=1.0),
            committed_side=-1,
            params=p,
            forced_mode="wall",
        )
        assert cmd[3] == -1
        assert cmd[1] < 0.0

    def test_trajectory_no_feasible_keeps_committed_side(self):
        p = RecoveryCommanderParams(wall_follow_side_lock_enabled=True)
        decision = RecoveryDecision(
            is_stuck=True, needs_recovery=True, reason="trajectory_no_feasible",
        )
        cmd = compute_recovery_command(
            decision, _rays(front=3.2, left=0.8, right=4.0),
            committed_side=-1, params=p, forced_mode="trajectory_no_feasible",
        )
        assert cmd[3] == -1
        assert cmd[1] < 0.0


# ── RecoveryStateMachine ──


class TestStateMachineEntry:
    def test_initial_state_is_apf_active(self):
        sm = RecoveryStateMachine()
        assert sm.state == RecoveryState.APF_ACTIVE

    def test_stuck_triggers_recovery_active(self):
        sm = RecoveryStateMachine()
        result = sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "enter"
        assert result.vx_body < 0  # backward

    def test_oscillation_triggers_recovery_active(self):
        sm = RecoveryStateMachine()
        result = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=2.0))
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "enter"
        assert result.vx_body == 0.0
        assert result.vy_body < 0  # left side more open

    def test_dead_end_locks_climb_mode_then_hands_back(self):
        p = RecoveryCommanderParams(
            max_duration_s=5.0,
            vertical_clearance_m=2.0,
            vertical_climb_duration_s=1.0,
            vertical_climb_delta_m=0.4,
        )
        sm = RecoveryStateMachine(p)
        r1 = sm.tick(1000.0, _none_decision(), _dead_end_rays(),
                     current_position=(0.0, 0.0, -1.0))
        assert r1.event == "enter"
        assert sm.mode == "climb"
        assert r1.vz_body < 0.0

        r2 = sm.tick(1000.5, _none_decision(), _dead_end_rays(),
                     current_position=(0.0, 0.0, -1.1))
        assert r2.should_override
        assert r2.vz_body < 0.0

        r3 = sm.tick(1001.1, _none_decision(), _dead_end_rays(),
                     current_position=(0.0, 0.0, -1.2))
        assert r3.event == "exit_climb"
        assert not r3.should_override

    def test_dead_end_switches_from_climb_to_fixed_side_wall(self):
        p = RecoveryCommanderParams(
            max_duration_s=5.0, vertical_clearance_m=2.0,
            vertical_climb_duration_s=2.0,
        )
        sm = RecoveryStateMachine(p)
        sm.tick(1000.0, _none_decision(), _dead_end_rays(),
                current_position=(0.0, 0.0, -1.0))
        blocked_up = _dead_end_rays(up=1.0)
        r = sm.tick(1000.2, _none_decision(), blocked_up,
                    current_position=(0.0, 0.0, -1.0))
        assert sm.mode == "wall"
        assert r.should_override
        assert r.vz_body == 0.0
        assert r.vy_body != 0.0

    def test_no_recovery_no_transition(self):
        sm = RecoveryStateMachine()
        result = sm.tick(1000.0, _none_decision(), _rays())
        assert result.state == RecoveryState.APF_ACTIVE
        assert not result.should_override

    def test_second_tick_stays_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        result = sm.tick(1000.5, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_ACTIVE
        assert result.should_override
        assert result.event == "active"
        assert result.elapsed_s == pytest.approx(0.5)

    def test_commands_persist_across_ticks(self):
        """Once entered, the command is held for the full active period."""
        sm = RecoveryStateMachine()
        r1 = sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        # Feed another stuck decision — command should still be stuck-backward
        r2 = sm.tick(1000.3, _stuck_decision(), _rays(left=10.0, right=2.0))
        assert r2.should_override
        assert r2.vx_body < 0  # stuck command (backward) persisted

    def test_trajectory_no_feasible_exits_when_front_opens_in_narrow_gap(self):
        # Side pillars may remain within the clearance threshold while the
        # forward opening is already safe. Do not wait for the full timeout.
        p = RecoveryCommanderParams(
            max_duration_s=5.0, required_progress_m=0.5, clear_distance_m=2.0,
        )
        decision = RecoveryDecision(
            is_stuck=True, needs_recovery=True, reason="trajectory_no_feasible",
        )
        sm = RecoveryStateMachine(p)
        r1 = sm.tick(
            1000.0, decision, _rays(front=1.0, left=3.0, right=3.0),
            current_position=(0.0, 0.0, -1.0),
        )
        assert r1.should_override
        r2 = sm.tick(
            1001.0, decision, _rays(front=3.0, left=1.5, right=1.5),
            current_position=(0.6, 0.0, -1.0),
        )
        assert r2.event == "exit_progress"
        assert not r2.should_override


class TestStateMachineTimeout:
    def test_timeout_returns_to_cooldown(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        result = sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override
        assert result.event == "exit_timeout"
        assert result.elapsed_s == pytest.approx(RECOVERY_MAX_ACTIVE_S)

    def test_timeout_beyond_1s(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        result = sm.tick(1002.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override

    def test_cooldown_prevents_reentry(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S, _stuck_decision(), _rays(left=8.0, right=3.0))
        result = sm.tick(1000.0 + RECOVERY_MAX_ACTIVE_S + 0.5, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override

    def test_cooldown_expires_allows_reentry(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        now = 1000.0 + RECOVERY_MAX_ACTIVE_S
        sm.tick(now, _stuck_decision(), _rays(left=8.0, right=3.0))
        now += RECOVERY_COOLDOWN_S
        # First tick: cooldown expires → APF_ACTIVE (event="cooldown_expired")
        r_expire = sm.tick(now, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r_expire.event == "cooldown_expired"
        # Next tick: stuck decision triggers fresh entry
        result = sm.tick(now + 0.1, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.should_override
        assert result.event == "enter"

    def test_cooldown_state_reports_remaining(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.tick(1001.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        now = 1001.0 + 0.3
        result = sm.tick(now, _none_decision(), _rays())
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert result.cooldown_remaining_s == pytest.approx(RECOVERY_COOLDOWN_S - 0.3, abs=0.01)


class TestStateMachineSafety:
    def test_force_exit_from_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
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
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.tick(1001.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        result = sm.force_exit("geofence", 1001.5)
        assert result.state == RecoveryState.RECOVERY_COOLDOWN

    def test_after_force_exit_cooldown_applies(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.force_exit("collision", 1000.3)
        result = sm.tick(1000.4, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert result.state == RecoveryState.RECOVERY_COOLDOWN
        assert not result.should_override


class TestStateMachineReset:
    def test_reset_from_active(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.reset()
        assert sm.state == RecoveryState.APF_ACTIVE
        result = sm.tick(1001.0, _none_decision(), _rays())
        assert not result.should_override

    def test_reset_from_cooldown(self):
        sm = RecoveryStateMachine()
        sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        sm.tick(1001.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert sm.state == RecoveryState.RECOVERY_COOLDOWN
        sm.reset()
        assert sm.state == RecoveryState.APF_ACTIVE


class TestStateMachineEdgeCases:
    def test_decision_not_needing_recovery_does_not_transition(self):
        sm = RecoveryStateMachine()
        d = RecoveryDecision(is_stuck=True, needs_recovery=False, reason="stuck")
        result = sm.tick(1000.0, d, _rays())
        assert not result.should_override
        assert result.state == RecoveryState.APF_ACTIVE

    def test_multiple_complete_cycles(self):
        sm = RecoveryStateMachine()
        t = 1000.0
        # Cycle 1: enter
        r1 = sm.tick(t, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r1.event == "enter"
        # Cycle 1: timeout → cooldown
        t += RECOVERY_MAX_ACTIVE_S
        r2 = sm.tick(t, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r2.event == "exit_timeout"
        # Cycle 1: cooldown expiry → APF_ACTIVE
        t += RECOVERY_COOLDOWN_S
        r_expire = sm.tick(t, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r_expire.event == "cooldown_expired"
        # Cycle 2: next tick re-enters
        t += 0.1
        r3 = sm.tick(t, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r3.event == "enter"
        assert r3.should_override

    def test_vz_always_zero_from_state_machine(self):
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=3.0))
        assert r.vz_body == 0.0
        sm.reset()
        r = sm.tick(1000.0, _osc_decision(), _rays(left=8.0, right=3.0))
        assert r.vz_body == 0.0

    def test_oscillation_picks_right_when_left_greater(self):
        """left=10, right=2 → left more open → go left."""
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=2.0))
        assert r.vy_body < 0   # go left

    def test_oscillation_picks_left_when_right_greater(self):
        """right=10, left=2 → right more open → go right."""
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _osc_decision(), _rays(left=2.0, right=10.0))
        assert r.vy_body > 0   # go right

    def test_invalid_decision_not_overriding(self):
        sm = RecoveryStateMachine()
        d = RecoveryDecision(needs_recovery=False, reason="invalid_input:px")
        r = sm.tick(1000.0, d, _rays())
        assert not r.should_override

    def test_side_commitment_persists_across_ticks(self):
        """Side chosen on entry is held for the entire active period."""
        sm = RecoveryStateMachine()
        r1 = sm.tick(1000.0, _stuck_decision(), _rays(left=8.0, right=2.0))
        side1 = r1.committed_side
        assert side1 is not None
        # Active tick — same side should persist
        r2 = sm.tick(1000.5, _stuck_decision(), _rays(left=8.0, right=2.0))
        assert r2.committed_side == side1  # side persisted

    def test_committed_side_persists_when_safe(self):
        """Committed side persists if that side still has adequate clearance."""
        sm = RecoveryStateMachine()
        r1 = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=3.0))
        # Chose left because more open
        assert r1.committed_side == -1
        # Next tick: left still has clearance (3.0 >= 1.5), should persist
        r2 = sm.tick(1000.5, _osc_decision(), _rays(left=3.0, right=10.0))
        assert r2.committed_side == -1  # committed side persisted

    def test_committed_side_changes_when_unsafe(self):
        """Committed side changes when the chosen side is no longer safe."""
        sm = RecoveryStateMachine()
        r1 = sm.tick(1000.0, _osc_decision(), _rays(left=10.0, right=3.0))
        assert r1.committed_side == -1  # chose left
        # Next tick: left is now unsafe (1.0 < 1.5) but right is open
        r2 = sm.tick(1000.5, _osc_decision(), _rays(left=1.0, right=10.0))
        assert r2.committed_side == 1  # switched to right


# ── Constants ──

class TestConstants:
    def test_max_speed_limits(self):
        assert MAX_HORIZONTAL_SPEED_MPS == 0.35
        assert abs(STUCK_BACKWARD_VX) <= MAX_HORIZONTAL_SPEED_MPS
        assert abs(OSCILLATION_LATERAL_VY) <= MAX_HORIZONTAL_SPEED_MPS

    def test_timing_bounds(self):
        assert RECOVERY_MAX_ACTIVE_S == 1.0
        assert RECOVERY_COOLDOWN_S == 2.5
        assert RECOVERY_COOLDOWN_S > RECOVERY_MAX_ACTIVE_S


# ── RecoveryCommanderParams ──

class TestRecoveryCommanderParams:
    def test_default_params(self):
        p = RecoveryCommanderParams()
        assert p.reverse_speed == 0.12
        assert p.lateral_speed == 0.12
        assert p.max_duration_s == 1.0
        assert p.cooldown_s == 2.5

    def test_custom_params_respected(self):
        p = RecoveryCommanderParams(
            reverse_speed=0.35, lateral_speed=0.35,
            max_duration_s=2.0, cooldown_s=5.0,
        )
        sm = RecoveryStateMachine(p)
        assert sm._params.reverse_speed == 0.35
        assert sm._params.max_duration_s == 2.0

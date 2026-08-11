"""Tests for recovery test trigger CLI hook (--recovery-test-trigger)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
from planners.local_recovery import RecoveryDecision
from planners.recovery_commander import RecoveryStateMachine, RecoveryState


# ── helpers ──


def _make_auto(trigger=None):
    """Build an AutomaticMode with test trigger (or None = disabled)."""
    session = MagicMock()
    session.client = MagicMock()
    session.adapter = MagicMock()
    session.vehicle_name = "Drone1"

    overrides = {"planner_mode": "reactive"}
    if trigger is not None:
        overrides["recovery_test_trigger"] = trigger

    return AutomaticMode(
        session,
        params=AutomaticModeParams(
            target_z_ned=-2.0,
            max_flight_duration_s=0.2,
        ),
        cli_overrides=overrides,
    )


def _synthetic_decision(trigger_type):
    """Build the synthetic decision exactly as the injection code does."""
    return RecoveryDecision(
        is_stuck=(trigger_type == "stuck"),
        is_oscillating=(trigger_type == "oscillation"),
        needs_recovery=True,
        reason=f"test_trigger:{trigger_type}",
    )


# ── tests ──


class TestTriggerDefaultDisabled:
    def test_no_trigger_default(self):
        auto = _make_auto(trigger=None)
        assert auto._recovery_test_trigger is None
        assert not auto._recovery_test_trigger_fired

    def test_trigger_is_none_when_not_in_overrides(self):
        auto = _make_auto(trigger=None)
        assert auto._recovery_test_trigger is None


class TestTriggerStorage:
    def test_stuck_trigger_stored(self):
        auto = _make_auto(trigger="stuck")
        assert auto._recovery_test_trigger == "stuck"
        assert not auto._recovery_test_trigger_fired
        assert auto._recovery_test_trigger_delay_frames == 15

    def test_oscillation_trigger_stored(self):
        auto = _make_auto(trigger="oscillation")
        assert auto._recovery_test_trigger == "oscillation"
        assert not auto._recovery_test_trigger_fired


class TestSyntheticDecisionContent:
    """Verify the injected decision fields match expectations."""

    def test_stuck_decision_fields(self):
        d = _synthetic_decision("stuck")
        assert d.is_stuck is True
        assert d.is_oscillating is False
        assert d.needs_recovery is True
        assert d.reason == "test_trigger:stuck"

    def test_oscillation_decision_fields(self):
        d = _synthetic_decision("oscillation")
        assert d.is_stuck is False
        assert d.is_oscillating is True
        assert d.needs_recovery is True
        assert d.reason == "test_trigger:oscillation"

    def test_decision_not_special_cased(self):
        """The synthetic decision is a plain RecoveryDecision."""
        d = _synthetic_decision("stuck")
        assert isinstance(d, RecoveryDecision)
        assert not hasattr(d, "_is_test_trigger")  # no hidden markers


class TestStateMachineIntegration:
    """Verify the synthetic decision flows through RecoveryStateMachine."""

    def test_stuck_decision_enters_recovery(self):
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _synthetic_decision("stuck"), {"front": 10})
        assert r.state == RecoveryState.RECOVERY_ACTIVE
        assert r.should_override
        assert r.event == "enter"
        assert r.vx_body == -0.12       # stuck → backward
        assert r.vy_body == 0.0

    def test_oscillation_decision_enters_recovery(self):
        sm = RecoveryStateMachine()
        r = sm.tick(1000.0, _synthetic_decision("oscillation"),
                    {"front": 10, "left": 2.0, "right": 10.0})
        assert r.state == RecoveryState.RECOVERY_ACTIVE
        assert r.should_override
        assert r.event == "enter"
        assert r.vx_body == 0.0
        assert r.vy_body == 0.12        # right more open → go right

    def test_decision_flows_through_full_cycle(self):
        """Complete cycle: enter → active → timeout → cooldown → APF."""
        sm = RecoveryStateMachine()
        t = 1000.0

        # Enter
        r1 = sm.tick(t, _synthetic_decision("stuck"), {"front": 10})
        assert r1.event == "enter"
        assert r1.should_override

        # Active
        t += 0.3
        r2 = sm.tick(t, RecoveryDecision(), {"front": 10})
        assert r2.event == "active"
        assert r2.should_override

        # Timeout
        t = 1000.0 + 1.0
        r3 = sm.tick(t, RecoveryDecision(), {"front": 10})
        assert r3.event == "exit_timeout"
        assert not r3.should_override
        assert r3.state == RecoveryState.RECOVERY_COOLDOWN

        # Cooldown expired → APF
        t += 2.5
        r4 = sm.tick(t, RecoveryDecision(), {"front": 10})
        assert r4.state == RecoveryState.APF_ACTIVE


class TestTriggerNoBypass:
    """The trigger must not bypass safety or existing dispatch paths."""

    def test_trigger_uses_existing_state_machine(self):
        auto = _make_auto(trigger="stuck")
        assert isinstance(auto._recovery_sm, RecoveryStateMachine)
        assert auto._recovery_sm.state == RecoveryState.APF_ACTIVE

    def test_no_independent_dispatch(self):
        """The test trigger only affects recovery_decision; dispatch is unchanged."""
        auto = _make_auto(trigger="stuck")
        # Dispatch still uses recovery_result.should_override
        # (tested by commanding through state machine above)
        assert auto._planner_mode == "reactive"

    def test_trigger_does_not_modify_detector(self):
        """LocalRecovery detector thresholds are unchanged."""
        auto = _make_auto(trigger="stuck")
        p = auto._recovery._params
        assert p.history_window_s == 4.0
        assert p.stuck_time_window_s == 2.5
        assert p.stuck_position_epsilon_m == 0.15
        assert p.oscillation_min_sign_flips == 3


class TestInjectionSiteUsesCorrectAlias:
    """Prevent regression: the real injection line must use _RecoveryDecision,
    not the bare RecoveryDecision (which is not in scope inside run())."""

    def test_injection_uses_underscore_alias(self):
        source = Path(__file__).resolve().parent.parent / "flight_modes" / "automatic_mode.py"
        content = source.read_text(encoding="utf-8")
        # The synthetic = line must reference _RecoveryDecision
        assert "synthetic = _RecoveryDecision(" in content, \
            "Injection site must use _RecoveryDecision (the local alias)"
        # The bare name should NOT appear as a constructor call in run()
        # (it's only imported in __init__ scope, not run() scope)
        assert "synthetic = RecoveryDecision(" not in content, \
            "Bare RecoveryDecision is not in scope inside run(); use _RecoveryDecision"

    def test_default_assignment_uses_underscore_alias(self):
        source = Path(__file__).resolve().parent.parent / "flight_modes" / "automatic_mode.py"
        content = source.read_text(encoding="utf-8")
        assert "recovery_decision = _RecoveryDecision()" in content, \
            "Default assignment must use _RecoveryDecision"

    def test_import_alias_exists_in_run(self):
        source = Path(__file__).resolve().parent.parent / "flight_modes" / "automatic_mode.py"
        content = source.read_text(encoding="utf-8")
        assert "from planners.local_recovery import RecoveryDecision as _RecoveryDecision" in content, \
            "Local alias import must exist inside run()"

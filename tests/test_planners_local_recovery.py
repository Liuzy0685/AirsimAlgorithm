"""Tests for planners/local_recovery.py — stuck/oscillation detection module."""

import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.local_recovery import (
    LocalRecovery, RecoveryDecision, RecoveryParams, _vy_sign,
)


# ── helpers ──

def _params(**overrides):
    """Build RecoveryParams with low history/stuck windows for fast tests."""
    defaults = dict(
        history_window_s=4.0,          # must be > stuck_time_window_s
        stuck_time_window_s=2.5,
        stuck_position_epsilon_m=0.15,
        stuck_min_frames=5,
        oscillation_time_window_s=2.0,
        oscillation_min_sign_flips=3,
        oscillation_lateral_epsilon_m=0.2,
    )
    defaults.update(overrides)
    # Ensure invariant
    assert defaults["history_window_s"] > defaults["stuck_time_window_s"]
    return RecoveryParams(**defaults)


# ── _vy_sign unit tests ──


class TestVySign:
    def test_positive(self):
        assert _vy_sign(1.0) == 1
        assert _vy_sign(0.03) == 1

    def test_negative(self):
        assert _vy_sign(-1.0) == -1
        assert _vy_sign(-0.03) == -1

    def test_dead_zone(self):
        assert _vy_sign(0.0) == 0
        assert _vy_sign(0.01) == 0
        assert _vy_sign(-0.01) == 0
        assert _vy_sign(0.019) == 0


# ── RecoveryDecision / RecoveryParams dataclasses ──


class TestRecoveryDecision:
    def test_defaults_all_false(self):
        d = RecoveryDecision()
        assert not d.is_stuck
        assert not d.is_oscillating
        assert not d.needs_recovery
        assert d.candidate_actions == []
        assert d.reason == ""

    def test_fields_independent_from_needs_recovery(self):
        d = RecoveryDecision(is_stuck=True, is_oscillating=False, needs_recovery=True)
        assert d.is_stuck
        assert not d.is_oscillating
        assert d.needs_recovery

    def test_position_fields_populated(self):
        """stuck_latest_position and stuck_oldest_position are set by update()."""
        d = RecoveryDecision(
            stuck_latest_position=(1.0, 2.0, -3.0),
            stuck_oldest_position=(0.0, 0.0, -3.0),
        )
        assert d.stuck_latest_position == (1.0, 2.0, -3.0)
        assert d.stuck_oldest_position == (0.0, 0.0, -3.0)


class TestRecoveryParams:
    def test_defaults(self):
        p = RecoveryParams()
        assert p.history_window_s == 4.0
        assert p.stuck_time_window_s == 2.5
        assert p.stuck_position_epsilon_m == 0.15
        assert p.stuck_min_frames == 10
        assert p.oscillation_time_window_s == 2.0
        assert p.oscillation_min_sign_flips == 3
        assert p.oscillation_lateral_epsilon_m == 0.2

    def test_candidate_actions_tuple(self):
        p = RecoveryParams()
        assert "escape_maneuver" in p.candidate_actions
        assert "vertical_climb" in p.candidate_actions
        assert "lateral_sidestep" in p.candidate_actions

    def test_history_larger_than_stuck(self):
        """history_window_s must be > stuck_time_window_s for correct operation."""
        p = RecoveryParams()
        assert p.history_window_s > p.stuck_time_window_s


# ── Normal flight — no false triggers ──


class TestNormalFlightNoTrigger:
    def test_moving_forward_no_stuck(self):
        """Drone moving forward at 0.2 m/s for 3s: stuck=False, delta > 0.15."""
        rec = LocalRecovery(_params(
            history_window_s=4.0,
            stuck_time_window_s=2.0,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(20):
            rec.update(
                timestamp=t + i * 0.15,
                position=(float(i) * 0.2, 0.0, -1.0),
                velocity_body=(1.0, 0.0, 0.0),
            )
        d = rec.update(t + 3.0, (4.0, 0.0, -1.0), (1.0, 0.0, 0.0))
        assert not d.is_stuck
        assert d.stuck_position_delta_m > 0.15, \
            f"Moving 0.2 m/s for 3s, expected delta > 0.15, got {d.stuck_position_delta_m:.3f}"

    def test_steady_vy_no_oscillation(self):
        """Steady vy sign → no oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.6,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(
                timestamp=t + i * 0.1,
                position=(float(i) * 0.05, 0.0, -1.0),
                velocity_body=(0.5, 0.15, 0.0),
            )
        d = rec.update(t + 1.0, (0.5, 0.0, -1.0), (0.5, 0.15, 0.0))
        assert not d.is_oscillating

    def test_window_not_full_no_false_positive(self):
        """Before min_frames reached, stuck should not trigger."""
        rec = LocalRecovery(_params(
            stuck_time_window_s=1.0,
            stuck_min_frames=10,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(5):
            d = rec.update(
                timestamp=t + i * 0.2,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.0, 0.0, 0.0),
            )
        assert not d.is_stuck
        assert d.window_size_frames == 5


# ── Stuck detection — basic ──


class TestStuckDetection:
    def test_position_unchanged_triggers_stuck(self):
        """Position stays within epsilon for >= time window → stuck."""
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=0.6,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(
                timestamp=t + i * 0.1,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.2, 0.0, 0.0),
            )
        d = rec.update(t + 1.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.is_stuck
        assert d.needs_recovery
        assert d.stuck_position_delta_m == pytest.approx(0.0, abs=1e-6)
        assert "stuck" in d.reason
        assert len(d.candidate_actions) > 0

    def test_small_movement_still_stuck(self):
        """Position moves less than epsilon → still stuck."""
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=0.6,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(
                timestamp=t + i * 0.1,
                position=(float(i) * 0.01, 0.0, -1.0),
                velocity_body=(0.05, 0.0, 0.0),
            )
        d = rec.update(t + 1.0, (0.09, 0.0, -1.0), (0.05, 0.0, 0.0))
        assert d.is_stuck

    def test_large_movement_not_stuck(self):
        """Position moves more than epsilon → not stuck."""
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=0.6,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(
                timestamp=t + i * 0.1,
                position=(float(i) * 0.2, 0.0, -1.0),
                velocity_body=(1.0, 0.0, 0.0),
            )
        d = rec.update(t + 1.0, (1.8, 0.0, -1.0), (1.0, 0.0, 0.0))
        assert not d.is_stuck

    def test_stuck_releases_when_moving_again(self):
        """After being stuck, moving again clears the flag."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        # Stationary → stuck
        for i in range(10):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 1.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.is_stuck

        # Now move forward
        for i in range(10):
            rec.update(t + 1.1 + i * 0.1, (float(i) * 0.3, 0.0, -1.0), (2.0, 0.0, 0.0))
        d2 = rec.update(t + 2.1, (2.7, 0.0, -1.0), (2.0, 0.0, 0.0))
        assert not d2.is_stuck

    def test_delta_always_computed_even_when_not_stuck(self):
        """stuck_position_delta_m is the real XY delta, even when not stuck."""
        rec = LocalRecovery(_params(
            history_window_s=4.0,
            stuck_time_window_s=2.5,  # large — won't be reached quickly
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        # Move 0.2 m/s for 1s → delta ≈ 0.2
        for i in range(7):
            rec.update(
                timestamp=t + i * 0.15,
                position=(float(i) * 0.2, 0.0, -1.0),
                velocity_body=(1.0, 0.0, 0.0),
            )
        d = rec.update(t + 1.0, (1.2, 0.0, -1.0), (1.0, 0.0, 0.0))
        # Not stuck (duration < 2.5s), but delta must still be the real value
        assert not d.is_stuck
        assert d.stuck_position_delta_m > 0.1, \
            f"Expected real delta, got {d.stuck_position_delta_m:.3f}"

    def test_delta_is_horizontal_xy_only(self):
        """stuck_delta uses only XY, ignores Z."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        # XY stationary, Z changing → should be stuck (XY delta = 0)
        for i in range(8):
            rec.update(
                timestamp=t + i * 0.1,
                position=(0.0, 0.0, -1.0 - float(i) * 0.5),
                velocity_body=(0.0, 0.0, 0.5),
            )
        d = rec.update(t + 0.8, (0.0, 0.0, -4.5), (0.0, 0.0, 0.5))
        # XY delta = 0 < 0.15 → stuck (Z movement does NOT count)
        assert d.is_stuck
        assert d.stuck_position_delta_m == pytest.approx(0.0, abs=0.01)

    def test_position_diagnostics_populated(self):
        """latest_position and oldest_position report the actual NED values."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(8):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 0.8, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.stuck_latest_position == (0.0, 0.0, -1.0)
        assert d.stuck_oldest_position[2] == pytest.approx(-1.0)


# ── FPS independence ──


class TestFpsIndependence:
    """Stuck detection must produce consistent results at 5, 10, and 20 Hz."""

    # Shared parameters: stuck requires stationarity for >= 2.5 s, delta < 0.15 m.
    # history_window_s = 4.0 ensures enough frames survive pruning at all rates.
    SHARED_PARAMS = dict(
        history_window_s=4.0,
        stuck_time_window_s=2.5,
        stuck_position_epsilon_m=0.15,
        stuck_min_frames=5,
        oscillation_min_sign_flips=10,   # high — disable oscillation in these tests
        oscillation_lateral_epsilon_m=0.01,
    )

    def _run(self, fps, duration_s, position_fn):
        """Feed frames at *fps* for *duration_s*, return final RecoveryDecision."""
        rec = LocalRecovery(_params(**self.SHARED_PARAMS))
        t = 1000.0
        dt = 1.0 / fps
        n = int(duration_s * fps)
        for i in range(n):
            rec.update(
                timestamp=t + i * dt,
                position=position_fn(i * dt),
                velocity_body=(0.0, 0.0, 0.0),
            )
        return rec.update(t + duration_s, position_fn(duration_s), (0.0, 0.0, 0.0))

    # ── stationary tests ──

    def _stationary(self, _t):
        return (0.0, 0.0, -1.0)

    def test_5hz_stationary_2_4s_not_stuck(self):
        """2.4s stationary at 5 Hz: duration < 2.5s → not stuck."""
        d = self._run(fps=5, duration_s=2.4, position_fn=self._stationary)
        assert not d.is_stuck, f"2.4s < 2.5s, should NOT be stuck, got stuck_dur={d.stuck_duration_s:.2f}"

    def test_5hz_stationary_2_6s_stuck(self):
        """2.6s stationary at 5 Hz: duration >= 2.5s → stuck."""
        d = self._run(fps=5, duration_s=2.6, position_fn=self._stationary)
        assert d.is_stuck, f"2.6s >= 2.5s, should be stuck, got stuck_dur={d.stuck_duration_s:.2f}"

    def test_10hz_stationary_2_4s_not_stuck(self):
        """2.4s stationary at 10 Hz: not stuck."""
        d = self._run(fps=10, duration_s=2.4, position_fn=self._stationary)
        assert not d.is_stuck

    def test_10hz_stationary_2_6s_stuck(self):
        """2.6s stationary at 10 Hz: stuck."""
        d = self._run(fps=10, duration_s=2.6, position_fn=self._stationary)
        assert d.is_stuck

    def test_20hz_stationary_2_4s_not_stuck(self):
        """2.4s stationary at 20 Hz: not stuck."""
        d = self._run(fps=20, duration_s=2.4, position_fn=self._stationary)
        assert not d.is_stuck

    def test_20hz_stationary_2_6s_stuck(self):
        """2.6s stationary at 20 Hz: stuck."""
        d = self._run(fps=20, duration_s=2.6, position_fn=self._stationary)
        assert d.is_stuck

    # ── moving tests ──

    def _moving_0_2(self, t_elapsed):
        """0.2 m/s forward along X."""
        return (t_elapsed * 0.2, 0.0, -1.0)

    def test_5hz_moving_3s_not_stuck_and_delta_gt_015(self):
        """0.2 m/s for 3s at 5 Hz: stuck=False, delta > 0.15."""
        d = self._run(fps=5, duration_s=3.0, position_fn=self._moving_0_2)
        assert not d.is_stuck
        assert d.stuck_position_delta_m > 0.15, \
            f"Moving 0.2 m/s for 3s, delta={d.stuck_position_delta_m:.3f} should be > 0.15"

    def test_10hz_moving_3s_not_stuck_and_delta_gt_015(self):
        """0.2 m/s for 3s at 10 Hz: stuck=False, delta > 0.15."""
        d = self._run(fps=10, duration_s=3.0, position_fn=self._moving_0_2)
        assert not d.is_stuck
        assert d.stuck_position_delta_m > 0.15

    def test_20hz_moving_3s_not_stuck_and_delta_gt_015(self):
        """0.2 m/s for 3s at 20 Hz: stuck=False, delta > 0.15."""
        d = self._run(fps=20, duration_s=3.0, position_fn=self._moving_0_2)
        assert not d.is_stuck
        assert d.stuck_position_delta_m > 0.15


# ── Oscillation detection ──


class TestOscillationDetection:
    def test_vy_sign_flips_no_lateral_progress(self):
        """vy flips + - + - with no lateral movement → oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=1.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_sequence = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_sequence):
            rec.update(
                timestamp=t + i * 0.15,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.1, vy, 0.0),
            )
        d = rec.update(t + 0.9, (0.0, 0.0, -1.0), (0.1, 0.3, 0.0))
        assert d.is_oscillating
        assert d.needs_recovery
        assert d.oscillation_vy_sign_flips >= 3
        assert d.oscillation_lateral_progress_m < 0.2
        assert "oscillation" in d.reason

    def test_vy_sign_flips_with_lateral_progress_no_oscillation(self):
        """vy flips but lateral position progresses → effective avoidance."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.8,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_sequence = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_sequence):
            rec.update(
                timestamp=t + i * 0.15,
                position=(0.05 * i, float(i) * 0.3, -1.0),
                velocity_body=(0.1, vy, 0.0),
            )
        d = rec.update(t + 0.9, (0.25, 1.5, -1.0), (0.1, 0.3, 0.0))
        assert not d.is_oscillating

    def test_insufficient_flips_no_oscillation(self):
        """Only 1 flip → not enough for oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=1.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_sequence = [0.3, -0.3, -0.3, -0.3]
        for i, vy in enumerate(vy_sequence):
            rec.update(
                timestamp=t + i * 0.2,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.1, vy, 0.0),
            )
        d = rec.update(t + 0.8, (0.0, 0.0, -1.0), (0.1, -0.3, 0.0))
        assert not d.is_oscillating

    def test_dead_zone_ignored_in_flips(self):
        """vy within dead zone does not count as sign change."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.8,
            oscillation_min_sign_flips=2,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_sequence = [0.3, 0.01, -0.3, -0.01, 0.3]
        for i, vy in enumerate(vy_sequence):
            rec.update(
                timestamp=t + i * 0.15,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.1, vy, 0.0),
            )
        d = rec.update(t + 0.75, (0.0, 0.0, -1.0), (0.1, 0.3, 0.0))
        assert d.oscillation_vy_sign_flips == 2

    # ── body-lateral vs total-XY progress (osc_lateral fix) ──

    def test_pure_forward_x_osc_lateral_zero(self):
        """Pure forward flight (X only, vy=0) with yaw=0: osc_lateral ≈ 0."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=2.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        # Forward at 0.5 m/s for 3s, vy=0, yaw=0 (heading North = +X)
        for i in range(20):
            rec.update(
                timestamp=t + i * 0.15,
                position=(float(i) * 0.5 * 0.15, 0.0, -1.0),
                velocity_body=(0.5, 0.0, 0.0),
                yaw_rad=0.0,
            )
        d = rec.update(t + 3.0, (1.5, 0.0, -1.0), (0.5, 0.0, 0.0), yaw_rad=0.0)
        # No vy flips → not oscillating.  osc_lateral should be ~0 (body-Y)
        assert not d.is_oscillating
        assert d.oscillation_lateral_progress_m == pytest.approx(0.0, abs=0.02), \
            f"Pure forward X, osc_lateral should be ≈0, got {d.oscillation_lateral_progress_m:.4f}"

    def test_vy_flips_no_body_lateral_oscillates(self):
        """X forward + vy flips, but body-Y stays near 0 → oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=1.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_seq = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_seq):
            rec.update(
                timestamp=t + i * 0.12,
                position=(float(i) * 0.2, 0.0, -1.0),  # X progresses, Y stays 0
                velocity_body=(0.2, vy, 0.0),
                yaw_rad=0.0,
            )
        d = rec.update(t + 0.96, (1.4, 0.0, -1.0), (0.2, 0.3, 0.0), yaw_rad=0.0)
        # vy flips ≥ 3, body-Y progress ≈ 0 → oscillating
        assert d.is_oscillating, \
            f"vy flips={d.oscillation_vy_sign_flips}, osc_lateral={d.oscillation_lateral_progress_m:.4f}"

    def test_unidirectional_lateral_no_oscillation(self):
        """Sustained lateral drift in one direction → no oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=1.0,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy = 0.4  # steady rightward
        for i in range(15):
            rec.update(
                timestamp=t + i * 0.1,
                position=(float(i) * 0.1, float(i) * 0.15, -1.0),  # X+Y progress
                velocity_body=(0.3, vy, 0.0),
                yaw_rad=0.0,
            )
        d = rec.update(t + 1.4, (1.4, 2.1, -1.0), (0.3, vy, 0.0), yaw_rad=0.0)
        # 0 sign flips → not oscillating
        assert d.oscillation_vy_sign_flips == 0
        assert not d.is_oscillating

    def test_yaw_90_body_lateral_correct(self):
        """Yaw=90° (heading East): body-lateral = -Δx (NED x becomes body -Y)."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.6,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_seq = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        yaw = math.pi / 2  # 90°, heading East
        for i, vy in enumerate(vy_seq):
            rec.update(
                timestamp=t + i * 0.1,
                # NED position: moving East (+Y in NED), no North (+X in NED)
                position=(0.0, float(i) * 0.15, -1.0),
                velocity_body=(0.2, vy, 0.0),  # body-FRD: forward=East
                yaw_rad=yaw,
            )
        d = rec.update(t + 0.5, (0.0, 0.75, -1.0), (0.2, 0.3, 0.0), yaw_rad=yaw)
        # yaw=90° → body-lateral axis = (-sin(90°), cos(90°)) = (-1, 0)
        # lateral = |dx*(-1) + dy*0| = |dx|
        # dx = 0 (no north change), so osc_lateral ≈ 0 → oscillation
        assert d.is_oscillating
        assert d.oscillation_lateral_progress_m == pytest.approx(0.0, abs=0.02)


# ── Combined stuck + oscillation ──


class TestCombinedDetection:
    def test_both_stuck_and_oscillating(self):
        """When both conditions hold, both flags are set."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
            oscillation_time_window_s=0.5,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_sequence = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_sequence):
            rec.update(
                timestamp=t + i * 0.1,
                position=(0.0, 0.0, -1.0),
                velocity_body=(0.1, vy, 0.0),
            )
        d = rec.update(t + 0.8, (0.0, 0.0, -1.0), (0.1, 0.3, 0.0))
        assert d.is_stuck
        assert d.is_oscillating
        assert d.needs_recovery
        assert "stuck" in d.reason
        assert "oscillation" in d.reason


# ── Invalid input handling ──


class TestInvalidInput:
    def test_nan_position_returns_gracefully(self):
        rec = LocalRecovery()
        d = rec.update(1000.0, (float("nan"), 0.0, -1.0), (0.0, 0.0, 0.0))
        assert not d.needs_recovery
        assert "invalid_input" in d.reason

    def test_nan_velocity_returns_gracefully(self):
        rec = LocalRecovery()
        d = rec.update(1000.0, (0.0, 0.0, -1.0), (float("nan"), 0.0, 0.0))
        assert not d.needs_recovery
        assert "invalid_input" in d.reason

    def test_inf_position_returns_gracefully(self):
        rec = LocalRecovery()
        d = rec.update(1000.0, (float("inf"), 0.0, -1.0), (0.0, 0.0, 0.0))
        assert not d.needs_recovery
        assert "invalid_input" in d.reason

    def test_inf_velocity_returns_gracefully(self):
        rec = LocalRecovery()
        d = rec.update(1000.0, (0.0, 0.0, -1.0), (0.0, float("-inf"), 0.0))
        assert not d.needs_recovery
        assert "invalid_input" in d.reason

    def test_window_preserved_after_invalid(self):
        """Invalid frame does not corrupt the window."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_min_frames=4,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(5):
            rec.update(t + i * 0.15, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        # Inject invalid
        rec.update(t + 0.8, (float("nan"), 0.0, -1.0), (0.2, 0.0, 0.0))
        # Continue valid
        for i in range(8):
            rec.update(t + 1.0 + i * 0.15, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 2.2, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.is_stuck


# ── Reset ──


class TestReset:
    def test_reset_clears_stuck_window(self):
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 1.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.is_stuck

        rec.reset()
        d2 = rec.update(t + 1.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert not d2.is_stuck
        assert d2.window_size_frames == 1

    def test_reset_clears_oscillation(self):
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.5,
            oscillation_min_sign_flips=3,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_seq = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_seq):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.1, vy, 0.0))
        d = rec.update(t + 0.6, (0.0, 0.0, -1.0), (0.1, 0.3, 0.0))
        assert d.is_oscillating

        rec.reset()
        d2 = rec.update(t + 0.7, (0.0, 0.0, -1.0), (0.1, 0.3, 0.0))
        assert not d2.is_oscillating


# ── Candidate actions ──


class TestCandidateActions:
    def test_candidates_empty_when_no_recovery(self):
        rec = LocalRecovery(_params(
            stuck_time_window_s=2.0,
            stuck_min_frames=10,
        ))
        t = 1000.0
        for i in range(5):
            rec.update(t + i * 0.2, (float(i) * 0.3, 0.0, -1.0), (1.0, 0.0, 0.0))
        d = rec.update(t + 1.0, (1.5, 0.0, -1.0), (1.0, 0.0, 0.0))
        assert not d.needs_recovery
        assert d.candidate_actions == []

    def test_candidates_populated_when_stuck(self):
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 1.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        assert d.needs_recovery
        assert "escape_maneuver" in d.candidate_actions
        assert "vertical_climb" in d.candidate_actions
        assert "lateral_sidestep" in d.candidate_actions


# ── Edge cases ──


class TestEdgeCases:
    def test_single_frame_no_crash(self):
        """Single frame should not crash."""
        rec = LocalRecovery()
        d = rec.update(1000.0, (1.0, 2.0, -1.0), (0.0, 0.0, 0.0))
        assert not d.needs_recovery
        assert d.window_size_frames == 1

    def test_history_window_prune(self):
        """Frames outside history_window_s are pruned."""
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=1.0,
            stuck_min_frames=5,
        ))
        t = 1000.0
        # Add frames over 5 seconds — only last 2s worth should survive
        for i in range(50):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 5.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        # history_window_s=2.0, 0.1s interval → ~21 frames max
        assert d.window_size_frames <= 25

    def test_stuck_window_smaller_than_history(self):
        """stuck_time_window_s=2.5 < history_window_s=4.0: stuck can still trigger
        because the 2.5s threshold is reachable within the 4.0s window."""
        rec = LocalRecovery(_params(
            history_window_s=4.0,
            stuck_time_window_s=2.5,
            stuck_position_epsilon_m=0.15,
            stuck_min_frames=10,
        ))
        t = 1000.0
        # 30 frames at 0.15s = 4.5s, history keeps ~4.0s = 27 frames
        for i in range(30):
            rec.update(t + i * 0.15, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 4.5, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        # 4.5s stationary with 2.5s threshold → stuck
        assert d.is_stuck

    def test_oscillation_uses_xy_only(self):
        """Lateral progress is XY-plane only, Z change does not prevent oscillation."""
        rec = LocalRecovery(_params(
            oscillation_time_window_s=0.5,
            oscillation_min_sign_flips=2,
            oscillation_lateral_epsilon_m=0.2,
        ))
        t = 1000.0
        vy_seq = [0.3, -0.3, 0.3, -0.3]
        for i, vy in enumerate(vy_seq):
            rec.update(
                timestamp=t + i * 0.15,
                position=(0.0, 0.0, -1.0 - float(i) * 0.5),
                velocity_body=(0.1, vy, -0.5),
            )
        d = rec.update(t + 0.6, (0.0, 0.0, -2.5), (0.1, 0.3, -0.5))
        assert d.is_oscillating
        assert d.oscillation_lateral_progress_m == pytest.approx(0.0, abs=0.01)

    def test_custom_params_accepted(self):
        """Custom RecoveryParams override defaults."""
        rec = LocalRecovery(_params(
            history_window_s=3.0,
            stuck_time_window_s=2.0,
            stuck_position_epsilon_m=0.05,
            stuck_min_frames=6,
        ))
        t = 1000.0
        for i in range(15):
            rec.update(t + i * 0.25, (0.01, 0.0, -1.0), (0.1, 0.0, 0.0))
        d = rec.update(t + 3.75, (0.01, 0.0, -1.0), (0.1, 0.0, 0.0))
        # history=3.0 keeps 12 frames, stuck_window=2.0 checks frame pair @ ~1.75s
        # duration = 3.75 - 1.75 = 2.0 >= 2.0 → stuck
        assert d.is_stuck


# ── APF independence (LocalRecovery does NOT modify APF) ──


class TestApfIndependence:
    def test_recovery_does_not_import_apf(self):
        """local_recovery does not depend on improved_potential_field."""
        import planners.local_recovery as lr
        source = lr.__file__
        if source is not None:
            with open(source, "r", encoding="utf-8") as fh:
                content = fh.read()
            assert "ImprovedPotentialField" not in content
            assert "ApfOutput" not in content

    def test_recovery_has_no_airsim_imports(self):
        """local_recovery has no AirSim API references (beyond docstring)."""
        import planners.local_recovery as lr
        source = lr.__file__
        if source is not None:
            with open(source, "r", encoding="utf-8") as fh:
                content = fh.read()
            body_start = content.find('\n"""')
            if body_start > 0:
                body_start = content.find('"""\n', body_start + 3)
                if body_start > 0:
                    body = content[body_start + 4:]
                else:
                    body = content
            else:
                body = content
            assert "moveByVelocity" not in body
            assert "enableApiControl" not in body

    def test_candidate_actions_are_strings_only(self):
        """Candidate actions are labels, not API calls."""
        rec = LocalRecovery(_params(
            history_window_s=2.0,
            stuck_time_window_s=0.5,
            stuck_position_epsilon_m=0.15,
        ))
        t = 1000.0
        for i in range(10):
            rec.update(t + i * 0.1, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        d = rec.update(t + 1.0, (0.0, 0.0, -1.0), (0.2, 0.0, 0.0))
        for action in d.candidate_actions:
            assert isinstance(action, str)
            assert "moveByVelocity" not in action
            assert "enableApiControl" not in action

"""Tests for bypass episode lifecycle, CBMBA planning bounds, and safety invariants.

Covers:
  A1-A5: Failure A — Bypass dead-end (bypass enter/enforce/release/veto, watchdog)
  B1-B8: Failure B — CBMBA path feasibility (planning bounds, validity gate)
  S1-S5: Safety invariants (emergency_distance, forward_sign_guard, committed_side)
"""

import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# A1-A5: Failure A — Bypass Dead-End
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassEpisodeLifecycle:
    """A1: BypassEpisode lifecycle — enter → enforce → release."""

    def test_bypass_episode_defaults(self):
        from flight_modes.automatic_mode import BypassEpisode
        ep = BypassEpisode()
        assert not ep.active
        assert ep.side is None
        assert ep.reason == ""

    def test_bypass_enter_activation(self):
        from flight_modes.automatic_mode import BypassEpisode
        ep = BypassEpisode(
            active=True, side=1, start_time=100.0,
            reason="enter(side=right)", min_duration_s=2.5,
        )
        assert ep.active
        assert ep.side == 1
        assert ep.reason == "enter(side=right)"

    def test_should_enter_bypass_both_sides_open(self):
        """Bypass should NOT enter when both sides have ample clearance."""
        from flight_modes.automatic_mode import AutomaticMode

        # Build a minimal AutomaticMode to access the bypass methods
        auto = _make_minimal_auto()
        should, reason = auto._should_enter_bypass(
            {"front": 10.0, "left": 10.0, "right": 10.0},
            (0.8, -0.5),  # guidance prefers left
        )
        assert not should
        assert "both_sides_open" in reason

    def test_should_enter_bypass_guidance_forward(self):
        """Bypass should NOT enter when guidance is forward-dominant."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        should, reason = auto._should_enter_bypass(
            {"front": 3.0, "left": 1.0, "right": 5.0},
            (0.98, 0.02),  # nearly pure forward
        )
        assert not should
        assert "guidance_forward_dominant" in reason

    def test_should_enter_bypass_constrained_with_lateral_guidance(self):
        """Bypass SHOULD enter when space is tight and guidance has lateral bias."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        should, reason = auto._should_enter_bypass(
            {"front": 3.0, "left": 1.5, "right": 1.0},
            (0.6, 0.5),  # rightward lateral component
        )
        assert should
        assert "enter" in reason

    def test_choose_bypass_side_follows_guidance(self):
        """Side choice follows CBMBA guidance when LiDAR confirms safety."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        # Right has more LiDAR clearance, but guidance says left
        side = auto._choose_bypass_side(
            {"left": 3.0, "right": 10.0},
            (0.5, -0.8),  # strong left guidance
        )
        assert side == -1  # guidance overrides LiDAR

    def test_enforce_bypass_side_right(self):
        """Vy is clamped to ≥0 when bypass_side=+1."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        # Positive vy passes through
        vx, vy = auto._enforce_bypass_side(0.15, 0.10, 1)
        assert vy >= 0
        # Negative vy is clamped to 0
        vx, vy = auto._enforce_bypass_side(0.15, -0.10, 1)
        assert vy == 0.0
        # vx is preserved
        assert vx == 0.15

    def test_enforce_bypass_side_left(self):
        """Vy is clamped to ≤0 when bypass_side=-1."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        # Negative vy passes through
        vx, vy = auto._enforce_bypass_side(0.15, -0.10, -1)
        assert vy <= 0
        # Positive vy is clamped to 0
        vx, vy = auto._enforce_bypass_side(0.15, 0.10, -1)
        assert vy == 0.0


class TestBypassVeto:
    """A2: Bypass veto on persistent unsafety."""

    def test_bypass_unsafe_timer_tracks(self):
        """Veto timer starts when chosen side becomes unsafe."""
        from flight_modes.automatic_mode import BypassEpisode, AutomaticMode

        auto = _make_minimal_auto()
        ep = BypassEpisode(
            active=True, side=1, start_time=100.0,
            reason="test", min_duration_s=2.5,
        )
        auto._bypass = ep
        auto._bypass_unsafe_start = None

        # Chosen side (right) is dangerously close
        should_rel, reason = auto._should_release_bypass(
            {"front": 5.0, "left": 5.0, "right": 0.5},
            100.5,
        )
        # Not yet released (unsafe timer just started)
        assert not should_rel
        assert auto._bypass_unsafe_start is not None

    def test_bypass_veto_fires_after_duration(self):
        """Veto fires after persistent unsafety exceeds threshold."""
        from flight_modes.automatic_mode import BypassEpisode, AutomaticMode

        auto = _make_minimal_auto()
        ep = BypassEpisode(
            active=True, side=1, start_time=100.0,
            reason="test", min_duration_s=2.5,
        )
        auto._bypass = ep
        auto._bypass_unsafe_start = 100.0

        # Persistent unsafety for 2.0s (veto threshold is 1.5s)
        should_rel, reason = auto._should_release_bypass(
            {"front": 5.0, "left": 5.0, "right": 0.5},
            102.0,
        )
        assert should_rel
        assert "veto" in reason
        assert "chosen_side_unsafe" in reason

    def test_bypass_release_both_sides_clear(self):
        """Normal release when both sides have adequate clearance."""
        from flight_modes.automatic_mode import BypassEpisode, AutomaticMode

        auto = _make_minimal_auto()
        ep = BypassEpisode(
            active=True, side=1, start_time=100.0,
            reason="test", min_duration_s=2.5,
        )
        auto._bypass = ep
        auto._bypass_unsafe_start = None

        # After min_duration, both sides clear
        should_rel, reason = auto._should_release_bypass(
            {"front": 8.0, "left": 5.0, "right": 5.0},
            103.0,
        )
        assert should_rel
        assert "both_sides_clear" in reason

    def test_bypass_release_obstacle_passed(self):
        """Obstacle passed → immediate release even before min_duration.

        The committed side opened up and the front cleared, but the OPPOSITE
        side is still blocked (the drone hugged one side of the corridor).
        This is the exact permanent-hold failure: release must fire on
        ``obstacle_passed`` without waiting for both sides.
        """
        from flight_modes.automatic_mode import BypassEpisode, AutomaticMode

        auto = _make_minimal_auto()
        ep = BypassEpisode(
            active=True, side=-1, start_time=100.0,
            reason="test", min_duration_s=2.5,
        )
        auto._bypass = ep
        auto._bypass_unsafe_start = None

        # elapsed = 0.5 < min_duration (2.5) — still releases immediately.
        should_rel, reason = auto._should_release_bypass(
            {"front": 4.0, "left": 5.0, "right": 1.0},
            100.5,
        )
        assert should_rel
        assert reason == "obstacle_passed"


class TestRejoinLifecycle:
    """A6: REJOIN exit logic (BYPASS → REJOIN → NORMAL).

    The exit rule is: elapsed >= min_duration_s AND cross-track path_error <
    exit_path_error_m, where path_error is measured against the FROZEN
    ``ep.reference_path_xy`` (never the live CBMBA path, whose first waypoint
    is re-seeded from the current UAV position each replan).  Goal heading
    error is diagnostic only and can never trigger an exit on its own.
    """

    @staticmethod
    def _mk_episode(start_time=100.0, reference=((0.0, 0.0), (42.0, 4.0))):
        from flight_modes.automatic_mode import RejoinEpisode
        return RejoinEpisode(
            active=True,
            start_time=start_time,
            reason="obstacle_passed",
            reference_path_xy=tuple(reference),
            reference_source="test",
        )

    def test_rejoin_exits_on_path_error_after_min_duration(self):
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode()

        st = MagicMock()
        st.position_ned_m = [0.0, 0.0, -2.0]  # exactly on the frozen reference
        st.yaw_rad = math.pi / 2.0  # heading off-goal, irrelevant now

        # elapsed = 1.0 >= 0.6 min_duration; path_error ~0 < 1.5 → exit.
        should_exit, reason = auto._should_exit_rejoin(
            st, (42.0, 4.0, -2.0), [], 101.0,
        )
        assert should_exit
        assert "path_error" in reason

    def test_rejoin_does_not_exit_on_goal_heading_alone(self):
        """Regression: a small heading error must NOT exit REJOIN.

        This is the exact bug — heading_err ≈ 0.22 caused 157/157 instant
        exits.  Now the exit needs cross-track path error, not yaw alignment.
        """
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode()

        st = MagicMock()
        st.position_ned_m = [0.0, -5.0, -2.0]  # 5 m off the frozen reference
        # Yaw points straight at the goal → heading error ~0 (would have
        # triggered the old exit).
        st.yaw_rad = math.atan2(9.0, 42.0)

        should_exit, reason = auto._should_exit_rejoin(
            st, (42.0, 4.0, -2.0), [], 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_rejoin_holds_during_min_duration(self):
        """Dwell: even if path_error is already small, exit must wait for
        min_duration_s (no same-frame enter→exit)."""
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode()

        st = MagicMock()
        st.position_ned_m = [0.0, 0.0, -2.0]  # on the reference
        st.yaw_rad = 0.0

        # elapsed = 0.3 < 0.6 min_duration → hold, despite path_error ~0.
        should_exit, reason = auto._should_exit_rejoin(
            st, (42.0, 4.0, -2.0), [], 100.3,
        )
        assert not should_exit
        assert "dwell" in reason

    def test_rejoin_holds_when_off_path_and_off_heading(self):
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode()

        st = MagicMock()
        st.position_ned_m = [0.0, -5.0, -2.0]  # 5 m off the reference
        st.yaw_rad = math.pi  # facing away from the goal

        should_exit, reason = auto._should_exit_rejoin(
            st, (42.0, 4.0, -2.0), [], 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_rejoin_path_error_uses_segment_distance(self):
        """A point on a sparse segment between two far waypoints reports ~0
        cross-track error (not min-waypoint distance, which would be ~21 m)."""
        from flight_modes.automatic_mode import AutomaticMode

        auto = _make_minimal_auto()
        # Midpoint of (0,0)→(42,4) lies ON the segment but ~21 m from either
        # endpoint.  Segment distance must be ~0.
        path_err = auto._rejoin_path_error(
            (21.0, 2.0), [[0.0, 0.0, -2.0], [42.0, 4.0, -2.0]],
        )
        assert path_err < 0.01

        # Empty path → inf (never exits).
        assert auto._rejoin_path_error((0.0, 0.0), None) == float("inf")

    # ── NEW: self-reference regression tests ──

    def test_rejoin_ignores_live_replanned_path_starting_at_current_position(self):
        """Regression: a live path re-seeded at the current position must NOT
        cause a false REJOIN exit.  Only the frozen reference counts."""
        auto = _make_minimal_auto()
        # Frozen reference: the y=0 corridor the UAV must actually return to.
        auto._rejoin = self._mk_episode(reference=((0.0, 0.0), (20.0, 0.0)))

        st = MagicMock()
        st.position_ned_m = [5.0, 5.0, -2.0]  # 5 m off the y=0 reference
        st.yaw_rad = 0.0
        # Live replanned path starts AT the current position → distance 0.
        live_path = [[5.0, 5.0, -2.0], [6.0, 5.0, -2.0], [20.0, 5.0, -2.0]]

        # elapsed 1.0 >= 0.6, but frozen reference cross-track = 5.0 > 1.5 → hold.
        should_exit, reason = auto._should_exit_rejoin(
            st, (20.0, 0.0, -2.0), live_path, 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_rejoin_reference_stable_across_replans(self):
        """Fresh replans (each anchored to a new current position) must not
        change the REJOIN exit decision — only the frozen reference matters."""
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode(reference=((0.0, 0.0), (20.0, 0.0)))

        # UAV sits ~5 m off the reference; each replan anchors to a NEW position.
        for idx, (px, py) in enumerate([(5.0, 5.0), (5.2, 5.1), (5.4, 5.2)]):
            st = MagicMock()
            st.position_ned_m = [px, py, -2.0]
            st.yaw_rad = 0.0
            live_path = [[px, py, -2.0], [px + 1.0, py, -2.0], [20.0, py, -2.0]]
            should_exit, reason = auto._should_exit_rejoin(
                st, (20.0, 0.0, -2.0), live_path, 101.0,
            )
            assert not should_exit, f"replan #{idx} must not exit"
            assert reason == "hold"

    def test_rejoin_holds_when_reference_empty(self):
        """Empty frozen reference → path_error inf → never exit (no false
        exit just because there is no reference)."""
        auto = _make_minimal_auto()
        auto._rejoin = self._mk_episode(reference=())

        st = MagicMock()
        st.position_ned_m = [0.0, 0.0, -2.0]
        st.yaw_rad = 0.0
        live_path = [[0.0, 0.0, -2.0], [20.0, 0.0, -2.0]]

        should_exit, reason = auto._should_exit_rejoin(
            st, (20.0, 0.0, -2.0), live_path, 101.0,
        )
        assert not should_exit
        assert reason == "hold"
        assert auto._rejoin_path_error((0.0, 0.0), ()) == float("inf")

    # ── NEW: BYPASS-side reference freeze (Round 4) ──
    # The reference is now frozen at bypass_enter (before the drone deviates)
    # and INHERITED by REJOIN — never re-frozen from the live path.

    def test_bypass_freezes_reference_before_rejoin(self):
        """Reference is frozen at BYPASS entry, so its first waypoint is the
        bypass-entry position — NOT the (later) rejoin position.  This is what
        kills the one-shot self-reference that made path_error ~0."""
        auto = _make_minimal_auto()
        auto._cbmba_path_generation = 7
        path_world = [[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [42.0, 4.0, -2.0]]
        ref_xy, source, gen, first = auto._freeze_reference_xy(
            (0.0, 0.0), path_world,
        )
        assert ref_xy[0] == (0.0, 0.0)
        assert source == "cbmba_path_world"
        assert gen == 7
        assert first == (0.0, 0.0)
        # The drone later rejoins elsewhere — the frozen first point is NOT the
        # rejoin position, so path_error stays meaningful (not ~0).
        assert (ref_xy[0][0], ref_xy[0][1]) != (5.0, 5.0)

    def test_bypass_reference_stable_across_replans(self):
        """Fresh replans (each re-anchored to a new current position) must not
        mutate the BYPASS episode's frozen reference."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._cbmba_path_generation = 1
        frozen_ref, _, _, _ = auto._freeze_reference_xy(
            (0.0, 0.0), [[0.0, 0.0, -2.0], [20.0, 0.0, -2.0]],
        )
        ep = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=frozen_ref,
            reference_source="cbmba_path_world",
            reference_generation_id=1,
            reference_first_xy=frozen_ref[0],
            reference_frozen_position_xy=(0.0, 0.0),
        )
        for px, py in [(5.0, 5.0), (5.2, 5.1), (5.4, 5.2)]:
            auto._cbmba_path_generation += 1
            new_ref, _, _, _ = auto._freeze_reference_xy(
                (px, py),
                [[px, py, -2.0], [px + 1.0, py, -2.0], [20.0, py, -2.0]],
            )
            # The live path moved …
            assert new_ref != frozen_ref
            # … but the frozen BYPASS reference did not.
            assert ep.reference_path_xy == frozen_ref

    def test_rejoin_inherits_exact_bypass_reference(self):
        """BYPASS → REJOIN handoff copies the SAME frozen snapshot (identity,
        not re-derived), and computes start_path_error against it."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        ref = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=ref,
            reference_source="cbmba_path_world",
            reference_generation_id=3,
            reference_first_xy=(0.0, 0.0),
            reference_frozen_position_xy=(0.0, 0.0),
        )
        rejoin = auto._build_rejoin_from_bypass((5.0, 5.0), 105.0, "obstacle_passed")
        assert rejoin.reference_path_xy == ref  # EXACT same snapshot
        assert rejoin.reference_source == "cbmba_path_world"
        assert rejoin.reference_generation_id == 3
        assert rejoin.reference_first_xy == (0.0, 0.0)
        assert rejoin.reference_frozen_position_xy == (0.0, 0.0)
        # (5,5) is 5 m off the y=0 reference corridor.
        assert rejoin.start_path_error == pytest.approx(5.0, abs=1e-6)

    def test_rejoin_live_path_cannot_exit(self):
        """After inheriting the bypass reference, a live path anchored at the
        current position must NOT cause an exit — only the frozen reference."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=((0.0, 0.0), (20.0, 0.0)),
            reference_source="cbmba_path_world",
            reference_generation_id=1,
            reference_first_xy=(0.0, 0.0),
            reference_frozen_position_xy=(0.0, 0.0),
        )
        auto._rejoin = auto._build_rejoin_from_bypass((5.0, 5.0), 100.0, "obstacle_passed")

        st = MagicMock()
        st.position_ned_m = [5.0, 5.0, -2.0]  # 5 m off the y=0 reference
        st.yaw_rad = 0.0
        live = [[5.0, 5.0, -2.0], [20.0, 5.0, -2.0]]  # anchored at current pos
        should_exit, reason = auto._should_exit_rejoin(
            st, (20.0, 0.0, -2.0), live, 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_rejoin_does_not_fallback_to_live_path_when_bypass_reference_empty(self):
        """Empty bypass reference is preserved (path_error inf), NOT silently
        replaced by the live path."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            # reference fields left empty — no valid path was ever frozen.
        )
        rejoin = auto._build_rejoin_from_bypass((5.0, 5.0), 105.0, "obstacle_passed")
        assert rejoin.reference_path_xy == ()
        assert rejoin.reference_source == "none"
        assert rejoin.reference_first_xy is None
        assert rejoin.reference_frozen_position_xy is None
        assert rejoin.start_path_error == float("inf")
        auto._rejoin = rejoin

        st = MagicMock()
        st.position_ned_m = [5.0, 5.0, -2.0]
        st.yaw_rad = 0.0
        live = [[5.0, 5.0, -2.0], [20.0, 5.0, -2.0]]  # available but MUST be ignored
        should_exit, reason = auto._should_exit_rejoin(
            st, (20.0, 0.0, -2.0), live, 106.0,
        )
        assert not should_exit
        assert reason == "hold"

    # ── Round-4 audit additions: deep immutability + polyline distance ──

    def test_freeze_snapshot_is_deep_immutable(self):
        """Mutating the original path_world lists after freezing must NOT change
        the frozen snapshot (it is a tuple of NEW tuples, not a view)."""
        auto = _make_minimal_auto()
        auto._cbmba_path_generation = 5
        path_world = [[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [42.0, 4.0, -2.0]]
        ref_xy, _, _, _ = auto._freeze_reference_xy((0.0, 0.0), path_world)
        # Mutate the ORIGINAL lists in place (as a planner might on a replan).
        path_world[0][0] = 99.0
        path_world[0][1] = -99.0
        path_world[1][0] = 88.0
        # The frozen snapshot is unchanged.
        assert ref_xy[0] == (0.0, 0.0)
        assert ref_xy[1] == (1.0, 0.0)
        assert ref_xy[2] == (42.0, 4.0)

    def test_rejoin_start_error_uses_whole_polyline_not_first_point(self):
        """start_path_error is distance-to-PATH (min over segments), NOT distance
        to reference[0].  A drone near a LATER segment but far from the first
        waypoint must get a small start_path_error."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0), (200.0, 0.0)),
            reference_source="cbmba_path_world",
            reference_generation_id=1,
            reference_first_xy=(0.0, 0.0),
            reference_frozen_position_xy=(0.0, 0.0),
        )
        # Drone at (150, 1.0): ~150 m from reference[0], but only ~1 m off the
        # (100,0)->(200,0) segment.
        rejoin = auto._build_rejoin_from_bypass((150.0, 1.0), 105.0, "obstacle_passed")
        assert rejoin.start_path_error == pytest.approx(1.0, abs=1e-6)


class TestForwardProgressWatchdog:
    """A3: Forward-progress watchdog fires when stalled."""

    def test_watchdog_defaults(self):
        from flight_modes.automatic_mode import ForwardProgressWatchdog
        wd = ForwardProgressWatchdog()
        assert wd.window_s == 8.0
        assert wd.min_progress_m == 1.0
        assert not wd._fired

    def test_watchdog_not_fired_with_progress(self):
        from flight_modes.automatic_mode import ForwardProgressWatchdog
        wd = ForwardProgressWatchdog(window_s=4.0, min_progress_m=1.0, check_interval_s=0.0)
        wd.reset(100.0, (0.0, 0.0))
        # Moved 2m in 4s — adequate progress
        fired = wd.update(104.0, (2.0, 0.0))
        assert not fired

    def test_watchdog_fires_without_progress(self):
        from flight_modes.automatic_mode import ForwardProgressWatchdog
        wd = ForwardProgressWatchdog(window_s=4.0, min_progress_m=1.0, check_interval_s=0.0)
        wd.reset(100.0, (0.0, 0.0))
        # Moved only 0.3m in 4s — insufficient progress
        fired = wd.update(104.0, (0.3, 0.0))
        assert fired
        assert wd._fired_count == 1

    def test_watchdog_resets_after_firing(self):
        from flight_modes.automatic_mode import ForwardProgressWatchdog
        wd = ForwardProgressWatchdog(window_s=4.0, min_progress_m=1.0, check_interval_s=0.0)
        wd.reset(100.0, (0.0, 0.0))
        # First check: fires (only 0.3m progress in 4s)
        assert wd.update(104.0, (0.3, 0.0))
        assert wd._fired_count == 1
        # After firing, baseline resets to (0.3, 0.0) at t=104
        # Next check at t=108 (4s later): moved to (1.5, 0.0), progress=1.2m ≥ 1.0m → OK
        fired2 = wd.update(108.0, (1.5, 0.0))
        assert not fired2  # adequate progress from new baseline

    def test_watchdog_early_exit_before_window(self):
        from flight_modes.automatic_mode import ForwardProgressWatchdog
        wd = ForwardProgressWatchdog(window_s=8.0, min_progress_m=1.0, check_interval_s=0.0)
        wd.reset(100.0, (0.0, 0.0))
        # Only 4s elapsed — window not yet complete
        fired = wd.update(104.0, (0.0, 0.0))
        assert not fired  # too early


class TestRecoveryBypassInheritance:
    """A4: Recovery inherits bypass side."""

    def test_recovery_inherits_bypass_side(self):
        from planners.recovery_commander import RecoveryStateMachine, RecoveryCommanderParams
        from planners.local_recovery import RecoveryDecision

        sm = RecoveryStateMachine(RecoveryCommanderParams(
            reverse_speed=0.35, lateral_speed=0.35,
            max_duration_s=1.0, cooldown_s=2.5,
        ))
        d = RecoveryDecision(is_stuck=False, is_oscillating=True,
                             needs_recovery=True, reason="oscillation")

        # With active bypass_side=+1 (right), LiDAR says left is more open
        result = sm.tick(
            1000.0, d, {"front": 5.0, "left": 10.0, "right": 2.0},
            bypass_side=1,
        )
        assert result.should_override
        # Should follow bypass_side (+1 → vy ≥ 0) not LiDAR (which would go left)
        assert result.vy_body >= 0
        assert result.committed_side == 1


class TestPathValidityGate:
    """A5: Path validity gate rejects out-of-bounds paths."""

    def test_valid_path_passes(self):
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=5.0)
        pl = CbmbaAStarPlanner(p)
        path = [[0, 0, 0], [2, 1, 0], [10, 0, 0]]  # max dev = 0.83m < 5m
        assert pl.is_path_in_bounds(path)

    def test_out_of_bounds_path_rejected(self):
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=5.0)
        pl = CbmbaAStarPlanner(p)
        # Must set corridor before is_path_in_bounds uses it
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [10.0, 0.0, 0.0]
        # waypoint at (5, 15) is 15m off the start→goal axis
        path = [[0, 0, 0], [5, 15, 0], [10, 0, 0]]
        assert not pl.is_path_in_bounds(path)

    def test_path_blocked_by_obstacle(self):
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(inflation_radius=1.5)
        pl = CbmbaAStarPlanner(p)
        path = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]
        obstacles = [{
            "position": [5.0, 0.5, 0.0],
            "size": 0.5,  # half-extent = 0.5 + inflation 1.5 = 2.0 > 0.5 → blocked
        }]
        assert pl._is_path_blocked(obstacles, path, 1.5)


# ═══════════════════════════════════════════════════════════════════════════
# B1-B8: Failure B — CBMBA Path Feasibility
# ═══════════════════════════════════════════════════════════════════════════


class TestPlanningBounds:
    """B1-B4: CBMBA planning bounds enforcement."""

    def test_bounds_block_cells_outside_corridor(self):
        """B1: Cells beyond planning_bounds_xy_m are treated as blocked."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=3.0, resolution=1.0)
        pl = CbmbaAStarPlanner(p)
        # Set corridor along X axis: (0,0)→(20,0)
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [20.0, 0.0, 0.0]
        pl.last_origin = [0.0, 0.0, 0.0]

        from planners.cbmba_astar import _Cell
        # Cell at (5, 2, 0): 2m off axis → < 3m → in bounds
        assert pl._cell_in_corridor(_Cell(5, 2, 0), pl.last_origin, 1.0)
        # Cell at (5, 5, 0): 5m off axis → > 3m → out of bounds
        assert not pl._cell_in_corridor(_Cell(5, 5, 0), pl.last_origin, 1.0)

    def test_plan_with_result_respects_bounds(self):
        """B2: A* search does not escape the planning corridor."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        # Tight bounds — path must stay within 2m of the axis
        p = CbmbaParams(
            planning_bounds_xy_m=2.0,
            resolution=1.0,
            max_search_nodes=500,
            inflation_radius=0.5,
        )
        pl = CbmbaAStarPlanner(p)
        # No obstacles — should produce a straight line
        result = pl.plan_with_result([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert result.success
        # All waypoints should be within bounds
        assert pl.is_path_in_bounds(result.path_world)

    def test_is_path_in_bounds_true_for_valid_path(self):
        """B3: is_path_in_bounds returns True for in-bounds paths."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=10.0)
        pl = CbmbaAStarPlanner(p)
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [10.0, 0.0, 0.0]
        path = [[0, 0, 0], [3, 2, 1], [7, -1, 0], [10, 0, 0]]
        assert pl.is_path_in_bounds(path)

    def test_path_max_lateral_deviation_correct(self):
        """B4: path_max_lateral_deviation returns the maximum deviation."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=10.0)
        pl = CbmbaAStarPlanner(p)
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [10.0, 0.0, 0.0]
        # Waypoint (5, 4, 0) is 4m off axis; (5, -3, 0) is 3m off
        path = [[0, 0, 0], [5, 4, 0], [5, -3, 0], [10, 0, 0]]
        max_dev = pl.path_max_lateral_deviation(path)
        assert max_dev == pytest.approx(4.0)


class TestPlanningBoundsCornerCases:
    """B5-B8: CBMBA bounds corner cases."""

    def test_bounds_disabled_when_zero(self):
        """B7: planning_bounds_xy_m=0 disables bounds checking."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams, _Cell
        p = CbmbaParams(planning_bounds_xy_m=0)
        pl = CbmbaAStarPlanner(p)
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [10.0, 0.0, 0.0]
        pl.last_origin = [0.0, 0.0, 0.0]
        # Any cell should pass when bounds disabled
        assert pl._cell_in_corridor(_Cell(100, 100, 0), pl.last_origin, 1.0)
        assert pl.is_path_in_bounds([[0, 0, 0], [5, 100, 0], [10, 0, 0]])

    def test_degenerate_corridor_handled(self):
        """B6: Degenerate corridor (start≈goal) handled gracefully."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=5.0)
        pl = CbmbaAStarPlanner(p)
        pl._corridor_start = [0.0, 0.0, 0.0]
        pl._corridor_end = [0.0, 0.0, 1.0]  # same XY
        # Should not crash
        dev = pl.path_max_lateral_deviation([[0, 5, 0], [0, 0, 1]])
        assert dev >= 0  # returns some value, doesn't crash

    def test_no_corridor_set_returns_true(self):
        """B8: is_path_in_bounds returns True when no corridor is set."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(planning_bounds_xy_m=5.0)
        pl = CbmbaAStarPlanner(p)
        # No corridor set — should pass
        assert pl._corridor_start is None
        assert pl.is_path_in_bounds([[0, 0, 0], [5, 100, 0]])

    def test_plan_with_obstacles_stays_in_bounds(self):
        """B8 (cont): Even with obstacles, path respects bounds."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(
            planning_bounds_xy_m=3.0,
            resolution=1.0,
            max_search_nodes=500,
            inflation_radius=0.5,
        )
        pl = CbmbaAStarPlanner(p)
        # Place obstacle near the direct path, but within bounds
        obstacles = [{
            "position": [5.0, 1.0, 0.0],
            "footprint_half_extents": [0.5, 0.5, 0.5],
        }]
        result = pl.plan_with_result(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        if result.success:
            assert pl.is_path_in_bounds(result.path_world)


# ═══════════════════════════════════════════════════════════════════════════
# S1-S5: Safety Invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetyInvariants:
    """Safety mechanisms must NOT be weakened by any new code."""

    def test_emergency_distance_still_terminates(self):
        """S1: emergency_distance still triggers termination."""
        from flight_modes.automatic_mode import choose_reactive_command
        dec = choose_reactive_command(
            front_m=10.0, left_m=10.0, right_m=10.0,
            minimum_distance_m=0.5,  # below default 0.8
            config={"emergency_distance_m": 0.8, "front_threshold_m": 2.5,
                    "forward_speed_mps": 0.2, "side_speed_mps": 0.15},
        )
        assert dec.should_terminate
        assert dec.termination_reason == "emergency_distance"

    def test_emergency_distance_not_triggered_above_threshold(self):
        """S1: emergency_distance does NOT trigger when minD ≥ threshold."""
        from flight_modes.automatic_mode import choose_reactive_command
        dec = choose_reactive_command(
            front_m=10.0, left_m=10.0, right_m=10.0,
            minimum_distance_m=1.0,  # above default 0.8
            config={"emergency_distance_m": 0.8, "front_threshold_m": 2.5,
                    "forward_speed_mps": 0.2, "side_speed_mps": 0.15},
        )
        assert not dec.should_terminate
        assert dec.vx_body_mps == 0.2  # forward (front > threshold)

    def test_recovery_committed_side_persists(self):
        """S3: RecoveryStateMachine committed_side persists across active ticks."""
        from planners.recovery_commander import RecoveryStateMachine, RecoveryCommanderParams
        from planners.local_recovery import RecoveryDecision

        sm = RecoveryStateMachine(RecoveryCommanderParams(
            reverse_speed=0.35, lateral_speed=0.35,
            max_duration_s=1.0, cooldown_s=2.5,
        ))
        d = RecoveryDecision(is_stuck=False, is_oscillating=True,
                             needs_recovery=True, reason="oscillation")

        # Enter: LiDAR says left is more open
        r1 = sm.tick(1000.0, d, {"front": 5.0, "left": 10.0, "right": 2.0})
        assert r1.event == "enter"
        assert r1.committed_side == -1  # left

        # Active: LiDAR flipped — right now more open
        r2 = sm.tick(1000.5, d, {"front": 5.0, "left": 2.0, "right": 10.0})
        assert r2.event == "active"
        # Committed side persists (left still has enough clearance)
        # left=2.0 ≥ min_clearance=1.5 → persists
        assert r2.committed_side == -1

    def test_bypass_enforce_preserves_vx_sign(self):
        """S4: bypass_enforce does NOT modify vx (forward direction)."""
        from flight_modes.automatic_mode import AutomaticMode
        auto = _make_minimal_auto()
        # Positive vx stays positive
        vx, vy = auto._enforce_bypass_side(0.15, -0.10, 1)
        assert vx == 0.15  # vx preserved
        # Negative vx stays negative
        vx, vy = auto._enforce_bypass_side(-0.12, 0.10, -1)
        assert vx == -0.12  # vx preserved

    def test_cooldown_prevents_reentry(self):
        """S5: Cooldown prevents re-entry during bypass."""
        from planners.recovery_commander import RecoveryStateMachine, RecoveryCommanderParams
        from planners.local_recovery import RecoveryDecision

        sm = RecoveryStateMachine(RecoveryCommanderParams(
            reverse_speed=0.35, lateral_speed=0.35,
            max_duration_s=1.0, cooldown_s=2.5,
        ))
        d = RecoveryDecision(is_stuck=True, needs_recovery=True, reason="stuck")

        # Enter recovery
        sm.tick(1000.0, d, {"front": 3.0, "left": 8.0, "right": 3.0})
        # Timeout → cooldown
        sm.tick(1001.0, d, {"front": 3.0, "left": 8.0, "right": 3.0})
        # During cooldown, re-entry is blocked
        result = sm.tick(1001.5, d, {"front": 3.0, "left": 8.0, "right": 3.0})
        assert not result.should_override
        assert result.state == "RECOVERY_COOLDOWN"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_minimal_auto():
    """Build a minimal AutomaticMode for testing bypass methods."""
    from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams

    session = MagicMock()
    session.client = MagicMock()
    session.adapter = MagicMock()
    session.vehicle_name = "Drone1"

    return AutomaticMode(
        session,
        params=AutomaticModeParams(
            target_z_ned=-2.0,
            max_flight_duration_s=0.2,
        ),
        cli_overrides={"planner_mode": "reactive"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Integration test: bypass + bounds together
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationBypassAndBounds:
    """Verify that bypass and planning bounds work together correctly."""

    def test_bounds_prevent_large_lateral_excursion(self):
        """CBMBA path with tight bounds cannot make large lateral detours."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        p = CbmbaParams(
            planning_bounds_xy_m=2.0,
            resolution=1.0,
            max_search_nodes=500,
            inflation_radius=0.5,
        )
        pl = CbmbaAStarPlanner(p)
        result = pl.plan_with_result([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert result.max_lateral_deviation_m <= 2.0 + p.resolution  # tolerance for grid discretization

    def test_bypass_side_label_safe_with_magic_mock(self):
        """_side_label handles MagicMock and non-int values safely."""
        from flight_modes.automatic_mode import AutomaticMode
        auto = _make_minimal_auto()
        assert auto._side_label(1) == "right"
        assert auto._side_label(-1) == "left"
        assert auto._side_label(0) == "hold"
        assert auto._side_label(None) == "unknown"
        assert auto._side_label(MagicMock()) == "unknown"

    def test_compute_recovery_command_returns_4_tuple(self):
        """New command format returns (vx, vy, vz, committed_side)."""
        from planners.recovery_commander import compute_recovery_command
        from planners.local_recovery import RecoveryDecision

        d = RecoveryDecision(is_stuck=True, needs_recovery=True,
                             reason="stuck")
        cmd = compute_recovery_command(d, {"front": 10.0, "left": 8.0, "right": 3.0})
        assert len(cmd) == 4
        assert cmd[2] == 0.0  # vz always 0
        assert cmd[3] in (-1, 0, 1)  # committed_side is valid


class TestRound7StateSemantics:
    """Deterministic state-semantics fixes (Round 7).

    P0-A: stale hover must not feed a false stuck / progress-watchdog fire.
    P1-B: recovery→bypass inheritance is gated by the formal entry gate.
    P1-C: a bypass that never deviated goes straight to NORMAL (no REJOIN).
    P1-D: REJOIN exit requires CONVERGENCE, not merely "inside the corridor".
    """

    def test_stale_hold_resets_stuck_and_progress_accumulators(self):
        """P0-A: an intentional stale hover re-anchors both the stuck detector
        and the forward-progress watchdog, so the hold period cannot later read
        as "stuck" / "no progress" once perception recovers."""
        auto = _make_minimal_auto()
        # Pre-fill the stuck detector with stationary frames (real history that
        # a hover gap would otherwise let span into a false "stuck").
        for i in range(30):
            auto._recovery.update(
                100.0 + i * 0.1, (0.0, 0.0, -2.0), (0.0, 0.0, 0.0), 0.0,
            )
        assert len(auto._recovery._window) > 10

        auto._reset_stale_hold_accumulators(101.0, (1.5, 2.5))

        # Stuck detector window is cleared — a fresh stationary frame is NOT
        # treated as a continuation of pre-hover history.
        assert len(auto._recovery._window) == 0
        d = auto._recovery.update(101.05, (1.5, 2.5, -2.0), (0.0, 0.0, 0.0), 0.0)
        assert not d.needs_recovery
        assert d.window_size_frames == 1
        # Progress watchdog baseline re-anchored to the hold position/time.
        assert auto._progress_watchdog._start_time == 101.0
        assert auto._progress_watchdog._start_position == (1.5, 2.5)

    def test_rejoin_does_not_exit_on_increasing_error(self):
        """P1-D: REJOIN must NOT exit when path_error is still INCREASING, even
        when it is below the exit threshold (start=0.011 → now=0.023 < 1.5)."""
        from flight_modes.automatic_mode import RejoinEpisode

        auto = _make_minimal_auto()
        auto._rejoin = RejoinEpisode(
            active=True, start_time=100.0, reason="obstacle_passed",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0)),
            reference_source="test",
            start_path_error=0.011,
        )
        st = MagicMock()
        st.position_ned_m = [50.0, 0.023, -2.0]  # 0.023 m off the y=0 reference
        st.yaw_rad = 0.0

        should_exit, reason = auto._should_exit_rejoin(
            st, (100.0, 0.0, -2.0), [], 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_rejoin_exits_on_converging_error(self):
        """P1-D positive control: a REJOIN whose error decreased below both the
        threshold and its entry value exits to NORMAL."""
        from flight_modes.automatic_mode import RejoinEpisode

        auto = _make_minimal_auto()
        auto._rejoin = RejoinEpisode(
            active=True, start_time=100.0, reason="obstacle_passed",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0)),
            reference_source="test",
            start_path_error=2.0,
        )
        st = MagicMock()
        st.position_ned_m = [50.0, 0.5, -2.0]  # 0.5 m off reference (< 2.0 start)
        st.yaw_rad = 0.0

        should_exit, reason = auto._should_exit_rejoin(
            st, (100.0, 0.0, -2.0), [], 101.0,
        )
        assert should_exit
        assert "path_error" in reason

    def test_rejoin_holds_when_below_threshold_but_diverging(self):
        """P1-D: path_error=1.0 < 1.5 threshold, but start=0.5 → error INCREASED.
        Must hold, not exit."""
        from flight_modes.automatic_mode import RejoinEpisode

        auto = _make_minimal_auto()
        auto._rejoin = RejoinEpisode(
            active=True, start_time=100.0, reason="obstacle_passed",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0)),
            reference_source="test",
            start_path_error=0.5,
        )
        st = MagicMock()
        st.position_ned_m = [50.0, 1.0, -2.0]  # 1.0 m off reference (> 0.5 start)
        st.yaw_rad = 0.0

        should_exit, reason = auto._should_exit_rejoin(
            st, (100.0, 0.0, -2.0), [], 101.0,
        )
        assert not should_exit
        assert reason == "hold"

    def test_bypass_excursion_tracks_peak_path_error(self):
        """P1-C: `_track_bypass_excursion` records the peak cross-track error
        against the frozen reference across the bypass episode."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0)),
            reference_source="test",
        )
        auto._track_bypass_excursion((50.0, 0.5))   # 0.5 m excursion
        auto._track_bypass_excursion((50.0, 2.0))   # 2.0 m excursion (peak)
        auto._track_bypass_excursion((50.0, 0.3))   # back down — peak unchanged
        assert auto._bypass.max_path_error_m == pytest.approx(2.0, abs=1e-6)

    def test_bypass_release_destination_requires_excursion(self):
        """P1-C: a bypass that never left the corridor goes straight to NORMAL;
        a bypass that actually deviated hands off to REJOIN."""
        from flight_modes.automatic_mode import BypassEpisode

        auto = _make_minimal_auto()
        auto._bypass = BypassEpisode(
            active=True, side=1, start_time=100.0, reason="test",
            reference_path_xy=((0.0, 0.0), (100.0, 0.0)),
        )
        auto._bypass.max_path_error_m = 0.05
        assert auto._bypass_release_destination("obstacle_passed") == "normal"

        auto._bypass.max_path_error_m = 2.0
        assert auto._bypass_release_destination("obstacle_passed") == "rejoin"

        # Non-obstacle releases never go to REJOIN.
        auto._bypass.max_path_error_m = 5.0
        assert auto._bypass_release_destination("both_sides_clear") == "normal"

    def test_recovery_inherit_gated_by_formal_entry(self):
        """P1-B: recovery→bypass side inheritance is blocked when there is no
        real corridor constraint (both sides open) or no guidance."""
        auto = _make_minimal_auto()
        auto._guided_apf_control = True

        # Both sides open → no inheritance.
        ok, reason = auto._inheritance_formal_entry(
            {"front": 10.0, "left": 10.0, "right": 10.0}, (0.8, -0.5),
        )
        assert not ok
        assert "both_sides_open" in reason

        # No guidance → no inheritance.
        ok, reason = auto._inheritance_formal_entry(
            {"front": 3.0, "left": 1.0, "right": 1.0}, None,
        )
        assert not ok
        assert reason == "guided_apf_unavailable"

        # A real corridor constraint → inheritance allowed.
        ok, reason = auto._inheritance_formal_entry(
            {"front": 3.0, "left": 1.5, "right": 1.0}, (0.6, 0.5),
        )
        assert ok

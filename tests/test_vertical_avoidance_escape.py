"""Tests for the persistent vertical-escape state machine.

Exercises ``AutomaticMode._update_vertical_avoidance`` directly (no AirSim
client).  The state machine lives on ``AutomaticMode`` because it reads the
episode across loop iterations and drives the recovery suppression / altitude
relax; the pure selector is covered separately in ``test_vertical_avoidance.py``.

NED convention: Z is positive downward, so climbing = more-negative Z and the
altitude-hold velocity toward a higher target is negative.
"""

import types
from unittest.mock import MagicMock

import pytest

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.automatic_mode import AutomaticMode, VerticalAvoidEpisode
from planners.vertical_avoidance import (
    VerticalAvoidanceParams,
    VerticalAvoidanceSelector,
)


def _rays(**overrides):
    base = {
        "front": 10.0,
        "frontLeft": 10.0,
        "frontRight": 10.0,
        "back": 10.0,
        "backLeft": 10.0,
        "backRight": 10.0,
        "left": 10.0,
        "right": 10.0,
        "up": 10.0,
        "down": 10.0,
        "frontUp": 10.0,
        "frontDown": 10.0,
        "leftUp": 10.0,
        "rightUp": 10.0,
        "leftDown": 10.0,
        "rightDown": 10.0,
    }
    base.update(overrides)
    return base


def _mode(target_z=-10.0, **param_overrides):
    mode = AutomaticMode.__new__(AutomaticMode)
    params = VerticalAvoidanceParams(**param_overrides)
    mode._va_params = params
    mode._va_selector = VerticalAvoidanceSelector(params)
    mode._va_episode = VerticalAvoidEpisode()
    mode._params = types.SimpleNamespace(target_z_ned=target_z)
    mode._recovery = types.SimpleNamespace(reset=MagicMock())
    mode._traj_force_replan = False
    mode._log_throttled = lambda *a, **k: None
    return mode


def _climbing_mode(start_z=-10.0, target_z=-10.0, **param_overrides):
    """A mode already in CLIMBING, with a known climb-entry altitude."""
    mode = _mode(target_z=target_z, **param_overrides)
    mode._va_episode = VerticalAvoidEpisode(
        state="CLIMBING", start_time=0.0, start_z=start_z,
        target_z=-10.6, reason="test",
    )
    return mode


_CLIMB_RAYS = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                    left=1.0, right=1.5)


# ── entry ──


def test_climb_entry_sets_state_and_resets_recovery():
    mode = _mode()
    mode._update_vertical_avoidance(_CLIMB_RAYS, [0.0, 0.0, -10.0])
    assert mode._va_episode.state == "CLIMBING"
    assert mode._va_episode.active is True
    assert mode._va_episode.start_z == -10.0
    # The XY stuck window is cleared so a vertical climb (XY ≈ 0) is never
    # mistaken for being stuck.
    mode._recovery.reset.assert_called_once()


def test_ned_climb_target_is_upward_and_bounded():
    mode = _mode()
    mode._update_vertical_avoidance(_CLIMB_RAYS, [0.0, 0.0, -10.0])
    assert mode._va_episode.state == "CLIMBING"
    # Climbing = more negative NED z (up), bounded by target_z - max_climb_m.
    assert mode._va_episode.target_z < -10.0
    assert mode._va_episode.target_z >= -12.0


# ── exit gating (no single-frame LiDAR flicker) ──


def test_leaf_flicker_does_not_exit():
    mode = _climbing_mode()
    clear = _rays(front=5.0, up=5.0, frontUp=5.0, leftUp=5.0, rightUp=5.0)
    blocked = _rays(front=1.0, up=5.0, frontUp=5.0, leftUp=5.0, rightUp=5.0)
    # climbed 0.9 m (> 0.8 min) and front clears for exactly one frame.
    mode._update_vertical_avoidance(clear, [0.0, 0.0, -10.9])
    assert mode._va_episode.state == "CLIMBING"
    assert mode._va_episode.clear_frames == 1
    # front blocks again → the consecutive-clear counter resets, no exit.
    mode._update_vertical_avoidance(blocked, [0.0, 0.0, -10.9])
    assert mode._va_episode.state == "CLIMBING"
    assert mode._va_episode.clear_frames == 0


def test_below_min_height_stays_climbing_even_if_clear():
    mode = _climbing_mode()
    clear = _rays(front=5.0, up=5.0, frontUp=5.0, leftUp=5.0, rightUp=5.0)
    # climbed only 0.5 m (< 0.8 min): front clearance is ignored entirely.
    for _ in range(10):
        mode._update_vertical_avoidance(clear, [0.0, 0.0, -10.5])
    assert mode._va_episode.state == "CLIMBING"
    assert mode._va_episode.clear_frames == 0


def test_consecutive_clear_plus_min_height_enters_forward_clearance():
    mode = _climbing_mode()
    clear = _rays(front=5.0, up=5.0, frontUp=5.0, leftUp=5.0, rightUp=5.0)
    for _ in range(VerticalAvoidanceParams().clear_frames_required):
        mode._update_vertical_avoidance(clear, [0.0, 0.0, -10.9])
    assert mode._va_episode.state == "FORWARD_CLEARANCE"
    assert mode._va_episode.forward_start_xy == (0.0, 0.0)


# ── forward clearance ──


def test_forward_clearance_holds_altitude_then_crosses():
    import time
    mode = _mode()
    mode._va_episode = VerticalAvoidEpisode(
        state="FORWARD_CLEARANCE", start_time=time.monotonic(), start_z=-10.9,
        target_z=-10.9, forward_start_xy=(0.0, 0.0), reason="test",
    )
    open_rays = _rays()
    # Not yet crossed: holds the cleared altitude (target == current z).
    mode._update_vertical_avoidance(open_rays, [0.5, 0.0, -10.9])
    assert mode._va_episode.state == "FORWARD_CLEARANCE"
    assert mode._va_episode.target_z == pytest.approx(-10.9)
    # Crosses the 1.5 m forward-clearance distance → exit + replan.
    mode._update_vertical_avoidance(open_rays, [1.6, 0.0, -10.9])
    assert mode._va_episode.state == "NORMAL"
    assert mode._traj_force_replan is True


def test_forward_clearance_timeout_exits():
    import time
    mode = _mode()
    mode._va_episode = VerticalAvoidEpisode(
        state="FORWARD_CLEARANCE", start_time=time.monotonic() - 9.0,
        start_z=-10.9, target_z=-10.9, forward_start_xy=(0.0, 0.0),
        reason="test",
    )
    # Has moved little but the 8 s bound is exceeded → exit.
    mode._update_vertical_avoidance(_rays(), [0.1, 0.0, -10.9])
    assert mode._va_episode.state == "NORMAL"
    assert mode._traj_force_replan is True


# ── bounded failure ──


def test_max_climb_exits_with_replan():
    mode = _climbing_mode()  # max_climb_m=2.0, ceiling_margin_m=0.3 → cap -11.7
    mode._update_vertical_avoidance(_CLIMB_RAYS, [0.0, 0.0, -11.8])
    assert mode._va_episode.state == "NORMAL"
    assert mode._traj_force_replan is True


def test_up_clearance_lost_exits_climb():
    mode = _climbing_mode()
    no_headroom = _rays(front=1.0, up=1.0, frontUp=1.0, leftUp=1.0, rightUp=1.0)
    mode._update_vertical_avoidance(no_headroom, [0.0, 0.0, -10.9])
    assert mode._va_episode.state == "NORMAL"
    assert mode._traj_force_replan is True


# ── NED velocity sign ──


def test_altitude_hold_velocity_negative_when_above():
    mode = AutomaticMode.__new__(AutomaticMode)
    vz = mode._altitude_hold_velocity(-10.0, -10.6, 1.0)
    assert vz < 0.0  # target above current → climb = negative NED vz


# ── command-source accounting ──


def test_vertical_escape_is_a_maneuver_not_a_hover():
    mode = AutomaticMode.__new__(AutomaticMode)
    mode._trajectory_dispatch_count = 0
    mode._hover_dispatch_count = 0
    mode._hover_due_control_overrun = 0
    mode._hover_due_trajectory_stale = 0
    mode._hover_due_no_feasible = 0
    mode._hover_due_other = 0
    mode._record_dispatch_source("vertical_escape")
    assert mode._hover_dispatch_count == 0
    assert mode._trajectory_dispatch_count == 0

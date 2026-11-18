"""Tests for planners/vertical_avoidance.py — climb-over vs. go-around selector.

Pure unit tests: no AirSim client is imported.  The selector is stateless; the
per-frame hysteresis (stay in climb until ``front`` clears) lives in
``AutomaticMode._update_vertical_avoidance`` and is exercised there.
"""

import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.vertical_avoidance import (
    VerticalAvoidanceParams,
    VerticalAvoidanceSelector,
    safe_ray,
    upward_clearance,
)


def _rays(**overrides):
    """Return a plausible full sector dict, overriding just the tested keys."""
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


@pytest.fixture
def selector():
    return VerticalAvoidanceSelector()


# ── safe_ray ──


def test_safe_ray_coerces_missing_and_non_finite():
    assert safe_ray(None) == 0.0
    assert safe_ray(float("nan")) == 0.0
    assert safe_ray(float("inf")) == 0.0
    assert safe_ray(float("-inf")) == 0.0
    assert safe_ray("not-a-number") == 0.0
    assert safe_ray("3.5") == 3.5
    assert safe_ray(2.0) == 2.0


def test_upward_clearance_takes_most_restrictive_sector():
    rays = _rays(up=1.0, frontUp=4.0, leftUp=5.0, rightUp=6.0)
    assert upward_clearance(rays) == 1.0


def test_upward_clearance_missing_sector_is_zero():
    assert upward_clearance({"up": 5.0, "frontUp": 5.0, "leftUp": 5.0}) == 0.0


# ── lateral decisions ──


def test_front_clear_is_lateral(selector):
    d = selector.select(_rays(front=5.0), current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "front_clear"


def test_low_obstacle_but_wide_sides_stays_lateral(selector):
    # Obstacle ahead is low enough to climb, headroom is fine, but the sides
    # are wide → the default lateral bypass is preferred.
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=5.0, right=5.0)
    d = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "sides_open"


def test_tall_obstacle_is_lateral(selector):
    rays = _rays(front=1.0, frontUp=1.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "obstacle_tall"


def test_insufficient_headroom_is_lateral(selector):
    rays = _rays(front=1.0, frontUp=5.0, up=1.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "up_clearance_insufficient"


def test_ceiling_reached_is_lateral(selector):
    # Already climbed to the cap (target_z - max_climb_m + margin).
    params = VerticalAvoidanceParams(max_climb_m=2.0, ceiling_margin_m=0.3)
    sel = VerticalAvoidanceSelector(params)
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    # climb_cap = -12.0, +0.3 → -11.7; current -11.9 is already above it.
    d = sel.select(rays, current_z=-11.9, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "ceiling_reached"


def test_disabled_is_lateral(selector):
    sel = VerticalAvoidanceSelector(VerticalAvoidanceParams(enabled=False))
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = sel.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "disabled"


def test_invalid_altitude_is_lateral(selector):
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = selector.select(rays, current_z=float("nan"), target_z=-10.0)
    assert d.action == "lateral"
    assert d.reason == "invalid_altitude"


# ── climb decision ──


def test_low_obstacle_narrow_sides_and_headroom_climbs(selector):
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.5)
    d = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "climb"
    assert d.reason == "climb_over"
    # Incremental step: max(climb_cap=-12.0, -10.0 - 0.6) = -10.6.
    assert d.target_z == pytest.approx(-10.6)
    # Target stays within [target_z - max_climb_m, current_z].
    assert -12.0 <= d.target_z <= -10.0


def test_climb_target_never_exceeds_ceiling(selector):
    # Close to the ceiling but not past it; the step must clamp at the cap.
    params = VerticalAvoidanceParams(max_climb_m=2.0)
    sel = VerticalAvoidanceSelector(params)
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = sel.select(rays, current_z=-11.6, target_z=-10.0)
    assert d.action == "climb"
    # climb_cap = -12.0; step would be -12.2 but is clamped up to -12.0.
    assert d.target_z == pytest.approx(-12.0)


def test_missing_vertical_sector_never_climbs(selector):
    # Fail-closed: an absent/Nan upward sector must force lateral.
    rays = _rays(front=1.0, frontUp=5.0, up=None, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    d = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert d.action == "lateral"

    rays2 = _rays(front=1.0, frontUp=float("nan"), up=5.0, leftUp=5.0,
                  rightUp=5.0, left=1.0, right=1.0)
    d2 = selector.select(rays2, current_z=-10.0, target_z=-10.0)
    assert d2.action == "lateral"


# ── statelessness (hysteresis is the caller's job) ──


def test_selector_is_stateless_and_repeatable(selector):
    rays = _rays(front=1.0, frontUp=5.0, up=5.0, leftUp=5.0, rightUp=5.0,
                 left=1.0, right=1.0)
    first = selector.select(rays, current_z=-10.0, target_z=-10.0)
    second = selector.select(rays, current_z=-10.0, target_z=-10.0)
    assert first.action == second.action == "climb"
    assert first.target_z == pytest.approx(second.target_z)

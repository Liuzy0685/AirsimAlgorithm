"""Tests for CBMBA → horizontal guidance adapter (v2 — segment crossing)."""

import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.cbmba_guidance import (
    CbmbaGuidance,
    CbmbaGuidanceParams,
    CbmbaGuidanceResult,
)


# ── helpers ──


def _path(*points):
    """Build a path_world list from (x, y, z) tuples."""
    return [[p[0], p[1], p[2]] for p in points]


def _seg(path, i, j):
    """Return the (world_i, world_j) pair for segment i→j."""
    return (tuple(path[i]), tuple(path[j]))


# ── test classes ──


class TestSegmentCrossingPrimary:
    """Primary strategy: earliest segment crossing body_x = guidance_lookahead_x."""

    def test_backward_then_forward_crosses_with_lateral_preserved(self):
        """Segment (-2,-1.8)→(3,-0.5) crosses lookahead=1.0; lateral preserved."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        # Simulates real CBMBA: first waypoint goes backward+left, then forward
        path = _path(
            (0.0, 0.0, -0.5),       # 0: start
            (-2.0, -1.8, -2.5),     # 1: backward + left + up
            (3.0, -0.5, -2.5),      # 2: forward (crosses lookahead)
            (8.0, 1.2, -0.5),       # 3: further forward
            (15.0, 0.0, -0.5),      # 4: goal
        )
        result = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        assert result.valid
        assert result.interpolated is True
        assert result.source_segment == (1, 2)
        # t = (1.0 - (-2.0)) / (3.0 - (-2.0)) = 3/5 = 0.6
        # target_body_y = -1.8 + 0.6 * (-0.5 - (-1.8)) = -1.8 + 0.78 = -1.02
        assert result.target_body_xy[0] == pytest.approx(1.0, abs=0.01)
        assert result.target_body_xy[1] == pytest.approx(-1.02, abs=0.05)
        assert result.forward_progress_m == pytest.approx(1.0, abs=0.01)
        assert result.lateral_offset_m < 0  # left avoidance preserved

    def test_multiple_backward_then_forward_earliest_crossing(self):
        """Multiple backward waypoints; earliest crossing segment wins."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (-1.0, -0.5, 0.0),      # 1: backward
            (-2.0, -1.0, 0.0),      # 2: backward
            (-2.5, -1.2, 0.0),      # 3: backward
            (4.0, 0.5, 0.0),        # 4: forward — crosses with segment 3→4
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.interpolated is True
        assert result.source_segment == (3, 4)  # earliest crossing
        assert result.forward_progress_m == pytest.approx(1.0, abs=0.01)

    def test_straight_path_target_near_lookahead(self):
        """Straight-forward path: target_body_x ≈ lookahead, body_y ≈ 0."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.interpolated is True
        assert result.source_segment == (0, 1)
        assert result.target_body_xy[0] == pytest.approx(1.0, abs=0.01)
        assert result.target_body_xy[1] == pytest.approx(0.0, abs=0.01)

    def test_left_detour_interpolated_body_y_negative(self):
        """Path with a leftward detour: interpolated body_y < 0."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (5.0, -3.0, 0.0),       # forward + left
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.source_segment == (0, 1)
        # t = 1.0 / 5.0 = 0.2; body_y = 0 + 0.2 * (-3.0) = -0.6
        assert result.target_body_xy[1] == pytest.approx(-0.6, abs=0.05)
        assert result.lateral_offset_m < 0

    def test_right_detour_interpolated_body_y_positive(self):
        """Path with a rightward detour: interpolated body_y > 0."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (5.0, 3.0, 0.0),        # forward + right
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.source_segment == (0, 1)
        # t = 1.0 / 5.0 = 0.2; body_y = 0 + 0.2 * 3.0 = 0.6
        assert result.target_body_xy[1] == pytest.approx(0.6, abs=0.05)
        assert result.lateral_offset_m > 0

    def test_large_z_changes_do_not_affect_xy_interpolation(self):
        """Z jumps do not affect the horizontal interpolation result."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, -0.5),
            (-2.0, -1.8, -20.0),    # huge Z drop
            (5.0, 2.0, 15.0),        # huge Z rise — crosses lookahead
            (15.0, 0.0, -0.5),
        )
        result = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        assert result.valid
        assert result.interpolated is True
        # Horizontal interpolation uses XY only
        assert result.target_body_xy[0] == pytest.approx(1.0, abs=0.01)
        # Z is preserved in source_waypoint but doesn't affect XY
        assert result.source_waypoint is not None
        assert len(result.source_waypoint) == 3

    def test_custom_lookahead_distance(self):
        """Custom guidance_lookahead_x changes the crossing point."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=3.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (-1.0, 0.5, 0.0),       # backward
            (8.0, 2.0, 0.0),        # forward — crosses at body_x=3.0
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.source_segment == (1, 2)
        assert result.forward_progress_m == pytest.approx(3.0, abs=0.01)


class TestFallbackFirstForward:
    """Fallback: no segment crosses lookahead → try first-forward-waypoint rule."""

    def test_no_crossing_uses_fallback_intermediate_waypoint(self):
        """When no segment crosses lookahead, an intermediate forward waypoint works."""
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=5.0,       # far ahead
            min_forward_progress=2.0,        # > body_x of first waypoint
            min_waypoint_distance=0.5,
        ))
        # All body_x <= 4 < 5.0, so NO segment crosses.
        # idx=1 body_x=1.0 < 2.0 (min_forward_progress) → skipped
        # idx=2 body_x=3.0 >= 2.0, intermediate → fallback selects
        path = _path(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),        # body_x=1 < 2.0, skipped
            (3.0, 0.5, 0.0),        # body_x=3 >= 2.0, intermediate
            (4.0, 0.0, 0.0),        # body_x=4, last (rejected by fallback)
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.interpolated is False
        assert result.source_segment is None
        assert result.reason == "fallback_first_forward"
        assert result.source_waypoint[0] == pytest.approx(3.0)
        assert result.source_waypoint[1] == pytest.approx(0.5)

    def test_fallback_rejects_last_waypoint_goal(self):
        """Fallback rejects the last waypoint even if it has forward progress."""
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=20.0,      # beyond everything
            min_forward_progress=0.25,
        ))
        path = _path(
            (0.0, 0.0, 0.0),
            (15.0, 0.0, 0.0),       # only waypoint; forward but it's the goal
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert not result.valid
        assert result.reason == "no_forward_path_intersection"

    def test_no_crossing_no_fallback_returns_invalid(self):
        """Neither primary nor fallback finds a valid target."""
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=20.0,
            min_forward_progress=0.25,
        ))
        path = _path(
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),       # backward
            (-3.0, 0.0, 0.0),       # backward
            (-5.0, 0.0, 0.0),       # backward (last, rejected)
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert not result.valid
        assert result.reason == "no_forward_path_intersection"


class TestDegenerateCases:
    """Edge cases: degenerate segments, yaw rotations, short paths."""

    def test_degenerate_vertical_segment_safe(self):
        """Segment with A.x == B.x (vertical in body X) is skipped safely."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),        # pure lateral (body_x stays 0)
            (5.0, 2.0, 0.0),        # forward — this segment crosses
            (15.0, 0.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        # segment 0→1 has body (0,0)→(0,2): A.x == B.x == 0, not < 1.0 → skip
        # segment 1→2: body (0,2)→(5,2): 0 < 1.0 <= 5 → crossing!
        assert result.source_segment == (1, 2)

    def test_yaw_90_deg_segment_crossing(self):
        """Segment crossing works correctly with yaw ≠ 0."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        # Drone heading East (yaw=π/2)
        # World waypoints: forward in world-Y (East), lateral in world-X
        path = _path(
            (0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),        # 5m East (body +X = forward when heading East)
            (0.0, 10.0, 0.0),
        )
        result = g.select_waypoint((0.0, 0.0, 0.0), math.pi / 2, path)
        assert result.valid
        assert result.interpolated is True
        # body_x should be ~1.0 (lookahead)
        assert result.forward_progress_m == pytest.approx(1.0, abs=0.01)
        # body_y should be ~0 (no lateral offset)
        assert result.lateral_offset_m == pytest.approx(0.0, abs=0.01)

    def test_single_point_path_fails(self):
        g = CbmbaGuidance()
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, [[0.0, 0.0, 0.0]])
        assert not result.valid
        assert result.reason == "empty_path"

    def test_none_path_fails(self):
        g = CbmbaGuidance()
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, None)
        assert not result.valid

    def test_nan_waypoint_skipped_segment(self):
        """NaN waypoint breaks its segments; falls back to first-forward rule."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = [
            [0.0, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],     # NaN → segments 0→1 and 1→2 both broken
            [5.0, 0.0, 0.0],              # 2: first valid, body_x=5 >= 1.0 → no crossing
            [10.0, 0.0, 0.0],             # 3: last (rejected by fallback)
        ]
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        # Segment 2→3 has A.x=5 >= 1.0 → doesn't cross.  Falls back to
        # first-forward waypoint at idx=2 (intermediate, not last).
        assert result.source_segment is None
        assert result.interpolated is False
        assert result.reason == "fallback_first_forward"

    def test_none_waypoint_skipped_segment(self):
        """None waypoint breaks surrounding segments; falls back to first-forward."""
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = [
            [0.0, 0.0, 0.0],
            None,
            [5.0, 0.0, 0.0],       # first valid, body_x=5 >= 1.0 → no segment crosses
            [10.0, 0.0, 0.0],      # last (rejected by fallback)
        ]
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.source_segment is None
        assert result.interpolated is False
        assert result.reason == "fallback_first_forward"


class TestInputValidation:
    """NaN / Inf / malformed inputs must fail safely without crashing."""

    def test_nan_drone_position_fails(self):
        g = CbmbaGuidance()
        path = _path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        result = g.select_waypoint((float("nan"), 0.0, 0.0), 0.0, path)
        assert not result.valid
        assert "nonfinite" in result.reason

    def test_inf_drone_position_fails(self):
        g = CbmbaGuidance()
        path = _path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        result = g.select_waypoint((float("inf"), 0.0, 0.0), 0.0, path)
        assert not result.valid
        assert "nonfinite" in result.reason

    def test_nan_yaw_fails(self):
        g = CbmbaGuidance()
        path = _path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), float("nan"), path)
        assert not result.valid
        assert "nonfinite" in result.reason

    def test_inf_waypoint_skipped(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = [
            [0.0, 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
            [5.0, 0.0, 0.0],       # first valid, body_x=5 >= 1.0 → no crossing
            [10.0, 0.0, 0.0],      # last (rejected by fallback)
        ]
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.source_segment is None
        assert result.interpolated is False
        assert result.reason == "fallback_first_forward"


class TestDeterministicBehavior:
    """Guidance output must be deterministic."""

    def test_same_input_same_output(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, -0.5),
            (-2.0, -1.8, -2.5),
            (3.0, -0.5, -2.5),
            (15.0, 0.0, -0.5),
        )
        r1 = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        r2 = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        assert r1.valid == r2.valid
        assert r1.source_segment == r2.source_segment
        assert r1.interpolated == r2.interpolated
        assert r1.forward_progress_m == pytest.approx(r2.forward_progress_m)
        assert r1.lateral_offset_m == pytest.approx(r2.lateral_offset_m)
        assert r1.reason == r2.reason


class TestNonModification:
    """Guidance adapter must never mutate the input path."""

    def test_original_path_unchanged(self):
        g = CbmbaGuidance()
        path = _path(
            (0.0, 0.0, -0.5),
            (-2.0, -1.8, -2.5),
            (3.0, -0.5, -2.5),
            (15.0, 0.0, -0.5),
        )
        path_copy = [list(pt) for pt in path]
        g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        for i, pt in enumerate(path):
            assert pt == path_copy[i], f"Waypoint {i} was modified"


class TestNoAirSimImport:
    """CbmbaGuidance must not import or reference external simulation APIs."""

    def test_no_airsim_import(self):
        import inspect
        import planners.cbmba_guidance as mod
        src = inspect.getsource(mod)
        assert "airsim" not in src.lower()

    def test_no_velocity_command_interface(self):
        fields = {f.name for f in CbmbaGuidanceResult.__dataclass_fields__.values()}
        velocity_terms = {"vx", "vy", "vz", "velocity", "speed", "command", "throttle"}
        overlap = fields & velocity_terms
        assert not overlap, f"CbmbaGuidanceResult has velocity fields: {overlap}"


class TestSourceWaypointPreservesZ:
    """source_waypoint must include interpolated or raw Z for diagnostics."""

    def test_interpolated_z_preserved(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path(
            (0.0, 0.0, -0.5),
            (5.0, 2.0, -7.3),
            (15.0, 0.0, -0.5),
        )
        result = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        assert result.valid
        assert result.source_waypoint is not None
        assert len(result.source_waypoint) == 3
        # t = 1.0/5.0 = 0.2; z = -0.5 + 0.2 * (-7.3 - (-0.5)) = -0.5 + 0.2 * (-6.8) = -1.86
        expected_z = -0.5 + 0.2 * (-7.3 + 0.5)
        assert result.source_waypoint[2] == pytest.approx(expected_z, abs=0.05)

    def test_fallback_z_preserved(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=20.0,
            min_forward_progress=0.25,
        ))
        path = _path(
            (0.0, 0.0, -0.5),
            (3.0, 1.0, -9.0),       # intermediate, not last
            (15.0, 0.0, -0.5),
        )
        result = g.select_waypoint((0.0, 0.0, -0.5), 0.0, path)
        assert result.valid
        assert result.interpolated is False
        assert result.source_waypoint[2] == pytest.approx(-9.0)


class TestDirectionBodyXY:
    """direction_body_xy must be a unit vector in body frame."""

    def test_direction_is_unit_length(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        dx, dy = result.direction_body_xy
        assert math.hypot(dx, dy) == pytest.approx(1.0, abs=1e-9)

    def test_direction_at_yaw_90(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), math.pi / 2, path)
        assert result.valid
        assert result.direction_body_xy[0] == pytest.approx(1.0, abs=1e-9)
        assert result.direction_body_xy[1] == pytest.approx(0.0, abs=1e-9)


class TestBodyFrameConversion:
    """World → body conversion is correct and has no X/Y swap."""

    def test_yaw_zero_world_x_becomes_body_x(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.lateral_offset_m == pytest.approx(0.0, abs=1e-9)

    def test_yaw_zero_world_y_becomes_body_y(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=1.0,
            min_forward_progress=0.0,
        ))
        # Pure lateral path → segment won't cross lookahead (body_x stays 0),
        # but fallback with min_forward_progress=0 will select it.
        path = _path((0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 10.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), 0.0, path)
        assert result.valid
        assert result.target_body_xy[1] == pytest.approx(5.0)

    def test_yaw_90_world_north_is_body_left(self):
        """yaw=90°: world +X (North) is left of drone → body_y < 0."""
        g = CbmbaGuidance(CbmbaGuidanceParams(
            guidance_lookahead_x=1.0,
            min_forward_progress=0.0,
        ))
        path = _path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), math.pi / 2, path)
        assert result.valid
        assert result.lateral_offset_m < 0  # North is left of East heading

    def test_no_xy_swap(self):
        g = CbmbaGuidance(CbmbaGuidanceParams(guidance_lookahead_x=1.0))
        path = _path((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        result = g.select_waypoint((0.0, 0.0, 0.0), math.pi / 4, path)
        assert result.valid
        # With yaw=45°, world (10,0) should have body_x > 0 and body_y < 0
        assert result.target_body_xy[0] > 0
        assert result.target_body_xy[1] < 0

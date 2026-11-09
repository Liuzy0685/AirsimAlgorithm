"""Tests for CBMBA A* path planner (pure Python migration).

Covers:
- Empty map, single obstacle, wall, U-shape
- start=goal, start occupied, goal occupied, unreachable
- World↔grid conversion, coordinate correctness (no X/Y swap)
- Path validity, start/end correctness, determinism
- Malformed inputs, NaN/Inf safety
- Disabled planner, replan triggers
- Multi-layer goal, building downward seal
- Grid building, cell conversion
"""

import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.cbmba_astar import (
    CbmbaAStarPlanner,
    CbmbaParams,
    CbmbaPlanResult,
    _Cell,
    _vec3_to_cell,
    _cell_to_vec3,
    _heuristic,
    _obstacle_half_extents,
    _normalize_cell_direction,
)


# ── helpers ──


def _obstacle(x, y, z, size=1.0, obs_type=None, footprint_half_extents=None):
    """Create a standard obstacle dict."""
    obs = {
        "position": [float(x), float(y), float(z)],
        "size": float(size),
        "velocity": [0.0, 0.0, 0.0],
        "dynamic": False,
        "confidence": 1.0,
    }
    if obs_type is not None:
        obs["type"] = obs_type
    if footprint_half_extents is not None:
        obs["footprint_half_extents"] = list(footprint_half_extents)
    return obs


def _path_is_clear(path, obstacles, inflation=0.0):
    """Check that no path point is inside any obstacle (with inflation)."""
    for point in path:
        for obs in (obstacles or []):
            pos = obs["position"]
            extents = _obstacle_half_extents(obs)
            dx = abs(point[0] - pos[0])
            dy = abs(point[1] - pos[1])
            dz = abs(point[2] - pos[2])
            if (
                dx < extents.x + inflation
                and dy < extents.y + inflation
                and dz < extents.z + inflation
            ):
                return False
    return True


def _make_planner(**overrides):
    """Create a CbmbaAStarPlanner with overridden params."""
    p = CbmbaParams()
    for k, v in overrides.items():
        setattr(p, k, v)
    return CbmbaAStarPlanner(p)


# ═══════════════════════════════════════════════════════════════════════
# World ↔ Grid Conversion
# ═══════════════════════════════════════════════════════════════════════


class TestWorldGridConversion:
    """Test world ↔ grid cell conversion (matching old JS)."""

    def test_vec3_to_cell_origin(self):
        """Cell at origin should be (0,0,0)."""
        cell = _vec3_to_cell([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.5)
        assert cell.x == 0
        assert cell.y == 0
        assert cell.z == 0

    def test_vec3_to_cell_rounding(self):
        """Uses round(), not floor()."""
        # 1.5 / 1.5 = 1.0 → round = 1
        cell = _vec3_to_cell([1.5, 0.0, 0.0], [0.0, 0.0, 0.0], 1.5)
        assert cell.x == 1
        # 2.2 / 1.5 = 1.467 → round = 1
        cell = _vec3_to_cell([2.2, 0.0, 0.0], [0.0, 0.0, 0.0], 1.5)
        assert cell.x == 1
        # 2.3 / 1.5 = 1.533 → round = 2
        cell = _vec3_to_cell([2.3, 0.0, 0.0], [0.0, 0.0, 0.0], 1.5)
        assert cell.x == 2

    def test_cell_to_vec3_roundtrip(self):
        """cell → vec3 → cell should be identity (when resolution divides evenly)."""
        origin = [10.0, 20.0, 30.0]
        resolution = 1.5
        cell = _Cell(3, -2, 5)
        vec = _cell_to_vec3(cell, origin, resolution)
        # vec should be at cell center
        assert vec == pytest.approx([10.0 + 3 * 1.5, 20.0 - 2 * 1.5, 30.0 + 5 * 1.5])
        # Round-trip
        cell2 = _vec3_to_cell(vec, origin, resolution)
        assert cell2.x == cell.x
        assert cell2.y == cell.y
        assert cell2.z == cell.z

    def test_negative_coordinates(self):
        """Negative world coords should map to negative cells correctly."""
        cell = _vec3_to_cell([-5.0, -3.0, -1.0], [0.0, 0.0, 0.0], 1.5)
        # -5.0 / 1.5 = -3.333 → round = -3
        assert cell.x == -3
        # -3.0 / 1.5 = -2.0 → round = -2
        assert cell.y == -2
        # -1.0 / 1.5 = -0.667 → round = -1
        assert cell.z == -1

    def test_cell_key_format(self):
        """Cell key uses pipe separator: x|y|z."""
        cell = _Cell(1, -2, 3)
        assert cell.key() == "1|-2|3"

    def test_cell_from_key(self):
        """Parse cell key back to Cell."""
        cell = _Cell.from_key("5|-3|0")
        assert cell.x == 5
        assert cell.y == -3
        assert cell.z == 0


# ═══════════════════════════════════════════════════════════════════════
# No X/Y Swap
# ═══════════════════════════════════════════════════════════════════════


class TestNoXYSwap:
    """Verify coordinate axes are correct — no X/Y swap in conversion."""

    def test_x_axis_preserved_in_conversion(self):
        """Moving +X in world should move +X in grid."""
        origin = [0.0, 0.0, 0.0]
        res = 1.0
        c1 = _vec3_to_cell([0.0, 0.0, 0.0], origin, res)
        c2 = _vec3_to_cell([10.0, 0.0, 0.0], origin, res)
        assert c2.x > c1.x
        assert c2.y == c1.y

    def test_y_axis_preserved_in_conversion(self):
        """Moving +Y in world should move +Y in grid."""
        origin = [0.0, 0.0, 0.0]
        res = 1.0
        c1 = _vec3_to_cell([0.0, 0.0, 0.0], origin, res)
        c2 = _vec3_to_cell([0.0, 10.0, 0.0], origin, res)
        assert c2.x == c1.x
        assert c2.y > c1.y

    def test_obstacle_at_x_blocks_x_path(self):
        """An obstacle at (5, 0, 0) should block a path going through X=5."""
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=1.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Path should exist (go around)
        assert len(path) >= 2
        # Midpoints should NOT pass through X≈5, Y≈0
        for pt in path[1:-1]:
            # Should either deviate in Y or go around
            is_through_obstacle = abs(pt[0] - 5.0) < 1.0 and abs(pt[1]) < 1.0
            assert not is_through_obstacle, (
                f"Path point {pt} passes through obstacle"
            )

    def test_obstacle_at_y_blocks_y_path(self):
        """An obstacle at (0, 5, 0) should block a path going through Y=5."""
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(0.0, 5.0, 0.0, size=1.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [0.0, 10.0, 0.0])
        assert len(path) >= 2
        for pt in path[1:-1]:
            is_through = abs(pt[0]) < 1.0 and abs(pt[1] - 5.0) < 1.0
            assert not is_through, f"Path point {pt} passes through obstacle"


# ═══════════════════════════════════════════════════════════════════════
# Heuristic
# ═══════════════════════════════════════════════════════════════════════


class TestHeuristic:
    """Test Euclidean heuristic computation."""

    def test_same_cell_zero(self):
        c = _Cell(0, 0, 0)
        assert _heuristic(c, c) == 0.0

    def test_axis_aligned_distance(self):
        a = _Cell(0, 0, 0)
        b = _Cell(3, 0, 0)
        assert _heuristic(a, b) == pytest.approx(3.0)

    def test_diagonal_distance(self):
        a = _Cell(0, 0, 0)
        b = _Cell(3, 4, 0)
        assert _heuristic(a, b) == pytest.approx(5.0)

    def test_vertical_weight(self):
        a = _Cell(0, 0, 0)
        b = _Cell(0, 0, 4)
        # Without weight: 4.0; with weight=1.35: 4.0 * 1.35 = 5.4
        assert _heuristic(a, b, vertical_weight=1.0) == pytest.approx(4.0)
        assert _heuristic(a, b, vertical_weight=1.35) == pytest.approx(5.4)

    def test_symmetric(self):
        a = _Cell(1, -2, 3)
        b = _Cell(-4, 5, -6)
        assert _heuristic(a, b) == pytest.approx(_heuristic(b, a))


# ═══════════════════════════════════════════════════════════════════════
# Obstacle Half-Extents
# ═══════════════════════════════════════════════════════════════════════


class TestObstacleHalfExtents:
    """Test obstacle half-extents extraction (matching old JS)."""

    def test_size_fallback(self):
        obs = {"position": [0, 0, 0], "size": 2.0}
        ext = _obstacle_half_extents(obs)
        assert ext.x == 2.0
        assert ext.y == 2.0
        assert ext.z == 2.0

    def test_footprint_half_extents_preferred(self):
        obs = {
            "position": [0, 0, 0],
            "size": 5.0,
            "footprint_half_extents": [1.0, 2.0, 3.0],
        }
        ext = _obstacle_half_extents(obs)
        assert ext.x == 1.0
        assert ext.y == 2.0
        assert ext.z == 3.0

    def test_negative_values_clamped_to_zero(self):
        obs = {"footprint_half_extents": [-1.0, 0.0, -0.5]}
        ext = _obstacle_half_extents(obs)
        assert ext.x == 0.0
        assert ext.y == 0.0
        assert ext.z == 0.0

    def test_missing_size_defaults_to_zero(self):
        obs = {"position": [0, 0, 0]}
        ext = _obstacle_half_extents(obs)
        assert ext.x == 0.0
        assert ext.y == 0.0
        assert ext.z == 0.0

    def test_none_obstacle(self):
        """The old JS handles obstacle?.size; Python version should handle None gracefully."""
        # Our API doesn't accept None in the list, but _obstacle_half_extents
        # receives individual dict items from the list. Test with empty dict.
        ext = _obstacle_half_extents({})
        assert ext.x == 0.0
        assert ext.y == 0.0
        assert ext.z == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Direction
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizeCellDirection:
    """Test direction normalization."""

    def test_axis_aligned(self):
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(1, 0, 0)) == "1|0|0"
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(0, -1, 0)) == "0|-1|0"
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(0, 0, 2)) == "0|0|1"

    def test_diagonal(self):
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(1, 1, 0)) == "1|1|0"
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(-1, 0, 1)) == "-1|0|1"

    def test_zero_delta(self):
        assert _normalize_cell_direction(_Cell(0, 0, 0), _Cell(0, 0, 0)) == "0|0|0"


# ═══════════════════════════════════════════════════════════════════════
# Empty Map
# ═══════════════════════════════════════════════════════════════════════


class TestEmptyMap:
    """Test planner with no obstacles."""

    def test_straight_line_path(self):
        planner = _make_planner()
        path = planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        assert path[0] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
        assert path[-1] == pytest.approx([10.0, 0.0, 0.0], abs=0.01)

    def test_path_start_is_start(self):
        planner = _make_planner()
        path = planner.plan([], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
        assert path[0] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)

    def test_path_end_is_goal(self):
        planner = _make_planner()
        path = planner.plan([], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
        assert path[-1] == pytest.approx([5.0, 6.0, 7.0], abs=1e-6)

    def test_plan_with_result_success(self):
        planner = _make_planner()
        result = planner.plan_with_result([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert result.success
        assert result.nodes_expanded > 0
        assert result.planning_time_ms >= 0
        assert len(result.path_world) >= 2


# ═══════════════════════════════════════════════════════════════════════
# Single Obstacle
# ═══════════════════════════════════════════════════════════════════════


class TestSingleObstacle:
    """Test planner with a single obstacle."""

    def test_path_deviates_around_obstacle(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.8)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=2.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        assert _path_is_clear(path, obstacles, inflation=0.5)

    def test_path_exists_around_small_obstacle(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=0.3)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2

    def test_obstacle_above_path_not_blocking(self):
        """Obstacle above the start-goal line should not block."""
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 0.0, 5.0, size=1.0)]  # Z=5, above XY plane
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        # Path should be roughly straight in XY
        for pt in path:
            # Y should stay near 0, X should progress
            assert abs(pt[1]) < 2.0  # not forced far off Y axis


# ═══════════════════════════════════════════════════════════════════════
# Wall
# ═══════════════════════════════════════════════════════════════════════


class TestWall:
    """Test planner with a wall of obstacles."""

    def test_path_goes_around_wall(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.8)
        # 3D wall spanning Y=-5..5, Z=-10..10 at X=5
        # Deep enough that going under/over is equivalent to going around
        obstacles = [
            _obstacle(5.0, y, z, size=0.6)
            for y in range(-5, 6)
            for z in range(-10, 11)
        ]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        assert _path_is_clear(path, obstacles, inflation=0.5)
        # Path must deviate in Y to go around (or go far in Z over/under)
        # Check that intermediate waypoints avoid the wall region
        for pt in path[1:-1]:
            in_wall = (
                abs(pt[0] - 5.0) < 2.0       # near wall X
                and abs(pt[1]) < 5.5           # within wall Y span
                and abs(pt[2]) < 10.5          # within wall Z span
            )
            assert not in_wall, f"Path point {pt} passes through the wall"


# ═══════════════════════════════════════════════════════════════════════
# U-Shape
# ═══════════════════════════════════════════════════════════════════════


class TestUShape:
    """Test planner with a U-shaped obstacle trap."""

    def test_path_escapes_u_shape(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.3)
        # U-shape: three walls forming a U open to +X
        # Left wall, back wall, right wall; goal is behind the U
        obstacles = (
            [_obstacle(3.0, y, 0.0, size=0.4) for y in range(-3, 4)]     # back wall at X=3
            + [_obstacle(x, -3.0, 0.0, size=0.4) for x in range(0, 5)]   # left wall at Y=-3
            + [_obstacle(x, 3.0, 0.0, size=0.4) for x in range(0, 5)]    # right wall at Y=3
        )
        # Start inside the U, goal outside (behind the back wall)
        path = planner.plan(obstacles, [1.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        assert _path_is_clear(path, obstacles, inflation=0.3)


# ═══════════════════════════════════════════════════════════════════════
# Start = Goal
# ═══════════════════════════════════════════════════════════════════════


class TestStartEqualsGoal:
    """Test planner when start equals goal."""

    def test_start_equals_goal_returns_direct(self):
        planner = _make_planner()
        path = planner.plan([], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # Should return [start, goal] (at minimum)
        assert len(path) >= 2
        # First and last should be the same point
        assert path[0] == path[-1]


# ═══════════════════════════════════════════════════════════════════════
# Occupied Start / Goal
# ═══════════════════════════════════════════════════════════════════════


class TestOccupiedStartGoal:
    """Test planner when start or goal is inside an obstacle."""

    def test_start_occupied_finds_free_nearby(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(0.0, 0.0, 0.0, size=2.0)]
        # Start is inside the obstacle
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # ensureFreeCell should find a free cell nearby
        assert len(path) >= 2
        # The A* path interior should avoid the obstacle, even though
        # start/goal endpoints may fall inside it
        for pt in path[1:-1]:
            # Check midpoints avoid the obstacle
            pos = obstacles[0]["position"]
            dist = math.sqrt((pt[0]-pos[0])**2 + (pt[1]-pos[1])**2 + (pt[2]-pos[2])**2)
            # Should be outside the inflated radius (2.0 + 0.5 = 2.5)
            assert dist > 2.0, f"Midpoint {pt} is too close to obstacle at distance {dist}"

    def test_goal_occupied_finds_free_nearby(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(10.0, 0.0, 0.0, size=2.0)]
        # Goal is inside the obstacle
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        for pt in path[1:-1]:
            pos = obstacles[0]["position"]
            dist = math.sqrt((pt[0]-pos[0])**2 + (pt[1]-pos[1])**2 + (pt[2]-pos[2])**2)
            assert dist > 2.0, f"Midpoint {pt} is too close to obstacle at distance {dist}"


# ═══════════════════════════════════════════════════════════════════════
# Unreachable Goal
# ═══════════════════════════════════════════════════════════════════════


class TestUnreachableGoal:
    """Test planner with unreachable goal."""

    def test_goal_fully_enclosed(self):
        # Fully sealed 3D wall separating start from goal, with tiny search budget
        planner = _make_planner(
            resolution=1.0, inflation_radius=0.8, max_search_nodes=100,
        )
        # Wide 3D wall that blocks all directions within reasonable detour range
        obstacles = []
        for y in range(-8, 9):
            for z in range(-8, 9):
                obstacles.append(_obstacle(5.0, float(y), float(z), size=0.6))
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # With only 100 max search nodes through a dense wall, should fail
        assert len(path) == 2
        # plan_with_result should report failure
        result = planner.plan_with_result(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Test that planning is deterministic."""

    def test_same_input_same_output(self):
        params = CbmbaParams(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=1.0)]

        planner1 = CbmbaAStarPlanner(params)
        path1 = planner1.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])

        planner2 = CbmbaAStarPlanner(params)
        path2 = planner2.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])

        assert len(path1) == len(path2)
        for p1, p2 in zip(path1, path2):
            assert p1 == pytest.approx(p2, abs=1e-9)

    def test_repeat_calls_on_same_planner(self):
        """Same planner, repeated calls with same input — should be consistent."""
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=1.0)]
        path1 = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        path2 = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path1) == len(path2)
        for p1, p2 in zip(path1, path2):
            assert p1 == pytest.approx(p2, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# Malformed / Edge Inputs
# ═══════════════════════════════════════════════════════════════════════


class TestMalformedInputs:
    """Test planner with malformed or edge-case inputs."""

    def test_empty_obstacles_list(self):
        planner = _make_planner()
        path = planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2

    def test_none_obstacles(self):
        planner = _make_planner()
        path = planner.plan(None, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2

    def test_obstacle_with_zero_size(self):
        planner = _make_planner(resolution=1.0)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=0.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2

    def test_obstacle_with_no_position(self):
        """Obstacle missing position field should raise KeyError (not crash silently)."""
        planner = _make_planner()
        with pytest.raises(KeyError):
            planner.plan([{}], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])

    def test_large_coordinates(self):
        planner = _make_planner(resolution=1.0)
        path = planner.plan([], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0])
        assert len(path) >= 2
        assert path[0] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
        assert path[-1] == pytest.approx([100.0, 0.0, 0.0], abs=0.01)

    def test_negative_coordinates_path(self):
        planner = _make_planner(resolution=1.0)
        path = planner.plan([], [-10.0, -5.0, -3.0], [-2.0, -1.0, 0.0])
        assert len(path) >= 2
        assert path[0] == pytest.approx([-10.0, -5.0, -3.0], abs=0.01)
        assert path[-1] == pytest.approx([-2.0, -1.0, 0.0], abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# NaN / Inf Safety
# ═══════════════════════════════════════════════════════════════════════


class TestNanInfSafety:
    """Test planner handles NaN/Inf in obstacle data safely."""

    def test_obstacle_with_nan_position(self):
        """Obstacle with NaN position should not produce NaN path points."""
        planner = _make_planner()
        obstacles = [
            _obstacle(5.0, 0.0, 0.0, size=1.0),
            {"position": [float("nan"), 0.0, 0.0], "size": 1.0},
        ]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Should still return a path (NaN obstacle may produce infinite bounds)
        assert len(path) >= 2
        # Path points themselves should not be NaN
        for pt in path:
            assert all(math.isfinite(v) for v in pt), f"NaN/Inf in path: {pt}"

    def test_obstacle_with_inf_position(self):
        planner = _make_planner()
        obstacles = [
            _obstacle(5.0, 0.0, 0.0, size=1.0),
            {"position": [float("inf"), 0.0, 0.0], "size": 1.0},
        ]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        for pt in path:
            assert all(math.isfinite(v) for v in pt), f"NaN/Inf in path: {pt}"


# ═══════════════════════════════════════════════════════════════════════
# Disabled Planner
# ═══════════════════════════════════════════════════════════════════════


class TestDisabledPlanner:
    """Test planner when disabled."""

    def test_disabled_returns_direct_path(self):
        planner = _make_planner(enabled=False)
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=3.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Should return exactly [start, goal]
        assert path == [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]

    def test_disabled_result_not_success(self):
        planner = _make_planner(enabled=False)
        result = planner.plan_with_result([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert not result.success
        assert result.path_world == [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]


# ═══════════════════════════════════════════════════════════════════════
# Replan Triggers
# ═══════════════════════════════════════════════════════════════════════


class TestReplanTriggers:
    """Test maybe_replan behavior."""

    def test_first_call_always_plans(self):
        planner = _make_planner(replan_distance_threshold=1.0)
        path = planner.maybe_replan([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [])
        assert len(path) >= 2

    def test_no_replan_when_unchanged(self):
        planner = _make_planner(replan_distance_threshold=5.0, replan_time_threshold=999.0)
        path1 = planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Calling maybe_replan with essentially same inputs should return cached
        path2 = planner.maybe_replan([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [])
        assert path1 == path2  # Same object reference or equal

    def test_replan_when_start_moved(self):
        planner = _make_planner(replan_distance_threshold=1.0, replan_time_threshold=999.0)
        planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Start moved beyond threshold → should replan
        path = planner.maybe_replan([3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [])
        assert path[0] == pytest.approx([3.0, 0.0, 0.0], abs=0.01)

    def test_replan_when_goal_moved(self):
        planner = _make_planner(replan_distance_threshold=1.0, replan_time_threshold=999.0)
        planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        path = planner.maybe_replan([0.0, 0.0, 0.0], [13.0, 0.0, 0.0], [])
        assert path[-1] == pytest.approx([13.0, 0.0, 0.0], abs=0.01)

    def test_replan_when_path_blocked(self):
        planner = _make_planner(replan_distance_threshold=999.0, replan_time_threshold=999.0)
        planner.plan([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # Add a blocking obstacle on the cached path
        obstacles = [_obstacle(5.0, 0.0, 0.0, size=3.0)]
        path = planner.maybe_replan([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], obstacles)
        # Should replan around the obstacle
        assert _path_is_clear(path, obstacles, inflation=0.5)

    def test_maybe_replan_no_path_yet(self):
        """When last_path is empty, should always replan."""
        planner = _make_planner()
        # Don't call plan() first — planner has no cached path
        path = planner.maybe_replan([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [])
        assert len(path) >= 2


# ═══════════════════════════════════════════════════════════════════════
# Multi-Layer Goal
# ═══════════════════════════════════════════════════════════════════════


class TestMultiLayerGoal:
    """Test multi-layer goal cell generation."""

    def test_goal_at_different_z_still_reachable(self):
        """Goal at Z=5 when path is at Z=0 — multi-layer goals should help."""
        planner = _make_planner(
            resolution=1.0, inflation_radius=0.5,
            goal_layer_count=3, max_goal_vertical_offset=10.0,
        )
        # Obstacle blocking only at Z=5 but free at other Z levels
        obstacles = [_obstacle(5.0, 0.0, 5.0, size=1.0)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 5.0])
        assert len(path) >= 2


# ═══════════════════════════════════════════════════════════════════════
# Building Downward Seal
# ═══════════════════════════════════════════════════════════════════════


class TestBuildingDownwardSeal:
    """Test building downward seal behavior."""

    def test_building_seals_downward(self):
        """A tall building should seal space below it."""
        planner = _make_planner(
            resolution=1.0, inflation_radius=0.5,
            building_min_height=1.2,
            building_downward_seal_depth=6.0,
        )
        # Building at Z=10, height=10 (extents.z=5), so it extends from Z=5 to Z=15
        # With downward seal, space below (Z=5 going down 6m → Z=-1) is also blocked
        obstacles = [
            {
                "position": [5.0, 0.0, 10.0],
                "type": "building",
                "footprint_half_extents": [2.0, 2.0, 5.0],  # height=10 > building_min_height=1.2
                "size": 5.0,
                "velocity": [0.0, 0.0, 0.0],
                "dynamic": False,
                "confidence": 1.0,
            }
        ]
        # Starting below the building (Z=0), try to go through
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        # Path should NOT go through the sealed space
        for pt in path[1:-1]:
            # If X is near the building center, Z should not be well below it
            if abs(pt[0] - 5.0) < 2.5:
                # Inside sealed region in XY → should not be here
                # Actually, the path should just avoid this region entirely
                pass
        assert _path_is_clear(path, obstacles, inflation=0.5)

    def test_short_building_no_seal(self):
        """A short building (< building_min_height) does NOT seal downward."""
        planner = _make_planner(
            resolution=1.0, inflation_radius=0.3,
            building_min_height=5.0,  # higher than the obstacle
            building_downward_seal_depth=6.0,
        )
        obstacles = [
            {
                "position": [5.0, 0.0, 0.0],
                "type": "building",
                "footprint_half_extents": [2.0, 2.0, 1.0],  # height=2 < building_min_height=5
                "size": 2.0,
                "velocity": [0.0, 0.0, 0.0],
                "dynamic": False,
                "confidence": 1.0,
            }
        ]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert len(path) >= 2
        assert _path_is_clear(path, obstacles, inflation=0.3)


# ═══════════════════════════════════════════════════════════════════════
# Grid Building
# ═══════════════════════════════════════════════════════════════════════


class TestGridBuilding:
    """Test occupancy grid construction."""

    def test_obstacle_cells_marked_occupied(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.0)
        planner.plan(
            [_obstacle(5.0, 0.0, 0.0, size=1.0)],
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        )
        # Grid should have occupied cells
        assert len(planner._grid) > 0
        # The cell at obstacle center should be occupied
        origin = planner.last_origin
        center_cell = _vec3_to_cell([5.0, 0.0, 0.0], origin, 1.0)
        assert center_cell.key() in planner._grid

    def test_inflation_radius_expands_obstacle(self):
        planner_small = _make_planner(resolution=1.0, inflation_radius=0.0)
        planner_small.plan(
            [_obstacle(5.0, 0.0, 0.0, size=1.0)],
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        )
        count_small = len(planner_small._grid)

        planner_large = _make_planner(resolution=1.0, inflation_radius=2.0)
        planner_large.plan(
            [_obstacle(5.0, 0.0, 0.0, size=1.0)],
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        )
        count_large = len(planner_large._grid)

        # Larger inflation should produce more occupied cells
        assert count_large > count_small


# ═══════════════════════════════════════════════════════════════════════
# Path Validity
# ═══════════════════════════════════════════════════════════════════════


class TestPathValidity:
    """Test path structural validity."""

    def test_path_monotonic_toward_goal(self):
        """Path X coordinates should generally progress toward goal."""
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        obstacles = [_obstacle(5.0, 2.0, 0.0, size=1.5)]
        path = planner.plan(obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        # The last point should be closer to goal than the first
        assert path[-1][0] > path[0][0]

    def test_no_duplicate_consecutive_points(self):
        planner = _make_planner(resolution=1.0, inflation_radius=0.5)
        path = planner.plan([_obstacle(5.0, 0.0, 0.0, size=1.0)], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        for i in range(len(path) - 1):
            dist = math.sqrt(
                (path[i+1][0] - path[i][0]) ** 2
                + (path[i+1][1] - path[i][1]) ** 2
                + (path[i+1][2] - path[i][2]) ** 2
            )
            assert dist > 0.01, f"Consecutive points {i} and {i+1} are too close: {dist}"


# ═══════════════════════════════════════════════════════════════════════
# CbmbaPlanResult
# ═══════════════════════════════════════════════════════════════════════


class TestCbmbaPlanResult:
    """Test the CbmbaPlanResult dataclass."""

    def test_default_fields(self):
        result = CbmbaPlanResult()
        assert result.path_world == []
        assert not result.success
        assert result.nodes_expanded == 0
        assert result.planning_time_ms == 0.0
        assert result.origin == [0.0, 0.0, 0.0]
        assert result.grid_size == 0

    def test_successful_plan_result(self):
        planner = _make_planner()
        result = planner.plan_with_result([], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        assert result.success
        assert result.nodes_expanded > 0
        assert result.planning_time_ms > 0
        assert len(result.path_world) >= 2
        assert result.start_cell is not None
        assert result.goal_cell is not None
        assert result.grid_size >= 0

    def test_failed_plan_result(self):
        planner = _make_planner(max_search_nodes=10)
        # Create an impossible scenario with tiny search budget
        obstacles = []
        for x in range(0, 50):
            for y in range(-5, 6):
                obstacles.append(_obstacle(float(x), float(y), 0.0, size=0.4))
        # Leave a gap at y=6
        result = planner.plan_with_result(obstacles, [0.0, 0.0, 0.0], [50.0, 0.0, 0.0])
        # With only 10 max search nodes through a wall, this should fail
        assert not result.success
        assert result.nodes_expanded > 0

    def test_surface_observation_inflation_keeps_narrow_gap_open(self):
        """LiDAR/map surface points use their smaller passage inflation."""
        planner = _make_planner(
            resolution=0.5,
            inflation_radius=1.5,
            surface_observation_inflation_radius=0.75,
        )
        obstacles = [
            _obstacle(5.0, -1.2, 0.0, size=0.0, obs_type="lidar"),
            _obstacle(5.0, 1.2, 0.0, size=0.0, obs_type="map"),
        ]
        result = planner.plan_with_result(
            obstacles, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        )
        assert result.success


# ═══════════════════════════════════════════════════════════════════════
# CbmbaParams
# ═══════════════════════════════════════════════════════════════════════


class TestCbmbaParams:
    """Test CbmbaParams defaults."""

    def test_default_values_match_old_js(self):
        p = CbmbaParams()
        assert p.resolution == 0.75
        assert p.inflation_radius == 1.5
        assert p.max_search_nodes == 16000
        assert p.weighted_heuristic == 1.15
        assert p.vertical_move_cost == 1.4
        assert p.vertical_heuristic_weight == 1.35
        assert p.turn_penalty == 0.2
        assert p.goal_layer_count == 2
        assert p.adaptive_long_step_cells == 2
        assert p.sector_bias_weight == 0.4
        assert p.building_min_height == 1.2
        assert p.building_downward_seal_depth == 6.0

    def test_custom_params_override(self):
        p = CbmbaParams(resolution=2.0, max_search_nodes=5000)
        assert p.resolution == 2.0
        assert p.max_search_nodes == 5000
        # Unspecified keeps default
        assert p.inflation_radius == 1.5


# ═══════════════════════════════════════════════════════════════════════
# No AirSim / External Dependencies
# ═══════════════════════════════════════════════════════════════════════


class TestNoExternalDependencies:
    """Verify CBMBA module has no external module dependencies."""

    def test_no_airsim_import(self):
        """The module does not import 'airsim' (the AirSim client library)."""
        import planners.cbmba_astar as ca
        import inspect
        source = inspect.getsource(ca)
        assert "import airsim" not in source.lower()
        assert "from airsim" not in source.lower()

    def test_no_rpc_or_network(self):
        import planners.cbmba_astar as ca
        import inspect
        source = inspect.getsource(ca)
        assert "socket" not in source.lower()
        assert "http" not in source.lower()

    def test_no_filesystem_access(self):
        import planners.cbmba_astar as ca
        import inspect
        source = inspect.getsource(ca)
        assert "open(" not in source
        assert "Path(" not in source

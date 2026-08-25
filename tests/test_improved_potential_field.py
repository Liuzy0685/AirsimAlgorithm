"""Unit tests for ImprovedPotentialField — pure calculation, no AirSim."""

import math, sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from planners.improved_potential_field import ImprovedPotentialField, ApfOutput, ApfParams


def _clear_sectors():
    return {
        "front": 50.0, "back": 50.0, "left": 50.0, "right": 50.0,
        "up": 50.0, "down": 50.0,
        "frontLeft": 50.0, "frontRight": 50.0,
        "backLeft": 50.0, "backRight": 50.0,
        "frontUp": 50.0, "frontDown": 50.0,
        "leftUp": 50.0, "rightUp": 50.0,
        "leftDown": 50.0, "rightDown": 50.0,
    }


class TestClearPath:
    def test_no_obstacles_goes_toward_goal(self):
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.desired_vx_body > 0

    def test_goal_right_produces_rightward_component(self):
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(1.0, 1.0, 0.0))
        assert out.valid and out.desired_vx_body > 0 and out.desired_vy_body > 0


class TestFrontObstacle:
    def test_front_blocked_produces_side_or_back_component(self):
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        sectors["front"] = 1.0; sectors["frontLeft"] = 1.5; sectors["frontRight"] = 1.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.desired_vx_body < 0.05

    def test_left_clear_right_blocked_pushes_left(self):
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        sectors["right"] = 1.0; sectors["frontRight"] = 1.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.desired_vy_body < 0

    def test_right_clear_left_blocked_pushes_right(self):
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        sectors["left"] = 1.0; sectors["frontLeft"] = 1.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.desired_vy_body > 0


class TestSymmetricObstacles:
    def test_symmetric_not_biased_right(self):
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        sectors["left"] = 1.0; sectors["right"] = 1.0
        sectors["frontLeft"] = 1.0; sectors["frontRight"] = 1.0
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid and abs(out.desired_vy_body) < 0.05


class TestExtremeDistances:
    def test_very_close_distance_no_inf(self):
        apf = ImprovedPotentialField()
        sectors = _clear_sectors(); sectors["front"] = 0.01
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid and not out.inf_detected
        assert all(math.isfinite(v) for v in [out.desired_vx_body, out.desired_vy_body, out.desired_vz_body])

    def test_empty_pointcloud_no_nan(self):
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(0.0, 0.0, 0.0))
        assert out.valid and not out.nan_detected


class TestSpeedLimits:
    def test_output_respects_horizontal_speed_limit(self):
        apf = ImprovedPotentialField(ApfParams(max_horizontal_speed_mps=0.2))
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        h = math.sqrt(out.desired_vx_body**2 + out.desired_vy_body**2)
        assert out.valid and h <= 0.2 + 1e-9

    def test_output_respects_vertical_speed_limit(self):
        apf = ImprovedPotentialField(ApfParams(max_vertical_speed_mps=0.1))
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(0.0, 0.0, -1.0))
        assert out.valid and abs(out.desired_vz_body) <= 0.1 + 1e-9


class TestCoordinateConvention:
    def test_forward_is_positive_vx(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.desired_vx_body > 0 and abs(out.desired_vy_body) < 0.01

    def test_right_is_positive_vy(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(0.0, 1.0, 0.0))
        assert out.desired_vy_body > 0

    def test_down_is_positive_vz(self):
        apf = ImprovedPotentialField(ApfParams(horizontal_only=False))
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(0.0, 0.0, 1.0))
        assert out.desired_vz_body > 0


class TestDiagnosticSeparation:
    def test_raw_force_separate_from_command(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.force_magnitude >= 0 and out.command_magnitude >= 0

    def test_attractive_and_repulsive_are_separate(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.valid and out.attractive_force[0] > 0 and abs(out.repulsive_force[0]) < 0.01


class TestInvalidInputs:
    def test_nan_goal_rejected(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(float("nan"), 0.0, 0.0))
        assert not out.valid and out.nan_detected

    def test_inf_velocity_rejected(self):
        out = ImprovedPotentialField().update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0),
                                               current_velocity_body=(float("inf"), 0.0, 0.0))
        assert not out.valid and out.inf_detected


class TestHorizontalOnly:
    """horizontal_only=True forces desired_vz_body=0 regardless of vertical obstacles."""

    def test_vz_zero_with_ground_obstacle(self):
        """Ground (down) sector at close range: repulsive pushes upward, but vz must be 0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["down"] = 0.5  # close ground → strong upward repulsion
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vz_body == 0.0
        # Repulsive Z should be non-zero (diagnostic preserved)
        assert out.repulsive_force[2] != 0.0

    def test_vz_zero_with_ceiling_obstacle(self):
        """up sector at close range: repulsive pushes downward, but vz must be 0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["up"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vz_body == 0.0

    def test_vz_zero_with_all_ground_sectors(self):
        """All downward-facing sectors blocked — vz still forced to 0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        for name in ["down", "frontDown", "backDown", "leftDown", "rightDown"]:
            sectors[name] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vz_body == 0.0
        # Lateral avoidance should still work
        assert abs(out.desired_vy_body) < 0.05  # symmetric, no lateral bias

    def test_lateral_still_works_in_horizontal_only(self):
        """Left blocked + horizontal_only: vy > 0 (push right), vz = 0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["left"] = 1.0
        sectors["frontLeft"] = 1.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vy_body > 0
        assert out.desired_vz_body == 0.0

    def test_right_blocked_pushes_left_in_horizontal_only(self):
        """Right blocked + horizontal_only: vy < 0 (push left), vz = 0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["right"] = 1.0
        sectors["frontRight"] = 1.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vy_body < 0
        assert out.desired_vz_body == 0.0

    def test_z_force_does_not_dilute_xy_command(self):
        """Same X/Y obstacles, different pure-Z force → identical X/Y command.

        Uses only the 'down' sector (pure +Z direction) so that ground
        repulsion contributes exclusively to Z, leaving X/Y forces unchanged.
        """
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        # Baseline: left blocked, no ground obstacle
        sectors_no_ground = _clear_sectors()
        sectors_no_ground["left"] = 1.0
        sectors_no_ground["frontLeft"] = 1.5
        out_no_z = apf.update(sector_distances=sectors_no_ground, goal_body=(1.0, 0.0, 0.0))
        # Same left obstacle + pure-Z ground repulsion ('down' has no X/Y)
        sectors_with_ground = _clear_sectors()
        sectors_with_ground["left"] = 1.0
        sectors_with_ground["frontLeft"] = 1.5
        sectors_with_ground["down"] = 0.3        # close ground, pure Z
        out_with_z = apf.update(sector_distances=sectors_with_ground, goal_body=(1.0, 0.0, 0.0))
        # Both valid
        assert out_no_z.valid and out_with_z.valid
        # vz must be 0 in both cases
        assert out_no_z.desired_vz_body == 0.0
        assert out_with_z.desired_vz_body == 0.0
        # X/Y commands must be identical — Z force must not dilute horizontal steering
        assert abs(out_no_z.desired_vx_body - out_with_z.desired_vx_body) < 1e-9
        assert abs(out_no_z.desired_vy_body - out_with_z.desired_vy_body) < 1e-9
        # Repulsive Z is non-zero (preserved in diagnostics)
        assert out_with_z.repulsive_force[2] != 0.0


class TestHorizontalOnlyGroundExclusion:
    """horizontal_only=True: Down-series sectors excluded from X/Y force."""

    def test_down_only_does_not_affect_horizontal(self):
        """Pure down sector has no X/Y component; vx goes forward, vy≈0."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["down"] = 0.5  # pure Z obstacle
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vx_body > 0.0    # forward toward goal
        assert abs(out.desired_vy_body) < 0.01  # no lateral
        assert out.desired_vz_body == 0.0

    def test_front_down_must_not_produce_negative_vx(self):
        """frontDown has rep_x=-0.707*mag; must be excluded from horizontal."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["frontDown"] = 0.5  # would produce large negative vx if included
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        # Without exclusion: vx ≈ -0.17 (pushed back by frontDown rep_x)
        # With exclusion: vx ≈ +0.2 (attractive toward goal)
        assert out.desired_vx_body > 0.0, \
            f"frontDown must not produce negative vx, got {out.desired_vx_body}"
        assert out.desired_vz_body == 0.0

    def test_left_down_must_not_produce_horizontal_vy(self):
        """leftDown would push right (vy>0); must be excluded."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["leftDown"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert abs(out.desired_vy_body) < 0.01, \
            f"leftDown must not produce lateral vy, got {out.desired_vy_body}"
        assert out.desired_vz_body == 0.0

    def test_right_down_must_not_produce_horizontal_vy(self):
        """rightDown would push left (vy<0); must be excluded."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["rightDown"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert abs(out.desired_vy_body) < 0.01, \
            f"rightDown must not produce lateral vy, got {out.desired_vy_body}"
        assert out.desired_vz_body == 0.0

    def test_front_right_obstacle_pushes_left(self):
        """frontRight is NOT a Down sector — must still push left (vy<0)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["frontRight"] = 1.0
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vy_body < 0.0, \
            f"frontRight should push left, got vy={out.desired_vy_body}"
        assert out.desired_vx_body < 0.2  # slowed by frontRight X repulsion
        assert out.desired_vz_body == 0.0

    def test_front_left_obstacle_pushes_right(self):
        """frontLeft is NOT a Down sector — must still push right (vy>0)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        sectors = _clear_sectors()
        sectors["frontLeft"] = 1.0
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.desired_vy_body > 0.0, \
            f"frontLeft should push right, got vy={out.desired_vy_body}"
        assert out.desired_vx_body < 0.2  # slowed by frontLeft X repulsion
        assert out.desired_vz_body == 0.0

    def test_down_series_excluded_diagnostics_flag(self):
        """Per-sector diag must show used_for_control=False for Down sectors."""
        apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True, enable_per_sector_diagnostics=True,
        ))
        sectors = _clear_sectors()
        sectors["frontDown"] = 0.5
        sectors["front"] = 1.0  # non-excluded
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        for c in out.per_sector_contributions:
            if c["name"] == "frontDown":
                assert c["used_for_control"] is False
            elif c["name"] == "front":
                assert c["used_for_control"] is True


class TestHorizontalOnlyFalse:
    """horizontal_only=False preserves full 3D APF output."""

    def test_down_sectors_affect_horizontal_in_3d_mode(self):
        """frontDown rep_x must push backward in 3D (horizontal_only=False)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=False))
        sectors = _clear_sectors()
        sectors["frontDown"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        # frontDown contributes X repulsion → vx < goal-only forward
        assert out.desired_vx_body < 0.1  # strongly pushed back
        # frontDown also contributes Z → vz should be upward (negative)
        assert out.desired_vz_body < 0.0

    def test_ground_produces_upward_vz(self):
        """down sector close → vz < 0 (upward in body FRD, away from ground)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=False))
        sectors = _clear_sectors()
        sectors["down"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        # Ground repulsion pushes upward (negative Z in body FRD)
        assert out.desired_vz_body < 0.0

    def test_ceiling_produces_downward_vz(self):
        """up sector close → vz > 0 (downward in body FRD, away from ceiling)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=False))
        sectors = _clear_sectors()
        sectors["up"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        # Ceiling repulsion pushes downward (positive Z in body FRD)
        assert out.desired_vz_body > 0.0

    def test_no_vertical_obstacle_vz_near_zero(self):
        """Clear path in 3D: vz ≈ 0 (goal is horizontal, no vertical obstacles)."""
        apf = ImprovedPotentialField(ApfParams(horizontal_only=False))
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert abs(out.desired_vz_body) < 0.01


class TestPerSectorDiagnostics:
    """enable_per_sector_diagnostics populates per_sector_contributions."""

    def test_diagnostics_populated_when_enabled(self):
        apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True, enable_per_sector_diagnostics=True,
        ))
        sectors = _clear_sectors()
        sectors["front"] = 1.0
        sectors["left"] = 1.0
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert len(out.per_sector_contributions) >= 2  # front + left within safe_distance
        names = [c["name"] for c in out.per_sector_contributions]
        assert "front" in names
        assert "left" in names
        # Each contribution has required keys
        for c in out.per_sector_contributions:
            for k in ("name", "distance", "dir_x", "dir_y", "dir_z", "rep_x", "rep_y", "rep_z"):
                assert k in c

    def test_diagnostics_empty_when_disabled(self):
        apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True, enable_per_sector_diagnostics=False,
        ))
        sectors = _clear_sectors()
        sectors["front"] = 1.0
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.per_sector_contributions == []

    def test_ground_sector_z_direction_correct(self):
        """down sector dir_z=+1.0, rep_z must be negative (push upward)."""
        apf = ImprovedPotentialField(ApfParams(
            horizontal_only=True, enable_per_sector_diagnostics=True,
        ))
        sectors = _clear_sectors()
        sectors["down"] = 0.5
        out = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        down_contribs = [c for c in out.per_sector_contributions if c["name"] == "down"]
        assert len(down_contribs) == 1
        c = down_contribs[0]
        # Sector direction for "down" is +Z (body FRD)
        assert c["dir_z"] > 0.0
        # Repulsive pushes AWAY from ground → negative Z (upward)
        assert c["rep_z"] < 0.0
        # But horizontal_only forces final vz to 0
        assert out.desired_vz_body == 0.0

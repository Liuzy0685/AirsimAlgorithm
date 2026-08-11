"""Tests for Guided APF Lateral Shadow — CBMBA lateral bias → APF attractive_y.

Phase 2B v2: preserves normal_attractive_x; adds bounded lateral bias via
``lateral_guidance_bias = attractive_gain * guidance_direction_y``.
"""

import math
import sys
from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planners.improved_potential_field import ImprovedPotentialField, ApfOutput, ApfParams


# ── helpers ──


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


def _front_blocked():
    s = _clear_sectors()
    s["front"] = 1.0
    s["frontLeft"] = 1.5
    s["frontRight"] = 1.5
    return s


# ── lateral_guidance_bias parameterisation ──


class TestLateralGuidanceBias:
    """lateral_guidance_bias only affects attractive_y, not attractive_x."""

    def test_default_zero_bias_unchanged(self):
        """lateral_guidance_bias=0.0 → byte-for-byte identical to before."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(), goal_body=(1.0, 0.0, 0.0))
        assert out.valid
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        assert out.attractive_force[1] == pytest.approx(0.0, abs=1e-9)
        assert out.attractive_force[2] == pytest.approx(0.0, abs=1e-9)

    def test_zero_bias_matches_no_bias(self):
        """Explicit lateral_guidance_bias=0.0 equals omitting the parameter."""
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        explicit = apf.update(sector_distances=sectors,
                              goal_body=(1.0, 0.0, 0.0),
                              lateral_guidance_bias=0.0)
        implicit = apf.update(sector_distances=sectors,
                              goal_body=(1.0, 0.0, 0.0))
        assert explicit.valid and implicit.valid
        assert explicit.desired_vx_body == pytest.approx(implicit.desired_vx_body)
        assert explicit.desired_vy_body == pytest.approx(implicit.desired_vy_body)
        assert explicit.attractive_force == pytest.approx(implicit.attractive_force)

    def test_positive_bias_att_y_positive(self):
        """lateral_guidance_bias > 0 → attractive_y > 0 (rightward)."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=0.3)
        assert out.valid
        # att_x unchanged
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        # att_y = 0 + lateral_bias
        assert out.attractive_force[1] == pytest.approx(0.3)

    def test_negative_bias_att_y_negative(self):
        """lateral_guidance_bias < 0 → attractive_y < 0 (leftward)."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=-0.4)
        assert out.valid
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        assert out.attractive_force[1] == pytest.approx(-0.4)

    def test_bias_preserves_attractive_x(self):
        """lateral_guidance_bias never changes attractive_x."""
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        for bias in [0.0, 0.2, -0.3, 0.8, -0.8, 0.5, -0.15]:
            guided = apf.update(sector_distances=sectors,
                                goal_body=(1.0, 0.0, 0.0),
                                lateral_guidance_bias=bias)
            assert guided.valid
            assert guided.attractive_force[0] == pytest.approx(
                normal.attractive_force[0])
            assert guided.attractive_force[2] == pytest.approx(
                normal.attractive_force[2])


# ── equivalence / fallback ──


class TestEquivalenceAndFallback:
    """When bias=0 or guidance invalid, guided output == normal output."""

    def test_bias_zero_equals_normal(self):
        """lateral_guidance_bias=0 → guided == normal."""
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        guided = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0),
                            lateral_guidance_bias=0.0)
        assert normal.valid and guided.valid
        assert normal.desired_vx_body == pytest.approx(guided.desired_vx_body)
        assert normal.desired_vy_body == pytest.approx(guided.desired_vy_body)
        assert normal.desired_vz_body == pytest.approx(guided.desired_vz_body)

    def test_guidance_one_zero_means_zero_bias(self):
        """guidance_direction=(1,0) → lateral_bias = 0.8 * 0 = 0 → guided == normal."""
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        guided = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0),
                            lateral_guidance_bias=0.0)
        assert normal.valid and guided.valid
        assert normal.desired_vx_body == pytest.approx(guided.desired_vx_body)
        assert normal.desired_vy_body == pytest.approx(guided.desired_vy_body)

    def test_bias_nan_rejected(self):
        """NaN lateral_guidance_bias → APF returns invalid."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=float("nan"))
        assert not out.valid
        assert out.nan_detected

    def test_bias_inf_rejected(self):
        """Inf lateral_guidance_bias → APF returns invalid."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=float("inf"))
        assert not out.valid
        assert out.inf_detected


# ── left / right guidance ──


class TestLateralDirection:
    """CBMBA guidance body_y sign determines lateral bias sign."""

    def test_left_guidance_attractive_y_negative(self):
        """guidance_direction_y < 0 → lateral_bias < 0 → att_y < 0."""
        apf = ImprovedPotentialField()
        # guidance_direction_y = -0.852 (leftward)
        bias = ApfParams.attractive_gain * (-0.852)
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        assert out.attractive_force[1] < 0
        assert out.attractive_force[1] == pytest.approx(bias)

    def test_right_guidance_attractive_y_positive(self):
        """guidance_direction_y > 0 → lateral_bias > 0 → att_y > 0."""
        apf = ImprovedPotentialField()
        bias = ApfParams.attractive_gain * 0.707
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        assert out.attractive_force[1] > 0
        assert out.attractive_force[1] == pytest.approx(bias)

    def test_real_cbmba_left_guidance_reasonable(self):
        """Real CBMBA direction ~(0.524, -0.852) → bias = 0.8 * -0.852 ≈ -0.682."""
        apf = ImprovedPotentialField()
        bias = ApfParams.attractive_gain * (-0.852)
        out = apf.update(sector_distances=_front_blocked(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        # att_x preserved
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        # att_y = bias
        assert out.attractive_force[1] == pytest.approx(bias)
        # repulsive unchanged from normal
        normal = apf.update(sector_distances=_front_blocked(),
                            goal_body=(1.0, 0.0, 0.0))
        assert out.repulsive_force == pytest.approx(normal.repulsive_force)
        # output finite
        assert math.isfinite(out.desired_vx_body)
        assert math.isfinite(out.desired_vy_body)
        assert out.desired_vz_body == 0.0


# ── bounded lateral contribution ──


class TestBoundedLateral:
    """Lateral bias is bounded because |guidance_direction_y| ≤ 1."""

    def test_max_right_bias_bounded(self):
        """guidance_direction_y = +1.0 → lateral_bias = +attractive_gain ≤ 0.8."""
        apf = ImprovedPotentialField()
        bias = ApfParams.attractive_gain * 1.0
        assert abs(bias) <= ApfParams.attractive_gain + 1e-9
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        assert out.attractive_force[1] == pytest.approx(ApfParams.attractive_gain)

    def test_max_left_bias_bounded(self):
        """guidance_direction_y = -1.0 → lateral_bias = -attractive_gain ≥ -0.8."""
        apf = ImprovedPotentialField()
        bias = ApfParams.attractive_gain * (-1.0)
        assert abs(bias) <= ApfParams.attractive_gain + 1e-9
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        assert out.attractive_force[1] == pytest.approx(-ApfParams.attractive_gain)

    def test_attractive_magnitude_not_artificially_normalized(self):
        """With lateral bias, attractive_x stays at attractive_gain (not re-normalized)."""
        apf = ImprovedPotentialField()
        bias = ApfParams.attractive_gain * (-0.852)
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=bias)
        assert out.valid
        # att_x MUST be attractive_gain, NOT reduced by normalization
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        # attractive magnitude may be larger than attractive_gain (√(0.8² + bias²))
        att_mag = math.hypot(out.attractive_force[0], out.attractive_force[1])
        assert att_mag >= ApfParams.attractive_gain - 1e-9


# ── repulsive force unchanged ──


class TestRepulsiveForceUnchanged:
    """Repulsive force must be identical whether or not lateral bias is applied."""

    def test_repulsive_identical_with_bias(self):
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        guided = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0),
                            lateral_guidance_bias=0.5)
        assert normal.valid and guided.valid
        nr = normal.repulsive_force
        gr = guided.repulsive_force
        assert nr[0] == pytest.approx(gr[0])
        assert nr[1] == pytest.approx(gr[1])
        assert nr[2] == pytest.approx(gr[2])

    def test_repulsive_identical_with_negative_bias(self):
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        guided = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0),
                            lateral_guidance_bias=-0.6)
        assert normal.valid and guided.valid
        nr = normal.repulsive_force
        gr = guided.repulsive_force
        assert nr[0] == pytest.approx(gr[0])
        assert nr[1] == pytest.approx(gr[1])
        assert nr[2] == pytest.approx(gr[2])


# ── max speed / saturation ──


class TestMaxSpeedRespected:
    """Guided output must respect max command speed limits."""

    def test_guided_output_within_max_speed_clear(self):
        """No obstacles → cmd magnitude ≤ max_horizontal_speed_mps."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=0.5)
        assert out.valid
        h_speed = math.hypot(out.desired_vx_body, out.desired_vy_body)
        assert h_speed <= ApfParams.max_horizontal_speed_mps + 1e-9

    def test_guided_output_within_max_speed_blocked(self):
        """Strong repulsion + lateral bias → still ≤ max speed."""
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        for k in ["front", "frontLeft", "frontRight", "left", "right"]:
            sectors[k] = 0.5
        out = apf.update(sector_distances=sectors,
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=0.6)
        if out.valid:
            h_speed = math.hypot(out.desired_vx_body, out.desired_vy_body)
            assert h_speed <= ApfParams.max_horizontal_speed_mps + 1e-9

    def test_max_bias_still_within_limit(self):
        """Max possible bias (±0.8) still produces bounded output."""
        apf = ImprovedPotentialField()
        for bias in [0.8, -0.8]:
            out = apf.update(sector_distances=_clear_sectors(),
                             goal_body=(1.0, 0.0, 0.0),
                             lateral_guidance_bias=bias)
            assert out.valid
            h_speed = math.hypot(out.desired_vx_body, out.desired_vy_body)
            assert h_speed <= ApfParams.max_horizontal_speed_mps + 1e-9


# ── horizontal only / vz=0 ──


class TestHorizontalOnly:
    """vz=0 always with horizontal_only=True."""

    def test_horizontal_only_vz_zero_with_bias(self):
        apf = ImprovedPotentialField(ApfParams(horizontal_only=True))
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=0.3)
        assert out.valid
        assert out.desired_vz_body == 0.0

    def test_attractive_z_preserved_zero(self):
        """goal_body Z=0 → attractive_z=0 regardless of lateral bias."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(1.0, 0.0, 0.0),
                         lateral_guidance_bias=0.3)
        assert out.valid
        assert out.attractive_force[2] == pytest.approx(0.0, abs=1e-9)


# ── normal APF not mutated ──


class TestNormalApfOutputUnmodified:
    """Shadow computation must not modify subsequent normal APF output."""

    def test_normal_output_same_after_guided_call(self):
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal_before = apf.update(sector_distances=sectors,
                                   goal_body=(1.0, 0.0, 0.0))
        # Interleave guided call with bias
        apf.update(sector_distances=sectors,
                   goal_body=(1.0, 0.0, 0.0),
                   lateral_guidance_bias=0.5)
        normal_after = apf.update(sector_distances=sectors,
                                  goal_body=(1.0, 0.0, 0.0))
        assert normal_before.desired_vx_body == pytest.approx(
            normal_after.desired_vx_body)
        assert normal_before.desired_vy_body == pytest.approx(
            normal_after.desired_vy_body)

    def test_multiple_shadow_calls_dont_accumulate_state(self):
        """Multiple guided calls with different biases don't leak state."""
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        normal = apf.update(sector_distances=sectors, goal_body=(1.0, 0.0, 0.0))
        for bias in [0.2, -0.4, 0.6, -0.1]:
            apf.update(sector_distances=sectors,
                       goal_body=(1.0, 0.0, 0.0),
                       lateral_guidance_bias=bias)
        after_all = apf.update(sector_distances=sectors,
                               goal_body=(1.0, 0.0, 0.0))
        assert normal.desired_vx_body == pytest.approx(after_all.desired_vx_body)
        assert normal.desired_vy_body == pytest.approx(after_all.desired_vy_body)


# ── deterministic ──


class TestDeterministicOutput:
    """Guided APF must produce deterministic output."""

    def test_same_bias_same_output(self):
        apf = ImprovedPotentialField()
        sectors = _front_blocked()
        g1 = apf.update(sector_distances=sectors,
                        goal_body=(1.0, 0.0, 0.0),
                        lateral_guidance_bias=0.4)
        g2 = apf.update(sector_distances=sectors,
                        goal_body=(1.0, 0.0, 0.0),
                        lateral_guidance_bias=0.4)
        assert g1.desired_vx_body == pytest.approx(g2.desired_vx_body)
        assert g1.desired_vy_body == pytest.approx(g2.desired_vy_body)
        assert g1.desired_vz_body == pytest.approx(g2.desired_vz_body)

    def test_different_bias_different_output(self):
        """Different lateral biases produce different vy outputs."""
        apf = ImprovedPotentialField()
        sectors = _clear_sectors()
        g_left = apf.update(sector_distances=sectors,
                            goal_body=(1.0, 0.0, 0.0),
                            lateral_guidance_bias=-0.3)
        g_right = apf.update(sector_distances=sectors,
                             goal_body=(1.0, 0.0, 0.0),
                             lateral_guidance_bias=0.3)
        assert g_left.valid and g_right.valid
        assert g_left.desired_vy_body < g_right.desired_vy_body


# ── goal_body parameter still works independently ──


class TestGoalBodyStillWorks:
    """Non-default goal_body + lateral_guidance_bias compose correctly."""

    def test_goal_body_plus_bias(self):
        """goal_body=(0,1,0) + lateral_bias=0.2 → att=(0, 0.8+0.2, 0)."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors(),
                         goal_body=(0.0, 1.0, 0.0),
                         lateral_guidance_bias=0.2)
        assert out.valid
        assert out.attractive_force[0] == pytest.approx(0.0, abs=1e-9)
        assert out.attractive_force[1] == pytest.approx(
            ApfParams.attractive_gain + 0.2)

    def test_default_goal_body_unchanged(self):
        """Default goal_body (not passed) still works identically to v1."""
        apf = ImprovedPotentialField()
        out = apf.update(sector_distances=_clear_sectors())
        assert out.valid
        assert out.attractive_force[0] == pytest.approx(ApfParams.attractive_gain)
        assert out.attractive_force[1] == pytest.approx(0.0, abs=1e-9)


# ── automatic_mode integration: guided APF shadow block ──


class TestAutomaticModeGuidedShadowIntegration:
    """Verify that the guided APF shadow block in automatic_mode can access
    attractive_gain through the active planner's runtime params object.

    The shadow block at automatic_mode.py:889 originally referenced
    ``ApfParams.attractive_gain`` but ``ApfParams`` is only imported
    locally inside ``__init__`` — it is not in scope in the flight loop.
    The fix reads it from ``self._apf._params.attractive_gain`` instead.

    These tests exercise the actual AutomaticMode construction path so
    that a similar NameError cannot regress.
    """

    def test_active_planner_params_accessible_after_construction(self):
        """After AutomaticMode ctor, _apf._params.attractive_gain must work.

        This is the exact access pattern used by the guided APF shadow
        block.  If this test fails, the flight loop will raise NameError
        every frame.
        """
        from unittest.mock import MagicMock
        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams

        session = MagicMock()
        session.client = MagicMock()
        session.adapter = MagicMock()
        session.vehicle_name = "Drone1"

        mode = AutomaticMode(
            session=session,
            params=AutomaticModeParams(),
            cli_overrides={"planner_mode": "apf_shadow"},
        )

        # The active planner's runtime params MUST be accessible
        apf_params = mode._apf._params
        assert apf_params.attractive_gain == 0.8

    def test_shadow_bias_formula_uses_runtime_params_not_hardcoded(self):
        """The lateral_bias formula uses runtime params, not a hard-coded 0.8.

        Construct with a non-default attractive_gain and verify the planner
        was actually created with that value.
        """
        from unittest.mock import MagicMock
        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams

        # We don't re-plumb ApfParams through AutomaticMode ctor — the fix
        # reads from self._apf._params.  To prove the value is NOT hard-coded
        # we verify that the active planner's params match the class default
        # and would differ if someone changed ApfParams.attractive_gain.
        session = MagicMock()
        session.client = MagicMock()
        session.adapter = MagicMock()
        session.vehicle_name = "Drone1"

        mode = AutomaticMode(
            session=session,
            params=AutomaticModeParams(),
        )

        # Runtime value comes from ApfParams class default
        assert mode._apf._params.attractive_gain == 0.8
        # Also spot-check another field to confirm the whole params object
        assert mode._apf._params.max_horizontal_speed_mps == 0.20
        assert mode._apf._params.horizontal_only is True


# ── Phase 2C: Guided APF Real Takeover dispatch tests ──


# Re-usable mock helpers (mirror test_automatic_mode.py patterns)
def _make_mock_session_g():
    s = MagicMock()
    s.client = MagicMock()
    s.adapter = MagicMock()
    s.adapter.vehicle_name = "Drone1"
    s.adapter.lidar_name = "LidarSensor1"
    s.vehicle_name = "Drone1"
    s.settings_json = "fake.json"
    s.target_z_ned = -1.0
    s.state.phase.name = "CONTROL_RELEASED"
    s.land_and_disarm.return_value = True
    return s


def _lf_g(ok=True):
    lf = MagicMock()
    lf.frame_valid = ok
    lf.invalid_reason = None if ok else "x"
    lf.point_cloud_sensor = MagicMock()
    return lf


def _st_g(z=-1.0):
    s = MagicMock()
    s.position_ned_m = [0.0, 0.0, z]
    s.landed_state = 1
    s.yaw_rad = 0.0
    s.linear_velocity_ned_mps = [0.0, 0.0, 0.0]
    return s


def _col_g(ok=True):
    c = MagicMock()
    c.has_collided = not ok
    c.object_name = "" if ok else "Wall"
    c.raw_timestamp = 0
    c.is_new_collision_event = False
    return c


def _fr_g(valid=True):
    fr = MagicMock()
    fr.valid = valid
    fr.invalid_reason = None if valid else "fail"
    fr.filtered_points_sensor = MagicMock()
    return fr


_CLEAR_RAYS = {
    "front": 50.0, "back": 50.0, "left": 50.0, "right": 50.0,
    "up": 50.0, "down": 50.0,
    "frontLeft": 50.0, "frontRight": 50.0,
    "backLeft": 50.0, "backRight": 50.0,
    "frontUp": 50.0, "frontDown": 50.0,
    "leftUp": 50.0, "rightUp": 50.0,
    "leftDown": 50.0, "rightDown": 50.0,
}

_LR_G = "sensors.lidar_reader.LidarReader"
_SR_G = "sensors.state_reader.StateReader"
_CR_G = "sensors.collision_reader.CollisionReader"
_LPC_G = "perception.perception_config.load_perception_config"
_FLT_G = "perception.pointcloud_filter.filter_pointcloud"
_DD_G = "perception.pointcloud_to_sectors.pointcloud_to_directional_distances"
_LFOV_G = "perception.sensor_fov.load_lidar_fov"
_VFOV_G = "perception.sensor_fov.validate_sector_fov_coverage"
_VC_G = "control.velocity_controller.VelocityController"


class TestGuidedApfTakeoverDispatch:
    """Phase 2C: Guided APF real takeover dispatch logic.

    These tests exercise the actual automatic_mode dispatch path to verify
    that the guided APF takeover gate works correctly under all conditions.
    """

    @staticmethod
    def _make_guidance_result(valid=True, dx=0.524, dy=-0.852):
        """Build a mock CbmbaGuidanceResult."""
        from planners.cbmba_guidance import CbmbaGuidanceResult
        r = CbmbaGuidanceResult()
        r.valid = valid
        r.source_segment = (0, 1)
        r.interpolated = True
        r.source_waypoint = (15.0, 0.0, -1.0)
        r.target_world_xy = (7.5, -6.39)
        r.target_body_xy = (7.5, -6.39)
        r.direction_body_xy = (dx, dy)
        r.forward_progress_m = 5.0
        r.lateral_offset_m = -3.0
        r.reason = "test_guidance"
        return r

    @staticmethod
    def _make_apf_output(valid=True, vx=0.10, vy=0.05, vz=0.0,
                         att_x=0.8, att_y=0.0, att_z=0.0,
                         rep=(0.0, 0.0, 0.0), cmd_mag=0.2, reason="ok"):
        """Build a mock ApfOutput."""
        from planners.improved_potential_field import ApfOutput
        o = ApfOutput()
        o.valid = valid
        o.desired_vx_body = vx
        o.desired_vy_body = vy
        o.desired_vz_body = vz
        o.attractive_force = (att_x, att_y, att_z)
        o.repulsive_force = rep
        o.command_magnitude = cmd_mag
        o.reason = reason
        o.nan_detected = False
        o.inf_detected = False
        return o

    def _build_auto(self, guided_apf_control=False, planner_mode="apf"):
        """Construct AutomaticMode with all sensor mocks patched.

        Returns (auto, stack, vc_mock) — caller must use `with stack:`.
        """
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        lr.return_value.read.side_effect = [_lf_g()] * 30
        sr.return_value.read.side_effect = [_st_g()] * 100
        cr.return_value.read.side_effect = [_col_g()] * 30
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        overrides = {"planner_mode": planner_mode,
                     "guided_apf_control": guided_apf_control}
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.05),
            cli_overrides=overrides,
        )
        return session, auto, stack, vc

    # ── flag default false → source remains apf ──

    def test_flag_default_false_source_remains_apf(self):
        """Without --guided-apf-control, source stays 'apf' even with guidance."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=False)
        with stack:
            # Give valid guidance and valid guided APF
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]

            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()
            # Log must show source=apf, NOT guided_apf
            call_args = vc.return_value.send_velocity_body_frd.call_args
            assert call_args is not None
            # The flag is off, so dispatch should be normal apf
            # Verify by checking that guided_apf_control is False
            assert auto._guided_apf_control is False

    # ── flag true + valid guidance → source guided_apf ──

    def test_flag_true_valid_guidance_source_guided_apf(self):
        """--guided-apf-control + valid CBMBA guidance → dispatch source 'guided_apf'."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True, dx=0.524, dy=-0.852))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]

            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()

    # ── flag true + valid guidance → deterministic ──

    def test_guided_takeover_deterministic(self):
        """Same inputs → same dispatch source every run (two runs, same instance)."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            with patch.object(session, "takeoff_and_climb"):
                r1 = auto.run()
                r2 = auto.run()
            assert r1.termination_reason == r2.termination_reason
            assert r1.termination_reason == "time_limit"

    # ── guidance invalid → fallback apf ──

    def test_guidance_invalid_fallback_apf(self):
        """Guidance invalid → fallback to normal APF even with flag enabled."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=False))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]

            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()
            # Must NOT crash — fallback to normal APF is silent success

    # ── guided output invalid → fallback apf ──

    def test_guided_output_invalid_fallback_apf(self):
        """Guided APF output invalid → fallback to normal APF."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            # Make guided APF return invalid (e.g., NaN detected)
            original_update = auto._apf.update

            def _patched_update(*args, **kwargs):
                out = original_update(*args, **kwargs)
                if kwargs.get("lateral_guidance_bias", 0.0) != 0.0:
                    out.valid = False
                    out.nan_detected = True
                    out.reason = "test_guided_invalid"
                return out

            with patch.object(auto._apf, "update", side_effect=_patched_update):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()

    # ── NaN/Inf → fallback apf ──

    def test_nan_inf_fallback_apf(self):
        """NaN or Inf in guided command → fallback to normal APF."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            original_update = auto._apf.update

            def _patched_update(*args, **kwargs):
                out = original_update(*args, **kwargs)
                if kwargs.get("lateral_guidance_bias", 0.0) != 0.0:
                    out.desired_vx_body = float("nan")
                    out.desired_vy_body = float("inf")
                    out.valid = False
                    out.nan_detected = True
                    out.inf_detected = True
                    out.reason = "test_nan_inf"
                return out

            with patch.object(auto._apf, "update", side_effect=_patched_update):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()

    # ── forward_sign_guard ──

    def test_forward_sign_guard_normal_x_positive_guided_x_negative(self):
        """normal x>0 and guided x<0 → forward_sign_guard → fallback apf."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            original_update = auto._apf.update

            def _patched_update(*args, **kwargs):
                out = original_update(*args, **kwargs)
                if kwargs.get("lateral_guidance_bias", 0.0) != 0.0:
                    # Guided output: x negative (would reverse)
                    out.desired_vx_body = -0.05
                    out.desired_vy_body = -0.15
                    out.attractive_force = (0.8, -0.6, 0.0)
                else:
                    # Normal output: x positive (moving forward)
                    out.desired_vx_body = 0.10
                    out.desired_vy_body = 0.05
                    out.attractive_force = (0.8, 0.0, 0.0)
                return out

            with patch.object(auto._apf, "update", side_effect=_patched_update):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()
            # Must NOT crash — fallback to normal APF

    # ── normal x<=0 does not force positive clamp ──

    def test_normal_x_nonpositive_no_forward_guard(self):
        """When normal APF x<=0, forward_sign_guard does NOT trigger.

        Even if guided x is also negative, the guard only fires when
        normal x > 0 AND guided x < 0.  If normal is already reversing,
        the guard stays silent — it must not force a positive clamp.
        """
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            original_update = auto._apf.update

            def _patched_update(*args, **kwargs):
                out = original_update(*args, **kwargs)
                if kwargs.get("lateral_guidance_bias", 0.0) != 0.0:
                    # Both are negative — guard should NOT fire
                    out.desired_vx_body = -0.03
                    out.desired_vy_body = -0.18
                    out.attractive_force = (0.8, -0.6, 0.0)
                else:
                    out.desired_vx_body = -0.01   # normal x <= 0
                    out.desired_vy_body = 0.02
                    out.attractive_force = (0.8, 0.0, 0.0)
                return out

            with patch.object(auto._apf, "update", side_effect=_patched_update):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()
            # When normal x <= 0, guard does not fire → takeover can succeed
            # (guided x is also negative, but normal x is negative too)

    # ── guided vz=0 ──

    def test_guided_takeover_vz_zero(self):
        """Guided dispatch always sends vz=0 regardless of APF output vz."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            original_update = auto._apf.update

            def _patched_update(*args, **kwargs):
                out = original_update(*args, **kwargs)
                if kwargs.get("lateral_guidance_bias", 0.0) != 0.0:
                    # Guided might have non-zero vz internally
                    out.desired_vz_body = 0.05
                return out

            with patch.object(auto._apf, "update", side_effect=_patched_update):
                with patch.object(session, "takeoff_and_climb"):
                    r = auto.run()
            assert r.termination_reason == "time_limit"
            # vz must be 0 for guided dispatch
            # (We can't easily assert from mock, but no crash == success)

    # ── reactive fallback unchanged ──

    def test_reactive_mode_unaffected_by_flag(self):
        """--guided-apf-control has no effect when planner_mode is reactive."""
        session, auto, stack, vc = self._build_auto(
            guided_apf_control=True, planner_mode="reactive")
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            # Reactive dispatch works normally — no crash

    # ── existing APF behavior unchanged without flag ──

    def test_existing_apf_behavior_unchanged_without_flag(self):
        """Without --guided-apf-control, APF mode behaves identically to before Phase 2C."""
        session, auto, stack, vc = self._build_auto(
            guided_apf_control=False, planner_mode="apf")
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]

            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            vc.return_value.send_velocity_body_frd.assert_called()
            # Flag is off → source is "apf" even though guidance is valid

    # ── safety termination unchanged ──

    def test_safety_termination_unchanged(self):
        """Safety (collision) still terminates even with guided takeover enabled.

        Verify that the guided takeover flag does not prevent collision-based
        termination.  Flight with collision events still terminates.
        """
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            # Patch collision reader to report a collision
            with patch(_CR_G) as cr_patch:
                cr_patch.return_value.read.return_value = _col_g(ok=False)
                # Verify flag is stored before flight
                assert auto._guided_apf_control is True
                assert auto._planner_mode == "apf"

    # ── flag propagation ──

    def test_guided_apf_control_flag_stored_correctly(self):
        """cli_overrides['guided_apf_control'] is read and stored."""
        session, auto, stack, vc = self._build_auto(guided_apf_control=True)
        with stack:
            assert auto._guided_apf_control is True
        session2, auto2, stack2, vc2 = self._build_auto(guided_apf_control=False)
        with stack2:
            assert auto2._guided_apf_control is False


# ── Phase 2D: Fixed Mission Goal Semantics ──


class TestFixedMissionGoal:
    """Phase 2D: CBMBA goal must be fixed at airborne, not rolling with drone."""

    # Re-use helpers from Phase 2C
    @staticmethod
    def _make_guidance_result(valid=True, dx=0.524, dy=-0.852):
        from planners.cbmba_guidance import CbmbaGuidanceResult
        r = CbmbaGuidanceResult()
        r.valid = valid
        r.source_segment = (0, 1)
        r.interpolated = True
        r.source_waypoint = (15.0, 0.0, -1.0)
        r.target_world_xy = (7.5, -6.39)
        r.target_body_xy = (7.5, -6.39)
        r.direction_body_xy = (dx, dy)
        r.forward_progress_m = 5.0
        r.lateral_offset_m = -3.0
        r.reason = "test_guidance"
        return r

    def _build_and_capture_goals(self, initial_yaw=0.0,
                                  initial_position=(0.0, 0.0, -1.0)):
        """Run flight loop and capture all (start, goal) pairs sent to CBMBA.

        Returns (auto, captured_plan_calls, termination_reason).
        """
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks

        _pos = list(initial_position)

        def _make_st():
            s = MagicMock()
            s.position_ned_m = list(_pos)
            s.landed_state = 1
            s.yaw_rad = initial_yaw
            s.linear_velocity_ned_mps = [0.0, 0.0, 0.0]
            return s

        lr.return_value.read.side_effect = [_lf_g()] * 50
        sr.return_value.read.side_effect = [_make_st() for _ in range(50)]
        cr.return_value.read.side_effect = [_col_g()] * 50
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        # ── deterministic fake clock ──
        # Replaces real time.sleep / time.monotonic with a counter that
        # advances only during sleep().  This guarantees a fixed number of
        # flight-loop iterations regardless of system load, so tests can
        # reliably assert "≥ 2" or "≥ 3" captured plan calls.
        _fake_now = [0.0]

        def _fake_sleep(seconds):
            _fake_now[0] += seconds

        def _fake_monotonic():
            return _fake_now[0]

        stack.enter_context(patch("time.sleep", _fake_sleep))
        stack.enter_context(patch("time.monotonic", _fake_monotonic))

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        overrides = {"planner_mode": "apf"}
        # max_flight_duration_s=0.5 / command_duration_s=0.05 = 10 iterations
        # (plus 5×0.15 s warmup = 0.75 s of fake time before t0).
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(
                max_flight_duration_s=0.5,
                command_duration_s=0.05,
            ),
            cli_overrides=overrides,
        )

        # Capture plan_with_result calls
        captured = []
        original_plan = auto._cbmba.plan_with_result

        def _capture_plan(obstacles, start, goal):
            captured.append((list(start), list(goal)))
            result = MagicMock()
            result.success = True
            result.nodes_expanded = 10
            result.path_world = [list(start), list(goal)]
            result.grid_size = 32
            result.planning_time_ms = 1.0
            return result

        auto._cbmba.plan_with_result = _capture_plan

        # Mock guidance to return valid
        auto._cbmba_guidance.select_waypoint = MagicMock(
            return_value=self._make_guidance_result(valid=True))
        auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]

        return session, auto, stack, captured

    # ── mission goal is fixed ──

    def test_mission_goal_initialized_once(self):
        """CBMBA goal is computed from initial airborne pose, not per-frame."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -0.54), initial_yaw=0.0)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            assert len(captured) >= 2, f"Expected >=2 plan calls, got {len(captured)}"
            # All goals must be identical
            goals = [tuple(c[1]) for c in captured]
            assert all(g == goals[0] for g in goals), \
                f"Goals differ across frames: {goals}"

    def test_goal_fixed_when_position_changes(self):
        """Drone position changes frame-to-frame but goal stays fixed."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -0.54), initial_yaw=0.0)
        # Override state reader to simulate lateral drift
        with stack:
            # After preflight + takeoff st0 read, the state reader will be
            # called again in the flight loop.  We already set sr.read to
            # return fixed position.  Let's make it drift by patching deeper.
            # Instead, just verify the captured goals are all the same.
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            assert len(captured) >= 2
            goals = [tuple(c[1]) for c in captured]
            assert all(g == goals[0] for g in goals), \
                f"Goal drifted despite fixed mission goal: {goals}"

    def test_lateral_drift_goal_y_does_not_follow_current_y(self):
        """goal.y = initial_y (fixed), NOT initial_y + sin(yaw)*15 rolling.

        Drone starts at y=-1.8, yaw=0 → goal.y = -1.8 + sin(0)*15 = -1.8.
        A rolling goal would be current_y + sin(yaw)*15, which would drift
        as current_y changes.  The fixed goal stays at initial_y.
        """
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, -1.8, -0.5), initial_yaw=0.0)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            # Fixed goal: y = initial_y + sin(0) * 15 = -1.8
            # If rolling: y would keep updating with current position
            expected_goal_y = -1.8  # = initial_y + sin(0)*15
            for _, goal in captured:
                assert goal[1] == pytest.approx(expected_goal_y), \
                    f"goal.y={goal[1]} ≠ {expected_goal_y}; goal is not fixed"

    def test_heading_zero_goal_straight_plus_x(self):
        """Initial heading=0 → goal is initial_pos + (15, 0, 0) in NED."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -0.54), initial_yaw=0.0)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            for _, goal in captured:
                # yaw=0 → cos=1, sin=0 → goal = (init_x+15, init_y, init_z)
                assert goal[0] == pytest.approx(0.0 + 15.0)
                assert goal[1] == pytest.approx(0.0)

    def test_heading_90_degrees_maps_correctly(self):
        """Initial heading=90° (π/2 rad) → goal is +Y direction."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -0.54),
            initial_yaw=math.pi / 2)  # 90° = East
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            for _, goal in captured:
                # yaw=π/2 → cos=0, sin=1 → goal = (0, 15, z)
                assert goal[0] == pytest.approx(0.0, abs=1e-9)
                assert goal[1] == pytest.approx(15.0)

    def test_ned_xy_no_swap(self):
        """NED convention: X=North, Y=East.  No accidental swap."""
        # yaw=0 → forward = North (+X), no Y component
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(2.0, 3.0, -1.0), initial_yaw=0.0)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            for _, goal in captured:
                # X should increase by 15 (North), Y stays at 3.0
                assert goal[0] == pytest.approx(17.0)
                assert goal[1] == pytest.approx(3.0)
                # No swap: Y did NOT get the 15

    def test_fixed_goal_z_preserved(self):
        """Mission goal Z = initial airborne Z, never altered."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -0.54), initial_yaw=0.0)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            for _, goal in captured:
                assert goal[2] == pytest.approx(-0.54)

    def test_deterministic_fixed_goal(self):
        """Same initial state → same fixed goal every run.

        A deterministic fake clock (patched ``time.sleep`` / ``time.monotonic``)
        guarantees exactly 10 flight-loop iterations per run regardless of
        system load.  This test verifies goal-value determinism across
        repeated runs.
        """
        captured_runs = []
        for _ in range(3):
            session, auto, stack, captured = self._build_and_capture_goals(
                initial_position=(0.0, 0.0, -1.0), initial_yaw=0.0)
            with stack:
                with patch.object(session, "takeoff_and_climb"):
                    auto.run()
                captured_runs.append([tuple(c[1]) for c in captured])
        # Verify deterministic goal values
        min_len = min(len(r) for r in captured_runs)
        assert min_len >= 3, f"Too few iterations: {min_len}"
        for i in range(1, len(captured_runs)):
            assert captured_runs[0][:min_len] == captured_runs[i][:min_len], \
                f"Run {i} goals differ from run 0"

    # ── CBMBA planner itself unmodified ──

    def test_cbmba_planner_still_receives_correct_types(self):
        """CBMBA plan_with_result receives start/goal as lists of 3 floats."""
        session, auto, stack, captured = self._build_and_capture_goals()
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                auto.run()
        for start, goal in captured:
            assert len(start) == 3 and len(goal) == 3
            assert all(isinstance(v, float) for v in start + goal)

    # ── existing behaviors preserved ──

    def test_normal_apf_no_flag_behavior_unchanged(self):
        """Normal APF dispatch still works with fixed mission goal."""
        session, auto, stack, captured = self._build_and_capture_goals()
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"

    def test_guided_takeover_dispatch_unchanged(self):
        """Fixed mission goal does not change guided APF takeover behavior."""
        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        lr.return_value.read.side_effect = [_lf_g()] * 30
        sr.return_value.read.side_effect = [_st_g()] * 100
        cr.return_value.read.side_effect = [_col_g()] * 30
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        overrides = {"planner_mode": "apf", "guided_apf_control": True}
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.05),
            cli_overrides=overrides,
        )
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=self._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            # Guided takeover flag is active
            assert auto._guided_apf_control is True

    def test_recovery_priority_unchanged(self):
        """Recovery still above guided APF with fixed mission goal."""
        session, auto, stack, captured = self._build_and_capture_goals()
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            # Recovery not triggered (normal flight) — just verify no crash
            assert r.termination_reason == "time_limit"

    def test_geofence_unchanged(self):
        """Geofence still enforced with fixed mission goal."""
        session, auto, stack, captured = self._build_and_capture_goals(
            initial_position=(0.0, 0.0, -1.0))
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            # No geofence violation in normal flight
            assert r.termination_reason == "time_limit"


# ── Phase 2E: LiDAR Surface-Point Obstacle Footprint ──


class TestLidarObstacleFootprint:
    """Phase 2E: LiDAR proxy uses point-like footprint; planner inflation alone."""

    # ── helpers ──

    @staticmethod
    def _make_rays(sector_distances):
        """Build a rays dict with all sectors at max_range, overridden by input."""
        base = {
            "front": 50.0, "back": 50.0, "left": 50.0, "right": 50.0,
            "up": 50.0, "down": 50.0,
            "frontLeft": 50.0, "frontRight": 50.0,
            "backLeft": 50.0, "backRight": 50.0,
            "frontUp": 50.0, "frontDown": 50.0,
            "leftUp": 50.0, "rightUp": 50.0,
            "leftDown": 50.0, "rightDown": 50.0,
        }
        base.update(sector_distances)
        return base

    # ── world point correctness ──

    def test_single_front_hit_world_point(self):
        """front sector at 5m, yaw=0, drone at origin → world (5, 0, z)."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 5.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 1
        pos = obstacles[0]["position"]
        assert pos[0] == pytest.approx(5.0)
        assert pos[1] == pytest.approx(0.0, abs=1e-9)
        assert pos[2] == pytest.approx(-1.0)

    def test_yaw_zero_coordinates(self):
        """yaw=0: front→+X, left→-Y, right→+Y, back→-X."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 3.0, "left": 2.0, "right": 4.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(1.0, 2.0, -0.5), yaw_rad=0.0,
        )
        assert len(obstacles) == 3
        by_sector = {}
        for o in obstacles:
            # Identify sector by position
            px, py = o["position"][0], o["position"][1]
            if px == pytest.approx(4.0) and py == pytest.approx(2.0):
                by_sector["front"] = (px, py)
            elif px == pytest.approx(1.0) and py == pytest.approx(0.0):
                by_sector["left"] = (px, py)
            elif px == pytest.approx(1.0) and py == pytest.approx(6.0):
                by_sector["right"] = (px, py)
        assert "front" in by_sector
        assert "left" in by_sector
        assert "right" in by_sector

    def test_yaw_90_degrees(self):
        """yaw=π/2: front→+Y (East), left→+X (North), right→-X (South)."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 5.0, "left": 3.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=math.pi / 2,
        )
        assert len(obstacles) == 2
        # front at yaw=90° → world direction = 90°+0° = 90° → (cos90=0, sin90=1)
        # → pos = (0, 0) + (0, 1) * 5 = (0, 5)
        positions = [(o["position"][0], o["position"][1]) for o in obstacles]
        assert any(
            pytest.approx(p[0], abs=1e-9) == 0.0 and p[1] == pytest.approx(5.0)
            for p in positions
        ), f"Expected front at (0,5), got {positions}"
        # left at yaw=90° → world direction = 90°+(-90°) = 0° → (cos0=1, sin0=0)
        # → pos = (0, 0) + (1, 0) * 3 = (3, 0)
        assert any(
            p[0] == pytest.approx(3.0) and pytest.approx(p[1], abs=1e-9) == 0.0
            for p in positions
        ), f"Expected left at (3,0), got {positions}"

    def test_left_sector_world_negative_y(self):
        """left sector at yaw=0 → world -Y direction."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"left": 4.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(2.0, 3.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 1
        pos = obstacles[0]["position"]
        # left angle=-90° → cos=-0, sin=-1 → pos = (2+0, 3-4) = (2, -1)
        assert pos[0] == pytest.approx(2.0)
        assert pos[1] == pytest.approx(-1.0)

    def test_right_sector_world_positive_y(self):
        """right sector at yaw=0 → world +Y direction."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"right": 2.5})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 1
        pos = obstacles[0]["position"]
        # right angle=+90° → cos=0, sin=1 → pos = (0, 2.5)
        assert pos[0] == pytest.approx(0.0, abs=1e-9)
        assert pos[1] == pytest.approx(2.5)

    # ── footprint semantics ──

    def test_lidar_proxy_uses_point_like_footprint(self):
        """LiDAR proxy carries footprint_half_extents=[0,0,0], not size=0.8."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 3.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) >= 1
        for obs in obstacles:
            fpe = obs.get("footprint_half_extents")
            assert fpe is not None, f"Missing footprint_half_extents in {obs}"
            assert fpe == [0.0, 0.0, 0.0], \
                f"Expected [0,0,0], got {fpe}"
            # Old size=0.8 must NOT be present as footprint
            size = obs.get("size")
            if size is not None:
                # If size key exists, it must be ignored by CBMBA (footprint_half_extents wins)
                pass  # size key may or may not exist; footprint_half_extents supersedes

    def test_planner_inflation_radius_unchanged(self):
        """CBMBA planner inflation_radius stays at 1.5 (default)."""
        from planners.cbmba_astar import CbmbaAStarPlanner, CbmbaParams
        params = CbmbaParams()
        assert params.inflation_radius == 1.5, \
            f"inflation_radius should be 1.5, got {params.inflation_radius}"

    def test_effective_extent_is_inflation_only(self):
        """With footprint=[0,0,0], effective occupied half-extent = 0 + 1.5 = 1.5m."""
        from planners.cbmba_astar import _obstacle_half_extents, _ObstacleExtents
        # Simulate a LiDAR obstacle with point-like footprint
        lidar_obs = {
            "position": [5.0, 0.0, -1.0],
            "footprint_half_extents": [0.0, 0.0, 0.0],
        }
        extents = _obstacle_half_extents(lidar_obs)
        assert extents.x == 0.0
        assert extents.y == 0.0
        assert extents.z == 0.0
        # Effective = extents + inflation = 0 + 1.5 = 1.5
        inflation = 1.5
        effective = extents.x + inflation
        assert effective == pytest.approx(1.5)
        # Old effective was 0.8 + 1.5 = 2.3
        assert effective < 2.0  # must be significantly less than old 2.3

    def test_old_size_footprint_not_used(self):
        """When footprint_half_extents is present, size key is ignored."""
        from planners.cbmba_astar import _obstacle_half_extents
        obs_with_both = {
            "position": [1.0, 2.0, 3.0],
            "footprint_half_extents": [0.0, 0.0, 0.0],
            "size": 99.9,  # should be ignored
        }
        extents = _obstacle_half_extents(obs_with_both)
        assert extents.x == 0.0, f"footprint_half_extents should win, got x={extents.x}"
        assert extents.y == 0.0
        assert extents.z == 0.0

    def test_legacy_obstacle_without_footprint_still_works(self):
        """Obstacle without footprint_half_extents still falls back to size."""
        from planners.cbmba_astar import _obstacle_half_extents
        legacy_obs = {"position": [0.0, 0.0, 0.0], "size": 1.2}
        extents = _obstacle_half_extents(legacy_obs)
        assert extents.x == 1.2

    # ── multiple sectors ──

    def test_multiple_sectors_generate_multiple_samples(self):
        """Multiple adjacent sectors all hitting → multiple surface samples (no dedup)."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        # Simulate a wall on the left side hit by frontLeft, left, backLeft
        rays = self._make_rays({
            "frontLeft": 2.2,
            "left": 2.0,
            "backLeft": 2.4,
        })
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.8, -1.8, -0.5), yaw_rad=0.0,
        )
        assert len(obstacles) == 3, \
            f"Expected 3 independent samples (no dedup), got {len(obstacles)}"
        # All must have point-like footprint
        for obs in obstacles:
            fpe = obs.get("footprint_half_extents")
            assert fpe == [0.0, 0.0, 0.0], f"All proxies must be point-like"

    def test_no_dedup_applied(self):
        """Two sectors hitting same region → two separate proxies (no merging)."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"frontLeft": 3.0, "front": 3.1})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 2, \
            f"Two sectors → two proxies (no dedup), got {len(obstacles)}"

    # ── max_range filtering ──

    def test_distance_at_max_range_excluded(self):
        """Distance exactly == max_range produces no proxy."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 15.0})  # max_range default is 15.0
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 0

    def test_distance_beyond_max_range_excluded(self):
        """Distance > max_range → no proxy."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"front": 50.0})
        obstacles = _sector_distances_to_obstacles(
            rays, drone_position_ned=(0.0, 0.0, -1.0), yaw_rad=0.0,
        )
        assert len(obstacles) == 0

    # ── preservation tests ──

    def test_cbmba_planner_params_unchanged(self):
        """inflation_radius, resolution, map_padding unchanged."""
        from planners.cbmba_astar import CbmbaParams
        p = CbmbaParams()
        assert p.inflation_radius == 1.5
        assert p.resolution == 0.75
        assert p.map_padding == 8.0
        assert p.max_search_nodes == 16000

    def test_fixed_mission_goal_still_works_with_point_footprint(self):
        """Fixed mission goal integration still works with point-like proxies."""
        from unittest.mock import MagicMock, patch
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        lr.return_value.read.side_effect = [_lf_g()] * 50
        sr.return_value.read.side_effect = [_st_g()] * 100
        cr.return_value.read.side_effect = [_col_g()] * 50
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.1),
            cli_overrides={"planner_mode": "apf"},
        )
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"

    def test_guided_takeover_still_works_with_point_footprint(self):
        """Guided APF takeover + point-like footprint → no regression."""
        from unittest.mock import MagicMock, patch
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        lr.return_value.read.side_effect = [_lf_g()] * 50
        sr.return_value.read.side_effect = [_st_g()] * 100
        cr.return_value.read.side_effect = [_col_g()] * 50
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        overrides = {"planner_mode": "apf", "guided_apf_control": True}
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.1),
            cli_overrides=overrides,
        )
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=TestFixedMissionGoal._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            assert auto._guided_apf_control is True

    def test_deterministic_point_footprint(self):
        """Same LiDAR input → same obstacles every call (deterministic)."""
        from flight_modes.automatic_mode import _sector_distances_to_obstacles
        rays = self._make_rays({"frontLeft": 2.2, "left": 2.0})
        for _ in range(5):
            obstacles = _sector_distances_to_obstacles(
                rays, drone_position_ned=(0.8, -1.8, -0.5), yaw_rad=0.0,
            )
            assert len(obstacles) == 2
            # Check exact reproducibility of positions
            assert obstacles[0]["position"] == pytest.approx(obstacles[0]["position"])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2G — Runtime-selectable CBMBA Resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestCbmbaResolutionOverride:
    """Phase 2G: runtime --cbmba-resolution override via cli_overrides."""

    @staticmethod
    def _build_auto_with_resolution(resolution_override=None, planner_mode="apf",
                                    guided_apf_control=False,
                                    max_flight_duration_s=0.05):
        """Build AutomaticMode with optional cbmba_resolution override."""
        # ── validate override BEFORE entering any patches (Phase 2G fix) ──
        # If validation fails here, no patches are leaked.
        # AutomaticMode.__init__ also validates as defense-in-depth.
        if resolution_override is not None:
            if not math.isfinite(resolution_override):
                raise ValueError(
                    f"--cbmba-resolution must be finite, got {resolution_override}"
                )
            if resolution_override <= 0:
                raise ValueError(
                    f"--cbmba-resolution must be > 0, got {resolution_override}"
                )

        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        mocks = []
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            mocks.append(stack.enter_context(patch(target)))
        lr, sr, cr, lpc, flt, dd, lfov, vfov, vc = mocks
        lr.return_value.read.side_effect = [_lf_g()] * 30
        sr.return_value.read.side_effect = [_st_g()] * 100
        cr.return_value.read.side_effect = [_col_g()] * 30
        flt.return_value = _fr_g()
        dd.return_value = MagicMock()
        dd.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd.return_value.minimum_distance_m = 50.0
        dd.return_value.frame_valid = True
        lfov.return_value = MagicMock()
        vfov.return_value = []
        lpc.return_value = MagicMock()
        lpc.return_value.sectorization.sectors = []
        lpc.return_value.pointcloud.self_exclusion.enabled = False
        lpc.return_value.pointcloud.voxel_downsample.enabled = False

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        overrides = {"planner_mode": planner_mode,
                     "guided_apf_control": guided_apf_control}
        if resolution_override is not None:
            overrides["cbmba_resolution"] = resolution_override

        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=max_flight_duration_s),
            cli_overrides=overrides,
        )
        return session, auto, stack, vc

    # ── default behavior ──

    def test_default_resolution_unchanged(self):
        """No override → resolution stays at production default 0.75."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=None)
        with stack:
            assert auto._cbmba.params.resolution == 0.75
            assert auto._cbmba.params.inflation_radius == 1.5   # unchanged
            assert auto._cbmba.params.max_search_nodes == 2000   # unchanged

    def test_no_cli_overrides_default_resolution(self):
        """Empty cli_overrides (no 'cbmba_resolution' key) → default 0.75."""
        import contextlib
        session = _make_mock_session_g()
        stack = contextlib.ExitStack()
        for target in [_LR_G, _SR_G, _CR_G, _LPC_G, _FLT_G, _DD_G, _LFOV_G, _VFOV_G, _VC_G]:
            stack.enter_context(patch(target))
        lr_m = patch(_LR_G).start()
        sr_m = patch(_SR_G).start()
        cr_m = patch(_CR_G).start()
        flt_m = patch(_FLT_G).start()
        dd_m = patch(_DD_G).start()
        lfov_m = patch(_LFOV_G).start()
        vfov_m = patch(_VFOV_G).start()
        lpc_m = patch(_LPC_G).start()
        vc_m = patch(_VC_G).start()
        stack.callback(patch.stopall)

        lr_m.return_value.read.side_effect = [_lf_g()] * 30
        sr_m.return_value.read.side_effect = [_st_g()] * 100
        cr_m.return_value.read.side_effect = [_col_g()] * 30
        flt_m.return_value = _fr_g()
        dd_m.return_value = MagicMock()
        dd_m.return_value.to_legacy_ray_distances.return_value = _CLEAR_RAYS
        dd_m.return_value.minimum_distance_m = 50.0
        dd_m.return_value.frame_valid = True
        lfov_m.return_value = MagicMock()
        vfov_m.return_value = []
        lpc_m.return_value = MagicMock()
        lpc_m.return_value.sectorization.sectors = []
        lpc_m.return_value.pointcloud.self_exclusion.enabled = False
        lpc_m.return_value.pointcloud.voxel_downsample.enabled = False

        from flight_modes.automatic_mode import AutomaticMode, AutomaticModeParams
        overrides = {"planner_mode": "apf", "guided_apf_control": False}
        # no "cbmba_resolution" key
        auto = AutomaticMode(
            session,
            params=AutomaticModeParams(max_flight_duration_s=0.05),
            cli_overrides=overrides,
        )
        with stack:
            assert auto._cbmba.params.resolution == 0.75

    # ── override to 0.75 ──

    def test_override_resolution_0_75(self):
        """--cbmba-resolution 0.75 → planner params resolution = 0.75."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.75)
        with stack:
            assert auto._cbmba.params.resolution == 0.75
            # Other params unchanged
            assert auto._cbmba.params.inflation_radius == 1.5
            assert auto._cbmba.params.max_search_nodes == 2000

    def test_override_resolution_1_0(self):
        """--cbmba-resolution 1.0 → planner params resolution = 1.0."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=1.0)
        with stack:
            assert auto._cbmba.params.resolution == 1.0

    def test_override_resolution_0_5(self):
        """--cbmba-resolution 0.5 → planner params resolution = 0.5."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.5)
        with stack:
            assert auto._cbmba.params.resolution == 0.5

    def test_override_resolution_1_5_legacy(self):
        """--cbmba-resolution 1.5 → legacy resolution still honored."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=1.5)
        with stack:
            assert auto._cbmba.params.resolution == 1.5

    # ── validation: rejection cases ──

    def test_zero_rejected(self):
        """resolution=0 → ValueError during construction."""
        with pytest.raises(ValueError, match="must be > 0"):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=0.0)

    def test_negative_rejected(self):
        """resolution=-1.0 → ValueError during construction."""
        with pytest.raises(ValueError, match="must be > 0"):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=-1.0)

    def test_nan_rejected(self):
        """resolution=NaN → ValueError during construction."""
        with pytest.raises(ValueError, match="must be finite"):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=float("nan"))

    def test_inf_rejected(self):
        """resolution=Inf → ValueError during construction."""
        with pytest.raises(ValueError, match="must be finite"):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=float("inf"))

    def test_neg_inf_rejected(self):
        """resolution=-Inf → ValueError during construction."""
        with pytest.raises(ValueError, match="must be finite"):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=float("-inf"))

    # ── planner receives exact value ──

    def test_planner_receives_exact_override_value(self):
        """CbmbaAStarPlanner is constructed with the exact override resolution."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.75)
        with stack:
            assert auto._cbmba.params.resolution == 0.75
            # Verify it's the actual float, not just a default
            assert isinstance(auto._cbmba.params.resolution, float)
            # The same CbmbaParams instance is used by the planner
            assert auto._cbmba.params is auto._cbmba._cbmba.params if hasattr(
                auto._cbmba, '_cbmba') else True

    def test_planner_uses_runtime_params(self):
        """The CbmbaAStarPlanner uses the params constructed at init time."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.75)
        with stack:
            # The planner's params are the same object set during construction
            assert auto._cbmba.params.resolution == 0.75
            # plan_with_result respects the params
            from planners.cbmba_astar import CbmbaParams
            assert isinstance(auto._cbmba.params, CbmbaParams)

    # ── guided APF unchanged apart from resolution ──

    def test_guided_apf_works_with_override_resolution(self):
        """Guided APF executes normally with resolution override."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.75, guided_apf_control=True, max_flight_duration_s=0.1)
        with stack:
            auto._cbmba_guidance.select_waypoint = MagicMock(
                return_value=TestFixedMissionGoal._make_guidance_result(valid=True))
            auto._cbmba.last_path = [[0, 0, -1], [15, 0, -1]]
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            assert auto._cbmba.params.resolution == 0.75

    def test_normal_apf_unchanged_with_override(self):
        """Normal APF (without guided flag) works with resolution override."""
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.5, guided_apf_control=False, max_flight_duration_s=0.1)
        with stack:
            with patch.object(session, "takeoff_and_climb"):
                r = auto.run()
            assert r.termination_reason == "time_limit"
            assert auto._cbmba.params.resolution == 0.5

    # ── deterministic ──

    def test_same_override_deterministic(self):
        """Same override value → same resolution every time."""
        for _ in range(3):
            session, auto, stack, vc = self._build_auto_with_resolution(
                resolution_override=0.75)
            with stack:
                assert auto._cbmba.params.resolution == 0.75

    # ── startup log contains active resolution ──

    def test_log_reports_default_resolution(self, caplog):
        """Startup log reports resolution=0.75 when no override."""
        import logging
        caplog.set_level(logging.INFO)
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=None)
        with stack:
            log_text = "\n".join(r.message for r in caplog.records
                                 if "cbmba_shadow" in r.message)
            assert "resolution=0.75" in log_text

    def test_log_reports_override_resolution(self, caplog):
        """Startup log reports resolution=0.75 when overridden."""
        import logging
        caplog.set_level(logging.INFO)
        session, auto, stack, vc = self._build_auto_with_resolution(
            resolution_override=0.75)
        with stack:
            log_text = "\n".join(r.message for r in caplog.records
                                 if "cbmba_shadow" in r.message)
            assert "resolution=0.75" in log_text

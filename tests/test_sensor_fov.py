"""ROUND 3.3: sensor_fov.py tests — comprehensive FOV loading, metadata validation,
horizontal+vertical coverage, fail-closed behaviour, and no external dependencies."""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.sensor_fov import (
    load_lidar_fov,
    validate_sector_fov_coverage,
    check_max_range_against_fov,
    SensorFov,
    SectorFovStatus,
    _azimuth_intersection_deg,
    _elevation_intersection_deg,
)
from perception.perception_config import load_perception_config

FIXTURES = _PROJECT_ROOT / "tests" / "fixtures"


def _settings_json(content: str) -> str:
    t = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    t.write(content)
    t.close()
    return t.name


# ═══════════════════════════════════════════════════════════════════════
# Fixture-based FOV loading (no external file dependency)
# ═══════════════════════════════════════════════════════════════════════


class TestLoadFovFromFixtures:
    """All tests use tests/fixtures/ — never reference/airsim_runtime/."""

    def test_loads_pm15(self):
        fov = load_lidar_fov(
            str(FIXTURES / "settings_lidar_pm15.json"), "Drone1", "LidarSensor1"
        )
        assert fov.horizontal_full_circle is True
        assert fov.vertical_upper_deg == 15.0
        assert fov.vertical_lower_deg == -15.0
        assert fov.range_m == 40.0
        assert fov.vertical_span_deg == 30.0

    def test_loads_pm30(self):
        fov = load_lidar_fov(
            str(FIXTURES / "settings_lidar_pm30.json"), "Drone1", "LidarSensor1"
        )
        assert fov.horizontal_full_circle is True
        assert fov.vertical_upper_deg == 30.0
        assert fov.vertical_lower_deg == -30.0
        assert fov.range_m == 40.0
        assert fov.vertical_span_deg == 60.0

    def test_loads_front180(self):
        fov = load_lidar_fov(
            str(FIXTURES / "settings_lidar_front180.json"), "Drone1", "LidarSensor1"
        )
        assert fov.horizontal_full_circle is False
        assert fov.horizontal_start_deg == -90.0
        assert fov.horizontal_end_deg == 90.0

    def test_vehicle_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            load_lidar_fov(
                str(FIXTURES / "settings_lidar_pm15.json"), "NoSuchDrone", "LidarSensor1"
            )

    def test_lidar_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            load_lidar_fov(
                str(FIXTURES / "settings_lidar_pm15.json"), "Drone1", "NoSuchLidar"
            )


# ═══════════════════════════════════════════════════════════════════════
# LiDAR metadata validation (SensorType, Enabled, DataFrame)
# ═══════════════════════════════════════════════════════════════════════


class TestMetadataValidation:
    """ROUND 3.3: SensorType==6, Enabled==true, DataFrame==SensorLocalFrame."""

    def test_sensor_type_not_6_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":5,"Enabled":true,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="SensorType must be 6"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_sensor_type_bool_rejected(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":true,"Enabled":true,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="SensorType must be a number"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_sensor_type_string_rejected(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":"6","Enabled":true,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="SensorType must be a number"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_enabled_false_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":false,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="Enabled must be true"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_enabled_missing_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="Enabled must be true"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_dataframe_wrong_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":true,"DataFrame":"WorldFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="SensorLocalFrame"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_dataframe_missing_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":true,'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="DataFrame must be a string"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_dataframe_bool_rejected(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":true,"DataFrame":true,'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="DataFrame must be a string"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_zero_width_horizontal_fov_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":true,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":45,"HorizontalFOVEnd":45,'
            '"VerticalFOVUpper":15,"VerticalFOVLower":-15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="zero-width"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")

    def test_vertical_lower_not_less_than_upper_fails(self):
        s = _settings_json(
            '{"Vehicles":{"Drone1":{"Sensors":{"LidarSensor1":{'
            '"SensorType":6,"Enabled":true,"DataFrame":"SensorLocalFrame",'
            '"HorizontalFOVStart":-180,"HorizontalFOVEnd":180,'
            '"VerticalFOVUpper":-15,"VerticalFOVLower":15,"Range":40}}}}}'
        )
        with pytest.raises(ValueError, match="must be <"):
            load_lidar_fov(s, "Drone1", "LidarSensor1")


# ═══════════════════════════════════════════════════════════════════════
# FOV coverage — horizontal + vertical (ROUND 3.3)
# ═══════════════════════════════════════════════════════════════════════


class TestFovCoveragePm15:
    """±15° FOV: horizontal sectors OK, all 8 vertical sectors unobservable."""

    def test_pm15_vertical_sectors_unobservable(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(
            horizontal_start_deg=-180, horizontal_end_deg=180,
            vertical_lower_deg=-15, vertical_upper_deg=15, range_m=40,
        )
        statuses = validate_sector_fov_coverage(cfg, fov)

        # All 8 vertical legacy sectors should be unobservable or partially observable
        vert_names = {"up", "down", "frontUp", "frontDown",
                      "leftUp", "rightUp", "leftDown", "rightDown"}
        for s in statuses:
            if s.legacy_name in vert_names:
                assert not s.fully_observable, (
                    f"{s.legacy_name} should NOT be fully observable with ±15° FOV: {s.note}"
                )

    def test_pm15_horizontal_sectors_partially_observable(self):
        """With ±15° FOV and [-22.5, 22.5] elevation sectors: ~67% coverage."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -15, 15, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        horiz_names = {"front", "back", "left", "right",
                       "frontLeft", "frontRight", "backLeft", "backRight"}
        for s in statuses:
            if s.legacy_name in horiz_names:
                # These have partial vertical coverage (30° / 45° ≈ 67%)
                assert s.vertical_coverage_fraction >= 0.6, (
                    f"{s.legacy_name}: expected >=60% vertical coverage, "
                    f"got {s.vertical_coverage_fraction*100:.1f}%"
                )
                assert not s.fully_observable, (
                    f"{s.legacy_name}: should NOT be fully observable with ±15° FOV"
                )


class TestFovCoveragePm30:
    """±30° FOV with updated perception.yaml: all 16 sectors fully observable."""

    def test_all_16_sectors_fully_observable(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        unobs = [s for s in statuses if not s.fully_observable]
        assert unobs == [], (
            f"Expected all sectors fully observable with ±30° FOV, got: "
            f"{[(s.legacy_name, s.note) for s in unobs]}"
        )

    def test_not_just_positive_intersection_but_full_coverage(self):
        """Each vertical sector must have vertical_coverage_fraction >= 0.9999."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        for s in statuses:
            assert s.vertical_coverage_fraction >= 0.9999, (
                f"{s.legacy_name}: vertical coverage fraction {s.vertical_coverage_fraction} "
                f"should be >= 0.9999 (full coverage)"
            )


class TestFovCoverageFront180:
    """Forward 180° FOV: front observable, back unobservable, left/right boundaries."""

    def test_front_observable_back_unobservable(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        # Forward 180°: [-90, 90], ±30° vertical
        fov = SensorFov(-90, 90, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        by_name = {s.legacy_name: s for s in statuses}

        # Front sector: azimuth [-22.5, 22.5] — fully inside [-90, 90]
        assert by_name["front"].fully_observable, (
            f"front should be fully observable: {by_name['front'].note}"
        )

        # Back sector: azimuth [157.5, -157.5] wraps through back
        # With [-90, 90] FOV, the [-180, -90] and [90, 180] regions are NOT covered
        assert not by_name["back"].fully_observable, (
            f"back should NOT be fully observable with forward-180 FOV: {by_name['back'].note}"
        )
        assert by_name["back"].horizontal_coverage_fraction < 1.0

    def test_left_right_boundary(self):
        """Left [-112.5, -67.5] is partially covered by [-90, 90] FOV."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-90, 90, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        by_name = {s.legacy_name: s for s in statuses}

        # Left: [-112.5, -67.5]. FOV covers [-90, 90].
        # The portion [-90, -67.5] is covered (22.5°), total span 45°
        # Coverage ~50%
        assert by_name["left"].horizontal_coverage_fraction < 1.0
        assert by_name["left"].horizontal_coverage_fraction > 0.0
        assert not by_name["left"].fully_observable


# ═══════════════════════════════════════════════════════════════════════
# Coverage classification: fully / partially / unobservable
# ═══════════════════════════════════════════════════════════════════════


class TestCoverageClassification:
    """Verify the three-state classification (fully/partially/unobservable)."""

    def test_fully_observable_flags(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)
        for s in statuses:
            assert s.fully_observable
            assert not s.partially_observable
            assert not s.unobservable

    def test_partially_observable_flags(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        # Elevation FOV that partially covers the vertical sectors
        fov = SensorFov(-180, 180, 20, 28, 40)  # only covers [20,28] of [18,30]
        statuses = validate_sector_fov_coverage(cfg, fov)
        # The up sector [18,30] should be partially covered
        up_status = next(s for s in statuses if s.legacy_name == "up")
        assert up_status.partially_observable
        assert not up_status.fully_observable
        assert not up_status.unobservable

    def test_unobservable_flags(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        # FOV that misses vertical sectors entirely
        fov = SensorFov(-180, 180, -5, 5, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)
        up_status = next(s for s in statuses if s.legacy_name == "up")
        assert up_status.unobservable
        assert not up_status.fully_observable
        assert not up_status.partially_observable


# ═══════════════════════════════════════════════════════════════════════
# Intersection width: must be real, not always 0
# ═══════════════════════════════════════════════════════════════════════


class TestIntersectionWidth:
    """intersection_azimuth_deg must be true intersection widths, not 0."""

    def test_full_circle_azimuth_intersection(self):
        """With 360° FOV, intersection should match sector span."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        front = next(s for s in statuses if s.legacy_name == "front")
        # Front sector azimuth: [-22.5, 22.5] = 45° span
        assert front.intersection_azimuth_deg >= 44.0, (
            f"Expected ~45° intersection, got {front.intersection_azimuth_deg}"
        )

    def test_front180_back_zero_intersection(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-90, 90, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        back = next(s for s in statuses if s.legacy_name == "back")
        # Back [157.5, -157.5] with FOV [-90, 90]: some overlap
        # The back sector wraps from 157.5 through 180/-180 to -157.5 (45° span)
        # FOV [-90, 90]: the overlap is [-90, -157.5?] — actually that's not right
        # Back: [157.5, -157.5]. FOV [-90, 90]. Overlap: none in [157.5, 180),
        # and [-90, -157.5) — no, -90 > -157.5
        # The overlap is: non-wrapping. Back goes from 157.5 → 180 → -180 → -157.5
        # FOV is [-90, 90]. Overlap is empty because [157.5, 180) not in [-90, 90],
        # and [-180, -157.5] not in [-90, 90] either (both are < -90)
        assert back.intersection_azimuth_deg < 1.0, (
            f"Back sector should have near-zero azimuth intersection with forward-180 FOV, "
            f"got {back.intersection_azimuth_deg}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Max range: errors, not warnings
# ═══════════════════════════════════════════════════════════════════════


class TestMaxRangeError:
    """max_range > LiDAR Range is a configuration error, not a warning."""

    def test_range_exceeds_fov_produces_errors(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -15, 15, range_m=10)
        errors = check_max_range_against_fov(cfg, fov)
        assert len(errors) > 0

    def test_range_within_fov_no_errors(self):
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -15, 15, range_m=50)
        errors = check_max_range_against_fov(cfg, fov)
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════
# Azimuth intersection helper unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestAzimuthIntersection:
    def test_full_circle_covers_all(self):
        _, _, width = _azimuth_intersection_deg(-22.5, 22.5, -180, 180, True)
        assert width >= 44.0

    def test_wrap_sector_with_full_circle(self):
        """Back sector [157.5, -157.5] with full circle FOV."""
        _, _, width = _azimuth_intersection_deg(157.5, -157.5, -180, 180, True)
        assert width >= 44.0

    def test_forward_fov_covers_front(self):
        _, _, width = _azimuth_intersection_deg(-22.5, 22.5, -90, 90, False)
        assert width >= 44.0

    def test_forward_fov_misses_back(self):
        """Back sector should have zero intersection with [-90, 90] FOV."""
        _, _, width = _azimuth_intersection_deg(157.5, -157.5, -90, 90, False)
        assert width < 1.0, f"Expected near-zero, got {width}"

    def test_partial_overlap_left(self):
        """Left sector [-112.5, -67.5] partially covered by [-90, 90]."""
        _, _, width = _azimuth_intersection_deg(-112.5, -67.5, -90, 90, False)
        # [-112.5, -67.5] ∩ [-90, 90] = [-90, -67.5] = 22.5° out of 45°
        assert 15.0 < width < 30.0, f"Expected ~22.5°, got {width}"


class TestElevationIntersection:
    def test_full_overlap(self):
        lo, hi, width = _elevation_intersection_deg(-22.5, 22.5, -30, 30)
        assert width >= 44.0
        assert lo == -22.5
        assert hi == 22.5

    def test_partial_overlap(self):
        lo, hi, width = _elevation_intersection_deg(18, 30, 20, 28)
        assert width == 8.0
        assert lo == 20.0
        assert hi == 28.0

    def test_no_overlap(self):
        lo, hi, width = _elevation_intersection_deg(18, 30, -5, 5)
        assert width == 0.0

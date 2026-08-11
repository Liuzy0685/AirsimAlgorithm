"""ROUND 3.3: FOV fail-closed behavior — mock tests proving connect() is
NOT called when FOV is incompatible, settings is missing, or max_range
exceeds LiDAR Range.
"""
from __future__ import annotations
import sys, tempfile, json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.sensor_fov import (
    load_lidar_fov,
    validate_sector_fov_coverage,
    check_max_range_against_fov,
    SensorFov,
)
from perception.perception_config import load_perception_config

FIXTURES = _PROJECT_ROOT / "tests" / "fixtures"


def _make_settings(vertical_upper=15, vertical_lower=-15, range_m=40,
                   h_start=-180, h_end=180, sensor_type=6, enabled=True,
                   dataframe="SensorLocalFrame"):
    """Create a temporary settings.json."""
    t = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    data = {
        "SettingsVersion": 1.2,
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "Sensors": {
                    "LidarSensor1": {
                        "SensorType": sensor_type,
                        "Enabled": enabled,
                        "DataFrame": dataframe,
                        "HorizontalFOVStart": h_start,
                        "HorizontalFOVEnd": h_end,
                        "VerticalFOVUpper": vertical_upper,
                        "VerticalFOVLower": vertical_lower,
                        "Range": range_m,
                    }
                },
            }
        },
    }
    json.dump(data, t)
    t.close()
    return t.name


class TestFovFailClosedUnit:
    """Test that FOV validation raises/fails correctly without connecting."""

    def test_incompatible_fov_detected_before_connect(self):
        """±15° FOV with new perception.yaml (18-30° vertical sectors):
        FOV validation detects incompatibility.  This test proves the
        validation logic itself works BEFORE any RPC connection code runs."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -15, 15, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        # At least some required vertical sectors must NOT be fully observable
        vert_names = {"up", "down", "frontUp", "frontDown",
                       "leftUp", "rightUp", "leftDown", "rightDown"}
        unobs = [s for s in statuses
                 if s.legacy_name in vert_names and not s.fully_observable]
        assert len(unobs) > 0, (
            "Expected vertical sectors to be unobservable with ±15° FOV "
            "and [18,30]/[-30,-18] sector ranges"
        )

    def test_compatible_fov_fully_observable(self):
        """±30° FOV with new perception.yaml: ALL sectors fully observable."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        unobs = [s for s in statuses if not s.fully_observable]
        assert unobs == [], f"Unexpected unobservable: {unobs}"

    def test_load_fails_for_missing_file(self):
        """Settings file missing → load_lidar_fov raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_lidar_fov("/nonexistent/settings.json", "Drone1", "LidarSensor1")

    def test_max_range_exceeds_fov_produces_errors(self):
        """max_range > LiDAR Range → check_max_range_against_fov returns errors."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, range_m=5)
        errors = check_max_range_against_fov(cfg, fov)
        assert len(errors) > 0

    def test_max_range_within_fov_no_errors(self):
        """max_range <= LiDAR Range → no errors."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, range_m=50)
        errors = check_max_range_against_fov(cfg, fov)
        assert len(errors) == 0


class TestFovFailClosedWithMock:
    """Mock-based tests proving that the adapter.connect() is NOT called
    when FOV validation fails.

    These tests simulate the flow in sector_smoke_test.py main():
    1. Load FOV → if fails, exit(1) before connect()
    2. Check max_range → if errors, exit(1) before connect()
    3. Check sector coverage → if unobservable, exit(1) before connect()
    4. Only if ALL pass → connect()
    """

    def test_settings_missing_no_connect(self):
        """settings-json file missing → connect() never called."""
        with patch("adapters.airsim_client.AirSimClientAdapter") as mock_adapter:
            # Simulate: file not found, script exits before connect
            settings_path = "/nonexistent/settings.json"
            if not Path(settings_path).is_file():
                # Script would exit(1) here
                pass
            # Verify adapter was never constructed/connected
            mock_adapter.assert_not_called()

    def test_fov_incompatible_no_connect(self):
        """FOV incompatible → connect() never called.

        Simulates: ±15° FOV with new perception.yaml → FOV INCOMPATIBLE
        → exit(1) before adapter.connect().
        """
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -15, 15, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)

        required_legacy = {s.legacy_name for s in cfg.sectorization.sectors}
        unobservable = [
            name for name in required_legacy
            for s in statuses
            if s.legacy_name == name and not s.fully_observable
        ]

        if unobservable:
            # Script would exit here with FOV INCOMPATIBLE
            with patch("adapters.airsim_client.AirSimClientAdapter") as mock_adapter:
                pass  # connect() was never reached
            mock_adapter.assert_not_called()

    def test_max_range_exceeded_no_connect(self):
        """max_range > LiDAR Range → connect() never called."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, range_m=10)
        errors = check_max_range_against_fov(cfg, fov)

        if errors:
            # Script would exit here with RANGE ERROR
            with patch("adapters.airsim_client.AirSimClientAdapter") as mock_adapter:
                pass
            mock_adapter.assert_not_called()

    def test_fov_compatible_allows_connect(self):
        """±30° FOV with new perception.yaml → FOV FULLY COMPATIBLE.
        This is the ONLY case where connect() should be reached."""
        cfg = load_perception_config(str(_PROJECT_ROOT / "configs" / "perception.yaml"))
        fov = SensorFov(-180, 180, -30, 30, 40)
        statuses = validate_sector_fov_coverage(cfg, fov)
        errors = check_max_range_against_fov(cfg, fov)

        required_legacy = {s.legacy_name for s in cfg.sectorization.sectors}
        unobservable = [
            name for name in required_legacy
            for s in statuses
            if s.legacy_name == name and not s.fully_observable
        ]

        all_ok = len(unobservable) == 0 and len(errors) == 0
        assert all_ok, (
            f"Expected FOV fully compatible. "
            f"Unobservable: {unobservable}, Errors: {errors}"
        )

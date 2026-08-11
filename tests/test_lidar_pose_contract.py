"""ROUND 4.3: LidarReader → _validate_sensor_pose contract tests.

Proves that the internal sensor_pose format (x/y/z/w keys, as produced
by LidarReader from AirSim x_val/y_val/etc) is correctly consumed by
_validate_sensor_pose() in the dry-run script.
"""
import sys, math
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the validation function from the dry-run script
from scripts.run_local_avoidance_dry_run import _validate_sensor_pose


# ── LidarReader-format sensor_pose (the internal contract) ──

def _make_pose(px=0.2, py=0.0, pz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Create a sensor_pose dict in LidarReader internal format:
    position: {x, y, z}, orientation: {x, y, z, w}."""
    return {
        "position": {"x": px, "y": py, "z": pz},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


class TestValidateSensorPoseFromLidarReader:
    """Test _validate_sensor_pose with LidarReader-format dicts."""

    def test_valid_pose_passes(self):
        pose = _make_pose(px=0.2, py=1.0, pz=-3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        pos, orient, err = _validate_sensor_pose(pose)
        assert err is None
        np.testing.assert_allclose(pos, [0.2, 1.0, -3.0])
        np.testing.assert_allclose(orient, [0.0, 0.0, 0.0, 1.0])

    def test_position_keys_are_x_y_z(self):
        pose = _make_pose(px=1.0, py=2.0, pz=3.0)
        pos, _, err = _validate_sensor_pose(pose)
        assert err is None
        assert pos[0] == 1.0 and pos[1] == 2.0 and pos[2] == 3.0

    def test_orientation_keys_are_x_y_z_w_and_order_is_xyzw(self):
        # Use a unit quaternion so normalisation doesn't change values
        # 90° yaw: q = [0, 0, sin(45°), cos(45°)]
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        pose = _make_pose(qx=0.0, qy=0.0, qz=s, qw=c)
        _, orient, err = _validate_sensor_pose(pose)
        assert err is None
        # Output order must be [qx, qy, qz, qw] = [0, 0, s, c]
        assert orient[0] == pytest.approx(0.0, abs=1e-12)
        assert orient[1] == pytest.approx(0.0, abs=1e-12)
        assert orient[2] == pytest.approx(s, rel=1e-6)
        assert orient[3] == pytest.approx(c, rel=1e-6)

    def test_normalises_non_unit_quaternion(self):
        pose = _make_pose(qx=0.0, qy=0.0, qz=0.0, qw=2.0)
        _, orient, err = _validate_sensor_pose(pose)
        assert err is None
        np.testing.assert_allclose(orient, [0.0, 0.0, 0.0, 1.0])

    def test_x_val_rejected(self):
        """x_val keys are NOT the internal format — must fail."""
        pose = {
            "position": {"x_val": 0.2, "y_val": 0.0, "z_val": 0.0},
            "orientation": {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0, "w_val": 1.0},
        }
        _, _, err = _validate_sensor_pose(pose)
        assert err is not None
        assert "field_missing" in err

    def test_sensor_pose_none(self):
        _, _, err = _validate_sensor_pose(None)
        assert err == "sensor_pose_missing"

    def test_sensor_pose_not_dict(self):
        _, _, err = _validate_sensor_pose("not a dict")
        assert err == "sensor_pose_missing"

    def test_position_missing(self):
        _, _, err = _validate_sensor_pose({"orientation": {"x": 0, "y": 0, "z": 0, "w": 1}})
        assert "position_missing" in err

    def test_orientation_missing(self):
        _, _, err = _validate_sensor_pose({"position": {"x": 0, "y": 0, "z": 0}})
        assert "orientation_missing" in err

    def test_field_missing_x(self):
        pose = _make_pose(); del pose["position"]["x"]
        _, _, err = _validate_sensor_pose(pose)
        assert "field_missing:x" in err

    def test_field_missing_w(self):
        pose = _make_pose(); del pose["orientation"]["w"]
        _, _, err = _validate_sensor_pose(pose)
        assert "field_missing:w" in err

    def test_string_rejected(self):
        pose = _make_pose(px="0.2")
        _, _, err = _validate_sensor_pose(pose)
        assert "field_not_numeric" in err

    def test_bool_rejected(self):
        pose = _make_pose(px=True)
        _, _, err = _validate_sensor_pose(pose)
        assert "field_not_numeric" in err

    def test_none_rejected(self):
        pose = _make_pose(); pose["position"]["x"] = None
        _, _, err = _validate_sensor_pose(pose)
        assert "field_not_numeric" in err

    def test_nan_rejected(self):
        pose = _make_pose(px=float("nan"))
        _, _, err = _validate_sensor_pose(pose)
        assert err == "sensor_pose_nonfinite"

    def test_inf_rejected(self):
        pose = _make_pose(px=float("inf"))
        _, _, err = _validate_sensor_pose(pose)
        assert err == "sensor_pose_nonfinite"

    def test_zero_quaternion_rejected(self):
        pose = _make_pose(qw=0.0)
        _, _, err = _validate_sensor_pose(pose)
        assert err == "sensor_pose_zero_quaternion"

    def test_input_dict_not_modified(self):
        pose = _make_pose(px=5.0)
        original = {"position": dict(pose["position"]), "orientation": dict(pose["orientation"])}
        _validate_sensor_pose(pose)
        assert pose["position"] == original["position"]
        assert pose["orientation"] == original["orientation"]


class TestLidarReaderProducesValidContract:
    """Integration test: LidarReader.read() with mock AirSim → _validate_sensor_pose()."""

    def test_lidar_reader_output_passes_validation(self):
        from sensors.lidar_reader import LidarReader

        # Mock adapter that returns a fake getLidarData result
        mock_adapter = MagicMock()
        mock_adapter.vehicle_name = "Drone1"
        mock_adapter.lidar_name = "LidarSensor1"

        fake_raw = MagicMock()
        fake_raw.time_stamp = 1234567890
        fake_raw.point_cloud = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        fake_raw.pose.position.x_val = 0.2
        fake_raw.pose.position.y_val = 0.0
        fake_raw.pose.position.z_val = 0.0
        fake_raw.pose.orientation.w_val = 1.0
        fake_raw.pose.orientation.x_val = 0.0
        fake_raw.pose.orientation.y_val = 0.0
        fake_raw.pose.orientation.z_val = 0.0

        mock_adapter.get_raw_client.return_value.getLidarData.return_value = fake_raw

        reader = LidarReader(mock_adapter)
        frame = reader.read()
        assert frame.frame_valid
        assert frame.sensor_pose is not None

        # Now validate with the dry-run function
        pos, orient, err = _validate_sensor_pose(frame.sensor_pose)
        assert err is None, f"LidarReader output should pass validation, got: {err}"
        np.testing.assert_allclose(pos, [0.2, 0.0, 0.0])
        np.testing.assert_allclose(orient, [0.0, 0.0, 0.0, 1.0])

    def test_lidar_reader_output_no_x_val_contamination(self):
        """Verify LidarFrame.sensor_pose does NOT use x_val/y_val keys."""
        from sensors.lidar_reader import LidarReader

        mock_adapter = MagicMock()
        mock_adapter.vehicle_name = "Drone1"
        mock_adapter.lidar_name = "LidarSensor1"

        fake_raw = MagicMock()
        fake_raw.time_stamp = 1
        fake_raw.point_cloud = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        fake_raw.pose.position.x_val = 0.2
        fake_raw.pose.position.y_val = 0.0
        fake_raw.pose.position.z_val = 0.0
        fake_raw.pose.orientation.w_val = 1.0
        fake_raw.pose.orientation.x_val = 0.0
        fake_raw.pose.orientation.y_val = 0.0
        fake_raw.pose.orientation.z_val = 0.0

        mock_adapter.get_raw_client.return_value.getLidarData.return_value = fake_raw

        reader = LidarReader(mock_adapter)
        frame = reader.read()

        # Confirm internal keys are x/y/z/w, NOT x_val/y_val/etc
        assert "x" in frame.sensor_pose["position"]
        assert "x_val" not in frame.sensor_pose["position"]
        assert "w" in frame.sensor_pose["orientation"]
        assert "w_val" not in frame.sensor_pose["orientation"]

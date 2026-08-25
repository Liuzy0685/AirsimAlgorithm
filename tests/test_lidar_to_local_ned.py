"""ROUND 4: lidar_to_local_ned coordinate transform tests."""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transforms.lidar_to_local_ned import (
    sensor_to_local_ned,
    _quaternion_to_rotation_matrix,
    _validate_quaternion,
)


class TestQuaternionToRotationMatrix:
    def test_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        R = _quaternion_to_rotation_matrix(q)
        assert R.shape == (3, 3)
        assert np.allclose(R, np.eye(3), atol=1e-12)

    def test_yaw_90(self):
        # 90° yaw about Z: q = [0, 0, sin(45°), cos(45°)]
        half = math.radians(45)
        q = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
        R = _quaternion_to_rotation_matrix(q)
        # +X → +Y, +Y → -X
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        assert np.allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_yaw_180(self):
        # 180° yaw about Z
        q = np.array([0.0, 0.0, 1.0, 0.0])
        R = _quaternion_to_rotation_matrix(q)
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        assert np.allclose(result, [-1.0, 0.0, 0.0], atol=1e-12)


class TestValidateQuaternion:
    def test_normalises(self):
        q = _validate_quaternion(np.array([0.0, 0.0, 0.0, 2.0]))
        assert np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            _validate_quaternion(np.array([0.0, 0.0, 0.0, np.nan]))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="Inf"):
            _validate_quaternion(np.array([0.0, 0.0, np.inf, 0.0]))

    def test_wrong_size_rejected(self):
        with pytest.raises(ValueError, match="4 elements"):
            _validate_quaternion(np.array([0.0, 0.0, 0.0]))

    def test_zero_norm_becomes_identity(self):
        q = _validate_quaternion(np.array([0.0, 0.0, 0.0, 0.0]))
        assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])


class TestSensorToLocalNed:
    def test_identity_transform(self):
        """Unit quaternion, zero translation: output == input."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([0.0, 0.0, 0.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        assert np.allclose(result, pts, atol=1e-12)

    def test_pure_translation(self):
        pts = np.array([[1.0, 0.0, 0.0]])
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([10.0, 20.0, 30.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        assert np.allclose(result, [[11.0, 20.0, 30.0]], atol=1e-12)

    def test_yaw_90_deg(self):
        """SensorLocalFrame +X(fwd) → NED +Y(east) after 90° yaw."""
        pts = np.array([[5.0, 0.0, 0.0]])  # 5m forward in sensor frame
        half = math.radians(45)
        q = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([0.0, 0.0, 0.0]),
            sensor_orientation_xyzw=q,
        )
        # After 90° yaw: sensor +X → NED +Y
        assert np.allclose(result, [[0.0, 5.0, 0.0]], atol=1e-12)

    def test_yaw_180_deg(self):
        pts = np.array([[5.0, 1.0, 0.0]])
        q = np.array([0.0, 0.0, 1.0, 0.0])  # 180° yaw
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([0.0, 0.0, 0.0]),
            sensor_orientation_xyzw=q,
        )
        assert np.allclose(result, [[-5.0, -1.0, 0.0]], atol=1e-12)

    def test_roll_and_pitch(self):
        """45° roll about X."""
        pts = np.array([[10.0, 0.0, 0.0]])
        half = math.radians(22.5)
        q = np.array([math.sin(half), 0.0, 0.0, math.cos(half)])
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([0.0, 0.0, 0.0]),
            sensor_orientation_xyzw=q,
        )
        # X unchanged by roll, Y/Z rotate
        assert abs(result[0, 0] - 10.0) < 1e-12

    def test_multi_point_array(self):
        pts = np.random.randn(100, 3).astype(np.float64)
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([1.0, 2.0, 3.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        assert result.shape == (100, 3)
        assert np.allclose(result, pts + [1.0, 2.0, 3.0], atol=1e-12)

    def test_input_not_modified(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        original = pts.copy()
        _ = sensor_to_local_ned(
            pts,
            sensor_position=np.array([10.0, 0.0, 0.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        assert np.array_equal(pts, original)

    def test_non_unit_quaternion_handled(self):
        """Non-unit quaternion is normalised before use."""
        pts = np.array([[1.0, 0.0, 0.0]])
        result = sensor_to_local_ned(
            pts,
            sensor_position=np.array([0.0, 0.0, 0.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 2.0]),
        )
        assert np.allclose(result, pts, atol=1e-12)

    def test_nan_points_rejected(self):
        pts = np.array([[np.nan, 0.0, 0.0]])
        with pytest.raises(ValueError, match="NaN"):
            sensor_to_local_ned(
                pts,
                sensor_position=np.array([0.0, 0.0, 0.0]),
                sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            )

    def test_inf_position_rejected(self):
        pts = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Inf"):
            sensor_to_local_ned(
                pts,
                sensor_position=np.array([np.inf, 0.0, 0.0]),
                sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            )

    def test_empty_point_cloud(self):
        result = sensor_to_local_ned(
            np.empty((0, 3)),
            sensor_position=np.array([1.0, 2.0, 3.0]),
            sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        assert result.shape == (0, 3)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="N,3"):
            sensor_to_local_ned(
                np.array([1.0, 2.0, 3.0]),
                sensor_position=np.array([0.0, 0.0, 0.0]),
                sensor_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            )

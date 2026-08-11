"""LiDAR SensorLocalFrame → local NED coordinate transform.

Uses LidarFrame.sensor_pose (position + orientation quaternion xyzw)
to rotate and translate the point cloud from the LiDAR body frame
into AirSim local NED world frame.

Pure NumPy — no scipy dependency.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def _quaternion_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) xyzw → 3×3 rotation matrix.

    Args:
        q_xyzw: (4,) or (N,4) quaternion array [x, y, z, w].

    Returns:
        (3,3) or (N,3,3) rotation matrix.
    """
    q = q_xyzw
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    if q.ndim == 1:
        R = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
        ])
    else:
        R = np.zeros((q.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def _validate_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    """Validate and normalise quaternion.  Returns (4,) normalised.

    Raises ValueError on NaN/Inf inputs.  Handles non-unit quaternions
    by normalising (scaling to unit length).  Zero-norm quaternion
    is treated as identity.
    """
    q = np.asarray(q_xyzw, dtype=np.float64).ravel()
    if q.size != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {q.size}")
    if np.any(np.isnan(q)) or np.any(np.isinf(q)):
        raise ValueError(f"Quaternion contains NaN or Inf: {q}")
    norm = np.linalg.norm(q)
    if norm < 1e-15:
        # Degenerate: return identity quaternion
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def sensor_to_local_ned(
    point_cloud_sensor: np.ndarray,
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform point cloud from SensorLocalFrame to local NED.

    Formula:
        p_local_ned = R(q) · p_sensor + t

    where:
    - q = sensor_orientation_xyzw (LiDAR pose orientation, xyzw order)
    - t = sensor_position (LiDAR pose position in local NED)
    - R(q) = rotation matrix from quaternion

    Args:
        point_cloud_sensor: (N,3) float array in SensorLocalFrame.
            +X=forward, +Y=right, +Z=down.
        sensor_position: (3,) float array — LiDAR position in local NED.
        sensor_orientation_xyzw: (4,) float array — LiDAR orientation
            quaternion in xyzw order.

    Returns:
        (N,3) float64 array in local NED (+X=north, +Y=east, +Z=down).

    Raises:
        ValueError: If inputs have wrong shape or contain NaN/Inf.
    """
    # --- Validate inputs ---
    pts = np.asarray(point_cloud_sensor, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"point_cloud_sensor must be (N,3), got {pts.shape}"
        )
    if pts.size == 0:
        return pts.reshape(0, 3)

    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        raise ValueError("point_cloud_sensor contains NaN or Inf")

    t = np.asarray(sensor_position, dtype=np.float64).ravel()
    if t.size != 3:
        raise ValueError(f"sensor_position must be (3,), got {t.shape}")
    if np.any(np.isnan(t)) or np.any(np.isinf(t)):
        raise ValueError("sensor_position contains NaN or Inf")

    # --- Validate and normalise quaternion ---
    q = _validate_quaternion(sensor_orientation_xyzw)

    # --- Compute rotation matrix ---
    R = _quaternion_to_rotation_matrix(q)

    # --- Transform ---
    # p_ned = R @ p_sensor^T + t  →  result = p_sensor @ R^T + t
    result = pts @ R.T + t.reshape(1, 3)

    # --- Input array is never modified (pts is a copy) ---
    return result

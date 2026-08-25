"""
LiDAR frame data model.

Coordinate system (SensorLocalFrame):
    +X = forward
    +Y = right
    +Z = down

This is the native frame reported by AirSim when
``DataFrame`` is set to ``"SensorLocalFrame"`` in settings.json.

The point cloud is stored as:
- ``point_cloud_sensor`` : N×3 NumPy array in SensorLocalFrame.
- ``point_cloud_world``  : reserved for stage 2 (world-NED transform);
  currently ``None``.

Transform to world NED (future):
    point_cloud_world = sensor_pose @ point_cloud_sensor
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LidarFrame:
    """A single, validated LiDAR frame.

    Attributes:
        point_cloud_sensor: N×3 array in SensorLocalFrame (+X forward, +Y right, +Z down).
        raw_timestamp_ns: AirSim raw time_stamp (uint64, Unreal-time epoch — NOT Unix time).
        received_monotonic_seconds: Local monotonic clock value when the frame was received.
        sensor_pose: AirSim Pose of the LiDAR sensor at capture time.
        frame_valid: ``True`` if this frame passed all sanity checks.
        invalid_reason: Human-readable reason when ``frame_valid`` is ``False``.
        point_count: Number of valid points (``0`` when invalid).
        vehicle_name: Vehicle the LiDAR is attached to.
        lidar_name: LiDAR sensor name.
    """

    point_cloud_sensor: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    raw_timestamp_ns: int = 0
    received_monotonic_seconds: float = 0.0
    sensor_pose: Optional[dict] = None
    frame_valid: bool = False
    invalid_reason: Optional[str] = None
    point_count: int = 0
    vehicle_name: str = ""
    lidar_name: str = ""

    def __repr__(self) -> str:
        status = "valid" if self.frame_valid else f"INVALID ({self.invalid_reason})"
        return (
            f"LidarFrame(vehicle={self.vehicle_name}, lidar={self.lidar_name}, "
            f"{self.point_count} pts, ts_ns={self.raw_timestamp_ns}, "
            f"mono={self.received_monotonic_seconds:.4f}, {status})"
        )

"""
Vehicle state data model.

World coordinate system is **NED** (North-East-Down):

    +X = North / forward
    +Y = East  / right
    +Z = Down

All angles are stored internally in **radians**.
Display/log functions may convert to degrees for human readability,
but computation MUST use radians.

.. note::

    ``airsim.to_eularian_angles()`` returns ``(pitch, roll, yaw)`` — **NOT**
    ``(roll, pitch, yaw)``.  This was confirmed against the AirSim 1.8.1
    PythonClient source:

        ``airsim/utils.py:80`` → ``return (pitch, roll, yaw)``

    Always unpack in that order.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleState:
    """Snapshot of a single ``getMultirotorState()`` call.

    Attributes:
        position_ned_m: [x_north, y_east, z_down] in meters (NED).
        linear_velocity_ned_mps: [vx_north, vy_east, vz_down] in m/s.
        angular_velocity_body_radps: [wx, wy, wz] in rad/s (body frame).
        orientation_quaternion_xyzw: Quaternion as [x, y, z, w] (AirSim native order).
        roll_rad: Roll angle in radians.
        pitch_rad: Pitch angle in radians.
        yaw_rad: Yaw angle in radians.
        timestamp: AirSim raw timestamp (uint64).
        ready: Whether the drone reports ``ready``.
        can_arm: Whether the drone reports ``can_arm``.
        landed_state: LandedState enum value from AirSim.
    """

    position_ned_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_velocity_ned_mps: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity_body_radps: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_quaternion_xyzw: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    timestamp: int = 0
    ready: bool = False
    can_arm: bool = False
    landed_state: Optional[int] = None
    received_monotonic_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"VehicleState(pos=[{self.position_ned_m[0]:.2f}, "
            f"{self.position_ned_m[1]:.2f}, {self.position_ned_m[2]:.2f}] m, "
            f"vel_ned=[{self.linear_velocity_ned_mps[0]:.2f}, "
            f"{self.linear_velocity_ned_mps[1]:.2f}, {self.linear_velocity_ned_mps[2]:.2f}] m/s, "
            f"rpy=({self.roll_rad:.3f}, {self.pitch_rad:.3f}, {self.yaw_rad:.3f}) rad, "
            f"ready={self.ready}, can_arm={self.can_arm})"
        )

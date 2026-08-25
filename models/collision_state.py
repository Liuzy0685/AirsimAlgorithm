"""
Collision state data model.

All positions and vectors are in NED world coordinates:
    +X = North, +Y = East, +Z = Down

``is_new_collision_event`` is set by ``CollisionReader`` when the
collision timestamp differs from the previously-seen timestamp.
"""

from dataclasses import dataclass, field


@dataclass
class CollisionState:
    """Snapshot from ``simGetCollisionInfo()``.

    Attributes:
        has_collided: ``True`` if a collision is currently active.
        is_new_collision_event: ``True`` when this is a *new* collision
            (timestamp differs from previous).  Set by ``CollisionReader``.
        object_name: Name of the collided object (empty string if none).
        object_id: AirSim internal object ID (-1 if none).
        impact_point_ned_m: Impact point [x, y, z] in NED meters.
        normal_ned: Collision normal vector [x, y, z] in NED.
        position_ned_m: Drone position at collision [x, y, z] in NED meters.
        penetration_depth: Penetration depth in meters (0.0 if none).
        raw_timestamp: AirSim ``time_stamp`` as **int** (AirSim-internal
            time epoch — NOT Unix seconds).  Do NOT convert to float.
    """

    has_collided: bool = False
    is_new_collision_event: bool = False
    object_name: str = ""
    object_id: int = -1
    impact_point_ned_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal_ned: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_ned_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    penetration_depth: float = 0.0
    raw_timestamp: int = 0
    received_monotonic_seconds: float = 0.0

    def __repr__(self) -> str:
        new_tag = " [NEW]" if self.is_new_collision_event else ""
        if not self.has_collided:
            return "CollisionState(no collision)"
        return (
            f"CollisionState(object={self.object_name!r}, "
            f"id={self.object_id}, "
            f"impact={self.impact_point_ned_m}, "
            f"normal={self.normal_ned}, "
            f"depth={self.penetration_depth:.3f} m, "
            f"ts={self.raw_timestamp}){new_tag}"
        )

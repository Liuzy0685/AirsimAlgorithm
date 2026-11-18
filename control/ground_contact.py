"""Identify a stationary, landed ground event without ignoring new collisions."""
import math
from models.collision_state import CollisionState
from models.vehicle_state import VehicleState


def is_stationary_ground_contact(collision, state):
    if not isinstance(collision, CollisionState) or not isinstance(state, VehicleState):
        return False
    if (collision.raw_timestamp <= 0 or state.landed_state != 0
            or collision.object_name not in {"Floor", "Floor_3", "(null)", ""}):
        return False
    values = [*collision.normal_ned, collision.penetration_depth, *state.linear_velocity_ned_mps]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        return False
    nx, ny, nz = collision.normal_ned
    return (math.hypot(nx,ny) <= .2 and nz <= -.9
            and 0 <= collision.penetration_depth <= .02
            and math.sqrt(sum(v*v for v in state.linear_velocity_ned_mps)) <= .05)

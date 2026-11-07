"""Common constants shared across multi-UAV coordination modules.

Coordinate frame
----------------
All internal algorithms use **World NED**:

    +X = North
    +Y = East
    +Z = Down

Only the final AirSim adapter/controller layer converts between World NED and
the concrete AirSim API conventions. A/B/C modules must all read/write World
NED and must NOT each perform their own coordinate conversion.

These are defaults, not flight-authoritative limits.  Real flight limits come
from the vehicle/flight YAML configs; the SafetyFilter / Arbitrator will be
constructed with the authoritative values when they are introduced in a later
phase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Coordinate frame
# ---------------------------------------------------------------------------

COORD_FRAME = "world_ned"
PROTOCOL_VERSION = 1
COORDINATION_STATES = (
    "NORMAL", "CONFLICT_DETECTED", "AVOIDING", "CONFLICT_CLEARED",
    "FORMATION_RECOVERING", "EMERGENCY",
)

# Coordinate-transform validation tolerances (CoordinateTransformer.validate).
DEFAULT_TRANSFORM_POSITION_TOLERANCE_M = 0.5
DEFAULT_TRANSFORM_ORIENTATION_TOLERANCE_RAD = 0.05

# ---------------------------------------------------------------------------
# Drone fleet bounds (Phase 1 supports Drone1..Drone5)
# ---------------------------------------------------------------------------

MIN_DRONES = 1
MAX_DRONES = 5

# ---------------------------------------------------------------------------
# Inter-UAV safety defaults (aligned with configs/swarm.yaml semantics)
# ---------------------------------------------------------------------------

DEFAULT_SAFETY_DISTANCE_M = 2.0
DEFAULT_PREDICTION_HORIZON_S = 3.0
DEFAULT_PEER_POLL_INTERVAL_S = 0.2
DEFAULT_MAX_STATE_SKEW_S = 0.5

# ---------------------------------------------------------------------------
# Speed / acceleration / yaw-rate limits (used by SafetyFilter / Arbitrator)
# ---------------------------------------------------------------------------

DEFAULT_MAX_HORIZONTAL_SPEED_MPS = 4.0
DEFAULT_MAX_VERTICAL_SPEED_MPS = 1.0
DEFAULT_MAX_YAW_RATE_RADPS = 1.2
DEFAULT_MAX_HORIZONTAL_ACCEL_MPS2 = 2.0
DEFAULT_MAX_VERTICAL_ACCEL_MPS2 = 1.0

# ---------------------------------------------------------------------------
# Command source labels (aligned with automatic_mode command_source values)
# ---------------------------------------------------------------------------

SOURCE_NONE = "none"
SOURCE_APF = "apf"
SOURCE_RECOVERY = "recovery"
SOURCE_ORCA = "orca"
SOURCE_FORMATION = "formation"
SOURCE_SAFETY_HOLD = "safety_hold"

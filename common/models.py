"""Unified multi-UAV data models (Phase 1).

Coordinate system: every position / velocity vector in this module is
**World NED** (+X=North, +Y=East, +Z=Down).

``VehicleState`` is re-exported from :mod:`models.vehicle_state` — it is the
single source of truth and is *not* copied here.  This keeps the existing
import path ``from models.vehicle_state import VehicleState`` working while
also exposing ``from common.models import VehicleState`` for new modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from models.vehicle_state import VehicleState  # noqa: F401  (re-export)

# A 3-vector in World NED.
Vec3 = Tuple[float, float, float]


def _require_finite_3d(value: Optional[Vec3], label: str) -> None:
    """Raise ``ValueError`` when ``value`` is not a finite 3-vector."""
    if value is None:
        raise ValueError(f"{label} must be a 3-vector, got None")
    if len(value) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {len(value)}")
    if not all(math.isfinite(float(c)) for c in value):
        raise ValueError(f"{label} must be finite, got {value!r}")


@dataclass
class FormationCommand:
    """Desired target / velocity produced by the formation controller (B).

    B may only emit a desired target position or desired velocity in World
    NED.  It must **not** call any AirSim control API — that is the gateway's
    responsibility.

    Attributes:
        vehicle_name: Target vehicle.
        mode: ``"position"`` (hold a target position) or ``"velocity"``.
        target_position_ned_m: World-NED target (used when ``mode="position"``).
        desired_velocity_ned_mps: World-NED desired velocity (``mode="velocity"``).
        valid: ``False`` marks a rejected / absent formation command.
        priority: Arbitration priority (lower than safety / avoidance).
        formation_id: Optional formation identifier for diagnostics.
        timestamp_s: Emission time (wall clock), seconds.
    """

    vehicle_name: str
    mode: str = "velocity"
    target_position_ned_m: Vec3 = (0.0, 0.0, 0.0)
    desired_velocity_ned_mps: Vec3 = (0.0, 0.0, 0.0)
    valid: bool = True
    priority: int = 50
    formation_id: str = ""
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_name, str) or not self.vehicle_name.strip():
            raise ValueError("vehicle_name must be a non-empty string")
        if self.mode not in ("position", "velocity"):
            raise ValueError(f"mode must be 'position' or 'velocity', got {self.mode!r}")
        if self.mode == "position":
            _require_finite_3d(self.target_position_ned_m, "target_position_ned_m")
        else:
            _require_finite_3d(self.desired_velocity_ned_mps, "desired_velocity_ned_mps")


@dataclass
class AvoidanceCommand:
    """Reactive avoidance result (APF / ORCA / recovery) in World NED."""

    vehicle_name: str
    desired_velocity_ned_mps: Vec3 = (0.0, 0.0, 0.0)
    source: str = "none"
    valid: bool = True
    reason: str = ""
    priority: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_name, str) or not self.vehicle_name.strip():
            raise ValueError("vehicle_name must be a non-empty string")
        _require_finite_3d(self.desired_velocity_ned_mps, "desired_velocity_ned_mps")


@dataclass
class TelemetryMessage:
    """Snapshot published to the telemetry / frontend module (C).

    C reads this message; it must not modify flight-control state.
    """

    timestamp_s: float
    vehicles: Dict[str, VehicleState] = field(default_factory=dict)
    avoidance: Optional[AvoidanceCommand] = None
    formation: Optional[FormationCommand] = None
    coordination_state: str = "monitor"
    valid: bool = True

    def vehicle_names(self) -> Tuple[str, ...]:
        """Return the sorted vehicle names present in this snapshot."""
        return tuple(sorted(self.vehicles))

    def vehicle(self, name: str) -> Optional[VehicleState]:
        """Return one vehicle snapshot, or ``None`` when absent."""
        return self.vehicles.get(name)


__all__ = [
    "VehicleState",
    "Vec3",
    "FormationCommand",
    "AvoidanceCommand",
    "TelemetryMessage",
]

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


@dataclass(frozen=True)
class CommandLifecycle:
    """Required metadata when a command crosses a process boundary.

    ``issued_at`` and ``expires_at`` are AirSim simulation nanoseconds.
    ``ttl_s`` is a separate receiver-local monotonic timeout beginning when
    A accepts the message. Neither clock may be compared with the other.
    ``stream_id`` changes when the publisher restarts so sequence numbers can
    restart only after a new authenticated session is established.
    """

    command_id: str
    sequence_id: int
    issued_at: int
    expires_at: int
    source: str
    task_id: str
    stream_id: str
    ttl_s: float

    def __post_init__(self) -> None:
        for label in ("command_id", "source", "task_id", "stream_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("sequence_id must be a nonnegative integer")
        if (type(self.issued_at) is not int or type(self.expires_at) is not int
                or self.issued_at < 0 or self.expires_at <= self.issued_at):
            raise ValueError("issued_at/expires_at must be increasing AirSim nanoseconds")
        if not math.isfinite(self.ttl_s) or self.ttl_s <= 0:
            raise ValueError("ttl_s must be finite and positive")

    def is_current(self, *, sim_time_ns: int, received_monotonic_seconds: float,
                   now_monotonic_seconds: float) -> bool:
        """Require both simulation and receiver-local deadlines to be live."""
        if type(sim_time_ns) is not int or sim_time_ns < 0:
            raise ValueError("sim_time_ns must be AirSim nanoseconds")
        if not all(math.isfinite(v) for v in (received_monotonic_seconds,
                                               now_monotonic_seconds)):
            raise ValueError("monotonic times must be finite")
        age = now_monotonic_seconds - received_monotonic_seconds
        return (0 <= age < self.ttl_s and self.issued_at <= sim_time_ns
                < self.expires_at)


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
    lifecycle: Optional[CommandLifecycle] = None

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_name, str) or not self.vehicle_name.strip():
            raise ValueError("vehicle_name must be a non-empty string")
        if self.mode not in ("position", "velocity"):
            raise ValueError(f"mode must be 'position' or 'velocity', got {self.mode!r}")
        if self.mode == "position":
            _require_finite_3d(self.target_position_ned_m, "target_position_ned_m")
        else:
            _require_finite_3d(self.desired_velocity_ned_mps, "desired_velocity_ned_mps")
        if self.lifecycle is not None and self.lifecycle.source != "formation":
            raise ValueError("FormationCommand lifecycle source must be 'formation'")

    def require_lifecycle(self) -> CommandLifecycle:
        """Reject a legacy in-process command before IPC or dispatch."""
        if self.lifecycle is None:
            raise ValueError("FormationCommand requires lifecycle for IPC/dispatch")
        return self.lifecycle


@dataclass
class AvoidanceCommand:
    """Reactive avoidance result (APF / ORCA / recovery) in World NED."""

    vehicle_name: str
    desired_velocity_ned_mps: Vec3 = (0.0, 0.0, 0.0)
    source: str = "none"
    valid: bool = True
    reason: str = ""
    priority: int = 100
    lifecycle: Optional[CommandLifecycle] = None

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_name, str) or not self.vehicle_name.strip():
            raise ValueError("vehicle_name must be a non-empty string")
        _require_finite_3d(self.desired_velocity_ned_mps, "desired_velocity_ned_mps")
        if self.lifecycle is not None and self.lifecycle.source != self.source:
            raise ValueError("AvoidanceCommand lifecycle source must match source")

    def require_lifecycle(self) -> CommandLifecycle:
        """Reject a legacy in-process command before IPC or dispatch."""
        if self.lifecycle is None:
            raise ValueError("AvoidanceCommand requires lifecycle for IPC/dispatch")
        return self.lifecycle


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
    "CommandLifecycle",
    "FormationCommand",
    "AvoidanceCommand",
    "TelemetryMessage",
]

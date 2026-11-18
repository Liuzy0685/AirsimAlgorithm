"""VelocityArbitrator — priority arbitration of competing velocity sources (§5).

Protocol ordering (§5): 安全/紧急 > 避障/恢复 > 编队目标 > 普通导航.  The
arbitrator is pure and only *selects* the winning desired velocity; it never
touches AirSim.  The selected velocity still passes through the
:class:`control.safety_filter.SafetyFilter` and
:class:`control.vehicle_command_gateway.VehicleCommandGateway` before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from common.models import AvoidanceCommand, FormationCommand

Vec3 = tuple


@dataclass(frozen=True)
class VelocityCandidate:
    source: str
    desired_velocity_ned_mps: Vec3
    valid: bool = True


@dataclass(frozen=True)
class ArbitratedVelocity:
    source: str
    desired_velocity_ned_mps: Vec3
    valid: bool
    reason: str


# Higher wins.  ``none`` is the weakest (navigation/fallback) source.
PRIORITY = {
    "emergency": 600,
    "safety_hold": 500,
    "recovery": 400,
    "orca": 350,
    "apf": 350,
    "formation": 200,
    "navigation": 100,
    "none": 0,
}


def formation_candidate(command: Optional[FormationCommand]) -> Optional[VelocityCandidate]:
    """Normalize a ``FormationCommand`` into an arbitrable candidate."""
    if command is None or not getattr(command, "valid", False):
        return None
    return VelocityCandidate("formation", tuple(command.desired_velocity_ned_mps), True)


def avoidance_candidate(command: Optional[AvoidanceCommand]) -> Optional[VelocityCandidate]:
    """Normalize an ``AvoidanceCommand`` into an arbitrable candidate."""
    if command is None or not getattr(command, "valid", False):
        return None
    return VelocityCandidate(command.source, tuple(command.desired_velocity_ned_mps), True)


class VelocityArbitrator:
    """Select the highest-priority valid candidate."""

    def arbitrate(self, candidates: Sequence[Optional[VelocityCandidate]]) -> ArbitratedVelocity:
        best: Optional[VelocityCandidate] = None
        best_priority = -1
        for candidate in candidates:
            if candidate is None or not candidate.valid:
                continue
            priority = PRIORITY.get(candidate.source, 0)
            if priority > best_priority:
                best_priority = priority
                best = candidate
        if best is None:
            return ArbitratedVelocity("none", (0.0, 0.0, 0.0), False, "no_valid_candidate")
        return ArbitratedVelocity(
            best.source, best.desired_velocity_ned_mps, True, ""
        )

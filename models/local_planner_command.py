"""Local planner command data model — ROUND 4.

The output of the avoidance pipeline (APF + Supervisor + SafetySupervisor).
Explicitly names the coordinate system: world NED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LocalPlannerCommand:
    """Safety-validated local avoidance command.

    Attributes:
        command_valid: False if any safety check failed.
        velocity_world_ned_mps: (vx_north, vy_east, vz_down) in m/s.
        yaw_rate_radps: Optional yaw rate in rad/s (None if not produced).
        source: 'apf' | 'recovery' | 'hover' | 'none'.
        priority: Higher = more authoritative (100 = APF, 40 = recovery, 0 = hover).
        invalid_reason: Human-readable reason when command_valid=False.
    """

    command_valid: bool = False
    velocity_world_ned_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rate_radps: Optional[float] = None
    source: str = "none"
    priority: int = 0
    invalid_reason: Optional[str] = None


def hover_command(reason: str = "default hover") -> LocalPlannerCommand:
    """Factory for a safe hover (zero-velocity) command."""
    return LocalPlannerCommand(
        command_valid=True,
        velocity_world_ned_mps=(0.0, 0.0, 0.0),
        yaw_rate_radps=0.0,
        source="hover",
        priority=0,
        invalid_reason=None,
    )


def invalid_command(reason: str) -> LocalPlannerCommand:
    """Factory for an invalid (hold) command."""
    return LocalPlannerCommand(
        command_valid=False,
        velocity_world_ned_mps=(0.0, 0.0, 0.0),
        yaw_rate_radps=None,
        source="none",
        priority=0,
        invalid_reason=reason,
    )

"""FixtureResult — ROUND 4.8.  Separates mission result from cleanup result."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class FixtureResult:
    mission_success: bool = False
    cleanup_success: bool = False
    exit_code: int = 1
    exit_reason: str = ""
    primary_failure_reason: str = ""
    cleanup_failure_reason: Optional[str] = None
    api_control_enabled: bool = False
    api_control_released: bool = False
    armed: bool = False
    disarmed: bool = False
    takeoff_completed: bool = False
    target_altitude_reached: bool = False
    actual_altitude_achieved: Optional[float] = None
    hover_seconds: float = 0.0
    emergency_shutdown_attempted: bool = False
    shutdown_type: str = ""
    cleanup_errors: List[str] = field(default_factory=list)
    preflight_checks_passed: bool = False
    takeoff_allowed: bool = False
    collision_detected_during_flight: bool = False
    landing_confirmed: bool = False

    @property
    def success(self) -> bool:
        """Deprecated compatibility property — prefer mission_success."""
        return self.mission_success

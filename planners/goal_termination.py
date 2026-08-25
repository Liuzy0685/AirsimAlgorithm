"""
Goal / mission-complete termination checker.

Following the DreamFly "LiteStop" idea — termination is decided by an
**independent** module, not by the motion controller observing that its
velocity has decayed to ~0.  A velocity≈0 could equally mean the drone is
stuck; the motion planner must not decide the mission is done.

This module is pure computation: it takes a position + speed snapshot and a
goal, and returns whether the goal has been *dwelled* long enough to be
considered reached.

Coordinate system: world NED (+X North, +Y East, +Z Down).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GoalTerminationParams:
    """Termination thresholds."""

    enabled: bool = True
    distance_tolerance_m: float = 1.0
    altitude_tolerance_m: float = 0.4
    max_speed_mps: float = 0.25
    dwell_time_s: float = 1.0


@dataclass
class GoalTerminationResult:
    """Result of one check call."""

    reached: bool = False
    within_distance: bool = False
    within_altitude: bool = False
    speed_low: bool = False
    dwelled: bool = False
    dwell_elapsed_s: float = 0.0
    distance_to_goal_m: float = float("inf")


class GoalTerminationChecker:
    """Dwell-based goal-arrival detector.

    ``update(position_ned, speed_mps, goal_ned, now)`` returns True once the
    drone has been within tolerance *and* slow *and* held that state for
    ``dwell_time_s``.
    """

    def __init__(self, params: Optional[GoalTerminationParams] = None) -> None:
        self.params = params or GoalTerminationParams()
        self._entered_at: Optional[float] = None
        self._goal: Optional[Tuple[float, float, float]] = None

    def reset(self, goal_ned: Optional[Tuple[float, float, float]] = None) -> None:
        self._entered_at = None
        self._goal = goal_ned

    def update(
        self,
        position_ned: Tuple[float, float, float],
        speed_mps: float,
        goal_ned: Tuple[float, float, float],
        now: float,
    ) -> GoalTerminationResult:
        """Check one snapshot.  Returns a ``GoalTerminationResult``."""
        p = self.params
        res = GoalTerminationResult()

        dx = position_ned[0] - goal_ned[0]
        dy = position_ned[1] - goal_ned[1]
        dz = position_ned[2] - goal_ned[2]
        res.distance_to_goal_m = math.hypot(dx, dy)
        res.within_distance = res.distance_to_goal_m <= p.distance_tolerance_m
        res.within_altitude = abs(dz) <= p.altitude_tolerance_m
        res.speed_low = speed_mps <= p.max_speed_mps

        if res.within_distance and res.within_altitude and res.speed_low:
            if self._entered_at is None:
                self._entered_at = now
            res.dwell_elapsed_s = now - self._entered_at
            res.dwelled = res.dwell_elapsed_s >= p.dwell_time_s
        else:
            self._entered_at = None

        res.reached = res.dwelled
        return res

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
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class GoalTerminationParams:
    """Termination thresholds."""

    enabled: bool = True
    distance_tolerance_m: float = 1.0
    altitude_tolerance_m: float = 0.4
    max_speed_mps: float = 0.25
    max_vertical_speed_mps: float = 0.20
    position_std_tolerance_m: float = 0.20
    history_size_frames: int = 1
    dwell_time_s: float = 1.0


@dataclass
class GoalTerminationResult:
    """Result of one check call."""

    reached: bool = False
    within_distance: bool = False
    within_3d_distance: bool = False
    within_altitude: bool = False
    speed_low: bool = False
    position_stable: bool = False
    window_ready: bool = False
    dwelled: bool = False
    dwell_elapsed_s: float = 0.0
    distance_to_goal_m: float = float("inf")
    distance_3d_to_goal_m: float = float("inf")
    horizontal_speed_mean_mps: float = float("inf")
    vertical_speed_mean_mps: float = float("inf")
    position_std_m: float = float("inf")


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
        self._samples: Deque[Tuple[Tuple[float, float, float], float, float]] = deque(
            maxlen=max(1, int(self.params.history_size_frames))
        )

    def reset(self, goal_ned: Optional[Tuple[float, float, float]] = None) -> None:
        self._entered_at = None
        self._goal = goal_ned
        self._samples.clear()

    def update(
        self,
        position_ned: Tuple[float, float, float],
        speed_mps: float,
        goal_ned: Tuple[float, float, float],
        now: float,
        velocity_ned_mps: Optional[Tuple[float, float, float]] = None,
    ) -> GoalTerminationResult:
        """Check one snapshot.  Returns a ``GoalTerminationResult``."""
        p = self.params
        res = GoalTerminationResult()

        dx = position_ned[0] - goal_ned[0]
        dy = position_ned[1] - goal_ned[1]
        dz = position_ned[2] - goal_ned[2]
        res.distance_to_goal_m = math.hypot(dx, dy)
        res.distance_3d_to_goal_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        res.within_distance = res.distance_to_goal_m <= p.distance_tolerance_m
        res.within_3d_distance = res.distance_3d_to_goal_m <= p.distance_tolerance_m
        res.within_altitude = abs(dz) <= p.altitude_tolerance_m

        if velocity_ned_mps is not None:
            horizontal_speed = math.hypot(velocity_ned_mps[0], velocity_ned_mps[1])
            vertical_speed = abs(velocity_ned_mps[2])
        else:
            horizontal_speed = float(speed_mps)
            vertical_speed = 0.0
        self._samples.append((position_ned, horizontal_speed, vertical_speed))

        required = max(1, int(p.history_size_frames))
        res.window_ready = len(self._samples) >= required
        if res.window_ready:
            n = len(self._samples)
            res.horizontal_speed_mean_mps = sum(s[1] for s in self._samples) / n
            res.vertical_speed_mean_mps = sum(s[2] for s in self._samples) / n
            mean_x = sum(s[0][0] for s in self._samples) / n
            mean_y = sum(s[0][1] for s in self._samples) / n
            mean_z = sum(s[0][2] for s in self._samples) / n
            variance = sum(
                (s[0][0] - mean_x) ** 2
                + (s[0][1] - mean_y) ** 2
                + (s[0][2] - mean_z) ** 2
                for s in self._samples
            ) / n
            res.position_std_m = math.sqrt(variance)

        res.speed_low = (
            res.window_ready
            and res.horizontal_speed_mean_mps <= p.max_speed_mps
            and res.vertical_speed_mean_mps <= p.max_vertical_speed_mps
        )
        res.position_stable = (
            res.window_ready
            and res.position_std_m <= p.position_std_tolerance_m
        )

        if (
            res.within_3d_distance
            and res.within_altitude
            and res.speed_low
            and res.position_stable
        ):
            if self._entered_at is None:
                self._entered_at = now
            res.dwell_elapsed_s = now - self._entered_at
            res.dwelled = res.dwell_elapsed_s >= p.dwell_time_s
        else:
            self._entered_at = None

        res.reached = res.dwelled
        return res

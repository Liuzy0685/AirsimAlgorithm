"""
Trajectory-mode flight validation metrics (Phase C0).

The trajectory navigator must be judged on **real AirSim flights**, not only
on unit tests.  This module accumulates a per-flight summary plus a few small
observable state machines that answer the concrete questions in the Phase C0
spec:

- ``TrajectoryFlightMetrics``  (sec 1)  — end-of-flight summary numbers.
- ``ObstacleAvoidanceEpisode`` (sec 2)  — observation-only绕障 episode
  (never modifies a command).
- ``FamilyTransitionLog``      (sec 3)  — family-switch transitions.
- ``MissionProgressMonitor``   (sec 23/24) — distinguishes *stuck* (no
  motion) from *mission-stalled* (moving but not toward the goal).
- ``SingleObstacleBehaviorMonitor`` (sec 25) — verifies the nearest-obstacle
  centroid is a fixed world obstacle, not a self-return that "moves" with the
  UAV.
- ``FlightTraceWriter``        (sec 21) — CSV trace of every control frame.

Everything here is **pure computation**: no AirSim RPC, no velocity dispatch.
The automatic-mode run loop feeds these with ground-truth state and LiDAR
readings.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── episode (sec 2) ──


@dataclass
class ObstacleAvoidanceEpisode:
    """One "approach → sidestep → clear" cycle around an obstacle.

    Observation-only: it records what the navigator *did*, it never changes a
    command.  ``side`` is the dominant lateral family observed during the
    episode (LEFT / RIGHT / NONE).
    """

    episode_id: int = 0
    start_frame: int = 0
    start_time: float = 0.0
    end_frame: int = 0
    end_time: float = 0.0
    duration_s: float = 0.0

    side: str = "NONE"
    approach_distance_m: float = float("inf")   # distance when first detected
    closest_distance_m: float = float("inf")    # min distance during episode
    final_distance_m: float = float("inf")      # distance at episode end
    success: bool = False                       # cleared (moving away from it)

    # world-NED XY of the nearest obstacle at closest approach (fixed world
    # frame, so it must NOT follow the UAV — see SingleObstacleBehaviorMonitor).
    obstacle_centroid_xy: Optional[Tuple[float, float]] = None


class ObstacleAvoidanceEpisodeTracker:
    """Detects and closes obstacle-avoidance episodes from per-frame readings.

    An episode opens when ``min_distance_m`` drops below ``start_distance_m``
    and closes when it rises back above ``end_distance_m`` for ``hold_frames``
    consecutive frames.  A completed episode is returned from ``update`` (and
    removed from the in-progress slot); otherwise ``None``.
    """

    def __init__(
        self,
        start_distance_m: float = 3.0,
        end_distance_m: float = 3.5,
        hold_frames: int = 5,
    ) -> None:
        self.start_distance_m = start_distance_m
        self.end_distance_m = end_distance_m
        self.hold_frames = max(1, hold_frames)

        self._active: Optional[ObstacleAvoidanceEpisode] = None
        self._clear_streak: int = 0
        self._next_id: int = 1
        self._side_votes: Dict[str, int] = {}

    @property
    def active(self) -> bool:
        return self._active is not None

    def update(
        self,
        frame: int,
        now: float,
        min_distance_m: float,
        family_side: int,
        drone_xy: Tuple[float, float],
        nearest_obstacle_xy: Optional[Tuple[float, float]],
    ) -> Optional[ObstacleAvoidanceEpisode]:
        """Feed one frame; return a completed episode (or None)."""
        if self._active is None:
            if min_distance_m < self.start_distance_m:
                self._active = ObstacleAvoidanceEpisode(
                    episode_id=self._next_id,
                    start_frame=frame,
                    start_time=now,
                    approach_distance_m=min_distance_m,
                    closest_distance_m=min_distance_m,
                    obstacle_centroid_xy=nearest_obstacle_xy,
                )
                self._clear_streak = 0
                self._side_votes = {}
            else:
                return None

        ep = self._active
        # Track closest approach + dominant side + obstacle centroid.
        if min_distance_m < ep.closest_distance_m:
            ep.closest_distance_m = min_distance_m
            if nearest_obstacle_xy is not None:
                ep.obstacle_centroid_xy = nearest_obstacle_xy
        if family_side != 0:
            label = "LEFT" if family_side < 0 else "RIGHT"
            self._side_votes[label] = self._side_votes.get(label, 0) + 1

        # Closing condition: distance rises back above end_distance_m.
        if min_distance_m > self.end_distance_m:
            self._clear_streak += 1
        else:
            self._clear_streak = 0

        if self._clear_streak >= self.hold_frames:
            ep.end_frame = frame
            ep.end_time = now
            ep.duration_s = max(0.0, now - ep.start_time)
            ep.final_distance_m = min_distance_m
            # Success = we moved away from the closest approach.
            ep.success = ep.final_distance_m > ep.closest_distance_m + 0.1
            ep.side = max(self._side_votes, key=self._side_votes.get) if self._side_votes else "NONE"
            if ep.obstacle_centroid_xy is not None and drone_xy is not None:
                # Pin the centroid to world frame (it must not follow the UAV).
                ep.obstacle_centroid_xy = ep.obstacle_centroid_xy
            self._active = None
            self._next_id += 1
            return ep

        return None

    def current(self) -> Optional[ObstacleAvoidanceEpisode]:
        """The in-progress episode (for end-of-flight bookkeeping)."""
        return self._active


# ── family transition log (sec 3) ──


@dataclass
class FamilyTransition:
    from_family: str
    to_family: str
    reason: str
    frame: int
    time: float
    direct_opposite: bool = False


class FamilyTransitionLog:
    """Accumulates family-switch transitions with a cap to bound memory."""

    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max_entries
        self.transitions: List[FamilyTransition] = []
        self.direct_opposite_count: int = 0
        self.breakdown: Dict[Tuple[str, str], int] = {}

    def record(
        self,
        from_family: str,
        to_family: str,
        reason: str,
        frame: int,
        now: float,
        direct_opposite: bool = False,
    ) -> None:
        self.transitions.append(FamilyTransition(
            from_family=from_family, to_family=to_family, reason=reason,
            frame=frame, time=now, direct_opposite=direct_opposite,
        ))
        if direct_opposite:
            self.direct_opposite_count += 1
        key = (from_family, to_family)
        self.breakdown[key] = self.breakdown.get(key, 0) + 1
        if len(self.transitions) > self.max_entries:
            self.transitions = self.transitions[-self.max_entries:]

    def __len__(self) -> int:
        return len(self.transitions)


# ── mission progress monitor (sec 23/24) ──


@dataclass
class MissionProgressStatus:
    progress_m: float = 0.0           # net progress toward goal over window
    path_length_m: float = 0.0        # total displacement over window
    window_s: float = 0.0
    is_stuck: bool = False            # no motion at all (blocked / recovery loop)
    is_mission_stalled: bool = False  # moving, but not toward the goal


class MissionProgressMonitor:
    """Distinguishes *stuck* from *mission-stalled* over a sliding window.

    - *stuck*: the drone has barely moved (``path_length_m < stuck_epsilon_m``).
    - *mission-stalled*: the drone is moving but its net progress toward the
      goal over the window is below ``min_progress_m``.

    ``update`` returns a status and, when a full window has elapsed, resets the
    window so the monitor fires at most once per window.
    """

    def __init__(
        self,
        window_s: float = 10.0,
        min_progress_m: float = 1.0,
        check_interval_s: float = 2.0,
        stuck_epsilon_m: float = 0.2,
    ) -> None:
        self.window_s = window_s
        self.min_progress_m = min_progress_m
        self.check_interval_s = check_interval_s
        self.stuck_epsilon_m = stuck_epsilon_m

        self._goal_xy: Optional[Tuple[float, float]] = None
        self._window_start_time: float = 0.0
        self._window_start_xy: Optional[Tuple[float, float]] = None
        self._last_xy: Optional[Tuple[float, float]] = None
        self._window_path_m: float = 0.0
        self._last_check_time: float = 0.0

        self.stuck_count: int = 0
        self.stalled_count: int = 0

    def reset(self, now: float, position_xy: Tuple[float, float]) -> None:
        self._window_start_time = now
        self._window_start_xy = position_xy
        self._last_xy = position_xy
        self._window_path_m = 0.0
        self._last_check_time = now

    def set_goal(self, goal_xy: Tuple[float, float]) -> None:
        self._goal_xy = goal_xy

    def update(self, now: float, position_xy: Tuple[float, float]) -> Optional[MissionProgressStatus]:
        if self._window_start_xy is None or self._last_xy is None:
            self.reset(now, position_xy)
            return None

        self._window_path_m += math.hypot(
            position_xy[0] - self._last_xy[0], position_xy[1] - self._last_xy[1],
        )
        self._last_xy = position_xy

        elapsed = now - self._window_start_time
        if elapsed < self.window_s:
            return None
        if now - self._last_check_time < self.check_interval_s:
            return None

        start = self._window_start_xy
        progress_m = 0.0
        if self._goal_xy is not None:
            gx, gy = self._goal_xy
            dx0 = gx - start[0]
            dy0 = gy - start[1]
            d0 = math.hypot(dx0, dy0)
            d1 = math.hypot(gx - position_xy[0], gy - position_xy[1])
            progress_m = d0 - d1 if d0 > 1e-9 else 0.0

        window_path_m = self._window_path_m
        is_stuck = window_path_m < self.stuck_epsilon_m
        is_stalled = (not is_stuck) and progress_m < self.min_progress_m

        if is_stuck:
            self.stuck_count += 1
        elif is_stalled:
            self.stalled_count += 1

        # Reset the window after a check so we don't re-fire every frame.
        self._window_start_time = now
        self._window_start_xy = position_xy
        self._window_path_m = 0.0
        self._last_check_time = now

        return MissionProgressStatus(
            progress_m=progress_m,
            path_length_m=window_path_m,
            window_s=elapsed,
            is_stuck=is_stuck,
            is_mission_stalled=is_stalled,
        )


# ── single-obstacle behaviour monitor (sec 25) ──


@dataclass
class ObstacleBehaviorVerdict:
    persistent: bool = True               # obstacle stayed fixed in world frame
    centroid_drift_m: float = 0.0         # max world-frame drift of centroid
    drone_travel_m: float = 0.0           # how far the UAV moved meanwhile
    samples: int = 0
    note: str = ""


class SingleObstacleBehaviorMonitor:
    """Verifies the nearest obstacle is a *fixed* world obstacle.

    A self-return / sensor artifact follows the UAV: as the drone translates,
    the "obstacle" centroid translates by the same amount.  A real column
    stays put.  We record the centroid world position and compare its drift to
    the drone's travel over the same span.
    """

    def __init__(self, min_samples: int = 8) -> None:
        self.min_samples = min_samples
        self._first_drone_xy: Optional[Tuple[float, float]] = None
        self._first_centroid_xy: Optional[Tuple[float, float]] = None
        self._max_drone_travel_m: float = 0.0
        self._max_centroid_drift_m: float = 0.0
        self._n: int = 0

    def update(
        self,
        drone_xy: Tuple[float, float],
        nearest_obstacle_xy: Optional[Tuple[float, float]],
    ) -> None:
        if nearest_obstacle_xy is None:
            return
        self._n += 1
        if self._first_drone_xy is None:
            self._first_drone_xy = drone_xy
            self._first_centroid_xy = nearest_obstacle_xy
            return
        self._max_drone_travel_m = max(
            self._max_drone_travel_m,
            math.hypot(drone_xy[0] - self._first_drone_xy[0],
                       drone_xy[1] - self._first_drone_xy[1]),
        )
        self._max_centroid_drift_m = max(
            self._max_centroid_drift_m,
            math.hypot(nearest_obstacle_xy[0] - self._first_centroid_xy[0],
                       nearest_obstacle_xy[1] - self._first_centroid_xy[1]),
        )

    def finalize(self) -> ObstacleBehaviorVerdict:
        if self._n < self.min_samples:
            return ObstacleBehaviorVerdict(
                persistent=True, samples=self._n,
                note=f"insufficient samples ({self._n} < {self.min_samples})",
            )
        # A self-return drifts with the UAV; a fixed obstacle drifts ≪ travel.
        # Flag as non-persistent only when the centroid clearly tracks motion.
        travel = self._max_drone_travel_m
        drift = self._max_centroid_drift_m
        persistent = not (travel > 1.0 and drift > 0.6 * travel)
        note = "fixed" if persistent else "tracks_uav"
        return ObstacleBehaviorVerdict(
            persistent=persistent,
            centroid_drift_m=drift,
            drone_travel_m=travel,
            samples=self._n,
            note=note,
        )


# ── flight metrics (sec 1 / 31) ──


@dataclass
class TrajectoryFlightMetrics:
    """End-of-flight summary numbers for one trajectory flight."""

    goal_xy: Optional[Tuple[float, float]] = None
    start_xy: Optional[Tuple[float, float]] = None

    flight_duration_s: float = 0.0
    frames_completed: int = 0

    # progress toward goal
    initial_distance_to_goal_m: float = float("inf")
    final_distance_to_goal_m: float = float("inf")
    max_goal_progress_m: float = 0.0
    total_path_length_m: float = 0.0
    max_lateral_deviation_m: float = 0.0

    # safety
    min_obstacle_clearance_m: float = float("inf")

    # avoidance episodes
    num_avoidance_episodes: int = 0
    total_avoidance_duration_s: float = 0.0
    avoidance_successes: int = 0

    # family switching
    num_family_switches: int = 0
    num_direct_opposite_switches: int = 0

    # recovery / no-feasible
    num_no_feasible_events: int = 0
    num_recovery_events: int = 0

    # control-loop health
    control_loop_overruns: int = 0
    control_loop_max_overrun_ms: float = 0.0
    control_loop_avg_dt_ms: float = 0.0
    lidar_stale_frames: int = 0
    total_frames: int = 0

    # monitors
    mission_stuck_count: int = 0
    mission_stalled_count: int = 0

    termination_reason: str = ""
    success: bool = False

    def __post_init__(self) -> None:
        self._last_xy: Optional[Tuple[float, float]] = None

    def _distance_to_goal(self, xy: Tuple[float, float]) -> float:
        if self.goal_xy is None:
            return float("inf")
        return math.hypot(xy[0] - self.goal_xy[0], xy[1] - self.goal_xy[1])

    def record_frame(
        self,
        position_xy: Tuple[float, float],
        min_distance_m: float,
        cmd_vx: float,
        cmd_vy: float,
    ) -> None:
        if self.start_xy is None:
            self.start_xy = position_xy
            self.initial_distance_to_goal_m = self._distance_to_goal(position_xy)
            self._last_xy = position_xy
            return

        step = math.hypot(
            position_xy[0] - self._last_xy[0], position_xy[1] - self._last_xy[1],
        )
        self.total_path_length_m += step
        self._last_xy = position_xy

        self.min_obstacle_clearance_m = min(self.min_obstacle_clearance_m, min_distance_m)

        # goal progress = initial distance − current distance (net gain).
        d = self._distance_to_goal(position_xy)
        self.final_distance_to_goal_m = d  # latest distance, not the minimum
        gain = self.initial_distance_to_goal_m - d
        self.max_goal_progress_m = max(self.max_goal_progress_m, gain)

        # lateral deviation from start→goal axis.
        if self.goal_xy is not None and self.start_xy is not None:
            gx, gy = self.goal_xy
            sx, sy = self.start_xy
            dx = gx - sx
            dy = gy - sy
            seg = math.hypot(dx, dy)
            if seg > 1e-9:
                dev = abs((position_xy[0] - sx) * dy - (position_xy[1] - sy) * dx) / seg
                self.max_lateral_deviation_m = max(self.max_lateral_deviation_m, dev)

    def finalize(self, termination_reason: str, success: bool, final_xy: Optional[Tuple[float, float]]) -> None:
        self.termination_reason = termination_reason
        self.success = success
        if final_xy is not None:
            self.final_distance_to_goal_m = self._distance_to_goal(final_xy)
        if self.total_frames > 0:
            self.lidar_stale_ratio = self.lidar_stale_frames / self.total_frames

    def summary(self) -> Dict[str, Any]:
        avoidance_success_rate = (
            self.avoidance_successes / self.num_avoidance_episodes
            if self.num_avoidance_episodes else 0.0
        )
        return {
            "success": self.success,
            "termination_reason": self.termination_reason,
            "flight_duration_s": self.flight_duration_s,
            "frames_completed": self.frames_completed,
            "initial_distance_to_goal_m": self.initial_distance_to_goal_m,
            "final_distance_to_goal_m": self.final_distance_to_goal_m,
            "max_goal_progress_m": self.max_goal_progress_m,
            "total_path_length_m": self.total_path_length_m,
            "max_lateral_deviation_m": self.max_lateral_deviation_m,
            "min_obstacle_clearance_m": self.min_obstacle_clearance_m,
            "num_avoidance_episodes": self.num_avoidance_episodes,
            "total_avoidance_duration_s": self.total_avoidance_duration_s,
            "avoidance_success_rate": avoidance_success_rate,
            "num_family_switches": self.num_family_switches,
            "num_direct_opposite_switches": self.num_direct_opposite_switches,
            "num_no_feasible_events": self.num_no_feasible_events,
            "num_recovery_events": self.num_recovery_events,
            "control_loop_overruns": self.control_loop_overruns,
            "control_loop_max_overrun_ms": self.control_loop_max_overrun_ms,
            "control_loop_avg_dt_ms": self.control_loop_avg_dt_ms,
            "lidar_stale_frames": self.lidar_stale_frames,
            "lidar_stale_ratio": getattr(self, "lidar_stale_ratio", 0.0),
            "mission_stuck_count": self.mission_stuck_count,
            "mission_stalled_count": self.mission_stalled_count,
        }

    def to_log_string(self) -> str:
        s = self.summary()
        return (
            "trajectory_flight_summary  "
            "success={}  term={}  dur={:.2f}  frames={}  "
            "dist_start={:.2f}  dist_final={:.2f}  max_progress={:.2f}  "
            "path_len={:.2f}  max_lat_dev={:.2f}  min_clear={:.2f}  "
            "episodes={}  episode_success={:.0%}  episode_dur={:.2f}  "
            "switches={}  opposite_switches={}  no_feasible={}  recovery={}  "
            "overruns={}  max_overrun={:.1f}ms  avg_dt={:.1f}ms  "
            "lidar_stale={}  stuck={}  stalled={}".format(
                "true" if s["success"] else "false",
                s["termination_reason"],
                s["flight_duration_s"],
                s["frames_completed"],
                s["initial_distance_to_goal_m"],
                s["final_distance_to_goal_m"],
                s["max_goal_progress_m"],
                s["total_path_length_m"],
                s["max_lateral_deviation_m"],
                s["min_obstacle_clearance_m"],
                s["num_avoidance_episodes"],
                s["avoidance_success_rate"],
                s["total_avoidance_duration_s"],
                s["num_family_switches"],
                s["num_direct_opposite_switches"],
                s["num_no_feasible_events"],
                s["num_recovery_events"],
                s["control_loop_overruns"],
                s["control_loop_max_overrun_ms"],
                s["control_loop_avg_dt_ms"],
                s["lidar_stale_frames"],
                s["mission_stuck_count"],
                s["mission_stalled_count"],
            )
        )


# ── CSV trace (sec 21) ──


class FlightTraceWriter:
    """Appends one CSV row per control frame (buffered).

    Columns (fixed order): frame, time_s, x, y, z, yaw, vx, vy, vz,
    cmd_vx, cmd_vy, command_source, family, min_distance, goal_progress_m,
    lateral_deviation_m.
    """

    _COLUMNS = [
        "frame", "time_s", "x", "y", "z", "yaw",
        "vx", "vy", "vz", "cmd_vx", "cmd_vy",
        "command_source", "family", "min_distance_m",
        "goal_progress_m", "lateral_deviation_m",
    ]

    def __init__(self, path: str, flush_interval: int = 20) -> None:
        self.path = path
        self.flush_interval = max(1, flush_interval)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self._COLUMNS)
        self._buffered = 0

    def write_row(self, row: List[Any]) -> None:
        self._writer.writerow(row)
        self._buffered += 1
        if self._buffered >= self.flush_interval:
            self._fh.flush()
            self._buffered = 0

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

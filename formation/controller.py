"""Position-slot formation planning. AirSim control remains with A."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from common.constants import MAX_DRONES, MIN_DRONES
from common.models import CommandLifecycle, FormationCommand, VehicleState
from coordination.state_provider import SwarmStateSnapshot


@dataclass(frozen=True)
class FormationConfig:
    leader: str
    followers: tuple[str, ...]
    shape: str
    spacing_m: float
    max_snapshot_age_s: float
    max_sim_skew_s: float
    command_ttl_s: float
    command_sim_lifetime_s: float
    recovery_gain: float
    max_recovery_speed_mps: float
    recovery_tolerance_m: float

    def __post_init__(self) -> None:
        names = (self.leader, *self.followers)
        if not MIN_DRONES <= len(names) <= MAX_DRONES or len(set(names)) != len(names):
            raise ValueError("formation requires 1..5 distinct vehicles")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("vehicle names must be nonempty strings")
        if self.shape not in ("line", "column", "vee"):
            raise ValueError("shape must be line, column or vee")
        for label in ("spacing_m", "max_snapshot_age_s", "max_sim_skew_s",
                      "command_ttl_s", "command_sim_lifetime_s", "recovery_gain",
                      "max_recovery_speed_mps", "recovery_tolerance_m"):
            value = getattr(self, label)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")

    @property
    def vehicle_names(self) -> tuple[str, ...]:
        return (self.leader, *self.followers)


def load_formation_config(path: str | Path) -> FormationConfig:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("formation"), dict):
        raise ValueError("formation config must contain a formation mapping")
    values = data["formation"]
    if not isinstance(values.get("followers"), list):
        raise ValueError("formation.followers must be a list")
    return FormationConfig(followers=tuple(values["followers"]), **{
        key: value for key, value in values.items() if key != "followers"
    })


class FormationController:
    """Emit follower targets only from a complete, recently received snapshot.

    ``received_at_monotonic_s`` is recorded by B when it receives the snapshot;
    A's per-state monotonic timestamps cannot be compared across processes.
    A remains responsible for command acceptance and safety arbitration.
    """

    def __init__(self, config: FormationConfig, *, task_id: str,
                 stream_id: str | None = None) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonempty")
        self.config = config
        self.task_id = task_id
        self.stream_id = stream_id or str(uuid.uuid4())
        self._sequence = 0
        self._last_snapshot: tuple[str, int] | None = None
        self._recovering = False

    def switch_shape(self, shape: str) -> None:
        if shape not in ("line", "column", "vee"):
            raise ValueError("shape must be line, column or vee")
        self.config = replace(self.config, shape=shape)

    def plan(self, snapshot: SwarmStateSnapshot, *, received_at_monotonic_s: float,
             now_monotonic_s: float | None = None, coordination_state: str = "NORMAL",
             snapshot_stream_id: str = "local") -> dict[str, FormationCommand]:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        if coordination_state in ("CONFLICT_DETECTED", "AVOIDING", "EMERGENCY"):
            self._recovering = True
            return {}
        if coordination_state not in ("NORMAL", "CONFLICT_CLEARED", "FORMATION_RECOVERING"):
            raise ValueError("unknown coordination state")
        if self._recovering and coordination_state == "NORMAL":
            return {}
        if coordination_state == "CONFLICT_CLEARED":
            self._recovering = True
        if not self._valid_snapshot(snapshot, received_at_monotonic_s, now):
            return {}
        if not isinstance(snapshot_stream_id, str) or not snapshot_stream_id.strip():
            raise ValueError("snapshot_stream_id must be nonempty")
        identity = (snapshot_stream_id, snapshot.sequence_id)
        if self._last_snapshot == identity:
            return {}
        if self._last_snapshot and self._last_snapshot[0] == snapshot_stream_id and snapshot.sequence_id <= self._last_snapshot[1]:
            return {}
        self._last_snapshot = identity

        leader = snapshot.states[self.config.leader]
        issued_at = max(state.timestamp for state in snapshot.states.values())
        commands: dict[str, FormationCommand] = {}
        converged = all(
            math.dist(snapshot.states[name].position_ned_m, self._slot(leader, index))
            <= self.config.recovery_tolerance_m
            for index, name in enumerate(self.config.followers)
        )
        recovering = self._recovering and not converged
        for index, name in enumerate(self.config.followers):
            target = self._slot(leader, index)
            if recovering:
                position = snapshot.states[name].position_ned_m
                error = tuple(target[i] - position[i] for i in range(3))
                distance = math.sqrt(sum(component * component for component in error))
                if distance <= self.config.recovery_tolerance_m:
                    velocity = (0.0, 0.0, 0.0)
                else:
                    scale = min(self.config.recovery_gain,
                                self.config.max_recovery_speed_mps / distance)
                    velocity = tuple(component * scale for component in error)
                mode = "velocity"
            else:
                velocity = (0.0, 0.0, 0.0)
                mode = "position"
            self._sequence += 1
            lifecycle = CommandLifecycle(
                command_id=str(uuid.uuid4()), sequence_id=self._sequence,
                issued_at=issued_at,
                expires_at=issued_at + int(self.config.command_sim_lifetime_s * 1e9),
                source="formation", task_id=self.task_id,
                stream_id=self.stream_id, ttl_s=self.config.command_ttl_s,
            )
            commands[name] = FormationCommand(
                vehicle_name=name, mode=mode, target_position_ned_m=target,
                desired_velocity_ned_mps=velocity, formation_id=self.config.shape,
                lifecycle=lifecycle,
            )
        if converged:
            self._recovering = False
        return commands

    def _valid_snapshot(self, snapshot: SwarmStateSnapshot, received_at: float,
                        now: float) -> bool:
        if (not isinstance(snapshot, SwarmStateSnapshot)
                or not snapshot.collection_complete
                or set(snapshot.states) != set(self.config.vehicle_names)
                or snapshot.errors
                or not all(math.isfinite(v) for v in (received_at, now))
                or not 0 <= now - received_at <= self.config.max_snapshot_age_s
                or not math.isfinite(snapshot.max_sim_timestamp_skew)
                or snapshot.max_sim_timestamp_skew > self.config.max_sim_skew_s):
            return False
        for state in snapshot.states.values():
            if not isinstance(state, VehicleState) or not state.ready or type(state.timestamp) is not int or state.timestamp <= 0:
                return False
            for vector in (state.position_ned_m, state.linear_velocity_ned_mps):
                if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                    return False
            if not math.isfinite(state.yaw_rad):
                return False
        timestamps = [state.timestamp for state in snapshot.states.values()]
        return (max(timestamps) - min(timestamps)) / 1e9 <= self.config.max_sim_skew_s

    def _slot(self, leader: VehicleState, index: int) -> tuple[float, float, float]:
        spacing = self.config.spacing_m
        if self.config.shape == "line":
            lateral = (index // 2 + 1) * spacing * (1 if index % 2 == 0 else -1)
            forward = 0.0
        elif self.config.shape == "column":
            forward, lateral = -(index + 1) * spacing, 0.0
        else:
            rank = index // 2 + 1
            forward = -rank * spacing
            lateral = rank * spacing * (1 if index % 2 == 0 else -1)
        yaw = leader.yaw_rad
        x, y, z = leader.position_ned_m
        return (x + forward * math.cos(yaw) - lateral * math.sin(yaw),
                y + forward * math.sin(yaw) + lateral * math.cos(yaw), z)

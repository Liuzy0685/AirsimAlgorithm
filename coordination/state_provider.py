"""Read-only, validated World-NED state snapshots for one to five vehicles.

Readers use the existing ``StateReader.read()`` interface. Reads remain
sequential on the caller's RPC connection; no flight-control path uses this
module until the live coordinate transform has been validated in AirSim.
All times in this module are monotonic seconds; ``VehicleState.timestamp``
retains the original AirSim nanosecond timestamp.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

from common.constants import MAX_DRONES, MIN_DRONES
from common.models import VehicleState
from coordination.coordinate_transform import CoordinateTransformer


def _world_attitude(rotation, quaternion):
    """Compose home->world rotation with a body->home XYZW quaternion."""
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        rw = 0.25 * s
        rx = (rotation[2][1] - rotation[1][2]) / s
        ry = (rotation[0][2] - rotation[2][0]) / s
        rz = (rotation[1][0] - rotation[0][1]) / s
    else:
        i = max(range(3), key=lambda axis: rotation[axis][axis])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + rotation[i][i] - rotation[j][j] - rotation[k][k]) * 2
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (rotation[i][j] + rotation[j][i]) / s
        q[k] = (rotation[i][k] + rotation[k][i]) / s
        rx, ry, rz = q
        rw = (rotation[k][j] - rotation[j][k]) / s
    x, y, z, w = quaternion
    qx = rw*x + rx*w + ry*z - rz*y
    qy = rw*y - rx*z + ry*w + rz*x
    qz = rw*z + rx*y - ry*x + rz*w
    qw = rw*w - rx*x - ry*y - rz*z
    pitch = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))
    roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return [qx, qy, qz, qw], roll, pitch, yaw


@dataclass(frozen=True)
class SwarmStateSnapshot:
    sequence_id: int
    timestamp: float
    states: dict[str, VehicleState]
    collection_complete: bool
    collection_duration: float
    acquired_at: dict[str, float] = field(default_factory=dict)
    read_duration_s: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    max_time_skew: float = 0.0
    max_sim_timestamp_skew: float = 0.0


class StateProvider:
    """Collect states through existing readers and a validated transform.

    ``readers`` maps configured vehicle names to objects exposing ``read()``.
    ``get_snapshot`` is the sole collection operation. The other getters
    return the latest snapshot and never perform hidden extra RPC reads.
    Failed or stale vehicles are omitted from ``states`` and listed in
    ``errors``. A regressed AirSim timestamp never replaces a newer state.
    """

    def __init__(
        self,
        readers: Mapping[str, object],
        transformer: CoordinateTransformer,
        *,
        max_age_s: float = 0.5,
        max_skew_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        names = tuple(readers)
        if not MIN_DRONES <= len(names) <= MAX_DRONES:
            raise ValueError(f"StateProvider requires {MIN_DRONES}..{MAX_DRONES} vehicles")
        if set(names) != set(transformer.vehicle_names):
            raise ValueError("reader names must match transformer vehicle names")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("vehicle names must be nonempty strings")
        if any(not callable(getattr(reader, "read", None)) for reader in readers.values()):
            raise ValueError("each reader must expose read()")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if not math.isfinite(max_skew_s) or max_skew_s < 0:
            raise ValueError("max_skew_s must be finite and nonnegative")
        self._readers = dict(readers)
        self._transformer = transformer
        self._max_age_s = max_age_s
        self._max_skew_s = max_skew_s
        self._clock = clock
        self._latest: SwarmStateSnapshot | None = None
        self._last_raw_timestamp: dict[str, int] = {}
        self._sequence = 0

    @classmethod
    def from_adapter(cls, adapter, vehicle_names, transformer, *, euler_converter=None, **kwargs):
        """Reuse ``StateReader`` with the existing shared AirSim RPC adapter."""
        from sensors.state_reader import StateReader

        readers = {
            name: StateReader(adapter, vehicle_name=name, euler_converter=euler_converter)
            for name in vehicle_names
        }
        return cls(readers, transformer, **kwargs)

    def get_snapshot(self) -> SwarmStateSnapshot:
        # Coordinate validation is required even for an empty/partial result.
        report = self._transformer.last_validation
        if (not self._transformer.validated or report is None
                or set(report.vehicles) != set(self._readers)):
            raise RuntimeError("CoordinateTransformer.validate() must pass before state collection")
        started = self._clock()
        states: dict[str, VehicleState] = {}
        acquired: dict[str, float] = {}
        durations: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, reader in self._readers.items():
            read_started = self._clock()
            try:
                local = reader.read()
                read_at = self._clock()
                durations[name] = read_at - read_started
                self._validate_local(local)
                if local.timestamp <= self._last_raw_timestamp.get(name, -1):
                    raise ValueError("AirSim timestamp did not advance")
                world = copy.deepcopy(local)
                world.position_ned_m = list(self._transformer.to_common_ned(name, local.position_ned_m))
                world.linear_velocity_ned_mps = list(
                    self._transformer.to_common_velocity(name, local.linear_velocity_ned_mps)
                )
                (world.orientation_quaternion_xyzw, world.roll_rad,
                 world.pitch_rad, world.yaw_rad) = _world_attitude(
                    self._transformer.rotation(name), local.orientation_quaternion_xyzw
                )
                world.received_monotonic_seconds = read_at
                states[name] = world
                acquired[name] = read_at
                self._last_raw_timestamp[name] = local.timestamp
            except Exception as exc:
                durations[name] = self._clock() - read_started
                errors[name] = f"{type(exc).__name__}: {exc}"
        ended = self._clock()
        for name, at in tuple(acquired.items()):
            if ended - at > self._max_age_s:
                errors[name] = "state expired during collection"
                del acquired[name]
                del states[name]
        skew = max(acquired.values()) - min(acquired.values()) if acquired else 0.0
        sim_timestamps = [state.timestamp for state in states.values()]
        sim_skew = ((max(sim_timestamps) - min(sim_timestamps)) / 1e9
                    if sim_timestamps else 0.0)
        if skew > self._max_skew_s or sim_skew > self._max_skew_s:
            for name in tuple(states):
                errors[name] = (f"snapshot time skew {skew:.3f}s / AirSim skew "
                                f"{sim_skew:.3f}s exceeds {self._max_skew_s:.3f}s")
            states.clear()
        self._sequence += 1
        snapshot = SwarmStateSnapshot(
            sequence_id=self._sequence,
            timestamp=ended,
            states=states,
            collection_complete=len(states) == len(self._readers),
            collection_duration=ended - started,
            acquired_at=acquired,
            read_duration_s=durations,
            errors=errors,
            max_time_skew=skew,
            max_sim_timestamp_skew=sim_skew,
        )
        self._latest = snapshot
        return snapshot

    def get_vehicle_state(self, vehicle_name: str) -> VehicleState | None:
        if vehicle_name not in self._readers:
            raise KeyError(vehicle_name)
        if not self._transformer.validated:
            return None
        snapshot = self._latest
        if snapshot is None:
            return None
        state = snapshot.states.get(vehicle_name)
        return copy.deepcopy(state) if state and self.is_state_fresh(state) else None

    def get_all_vehicle_states(self) -> dict[str, VehicleState]:
        return {
            name: state for name in self._readers
            if (state := self.get_vehicle_state(name)) is not None
        }

    def get_neighbor_states(self, vehicle_name: str) -> list[VehicleState]:
        if vehicle_name not in self._readers:
            raise KeyError(vehicle_name)
        return [state for name, state in self.get_all_vehicle_states().items() if name != vehicle_name]

    def is_state_fresh(self, state: VehicleState) -> bool:
        at = state.received_monotonic_seconds
        age = self._clock() - at
        return math.isfinite(at) and at > 0 and 0 <= age <= self._max_age_s

    @staticmethod
    def _validate_local(state: VehicleState) -> None:
        if not isinstance(state, VehicleState):
            raise ValueError("reader must return VehicleState")
        if not state.ready:
            raise ValueError("vehicle is not ready")
        if not isinstance(state.timestamp, int) or state.timestamp <= 0:
            raise ValueError("AirSim timestamp must be positive")
        for label, values, length in (
            ("position", state.position_ned_m, 3),
            ("velocity", state.linear_velocity_ned_mps, 3),
            ("angular velocity", state.angular_velocity_body_radps, 3),
            ("orientation", state.orientation_quaternion_xyzw, 4),
        ):
            if len(values) != length or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{label} must contain {length} finite values")
        if abs(sum(float(v) ** 2 for v in state.orientation_quaternion_xyzw) - 1.0) > 0.01:
            raise ValueError("orientation quaternion must have unit length")
        if not all(math.isfinite(v) for v in (state.roll_rad, state.pitch_rad, state.yaw_rad)):
            raise ValueError("attitude angles must be finite")

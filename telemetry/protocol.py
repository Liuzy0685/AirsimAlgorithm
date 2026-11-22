"""Validation and conversion for the v1 A -> C telemetry contract.

This module deliberately contains no AirSim calls.  C consumes validated JSON
snapshots from A and converts them into the browser-facing ``TelemetryFrame``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from common.constants import MAX_DRONES, MIN_DRONES, PROTOCOL_VERSION

MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_SNAPSHOT_AGE_S = 3.0


class ProtocolError(ValueError):
    """Raised when an incoming frame violates the v1 contract."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be a finite number")
    return result


def _nonnegative_number(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result < 0:
        raise ProtocolError(f"{field} must be nonnegative")
    return result


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _vector(value: Any, field: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProtocolError(f"{field} must contain {length} numbers")
    return [_finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _required(mapping: Mapping[str, Any], field: str) -> Any:
    if field not in mapping:
        raise ProtocolError(f"missing field: {field}")
    return mapping[field]


def _optional_vector(value: Any, field: str, length: int) -> list[float] | None:
    if value is None:
        return None
    return _vector(value, field, length)


def _validate_optional_display(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate optional display-only fields without granting C control authority."""
    result: dict[str, Any] = {}
    goal = payload.get("goal")
    if goal is not None:
        if not isinstance(goal, dict):
            raise ProtocolError("payload.goal must be an object")
        result["goal"] = {"position_ned_m": _vector(
            _required(goal, "position_ned_m"), "payload.goal.position_ned_m", 3
        )}
        for key in ("label", "task_id"):
            if key in goal:
                result["goal"][key] = _string(goal[key], f"payload.goal.{key}")
    obstacles = payload.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise ProtocolError("payload.obstacles must be a list")
    normalized_obstacles = []
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict):
            raise ProtocolError(f"payload.obstacles[{index}] must be an object")
        normalized = {
            "position_ned_m": _vector(
                _required(obstacle, "position_ned_m"), f"payload.obstacles[{index}].position_ned_m", 3
            ),
            "radius_m": _nonnegative_number(
                _required(obstacle, "radius_m"), f"payload.obstacles[{index}].radius_m"
            ),
        }
        if "label" in obstacle:
            normalized["label"] = _string(obstacle["label"], f"payload.obstacles[{index}].label")
        normalized_obstacles.append(normalized)
    result["obstacles"] = normalized_obstacles
    coordination_state = payload.get("coordination_state", "UNKNOWN")
    result["coordination_state"] = _string(coordination_state, "payload.coordination_state")
    active_command = payload.get("active_command")
    if active_command is not None:
        if not isinstance(active_command, dict):
            raise ProtocolError("payload.active_command must be an object")
        result["active_command"] = {
            key: _string(active_command[key], f"payload.active_command.{key}")
            for key in ("source", "mode") if key in active_command
        }
    cameras = payload.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ProtocolError("payload.cameras must be an object")
    normalized_cameras = {}
    for name, camera in cameras.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(camera, dict):
            raise ProtocolError("payload.cameras entries must have a name and object value")
        normalized_camera = {"status": _string(camera.get("status", "offline"), f"payload.cameras.{name}.status")}
        if "image_data_url" in camera:
            normalized_camera["image_data_url"] = _string(camera["image_data_url"], f"payload.cameras.{name}.image_data_url")
        normalized_cameras[name] = normalized_camera
    result["cameras"] = normalized_cameras
    return result


def _validate_state(state: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ProtocolError(f"{prefix}.state must be an object")
    result = dict(state)
    result["position_ned_m"] = _vector(_required(state, "position_ned_m"), f"{prefix}.state.position_ned_m", 3)
    result["linear_velocity_ned_mps"] = _vector(
        _required(state, "linear_velocity_ned_mps"), f"{prefix}.state.linear_velocity_ned_mps", 3
    )
    result["orientation_quaternion_xyzw"] = _vector(
        _required(state, "orientation_quaternion_xyzw"), f"{prefix}.state.orientation_quaternion_xyzw", 4
    )
    result["angular_velocity_body_radps"] = _vector(
        _required(state, "angular_velocity_body_radps"), f"{prefix}.state.angular_velocity_body_radps", 3
    )
    timestamp = _required(state, "timestamp")
    if type(timestamp) is not int or timestamp < 0:
        raise ProtocolError(f"{prefix}.state.timestamp must be a nonnegative integer")
    result["timestamp"] = timestamp
    received = _required(state, "received_monotonic_seconds")
    result["received_monotonic_seconds"] = _finite_number(
        received, f"{prefix}.state.received_monotonic_seconds"
    )
    return result


def validate_swarm_snapshot(frame: Mapping[str, Any], *, expected_producer: str | None = None) -> dict[str, Any]:
    """Validate and normalize one A ``swarm.snapshot`` envelope."""
    if not isinstance(frame, Mapping):
        raise ProtocolError("frame must be a JSON object")
    if _required(frame, "protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol_version")
    if _required(frame, "type") != "swarm.snapshot":
        raise ProtocolError("unexpected message type")
    producer_id = _string(_required(frame, "producer_id"), "producer_id")
    if expected_producer is not None and producer_id != expected_producer:
        raise ProtocolError("unexpected producer_id")
    stream_id = _string(_required(frame, "stream_id"), "stream_id")
    sequence_id = _required(frame, "sequence_id")
    if type(sequence_id) is not int or sequence_id < 0:
        raise ProtocolError("sequence_id must be a nonnegative integer")
    sent_at = _required(frame, "sent_at_unix_ms")
    if type(sent_at) is not int or sent_at < 0:
        raise ProtocolError("sent_at_unix_ms must be a nonnegative integer")

    payload = _required(frame, "payload")
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    normalized = dict(frame)
    normalized_payload = dict(payload)
    snapshot_id = _required(payload, "snapshot_id")
    if type(snapshot_id) not in (str, int) or not str(snapshot_id).strip():
        raise ProtocolError("payload.snapshot_id must be a non-empty string or integer")
    normalized_payload["snapshot_id"] = str(snapshot_id)
    if type(_required(payload, "collection_complete")) is not bool:
        raise ProtocolError("payload.collection_complete must be boolean")
    for field in ("collection_duration_s", "max_receive_skew_s", "max_sim_skew_s"):
        normalized_payload[field] = _nonnegative_number(_required(payload, field), f"payload.{field}")

    vehicles = _required(payload, "vehicles")
    if not isinstance(vehicles, dict) or not MIN_DRONES <= len(vehicles) <= MAX_DRONES:
        raise ProtocolError("payload.vehicles must contain 1 to 5 vehicles")
    normalized_vehicles: dict[str, dict[str, Any]] = {}
    for name, vehicle in vehicles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(vehicle, dict):
            raise ProtocolError("each vehicle must have a name and object value")
        valid = _required(vehicle, "valid")
        if type(valid) is not bool:
            raise ProtocolError(f"payload.vehicles.{name}.valid must be boolean")
        reason = vehicle.get("validity_reason", "")
        if not isinstance(reason, str):
            raise ProtocolError(f"payload.vehicles.{name}.validity_reason must be a string")
        normalized_vehicle = dict(vehicle)
        normalized_vehicle["validity_reason"] = reason
        received_at = vehicle.get("received_at_unix_ms")
        if received_at is not None and (type(received_at) is not int or received_at < 0):
            raise ProtocolError(f"payload.vehicles.{name}.received_at_unix_ms must be a timestamp or null")
        normalized_vehicle["received_at_unix_ms"] = received_at
        if valid:
            normalized_vehicle["state"] = _validate_state(
                _required(vehicle, "state"), f"payload.vehicles.{name}"
            )
        else:
            if vehicle.get("state") is not None or not reason.strip():
                raise ProtocolError(f"invalid vehicle {name} must have null state and a reason")
            normalized_vehicle["state"] = None
        normalized_vehicles[name] = normalized_vehicle
    normalized_payload["vehicles"] = normalized_vehicles
    normalized_payload.update(_validate_optional_display(payload))
    normalized["payload"] = normalized_payload
    return normalized


@dataclass(frozen=True)
class TelemetryFrame:
    """Browser-facing frame used by the C web page."""

    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.data


def snapshot_to_telemetry(
    frame: Mapping[str, Any],
    *,
    received_at_unix_ms: int | None = None,
    now_monotonic: float | None = None,
) -> TelemetryFrame:
    """Convert a validated A snapshot to the browser's ``TelemetryFrame``.

    ``timestamp`` is wall-clock Unix milliseconds for display.  AirSim
    nanoseconds and A's monotonic clock remain nested inside each drone for
    diagnostics and are never used as the browser timestamp.
    """
    normalized = validate_swarm_snapshot(frame)
    if received_at_unix_ms is None:
        received_at_unix_ms = int(time.time() * 1000)
    if type(received_at_unix_ms) is not int or received_at_unix_ms < 0:
        raise ValueError("received_at_unix_ms must be a nonnegative integer")
    _ = now_monotonic  # Reserved for age tracking by the gateway.
    payload = normalized["payload"]
    drones: dict[str, dict[str, Any]] = {}
    for name, vehicle in payload["vehicles"].items():
        state = vehicle["state"]
        drone: dict[str, Any] = {
            "vehicle_name": name,
            "valid": vehicle["valid"],
            "validity_reason": vehicle["validity_reason"],
            "position_ned_m": None,
            "linear_velocity_ned_mps": None,
            "orientation_quaternion_xyzw": None,
            "angular_velocity_body_radps": None,
            "sim_timestamp_ns": None,
            "state_received_monotonic_seconds": None,
            "received_at_unix_ms": vehicle.get("received_at_unix_ms"),
            "coordination_state": "UNKNOWN",
            "coordination_reason": "",
            "active_command_id": None,
        }
        if state is not None:
            drone.update(
                position_ned_m=state["position_ned_m"],
                linear_velocity_ned_mps=state["linear_velocity_ned_mps"],
                orientation_quaternion_xyzw=state["orientation_quaternion_xyzw"],
                angular_velocity_body_radps=state["angular_velocity_body_radps"],
                sim_timestamp_ns=state["timestamp"],
                state_received_monotonic_seconds=state["received_monotonic_seconds"],
            )
        drones[name] = drone
    all_valid = all(vehicle["valid"] for vehicle in payload["vehicles"].values())
    status = "NORMAL" if payload["collection_complete"] and all_valid else "PARTIAL_STATE"
    return TelemetryFrame(
        {
            "type": "telemetry",
            "protocol_version": PROTOCOL_VERSION,
            "frame": "world_ned",
            "timestamp": received_at_unix_ms,
            "source": normalized["producer_id"],
            "stream_id": normalized["stream_id"],
            "sequence_id": normalized["sequence_id"],
            "drones": drones,
            "goal": payload.get("goal"),
            "status": status,
            "obstacles": payload.get("obstacles", []),
            "coordination_state": payload.get("coordination_state", "UNKNOWN"),
            "coordination": {},
            "active_command": payload.get("active_command"),
            "cameras": payload.get("cameras", {}),
            "diagnostics": {
                "source_sent_at_unix_ms": normalized["sent_at_unix_ms"],
                "received_at_unix_ms": received_at_unix_ms,
                "transport_latency_ms": max(0, received_at_unix_ms - normalized["sent_at_unix_ms"]),
            },
            "snapshot": {
                "snapshot_id": payload["snapshot_id"],
                "collection_complete": payload["collection_complete"],
                "collection_duration_s": payload["collection_duration_s"],
                "max_receive_skew_s": payload["max_receive_skew_s"],
                "max_sim_skew_s": payload["max_sim_skew_s"],
            },
        }
    )


class SnapshotValidator:
    """Track stream/sequence monotonicity and reject stale snapshots."""

    def __init__(
        self,
        *,
        expected_producer: str = "a-state-service",
        max_age_s: float = MAX_SNAPSHOT_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.expected_producer = expected_producer
        self.max_age_s = max_age_s
        self.clock = clock
        self.stream_id: str | None = None
        self.sequence_id: int | None = None
        self.last_received_monotonic: float | None = None

    def accept(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_swarm_snapshot(frame, expected_producer=self.expected_producer)
        stream_id = normalized["stream_id"]
        sequence_id = normalized["sequence_id"]
        if stream_id == self.stream_id and self.sequence_id is not None and sequence_id <= self.sequence_id:
            raise ProtocolError("snapshot sequence is duplicate or out of order")
        self.stream_id = stream_id
        self.sequence_id = sequence_id
        self.last_received_monotonic = self.clock()
        return normalized

    def is_fresh(self) -> bool:
        if self.last_received_monotonic is None:
            return False
        age = self.clock() - self.last_received_monotonic
        return 0 <= age <= self.max_age_s

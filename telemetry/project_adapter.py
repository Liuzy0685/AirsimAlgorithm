"""Adapters from the repository's real state models to the A/C wire contract.

This module is the boundary between the existing in-process project code and
the WebSocket protocol.  It deliberately does not read AirSim or send any
flight command.  A calls it after ``StateProvider.get_snapshot()``; C then
consumes the resulting ``swarm.snapshot`` without re-reading the simulator.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from common.constants import MAX_DRONES, MIN_DRONES, PROTOCOL_VERSION
from common.models import AvoidanceCommand, FormationCommand, TelemetryMessage
from coordination.state_provider import SwarmStateSnapshot
from models.vehicle_state import VehicleState


def _vector(values: Sequence[float], length: int, field: str) -> list[float]:
    if len(values) != length:
        raise ValueError(f"{field} must contain {length} values")
    return [float(value) for value in values]


def _vehicle_state(state: VehicleState) -> dict[str, Any]:
    if not isinstance(state, VehicleState):
        raise TypeError("snapshot states must contain VehicleState objects")
    return {
        "position_ned_m": _vector(state.position_ned_m, 3, "position_ned_m"),
        "linear_velocity_ned_mps": _vector(
            state.linear_velocity_ned_mps, 3, "linear_velocity_ned_mps"
        ),
        "orientation_quaternion_xyzw": _vector(
            state.orientation_quaternion_xyzw, 4, "orientation_quaternion_xyzw"
        ),
        "angular_velocity_body_radps": _vector(
            state.angular_velocity_body_radps, 3, "angular_velocity_body_radps"
        ),
        "timestamp": int(state.timestamp),
        "received_monotonic_seconds": float(state.received_monotonic_seconds),
    }


def _validate_vehicle_names(vehicle_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(vehicle_names)
    if not MIN_DRONES <= len(names) <= MAX_DRONES:
        raise ValueError(f"vehicle_names must contain {MIN_DRONES}..{MAX_DRONES} vehicles")
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("vehicle_names must contain unique non-empty strings")
    return names


def _display_fields(display: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy only protocol-approved display fields supplied by A."""
    if display is None:
        return {
            "goal": None,
            "obstacles": [],
            "coordination_state": "UNKNOWN",
            "active_command": None,
            "cameras": {},
        }
    allowed = {"goal", "obstacles", "coordination_state", "active_command", "cameras"}
    unknown = set(display) - allowed
    if unknown:
        raise ValueError(f"unsupported display fields: {sorted(unknown)}")
    result = {
        "goal": display.get("goal"),
        "obstacles": display.get("obstacles", []),
        "coordination_state": display.get("coordination_state", "UNKNOWN"),
        "active_command": display.get("active_command"),
        "cameras": display.get("cameras", {}),
    }
    return result


def snapshot_to_swarm_frame(
    snapshot: SwarmStateSnapshot,
    *,
    vehicle_names: Sequence[str],
    producer_id: str = "a-state-service",
    stream_id: str,
    sent_at_unix_ms: int | None = None,
    snapshot_id: str | None = None,
    display: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a real ``SwarmStateSnapshot`` as an A ``swarm.snapshot``.

    ``vehicle_names`` must come from A's configured fleet (for example
    ``StateProvider.vehicle_names``), rather than being inferred only from the
    successful states.  That preserves failed vehicles in the wire payload.
    """
    if not isinstance(snapshot, SwarmStateSnapshot):
        raise TypeError("snapshot must be SwarmStateSnapshot")
    names = _validate_vehicle_names(vehicle_names)
    if not isinstance(stream_id, str) or not stream_id.strip():
        raise ValueError("stream_id must be a non-empty string")
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise ValueError("producer_id must be a non-empty string")
    if sent_at_unix_ms is None:
        sent_at_unix_ms = int(time.time() * 1000)
    if type(sent_at_unix_ms) is not int or sent_at_unix_ms < 0:
        raise ValueError("sent_at_unix_ms must be a nonnegative integer")

    unknown_states = set(snapshot.states) - set(names)
    unknown_errors = set(snapshot.errors) - set(names)
    if unknown_states or unknown_errors:
        raise ValueError("snapshot contains a vehicle outside the configured fleet")

    vehicles: dict[str, dict[str, Any]] = {}
    for name in names:
        state = snapshot.states.get(name)
        if state is not None:
            vehicles[name] = {
                "valid": True,
                "validity_reason": "",
                "state": _vehicle_state(state),
            }
        else:
            reason = snapshot.errors.get(name, "state missing from snapshot")
            vehicles[name] = {
                "valid": False,
                "validity_reason": str(reason),
                "state": None,
            }

    payload = {
        "snapshot_id": snapshot_id or f"snapshot-{snapshot.sequence_id}",
        "collection_complete": bool(snapshot.collection_complete),
        "collection_duration_s": float(snapshot.collection_duration),
        "max_receive_skew_s": float(snapshot.max_time_skew),
        "max_sim_skew_s": float(snapshot.max_sim_timestamp_skew),
        "vehicles": vehicles,
        **_display_fields(display),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "swarm.snapshot",
        "producer_id": producer_id,
        "stream_id": stream_id,
        "sequence_id": int(snapshot.sequence_id),
        "sent_at_unix_ms": sent_at_unix_ms,
        "payload": payload,
    }


def _command_display(
    avoidance: AvoidanceCommand | None,
    formation: FormationCommand | None,
) -> dict[str, str] | None:
    if avoidance is not None and avoidance.valid:
        return {"source": str(avoidance.source), "mode": "velocity"}
    if formation is not None and formation.valid:
        return {"source": "formation", "mode": str(formation.mode)}
    return None


def telemetry_message_to_swarm_frame(
    message: TelemetryMessage,
    *,
    producer_id: str = "a-state-service",
    stream_id: str,
    sequence_id: int,
    sent_at_unix_ms: int | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Adapt the existing ``TelemetryMessage`` compatibility model.

    New A code should prefer ``SwarmStateSnapshot`` because it carries
    completeness, per-vehicle errors and timing diagnostics.  This function is
    kept for code paths that already emit ``TelemetryMessage``.
    """
    if not isinstance(message, TelemetryMessage):
        raise TypeError("message must be TelemetryMessage")
    names = _validate_vehicle_names(tuple(message.vehicles))
    snapshot = SwarmStateSnapshot(
        sequence_id=sequence_id,
        timestamp=float(message.timestamp_s),
        states=dict(message.vehicles),
        collection_complete=bool(message.valid),
        collection_duration=0.0,
    )
    return snapshot_to_swarm_frame(
        snapshot,
        vehicle_names=names,
        producer_id=producer_id,
        stream_id=stream_id,
        sent_at_unix_ms=sent_at_unix_ms,
        snapshot_id=snapshot_id,
        display={
            "coordination_state": str(message.coordination_state),
            "active_command": _command_display(message.avoidance, message.formation),
        },
    )


__all__ = [
    "snapshot_to_swarm_frame",
    "telemetry_message_to_swarm_frame",
]

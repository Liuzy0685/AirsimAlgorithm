"""Wire-format codec for the v1 A/B/C protocol (docs/多机通信与控制协议.md).

Pure, AirSim-free, transport-free helpers.  Nothing here opens a socket or
imports ``airsim``; the WebSocket transport lives in :mod:`coordination.ws`
and the service in :mod:`coordination.state_service`.

Every frame crossing a process boundary uses the envelope defined in §3:

    {
      "protocol_version": 1,
      "type": "swarm.snapshot",
      "producer_id": "a-state-service",
      "stream_id": "boot-uuid",
      "sequence_id": 102,
      "sent_at_unix_ms": 1793318400000,
      "payload": {}
    }

Unknown protocol versions / types, missing fields, non-finite numbers, or
oversized frames are rejected with a reason — never silently forwarded.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from common.constants import COORDINATION_STATES, PROTOCOL_VERSION
from common.models import CommandLifecycle, FormationCommand, VehicleState

# ---------------------------------------------------------------------------
# Protocol identifiers
# ---------------------------------------------------------------------------

TYPE_SWARM_SNAPSHOT = "swarm.snapshot"
TYPE_FORMATION_COMMAND = "formation.command"
TYPE_COMMAND_ACK = "command.ack"
TYPE_COORDINATION_STATUS = "coordination.status"
TYPE_COMMAND_REQUEST = "command.request"
TYPE_COMMAND_PERMIT = "command.permit"

# Connection roles (handshake).  The service authorizes message types by role
# and NEVER trusts a message body's self-reported ``source``.
ROLE_A_CONTROLLER = "a_controller"
ROLE_B_FORMATION = "b_formation"
ROLE_C_TELEMETRY = "c_telemetry"
ROLES = (ROLE_A_CONTROLLER, ROLE_B_FORMATION, ROLE_C_TELEMETRY)

# Roles allowed to *send* each uplink message type.
UPLINK_AUTHORIZED_ROLES = {
    TYPE_FORMATION_COMMAND: {ROLE_B_FORMATION},
    TYPE_COORDINATION_STATUS: {ROLE_A_CONTROLLER},
    TYPE_COMMAND_REQUEST: {ROLE_A_CONTROLLER},
}

MAX_FRAME_BYTES = 2_000_000

WallClock = Callable[[], float]  # wall-clock seconds (e.g. time.time)
MonoClock = Callable[[], float]  # monotonic seconds (e.g. time.monotonic)


class WireError(ValueError):
    """Raised when an incoming frame violates the protocol."""


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Envelope:
    protocol_version: int
    message_type: str
    producer_id: str
    stream_id: str
    sequence_id: int
    sent_at_unix_ms: int
    payload: dict


def _require_str(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise WireError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WireError(f"{label} must be an integer >= {minimum}")
    return value


def build_envelope(
    *,
    message_type: str,
    producer_id: str,
    stream_id: str,
    sequence_id: int,
    payload: Mapping[str, Any],
    sent_at_unix_ms: Optional[int] = None,
    wall_clock: WallClock = time.time,
) -> dict:
    """Build a valid envelope dict.

    ``sent_at_unix_ms`` defaults to the current wall clock.  It is log/display
    metadata only and never participates in flight-safety freshness checks.
    """
    _require_str(message_type, "message_type")
    _require_str(producer_id, "producer_id")
    _require_str(stream_id, "stream_id")
    _require_int(sequence_id, "sequence_id")
    if not isinstance(payload, dict):
        raise WireError("payload must be an object")
    if sent_at_unix_ms is None:
        sent_at_unix_ms = int(wall_clock() * 1000)
    _require_int(sent_at_unix_ms, "sent_at_unix_ms", minimum=0)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        "producer_id": producer_id,
        "stream_id": stream_id,
        "sequence_id": sequence_id,
        "sent_at_unix_ms": sent_at_unix_ms,
        "payload": dict(payload),
    }


def parse_envelope(raw: str) -> Envelope:
    """Parse and validate one UTF-8 JSON envelope, raising ``WireError``."""
    if not isinstance(raw, str):
        raise WireError("frame must be a UTF-8 string")
    if len(raw.encode("utf-8", "replace")) > MAX_FRAME_BYTES:
        raise WireError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WireError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WireError("envelope must be a JSON object")
    version = data.get("protocol_version")
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise WireError(f"unsupported protocol_version {version!r} (expected {PROTOCOL_VERSION})")
    message_type = _require_str(data.get("type"), "type")
    if message_type not in {
        TYPE_SWARM_SNAPSHOT, TYPE_FORMATION_COMMAND, TYPE_COMMAND_ACK,
        TYPE_COORDINATION_STATUS,
        TYPE_COMMAND_REQUEST, TYPE_COMMAND_PERMIT,
    }:
        raise WireError(f"unknown message type {message_type!r}")
    producer_id = _require_str(data.get("producer_id"), "producer_id")
    stream_id = _require_str(data.get("stream_id"), "stream_id")
    sequence_id = _require_int(data.get("sequence_id"), "sequence_id")
    sent_at = _require_int(data.get("sent_at_unix_ms"), "sent_at_unix_ms", minimum=0)
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise WireError("payload must be an object")
    return Envelope(
        protocol_version=version,
        message_type=message_type,
        producer_id=producer_id,
        stream_id=stream_id,
        sequence_id=sequence_id,
        sent_at_unix_ms=sent_at,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# swarm.snapshot payload serialization
# ---------------------------------------------------------------------------


def serialize_vehicle_state(state: VehicleState) -> dict:
    """Serialize the World-NED fields of one valid ``VehicleState``."""
    return {
        "position_ned_m": [float(c) for c in state.position_ned_m],
        "linear_velocity_ned_mps": [float(c) for c in state.linear_velocity_ned_mps],
        "orientation_quaternion_xyzw": [
            float(c) for c in state.orientation_quaternion_xyzw
        ],
        "angular_velocity_body_radps": [
            float(c) for c in state.angular_velocity_body_radps
        ],
        "timestamp": int(state.timestamp),
        "received_monotonic_seconds": float(state.received_monotonic_seconds),
    }


def serialize_swarm_snapshot_payload(
    snapshot,
    *,
    configured_names: Sequence[str],
    wall_clock: WallClock = time.time,
    mono_clock: MonoClock = time.monotonic,
) -> dict:
    """Serialize a ``SwarmStateSnapshot`` into the §3 ``swarm.snapshot`` payload.

    ``configured_names`` is the full configured vehicle set so the ``vehicles``
    map always covers every vehicle; an invalid/missing vehicle is emitted as
    ``{valid: false, validity_reason: ..., state: null}``.

    ``received_at_unix_ms`` is back-computed from the vehicle's monotonic
    receive time so it reflects A's actual RPC receive instant (display only).
    """
    names = list(configured_names)
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise WireError("configured_names must be distinct non-empty strings")

    mono_now = mono_clock()
    wall_now = wall_clock()
    vehicles: dict[str, dict] = {}
    for name in names:
        state = snapshot.states.get(name)
        if state is not None:
            received_mono = float(state.received_monotonic_seconds)
            received_at_unix_ms = int(
                (wall_now - max(0.0, mono_now - received_mono)) * 1000
            ) if math.isfinite(received_mono) and received_mono > 0 else None
            vehicles[name] = {
                "valid": True,
                "validity_reason": "",
                "received_at_unix_ms": received_at_unix_ms,
                "state": serialize_vehicle_state(state),
            }
        else:
            vehicles[name] = {
                "valid": False,
                "validity_reason": str(snapshot.errors.get(name, "no state collected")),
                "received_at_unix_ms": None,
                "state": None,
            }

    return {
        "snapshot_id": int(snapshot.sequence_id),
        "collection_complete": bool(snapshot.collection_complete),
        "collection_duration_s": float(snapshot.collection_duration),
        "max_receive_skew_s": float(snapshot.max_time_skew),
        "max_sim_skew_s": float(snapshot.max_sim_timestamp_skew),
        "vehicles": vehicles,
    }


def build_swarm_snapshot_envelope(
    snapshot,
    *,
    configured_names: Sequence[str],
    producer_id: str,
    stream_id: str,
    wall_clock: WallClock = time.time,
    mono_clock: MonoClock = time.monotonic,
) -> dict:
    """Build a complete ``swarm.snapshot`` envelope for one snapshot."""
    payload = serialize_swarm_snapshot_payload(
        snapshot,
        configured_names=configured_names,
        wall_clock=wall_clock,
        mono_clock=mono_clock,
    )
    return build_envelope(
        message_type=TYPE_SWARM_SNAPSHOT,
        producer_id=producer_id,
        stream_id=stream_id,
        sequence_id=int(snapshot.sequence_id),
        payload=payload,
        wall_clock=wall_clock,
    )


# ---------------------------------------------------------------------------
# formation.command payload parsing (B → A)
# ---------------------------------------------------------------------------


def parse_lifecycle(raw: Any) -> CommandLifecycle:
    """Build a ``CommandLifecycle`` from a payload dict, validating its fields."""
    if not isinstance(raw, dict):
        raise WireError("lifecycle must be an object")
    if set(raw) - {"command_id", "sequence_id", "issued_at", "expires_at", "source", "task_id", "stream_id", "ttl_s"}:
        raise WireError("unknown lifecycle control fields")
    if "ttl_s" in raw and type(raw["ttl_s"]) not in (int, float):
        raise WireError("ttl_s must be numeric")
    try:
        return CommandLifecycle(
            command_id=raw["command_id"],
            sequence_id=raw["sequence_id"],
            issued_at=raw["issued_at"],
            expires_at=raw["expires_at"],
            source=raw["source"],
            task_id=raw["task_id"],
            stream_id=raw["stream_id"],
            ttl_s=raw["ttl_s"],
        )
    except KeyError as exc:
        raise WireError(f"lifecycle missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise WireError(f"invalid lifecycle: {exc}") from exc


def parse_formation_command_payload(payload: Any) -> FormationCommand:
    """Parse a ``formation.command`` payload into a validated ``FormationCommand``.

    The dataclass ``__post_init__`` enforces non-empty vehicle_name, mode
    ``position``/``velocity``, finite vectors and ``lifecycle.source ==
    "formation"``.  A missing lifecycle yields a command whose
    ``require_lifecycle()`` fails — matching the "legacy in-process command is
    not IPC-eligible" rule.
    """
    if not isinstance(payload, dict):
        raise WireError("formation.command payload must be an object")
    lifecycle_raw = payload.get("lifecycle")
    unknown = set(payload) - {"vehicle_name", "mode", "target_position_ned_m",
        "desired_velocity_ned_mps", "valid", "priority", "formation_id", "timestamp_s", "lifecycle"}
    if unknown:
        raise WireError(f"unknown control fields: {sorted(unknown)}")
    if type(payload.get("valid", True)) is not bool or type(payload.get("priority", 50)) is not int:
        raise WireError("valid/priority have incorrect types")
    lifecycle = parse_lifecycle(lifecycle_raw) if lifecycle_raw is not None else None
    try:
        target = payload.get("target_position_ned_m", (0.0, 0.0, 0.0))
        velocity = payload.get("desired_velocity_ned_mps", (0.0, 0.0, 0.0))
        for vector in (target, velocity):
            if vector is not None and (len(vector) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError("command vectors must contain three finite numbers")
        if not math.isfinite(float(payload.get("timestamp_s", 0.0))):
            raise ValueError("timestamp_s must be finite")
        return FormationCommand(
            vehicle_name=payload.get("vehicle_name"),
            mode=payload.get("mode", "velocity"),
            target_position_ned_m=tuple(target) if target is not None else None,
            desired_velocity_ned_mps=tuple(velocity) if velocity is not None else None,
            valid=bool(payload.get("valid", True)),
            priority=int(payload.get("priority", 50)),
            formation_id=str(payload.get("formation_id", "")),
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
            lifecycle=lifecycle,
        )
    except (TypeError, ValueError) as exc:
        raise WireError(f"invalid FormationCommand: {exc}") from exc


# ---------------------------------------------------------------------------
# command.ack / coordination.status payloads (A → B/C)
# ---------------------------------------------------------------------------


def build_command_ack(
    *,
    accepted: bool,
    reason: str,
    command_id: str,
    producer_id: str,
    stream_id: str,
    sequence_id: int,
    wall_clock: WallClock = time.time,
) -> dict:
    """Build a structured ``command.ack`` (always returned, even on reject).

    ``command_id`` may be empty when the inbound frame was malformed enough
    that no identifier could be extracted.
    """
    _require_str(command_id, "command_id", allow_empty=True)
    payload = {
        "accepted": bool(accepted),
        "reason": str(reason),
        "command_id": command_id,
    }
    return build_envelope(
        message_type=TYPE_COMMAND_ACK,
        producer_id=producer_id,
        stream_id=stream_id,
        sequence_id=sequence_id,
        payload=payload,
        wall_clock=wall_clock,
    )


def parse_swarm_snapshot(envelope, *, received_at):
    """Decode World NED and rebase freshness to the subscriber's own clock."""
    from coordination.state_provider import SwarmStateSnapshot, _world_attitude
    from coordination.coordinate_transform import IDENTITY_ROTATION
    if envelope.message_type != TYPE_SWARM_SNAPSHOT:
        raise WireError("expected swarm.snapshot")
    p = envelope.payload
    try:
        if type(p["collection_complete"]) is not bool:
            raise ValueError("invalid snapshot identity/completeness")
        vehicles = p["vehicles"]
        if not isinstance(vehicles, dict) or not 1 <= len(vehicles) <= 5:
            raise ValueError("invalid vehicles")
        states, errors = {}, {}
        for name, item in vehicles.items():
            _require_str(name, "vehicle name")
            if type(item["valid"]) is not bool:
                raise ValueError("invalid validity flag")
            if not item["valid"]:
                if item["state"] is not None or not item["validity_reason"]:
                    raise ValueError("invalid vehicle must have reason and null state")
                errors[name] = item["validity_reason"]
                continue
            raw = item["state"]
            vectors = {}
            for key, length in (("position_ned_m", 3), ("linear_velocity_ned_mps", 3),
                                ("orientation_quaternion_xyzw", 4), ("angular_velocity_body_radps", 3)):
                vector = raw[key]
                if len(vector) != length or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    raise ValueError(f"invalid {key}")
                vectors[key] = list(vector)
            q = vectors["orientation_quaternion_xyzw"]
            if abs(sum(v*v for v in q) - 1) > 0.01:
                raise ValueError("quaternion must be normalized")
            _, roll, pitch, yaw = _world_attitude(IDENTITY_ROTATION, q)
            states[name] = VehicleState(**vectors,
                timestamp=_require_int(raw["timestamp"], "timestamp", minimum=1),
                received_monotonic_seconds=received_at, ready=True,
                roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw)
        durations = [p[k] for k in ("collection_duration_s", "max_receive_skew_s", "max_sim_skew_s")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in durations):
            raise ValueError("invalid snapshot timing")
        if p["collection_complete"] and errors:
            raise ValueError("complete snapshot contains invalid vehicles")
        return SwarmStateSnapshot(sequence_id=_require_int(p["snapshot_id"], "snapshot_id"), timestamp=received_at,
            states=states, errors=errors, collection_complete=p["collection_complete"],
            collection_duration=durations[0], max_time_skew=durations[1],
            max_sim_timestamp_skew=durations[2], acquired_at={n: received_at for n in states})
    except (KeyError, TypeError, ValueError) as exc:
        raise WireError(f"invalid snapshot: {exc}") from exc


def build_coordination_status_payload(
    *,
    vehicle_name: str,
    state: str,
    reason: str = "",
    active_command_id: Optional[str] = None,
    snapshot_id: Optional[int] = None,
    task_id: str = "",
    sim_timestamp_ns: int = 0,
) -> dict:
    """Build a ``coordination.status`` payload (§3)."""
    _require_str(vehicle_name, "vehicle_name")
    if state not in COORDINATION_STATES:
        raise WireError(
            f"state must be one of {COORDINATION_STATES}, got {state!r}"
        )
    return {
        "vehicle_name": vehicle_name,
        "state": state,
        "reason": str(reason),
        "active_command_id": active_command_id,
        "snapshot_id": snapshot_id,
        "task_id": str(task_id),
        "sim_timestamp_ns": int(sim_timestamp_ns),
    }


__all__ = [
    "Envelope",
    "WireError",
    "MAX_FRAME_BYTES",
    "ROLES",
    "ROLE_A_CONTROLLER",
    "ROLE_B_FORMATION",
    "ROLE_C_TELEMETRY",
    "UPLINK_AUTHORIZED_ROLES",
    "TYPE_SWARM_SNAPSHOT",
    "TYPE_FORMATION_COMMAND",
    "TYPE_COMMAND_ACK",
    "TYPE_COORDINATION_STATUS",
    "build_envelope",
    "parse_envelope",
    "serialize_vehicle_state",
    "serialize_swarm_snapshot_payload",
    "build_swarm_snapshot_envelope",
    "parse_lifecycle",
    "parse_formation_command_payload",
    "build_command_ack",
    "build_coordination_status_payload",
]

"""SwarmStateService handler tests (socket-free)."""

import json

import pytest

from common.models import FormationCommand
from coordination.command_ingress import FormationCommandIngress
from coordination.coordinate_transform import (
    CoordinateTransformer,
    CoordinateTransformValidation,
    VehicleTransformCheck,
)
from coordination.state_provider import StateProvider, SwarmStateSnapshot
from coordination.state_publisher import SwarmStatePublisher
from coordination.state_service import SwarmStateService, parse_handshake_query
from coordination.wire import (
    TYPE_COORDINATION_STATUS,
    TYPE_FORMATION_COMMAND,
    build_envelope,
)
from models.vehicle_state import VehicleState


class Clock:
    def __init__(self, now=10.0):
        self.now = now

    def __call__(self):
        return self.now


class Reader:
    def __init__(self, state):
        self.state = state

    def read(self):
        return self.state


def _state(timestamp=1_500_000_000, position=(1.0, 2.0, -3.0)):
    return VehicleState(
        position_ned_m=list(position), linear_velocity_ned_mps=[1.0, 0.0, 0.0],
        timestamp=timestamp, ready=True, received_monotonic_seconds=10.0,
    )


def _service():
    offsets = {"Drone1": (10.0, 0.0, 0.0), "Drone2": (20.0, 0.0, 0.0)}
    tf = CoordinateTransformer(offsets)
    tf._validated = True
    tf._last_validation = CoordinateTransformValidation(
        validated=True,
        vehicles={name: VehicleTransformCheck(name, ok=True) for name in offsets},
    )
    readers = {name: Reader(_state(timestamp=1_500_000_000)) for name in offsets}
    provider = StateProvider(readers, tf, clock=Clock())
    publisher = SwarmStatePublisher(provider, producer_id="a", stream_id="a-stream")
    publisher._latest_snapshot = SwarmStateSnapshot(
        sequence_id=1, timestamp=10.0,
        states={"Drone1": _state(timestamp=1_500_000_000), "Drone2": _state(timestamp=1_500_000_000)},
        collection_complete=True, collection_duration=0.1,
    )
    ingress = FormationCommandIngress(clock=Clock())
    service = SwarmStateService(publisher, ingress, producer_id="a", stream_id="a-stream", clock=Clock())
    return service, publisher, ingress


def _register(service, role="b_formation", stream_id="b-stream", send=None):
    collected = []
    send = send or collected.append
    cid = service.register_client(
        role=role, producer_id="b-formation-service" if role == "b_formation" else f"{role}-service", stream_id=stream_id, send=send,
    )
    return cid, collected


def _command_payload(command_id="c1", sequence_id=1, stream_id="b-stream"):
    return {
        "vehicle_name": "Drone1",
        "mode": "velocity",
        "desired_velocity_ned_mps": [1.0, 0.0, 0.0],
        "valid": True,
        "priority": 50,
        "formation_id": "line",
        "timestamp_s": 0.0,
        "lifecycle": {
            "command_id": command_id,
            "sequence_id": sequence_id,
            "issued_at": 1_000_000_000,
            "expires_at": 2_000_000_000,
            "source": "formation",
            "task_id": "task-1",
            "stream_id": stream_id,
            "ttl_s": 0.4,
        },
    }


def _raw_command(payload, sequence_id=None):
    env = build_envelope(
        message_type=TYPE_FORMATION_COMMAND,
        producer_id="b-formation-service",
        stream_id="b-stream",
        sequence_id=sequence_id if sequence_id is not None else payload["lifecycle"]["sequence_id"],
        payload=payload,
        sent_at_unix_ms=1793318400000,
    )
    return json.dumps(env)


def test_register_client_valid_and_invalid_role():
    service, _, _ = _service()
    cid, _ = _register(service)
    assert service.client_count == 1
    from coordination.ws import WebSocketError

    with pytest.raises(WebSocketError, match="role"):
        service.register_client(role="hacker", producer_id="x", stream_id="y", send=lambda m: None)


def test_formation_command_accepted():
    service, _, ingress = _service()
    cid, replies = _register(service)
    reply = service.handle_message(cid, _raw_command(_command_payload()))
    assert reply["type"] == "command.ack"
    assert reply["payload"] == {"accepted": True, "reason": "accepted", "command_id": "c1"}
    assert isinstance(service.latest_formation_command("Drone1"), FormationCommand)
    assert ingress.seen_command_ids == {"c1"}


def test_formation_command_duplicate_rejected():
    service, _, _ = _service()
    cid, _ = _register(service)
    service.handle_message(cid, _raw_command(_command_payload(command_id="dup", sequence_id=1)))
    reply = service.handle_message(cid, _raw_command(_command_payload(command_id="dup", sequence_id=2)))
    assert reply["payload"]["accepted"] is False
    assert reply["payload"]["reason"] == "duplicate_command_id"


def test_formation_command_stale_sequence_rejected():
    service, _, _ = _service()
    cid, _ = _register(service)
    service.handle_message(cid, _raw_command(_command_payload(command_id="a", sequence_id=5)))
    reply = service.handle_message(cid, _raw_command(_command_payload(command_id="b", sequence_id=5)))
    assert reply["payload"]["reason"] == "stale_sequence"


def test_formation_command_no_snapshot_rejected():
    service, publisher, _ = _service()
    publisher._latest_snapshot = None
    cid, _ = _register(service)
    reply = service.handle_message(cid, _raw_command(_command_payload()))
    assert reply["payload"]["reason"] == "no_authoritative_snapshot"


def test_formation_command_stream_mismatch_rejected():
    service, _, _ = _service()
    cid, _ = _register(service, stream_id="other-stream")
    reply = service.handle_message(
        cid, _raw_command(_command_payload(stream_id="b-stream"))
    )
    assert reply["payload"]["reason"] == "envelope_identity_mismatch"


def test_non_formation_message_from_b_rejected():
    service, _, _ = _service()
    cid, _ = _register(service)
    env = build_envelope(
        message_type=TYPE_COORDINATION_STATUS, producer_id="b", stream_id="b-stream",
        sequence_id=1, payload={}, sent_at_unix_ms=1793318400000,
    )
    reply = service.handle_message(cid, json.dumps(env))
    assert reply["payload"]["accepted"] is False
    assert "unauthorized_message_type" in reply["payload"]["reason"]


def test_malformed_frame_from_b_gets_ack():
    service, _, _ = _service()
    cid, _ = _register(service)
    reply = service.handle_message(cid, "not json")
    assert reply["type"] == "command.ack"
    assert "malformed_frame" in reply["payload"]["reason"]


def test_broadcast_snapshot_reaches_all_roles():
    service, _, _ = _service()
    _, a_out = _register(service, role="a_controller", stream_id="a-ctrl")
    _, b_out = _register(service, role="b_formation", stream_id="b-stream")
    _, c_out = _register(service, role="c_telemetry", stream_id="c-tele")
    env = {"type": "swarm.snapshot", "sequence_id": 3, "payload": {}}
    service.broadcast_snapshot(env)
    assert a_out == b_out == c_out
    assert a_out[0]["type"] == env["type"] and a_out[0]["payload"] == env["payload"]
    assert a_out[0]["sequence_id"] > 0


def test_unregister_client_stops_broadcast():
    service, _, _ = _service()
    cid, out = _register(service)
    service.unregister_client(cid)
    assert service.client_count == 0
    service.broadcast_snapshot({"type": "swarm.snapshot"})
    assert out == []


def test_publish_status_broadcast():
    service, _, _ = _service()
    _, out = _register(service, role="c_telemetry", stream_id="c-tele")
    service.publish_status(vehicle_name="Drone1", state="NORMAL", snapshot_id=1)
    assert out[-1]["type"] == TYPE_COORDINATION_STATUS
    assert out[-1]["payload"]["state"] == "NORMAL"


def test_parse_handshake_query():
    path, query = parse_handshake_query(
        "/ws/swarm?role=b_formation&producer_id=b&stream_id=s1"
    )
    assert path == "/ws/swarm"
    assert query == {"role": "b_formation", "producer_id": "b", "stream_id": "s1"}

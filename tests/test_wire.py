"""Wire codec tests (no AirSim / network)."""

import pytest

from coordination.state_provider import SwarmStateSnapshot
from coordination.wire import (
    MAX_FRAME_BYTES,
    TYPE_COORDINATION_STATUS,
    TYPE_SWARM_SNAPSHOT,
    WireError,
    build_coordination_status_payload,
    build_command_ack,
    build_envelope,
    build_swarm_snapshot_envelope,
    parse_envelope,
    parse_formation_command_payload,
    serialize_swarm_snapshot_payload,
)
from models.vehicle_state import VehicleState


def _state(name="Drone1", position=(1.0, 2.0, -3.0), timestamp=1_000_000_000):
    return VehicleState(
        position_ned_m=list(position),
        linear_velocity_ned_mps=[0.5, 0.0, 0.0],
        angular_velocity_body_radps=[0.0, 0.0, 0.1],
        orientation_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        timestamp=timestamp,
        ready=True,
        received_monotonic_seconds=100.0,
    )


def _snapshot(states=None, errors=None, **kw):
    return SwarmStateSnapshot(
        sequence_id=7,
        timestamp=100.0,
        states=states or {},
        collection_complete=kw.get("collection_complete", True),
        collection_duration=kw.get("collection_duration", 0.2),
        acquired_at=kw.get("acquired_at", {}),
        read_duration_s=kw.get("read_duration_s", {}),
        errors=errors or {},
        max_time_skew=kw.get("max_time_skew", 0.0),
        max_sim_timestamp_skew=kw.get("max_sim_timestamp_skew", 0.0),
    )


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_envelope_round_trip():
    env = build_envelope(
        message_type=TYPE_SWARM_SNAPSHOT,
        producer_id="a-state-service",
        stream_id="boot-1",
        sequence_id=3,
        payload={"a": 1},
        sent_at_unix_ms=1793318400000,
    )
    parsed = parse_envelope(env["type"] and _to_text(env))
    assert parsed.message_type == TYPE_SWARM_SNAPSHOT
    assert parsed.producer_id == "a-state-service"
    assert parsed.stream_id == "boot-1"
    assert parsed.sequence_id == 3
    assert parsed.sent_at_unix_ms == 1793318400000
    assert parsed.payload == {"a": 1}


def _to_text(env) -> str:
    import json

    return json.dumps(env)


def test_envelope_rejects_bad_version():
    import json

    env = build_envelope(
        message_type=TYPE_SWARM_SNAPSHOT, producer_id="a", stream_id="s",
        sequence_id=0, payload={},
    )
    env["protocol_version"] = 99
    with pytest.raises(WireError, match="protocol_version"):
        parse_envelope(json.dumps(env))


def test_envelope_rejects_unknown_type():
    import json

    env = build_envelope(
        message_type=TYPE_SWARM_SNAPSHOT, producer_id="a", stream_id="s",
        sequence_id=0, payload={},
    )
    env["type"] = "formation.teleport"
    with pytest.raises(WireError, match="unknown message type"):
        parse_envelope(json.dumps(env))


def test_envelope_rejects_missing_payload():
    import json

    env = build_envelope(
        message_type=TYPE_SWARM_SNAPSHOT, producer_id="a", stream_id="s",
        sequence_id=0, payload={},
    )
    del env["payload"]
    with pytest.raises(WireError, match="payload"):
        parse_envelope(json.dumps(env))


def test_envelope_rejects_oversized():
    with pytest.raises(WireError, match="exceeds"):
        parse_envelope("x" * (MAX_FRAME_BYTES + 1))


# ---------------------------------------------------------------------------
# swarm.snapshot serialization
# ---------------------------------------------------------------------------


def test_snapshot_payload_covers_all_configured_names():
    snap = _snapshot(
        states={"Drone1": _state("Drone1")},
        errors={"Drone2": "OSError: rpc down"},
        collection_complete=False,
    )
    payload = serialize_swarm_snapshot_payload(
        snap, configured_names=["Drone1", "Drone2"],
        wall_clock=lambda: 110.0, mono_clock=lambda: 100.0,
    )
    assert payload["snapshot_id"] == 7
    assert payload["collection_complete"] is False
    assert payload["vehicles"]["Drone1"]["valid"] is True
    assert payload["vehicles"]["Drone1"]["state"]["position_ned_m"] == [1.0, 2.0, -3.0]
    assert payload["vehicles"]["Drone2"]["valid"] is False
    assert payload["vehicles"]["Drone2"]["state"] is None
    assert "rpc down" in payload["vehicles"]["Drone2"]["validity_reason"]


def test_snapshot_payload_back_computes_received_at_unix_ms():
    snap = _snapshot(states={"Drone1": _state("Drone1")})
    payload = serialize_swarm_snapshot_payload(
        snap, configured_names=["Drone1"],
        wall_clock=lambda: 112.0, mono_clock=lambda: 101.5,
    )
    # received_monotonic=100.0, so receive wall = 112 - (101.5 - 100) = 110.5s
    assert payload["vehicles"]["Drone1"]["received_at_unix_ms"] == 110500


def test_snapshot_envelope_sequence_matches_snapshot_id():
    snap = _snapshot(states={"Drone1": _state("Drone1")})
    env = build_swarm_snapshot_envelope(
        snap, configured_names=["Drone1"], producer_id="a", stream_id="s",
        wall_clock=lambda: 110.0, mono_clock=lambda: 100.0,
    )
    assert env["sequence_id"] == 7
    assert env["payload"]["snapshot_id"] == 7


# ---------------------------------------------------------------------------
# formation.command parsing
# ---------------------------------------------------------------------------


def _lifecycle(command_id="c1", sequence_id=1, source="formation"):
    return {
        "command_id": command_id,
        "sequence_id": sequence_id,
        "issued_at": 1_000_000_000,
        "expires_at": 2_000_000_000,
        "source": source,
        "task_id": "task-1",
        "stream_id": "b-stream",
        "ttl_s": 0.4,
    }


def test_parse_formation_velocity_command():
    cmd = parse_formation_command_payload({
        "vehicle_name": "Drone1",
        "mode": "velocity",
        "desired_velocity_ned_mps": [1.0, 0.0, 0.0],
        "lifecycle": _lifecycle(),
    })
    assert cmd.vehicle_name == "Drone1"
    assert cmd.mode == "velocity"
    assert cmd.desired_velocity_ned_mps == (1.0, 0.0, 0.0)
    assert cmd.require_lifecycle().command_id == "c1"


def test_parse_formation_position_command():
    cmd = parse_formation_command_payload({
        "vehicle_name": "Drone2",
        "mode": "position",
        "target_position_ned_m": [5.0, 6.0, -1.0],
        "lifecycle": _lifecycle(),
    })
    assert cmd.target_position_ned_m == (5.0, 6.0, -1.0)


def test_parse_formation_requires_lifecycle_for_dispatch():
    cmd = parse_formation_command_payload({
        "vehicle_name": "Drone1",
        "mode": "velocity",
        "desired_velocity_ned_mps": [1.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="requires lifecycle"):
        cmd.require_lifecycle()


def test_parse_formation_rejects_source_mismatch():
    with pytest.raises(WireError, match="source"):
        parse_formation_command_payload({
            "vehicle_name": "Drone1",
            "mode": "velocity",
            "desired_velocity_ned_mps": [1.0, 0.0, 0.0],
            "lifecycle": _lifecycle(source="orca"),
        })


def test_parse_formation_rejects_missing_lifecycle_field():
    lc = _lifecycle()
    del lc["expires_at"]
    with pytest.raises(WireError, match="expires_at"):
        parse_formation_command_payload({
            "vehicle_name": "Drone1",
            "mode": "velocity",
            "lifecycle": lc,
        })


# ---------------------------------------------------------------------------
# command.ack / coordination.status
# ---------------------------------------------------------------------------


def test_command_ack_reject():
    env = build_command_ack(
        accepted=False, reason="duplicate_command_id", command_id="c1",
        producer_id="a", stream_id="a-stream", sequence_id=9,
    )
    assert env["type"] == "command.ack"
    assert env["payload"] == {
        "accepted": False, "reason": "duplicate_command_id", "command_id": "c1",
    }


def test_coordination_status_valid_and_invalid():
    payload = build_coordination_status_payload(
        vehicle_name="Drone1", state="NORMAL", snapshot_id=7,
        sim_timestamp_ns=1_000_000_000,
    )
    assert payload["state"] == "NORMAL"
    with pytest.raises(WireError, match="state"):
        build_coordination_status_payload(vehicle_name="Drone1", state="WEIRD")


def test_status_type_constant_present():
    assert TYPE_COORDINATION_STATUS == "coordination.status"

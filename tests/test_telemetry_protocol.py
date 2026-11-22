import time

import pytest

from telemetry.protocol import ProtocolError, SnapshotValidator, snapshot_to_telemetry


def valid_frame(sequence=1, stream="stream-1"):
    return {
        "protocol_version": 1,
        "type": "swarm.snapshot",
        "producer_id": "a-state-service",
        "stream_id": stream,
        "sequence_id": sequence,
        "sent_at_unix_ms": 1700000000000,
        "payload": {
            "snapshot_id": f"snap-{sequence}",
            "collection_complete": True,
            "collection_duration_s": 0.02,
            "max_receive_skew_s": 0.01,
            "max_sim_skew_s": 0.01,
            "vehicles": {
                "Drone1": {
                    "valid": True,
                    "validity_reason": "",
                    "state": {
                        "position_ned_m": [1, 2, -1],
                        "linear_velocity_ned_mps": [0.1, 0.2, 0],
                        "orientation_quaternion_xyzw": [0, 0, 0, 1],
                        "angular_velocity_body_radps": [0, 0, 0],
                        "timestamp": 123,
                        "received_monotonic_seconds": 10.0,
                    },
                }
            },
        },
    }


def test_snapshot_converts_to_frontend_contract():
    frame = snapshot_to_telemetry(valid_frame(), received_at_unix_ms=1700000000123).as_dict()
    assert frame["type"] == "telemetry"
    assert frame["frame"] == "world_ned"
    assert frame["timestamp"] == 1700000000123
    assert frame["drones"]["Drone1"]["position_ned_m"] == [1.0, 2.0, -1.0]
    assert frame["drones"]["Drone1"]["sim_timestamp_ns"] == 123
    assert frame["goal"] is None and frame["obstacles"] == []
    assert frame["coordination_state"] == "UNKNOWN"
    assert frame["diagnostics"]["transport_latency_ms"] == 123


def test_optional_display_fields_are_forwarded_without_control_authority():
    source = valid_frame()
    source["payload"].update(
        {
            "coordination_state": "AVOIDING",
            "active_command": {"source": "orca", "mode": "velocity"},
            "goal": {"position_ned_m": [10, 2, -3], "label": "目标点"},
            "obstacles": [{"position_ned_m": [4, 5, -3], "radius_m": 1.5}],
        }
    )
    converted = snapshot_to_telemetry(source, received_at_unix_ms=1700000000100).as_dict()
    assert converted["coordination_state"] == "AVOIDING"
    assert converted["active_command"]["source"] == "orca"
    assert converted["goal"]["position_ned_m"] == [10.0, 2.0, -3.0]
    assert converted["obstacles"][0]["radius_m"] == 1.5


def test_protocol_accepts_the_full_five_vehicle_fleet():
    source = valid_frame()
    source["payload"]["vehicles"] = {
        f"Drone{index}": {
            "valid": True,
            "validity_reason": "",
            "state": {
                "position_ned_m": [index, 0, -3],
                "linear_velocity_ned_mps": [0, 0, 0],
                "orientation_quaternion_xyzw": [0, 0, 0, 1],
                "angular_velocity_body_radps": [0, 0, 0],
                "timestamp": index,
                "received_monotonic_seconds": 1.0,
            },
        }
        for index in range(1, 6)
    }
    source["payload"]["cameras"] = {"front": {"status": "offline"}}
    converted = snapshot_to_telemetry(source).as_dict()
    assert len(converted["drones"]) == 5
    assert converted["cameras"]["front"]["status"] == "offline"


def test_invalid_vehicle_has_no_state():
    frame = valid_frame()
    vehicle = frame["payload"]["vehicles"]["Drone1"]
    vehicle["valid"] = False
    vehicle["validity_reason"] = "state timeout"
    vehicle["state"] = None
    converted = snapshot_to_telemetry(frame).as_dict()
    assert converted["drones"]["Drone1"]["position_ned_m"] is None
    assert converted["status"] == "PARTIAL_STATE"


def test_validator_rejects_duplicate_and_wrong_producer():
    validator = SnapshotValidator()
    validator.accept(valid_frame())
    with pytest.raises(ProtocolError, match="duplicate"):
        validator.accept(valid_frame())
    wrong = valid_frame(sequence=2)
    wrong["producer_id"] = "demo"
    with pytest.raises(ProtocolError, match="producer"):
        validator.accept(wrong)


def test_validator_allows_new_stream_and_tracks_freshness():
    now = [100.0]
    validator = SnapshotValidator(clock=lambda: now[0])
    validator.accept(valid_frame())
    assert validator.is_fresh()
    now[0] += 3.01
    assert not validator.is_fresh()
    validator.accept(valid_frame(sequence=1, stream="restarted"))
    assert validator.is_fresh()


def test_invalid_frame_is_rejected():
    frame = valid_frame()
    frame["payload"]["vehicles"] = {}
    with pytest.raises(ProtocolError):
        snapshot_to_telemetry(frame)


def test_a_wire_integer_snapshot_id_and_vehicle_collection_time_are_supported():
    frame = valid_frame()
    frame["payload"]["snapshot_id"] = 5
    frame["payload"]["vehicles"]["Drone1"]["received_at_unix_ms"] = 1699999999990
    normalized = snapshot_to_telemetry(frame).as_dict()
    assert normalized["snapshot"]["snapshot_id"] == "5"
    assert normalized["drones"]["Drone1"]["received_at_unix_ms"] == 1699999999990

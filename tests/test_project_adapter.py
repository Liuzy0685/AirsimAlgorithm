from common.models import AvoidanceCommand, FormationCommand, TelemetryMessage
from coordination.state_provider import SwarmStateSnapshot
from models.vehicle_state import VehicleState
from telemetry.project_adapter import (
    snapshot_to_swarm_frame,
    telemetry_message_to_swarm_frame,
)
from telemetry.protocol import validate_swarm_snapshot


def make_state(index: int) -> VehicleState:
    return VehicleState(
        position_ned_m=[float(index), 2.0, -3.0],
        linear_velocity_ned_mps=[0.5, 0.0, 0.0],
        angular_velocity_body_radps=[0.0, 0.1, 0.0],
        orientation_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        timestamp=100 + index,
        ready=True,
        received_monotonic_seconds=10.0 + index / 100.0,
    )


def make_snapshot(count: int = 1, *, failed: str | None = None, complete: bool | None = None):
    names = tuple(f"Drone{index}" for index in range(1, count + 1))
    states = {name: make_state(index) for index, name in enumerate(names, 1) if name != failed}
    errors = {failed: "rpc timeout"} if failed else {}
    return (
        SwarmStateSnapshot(
            sequence_id=7,
            timestamp=10.1,
            states=states,
            collection_complete=(complete if complete is not None else failed is None),
            collection_duration=0.03,
            errors=errors,
            max_time_skew=0.01,
            max_sim_timestamp_skew=0.02,
        ),
        names,
    )


def test_real_snapshot_adapter_accepts_one_and_five_vehicle_snapshots():
    for count in (1, 5):
        snapshot, names = make_snapshot(count)
        frame = snapshot_to_swarm_frame(snapshot, vehicle_names=names, stream_id="a-stream", sent_at_unix_ms=1)
        normalized = validate_swarm_snapshot(frame)
        assert set(normalized["payload"]["vehicles"]) == set(names)
        assert all(item["valid"] for item in normalized["payload"]["vehicles"].values())


def test_failed_vehicle_and_incomplete_collection_are_preserved():
    snapshot, names = make_snapshot(3, failed="Drone2", complete=False)
    frame = snapshot_to_swarm_frame(snapshot, vehicle_names=names, stream_id="a-stream", sent_at_unix_ms=1)
    normalized = validate_swarm_snapshot(frame)
    vehicle = normalized["payload"]["vehicles"]["Drone2"]
    assert normalized["payload"]["collection_complete"] is False
    assert vehicle == {
        "valid": False,
        "validity_reason": "rpc timeout",
        "received_at_unix_ms": None,
        "state": None,
    }


def test_display_command_is_forwarded_without_giving_c_control_authority():
    snapshot, names = make_snapshot()
    frame = snapshot_to_swarm_frame(
        snapshot,
        vehicle_names=names,
        stream_id="a-stream",
        sent_at_unix_ms=1,
        display={
            "coordination_state": "AVOIDING",
            "active_command": {"source": "orca", "mode": "velocity"},
        },
    )
    normalized = validate_swarm_snapshot(frame)
    assert normalized["payload"]["active_command"]["source"] == "orca"


def test_legacy_telemetry_message_can_be_adapted():
    message = TelemetryMessage(
        timestamp_s=12.0,
        vehicles={"Drone1": make_state(1)},
        avoidance=AvoidanceCommand("Drone1", source="orca"),
        formation=FormationCommand("Drone1", mode="velocity"),
        coordination_state="AVOIDING",
    )
    frame = telemetry_message_to_swarm_frame(
        message, stream_id="legacy-stream", sequence_id=3, sent_at_unix_ms=2
    )
    normalized = validate_swarm_snapshot(frame)
    assert normalized["sequence_id"] == 3
    assert normalized["payload"]["active_command"] == {"source": "orca", "mode": "velocity"}

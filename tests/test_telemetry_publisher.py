import asyncio

from aiohttp.test_utils import TestClient, TestServer

from coordination.state_provider import SwarmStateSnapshot
from models.vehicle_state import VehicleState
from telemetry.publisher import SwarmPublisher


def make_snapshot(count=1, *, failed=None, complete=None):
    names = tuple(f"Drone{index}" for index in range(1, count + 1))
    states = {
        name: VehicleState(
            position_ned_m=[float(index), 2.0, -3.0],
            linear_velocity_ned_mps=[0.5, 0.0, 0.0],
            orientation_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            timestamp=100 + index,
            ready=True,
            received_monotonic_seconds=10.0,
        )
        for index, name in enumerate(names, 1)
        if name != failed
    }
    return (
        SwarmStateSnapshot(
            sequence_id=7,
            timestamp=10.1,
            states=states,
            collection_complete=(complete if complete is not None else failed is None),
            collection_duration=0.03,
            errors={failed: "rpc timeout"} if failed else {},
        ),
        names,
    )


def test_publisher_accepts_c_handshake_and_broadcasts_real_snapshot():
    async def scenario():
        publisher = SwarmPublisher(stream_id="a-stream")
        client = TestClient(TestServer(publisher.create_app()))
        await client.start_server()
        try:
            socket = await client.ws_connect("/ws/swarm")
            await socket.send_json(
                {
                    "protocol_version": 1,
                    "type": "connection.hello",
                    "role": "c_telemetry",
                    "producer_id": "c-telemetry-gateway",
                    "stream_id": "c-stream",
                }
            )
            snapshot, names = make_snapshot(2)
            frame = await publisher.publish_snapshot(snapshot, vehicle_names=names, sent_at_unix_ms=1)
            received = await socket.receive_json()
            assert frame["type"] == "swarm.snapshot"
            assert received["sequence_id"] == 7
            assert received["payload"]["vehicles"]["Drone2"]["valid"] is True
            await socket.close()
        finally:
            await publisher.close()
            await client.close()

    asyncio.run(scenario())


def test_publisher_keeps_failed_vehicle_and_never_handles_control_messages():
    async def scenario():
        publisher = SwarmPublisher(stream_id="a-stream")
        client = TestClient(TestServer(publisher.create_app()))
        await client.start_server()
        try:
            socket = await client.ws_connect("/ws/swarm")
            await socket.send_json(
                {
                    "protocol_version": 1,
                    "type": "connection.hello",
                    "role": "c_telemetry",
                    "producer_id": "c-telemetry-gateway",
                    "stream_id": "c-stream",
                }
            )
            await socket.send_json({"type": "formation.command", "payload": {"mode": "velocity"}})
            snapshot, names = make_snapshot(2, failed="Drone2", complete=False)
            await publisher.publish_snapshot(snapshot, vehicle_names=names, sent_at_unix_ms=1)
            received = await socket.receive_json()
            assert received["payload"]["vehicles"]["Drone2"]["state"] is None
            assert received["payload"]["vehicles"]["Drone2"]["validity_reason"] == "rpc timeout"
            assert publisher.latest_frame["type"] == "swarm.snapshot"
            await socket.close()
        finally:
            await publisher.close()
            await client.close()

    asyncio.run(scenario())

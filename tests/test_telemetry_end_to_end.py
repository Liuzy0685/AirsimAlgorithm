import asyncio

from aiohttp.test_utils import TestClient, TestServer

from coordination.state_provider import SwarmStateSnapshot
from models.vehicle_state import VehicleState
from telemetry.publisher import SwarmPublisher
from telemetry.server import GATEWAY_KEY, TelemetryConfig, create_app


def test_real_snapshot_reaches_browser_telemetry_frame():
    async def scenario():
        publisher = SwarmPublisher(stream_id="e2e-a-stream")
        upstream_client = TestClient(TestServer(publisher.create_app()))
        await upstream_client.start_server()
        try:
            upstream_url = str(upstream_client.server.make_url("/ws/swarm")).replace(
                "http://", "ws://"
            )
            config = TelemetryConfig(
                upstream_url=upstream_url,
                frontend_host="127.0.0.1",
                frontend_port=8000,
                telemetry_path="/ws/telemetry",
                reconnect_delays_s=(0.01,),
                frontend_offline_after_s=3.0,
                expected_producer="a-state-service",
            )
            app = create_app(config)
            gateway = app[GATEWAY_KEY]
            frontend_client = TestClient(TestServer(app))
            await frontend_client.start_server()
            try:
                browser = await frontend_client.ws_connect("/ws/telemetry")
                initial = await browser.receive_json()
                assert initial["status"] == "OFFLINE"

                for _ in range(100):
                    if gateway.upstream_connected:
                        break
                    await asyncio.sleep(0.01)
                assert gateway.upstream_connected

                snapshot = SwarmStateSnapshot(
                    sequence_id=42,
                    timestamp=12.0,
                    states={
                        "Drone1": VehicleState(
                            position_ned_m=[1.0, 2.0, -3.0],
                            linear_velocity_ned_mps=[0.5, 0.0, 0.0],
                            orientation_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                            timestamp=123,
                            ready=True,
                            received_monotonic_seconds=12.0,
                        )
                    },
                    collection_complete=True,
                    collection_duration=0.01,
                )
                await publisher.publish_snapshot(snapshot, vehicle_names=("Drone1",))

                received = None
                for _ in range(20):
                    candidate = await browser.receive_json(timeout=1)
                    if candidate.get("sequence_id") == 42:
                        received = candidate
                        break
                assert received is not None
                assert received["type"] == "telemetry"
                assert received["status"] == "NORMAL"
                assert received["drones"]["Drone1"]["position_ned_m"] == [1.0, 2.0, -3.0]
                await browser.close()
            finally:
                await frontend_client.close()
        finally:
            await upstream_client.close()

    asyncio.run(scenario())

import asyncio

from coordination.state_provider import SwarmStateSnapshot
from models.vehicle_state import VehicleState
from telemetry.provider_service import StatePublisherService


class FakeProvider:
    vehicle_names = ("Drone1",)

    def get_snapshot(self):
        return SwarmStateSnapshot(
            sequence_id=1,
            timestamp=1.0,
            states={"Drone1": VehicleState(timestamp=1, ready=True)},
            collection_complete=True,
            collection_duration=0.01,
        )


class FakePublisher:
    def __init__(self):
        self.calls = []

    async def publish_snapshot(self, snapshot, *, vehicle_names, display=None):
        self.calls.append((snapshot, tuple(vehicle_names), display))
        return {"type": "swarm.snapshot"}


def test_provider_service_publishes_the_provider_snapshot_and_fleet_names():
    async def scenario():
        publisher = FakePublisher()
        service = StatePublisherService(FakeProvider(), publisher, interval_s=1.0)
        frame = await service.publish_once()
        assert frame["type"] == "swarm.snapshot"
        assert publisher.calls[0][0].sequence_id == 1
        assert publisher.calls[0][1] == ("Drone1",)

    asyncio.run(scenario())

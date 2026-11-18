"""SwarmStatePublisher tests (authoritative-state fail-closed)."""

import pytest

from coordination.coordinate_transform import (
    CoordinateTransformer,
    CoordinateTransformValidation,
    VehicleTransformCheck,
)
from coordination.state_provider import StateProvider
from coordination.state_publisher import SwarmStatePublisher
from models.vehicle_state import VehicleState


class Clock:
    def __init__(self, now=10.0):
        self.now = now

    def __call__(self):
        return self.now


class Reader:
    def __init__(self, state, error=None):
        self.state, self.error = state, error

    def read(self):
        if self.error:
            raise self.error
        return self.state


def _state(timestamp=1, position=(1.0, 2.0, -3.0)):
    return VehicleState(
        position_ned_m=list(position), linear_velocity_ned_mps=[1.0, 0.0, 0.0],
        timestamp=timestamp, ready=True,
    )


def _provider(validated=True):
    offsets = {"Drone1": (10.0, 0.0, 0.0), "Drone2": (20.0, 0.0, 0.0)}
    tf = CoordinateTransformer(offsets)
    if validated:
        tf._validated = True
        tf._last_validation = CoordinateTransformValidation(
            validated=True,
            vehicles={name: VehicleTransformCheck(name, ok=True) for name in offsets},
        )
    readers = {name: Reader(_state(timestamp=i)) for i, name in enumerate(offsets, 1)}
    return StateProvider(readers, tf, clock=Clock())


def test_publish_authoritative_snapshot():
    pub = SwarmStatePublisher(_provider(), producer_id="a", stream_id="s")
    env = pub.collect_and_publish()
    assert env is not None
    assert env["type"] == "swarm.snapshot"
    assert env["producer_id"] == "a"
    assert env["stream_id"] == "s"
    assert env["payload"]["snapshot_id"] == 1
    assert pub.authoritative is True
    assert pub.last_error is None
    assert pub.latest_envelope is env


def test_unvalidated_transform_is_not_published():
    pub = SwarmStatePublisher(_provider(validated=False), producer_id="a", stream_id="s")
    env = pub.collect_and_publish()
    assert env is None
    assert pub.authoritative is False
    assert pub.latest_envelope is None
    assert "validate" in pub.last_error


def test_publisher_rejects_bad_ids():
    with pytest.raises(ValueError, match="producer_id"):
        SwarmStatePublisher(_provider(), producer_id="", stream_id="s")
    with pytest.raises(ValueError, match="stream_id"):
        SwarmStatePublisher(_provider(), producer_id="a", stream_id="   ")

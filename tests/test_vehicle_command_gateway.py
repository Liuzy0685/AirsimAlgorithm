"""VehicleCommandGateway authority + safety gate tests."""

import pytest

from common.models import FormationCommand
from control.command_authority import CommandAuthority
from control.safety_filter import SafetyFilter
from control.vehicle_command_gateway import VehicleCommandGateway
from control.velocity_arbitrator import VelocityCandidate
from models.vehicle_state import VehicleState


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeDispatcher:
    def __init__(self):
        self.velocities = []
        self.hovers = 0
        self.lands = 0
        self.takeoffs = 0

    def send_velocity_world_ned(self, vx, vy, vz, duration=None, vehicle_name=None):
        self.velocities.append((vehicle_name, vx, vy, vz, duration))

    def hover(self, vehicle_name=None):
        self.hovers += 1

    def land(self, vehicle_name=None):
        self.lands += 1

    def takeoff(self, vehicle_name=None):
        self.takeoffs += 1


def _state(position=(0.0, 0.0, -1.0)):
    return VehicleState(position_ned_m=list(position), ready=True)


def _gateway(dispatcher=None, safety=None):
    clock = Clock()
    authority = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    authority.acquire("Drone1", "ctrl-a")
    return (
        VehicleCommandGateway(
            authority, dispatcher=dispatcher or FakeDispatcher(),
            safety_filter=safety or SafetyFilter(),
        ),
        authority,
    )


def test_dispatch_velocity_requires_authority():
    gw, _ = _gateway()
    result = gw.dispatch_velocity("Drone1", 1.0, 0.0, 0.0, controller_id="ctrl-x")
    assert result.dispatched is False
    assert result.reason == "unauthorized_controller"


def test_dispatch_velocity_rejects_non_finite():
    gw, _ = _gateway()
    result = gw.dispatch_velocity("Drone1", float("inf"), 0.0, 0.0, controller_id="ctrl-a")
    assert result.reason == "non_finite"


def test_dispatch_velocity_ok():
    dispatcher = FakeDispatcher()
    gw, _ = _gateway(dispatcher=dispatcher)
    result = gw.dispatch_velocity("Drone1", 1.0, 2.0, 3.0, controller_id="ctrl-a")
    assert result.dispatched is True
    assert dispatcher.velocities == [("Drone1", 1.0, 2.0, 3.0, None)]


def test_dispatch_formation_safety_rejected():
    dispatcher = FakeDispatcher()
    safety = SafetyFilter(safety_distance_m=2.0, prediction_horizon_s=3.0)
    gw, _ = _gateway(dispatcher=dispatcher, safety=safety)
    neighbor = _state(position=(3.0, 0.0, -1.0))
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(1.0, 0.0, 0.0))
    result = gw.dispatch_formation(
        cmd, controller_id="ctrl-a", own_state=_state(), neighbors=[neighbor],
    )
    assert result.dispatched is False
    assert result.reason.startswith("inter_vehicle_conflict")
    assert dispatcher.velocities == []


def test_dispatch_formation_safe_dispatches():
    dispatcher = FakeDispatcher()
    gw, _ = _gateway(dispatcher=dispatcher)
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(1.0, 0.0, 0.0))
    result = gw.dispatch_formation(
        cmd, controller_id="ctrl-a", own_state=_state(), neighbors=[],
    )
    assert result.dispatched is True
    assert dispatcher.velocities[0][0] == "Drone1"


def test_dispatch_arbitrated_selects_safety_hold():
    dispatcher = FakeDispatcher()
    gw, _ = _gateway(dispatcher=dispatcher)
    result = gw.dispatch_arbitrated(
        [
            VelocityCandidate("formation", (2.0, 0.0, 0.0)),
            VelocityCandidate("safety_hold", (0.0, 0.0, 0.0)),
        ],
        vehicle_name="Drone1", controller_id="ctrl-a",
        own_state=_state(), neighbors=[],
    )
    assert result.dispatched is True
    assert dispatcher.velocities[0][1:4] == (0.0, 0.0, 0.0)


def test_emergency_stop_maps_to_hover():
    dispatcher = FakeDispatcher()
    gw, _ = _gateway(dispatcher=dispatcher)
    result = gw.emergency_stop("Drone1", controller_id="ctrl-a")
    assert result.dispatched is True
    assert dispatcher.hovers == 1


def test_lifecycle_methods_require_authority():
    dispatcher = FakeDispatcher()
    gw, _ = _gateway(dispatcher=dispatcher)
    assert gw.land("Drone1", controller_id="ctrl-x").reason == "unauthorized_controller"
    assert dispatcher.lands == 0

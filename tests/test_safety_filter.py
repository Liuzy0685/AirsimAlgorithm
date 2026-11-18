"""SafetyFilter formation-target gate tests."""

import math

import pytest

from common.models import FormationCommand
from control.safety_filter import SafetyFilter
from models.vehicle_state import VehicleState


def _state(position=(0.0, 0.0, -1.0), velocity=(0.0, 0.0, 0.0)):
    return VehicleState(
        position_ned_m=list(position), linear_velocity_ned_mps=list(velocity),
        ready=True,
    )


def _filter(**kw):
    return SafetyFilter(**kw)


def test_invalid_command_rejected():
    f = _filter()
    cmd = FormationCommand(vehicle_name="Drone1", valid=False)
    decision = f.check(cmd, own_state=_state(), neighbors=[])
    assert decision.allowed is False
    assert decision.reason == "invalid_formation_command"


def test_incomplete_neighbors_rejected():
    f = _filter()
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(1.0, 0.0, 0.0))
    decision = f.check(cmd, own_state=_state(), neighbors=[], neighbors_complete=False)
    assert decision.reason == "neighbor_state_incomplete"


def test_non_finite_velocity_rejected():
    f = _filter()
    decision = f.check_velocity(
        (float("nan"), 0.0, 0.0), own_state=_state(), neighbors=[]
    )
    assert decision.reason == "non_finite"


def test_horizontal_speed_clamped():
    f = _filter(max_horizontal_speed_mps=4.0)
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(10.0, 0.0, 5.0))
    decision = f.check(cmd, own_state=_state(), neighbors=[])
    assert decision.allowed is True
    assert math.isclose(decision.desired_velocity_ned_mps[0], 4.0)
    assert math.isclose(decision.desired_velocity_ned_mps[2], 1.0)  # vertical clamp


def test_inter_vehicle_conflict_rejected():
    f = _filter(safety_distance_m=2.0, prediction_horizon_s=3.0)
    own = _state(position=(0.0, 0.0, -1.0))
    neighbor = _state(position=(3.0, 0.0, -1.0))
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(1.0, 0.0, 0.0))
    decision = f.check(cmd, own_state=own, neighbors=[neighbor])
    assert decision.allowed is False
    assert decision.reason.startswith("inter_vehicle_conflict")


def test_separation_respected_when_moving_away():
    f = _filter(safety_distance_m=2.0, prediction_horizon_s=3.0)
    own = _state(position=(0.0, 0.0, -1.0))
    neighbor = _state(position=(3.0, 0.0, -1.0))
    cmd = FormationCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(-1.0, 0.0, 0.0))
    decision = f.check(cmd, own_state=own, neighbors=[neighbor])
    assert decision.allowed is True


def test_position_mode_derives_velocity_toward_target():
    f = _filter(max_horizontal_speed_mps=2.0, max_vertical_speed_mps=1.0)
    own = _state(position=(0.0, 0.0, -1.0))
    cmd = FormationCommand(
        vehicle_name="Drone1", mode="position", target_position_ned_m=(10.0, 0.0, -1.0)
    )
    decision = f.check(cmd, own_state=own, neighbors=[])
    assert decision.allowed is True
    assert math.isclose(decision.desired_velocity_ned_mps[0], 2.0)
    assert math.isclose(decision.desired_velocity_ned_mps[1], 0.0)

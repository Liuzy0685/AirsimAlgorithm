"""Offline contract tests for B's formation planner."""

import math
from dataclasses import replace

import pytest

from common.models import VehicleState
from coordination.state_provider import SwarmStateSnapshot
from formation import FormationConfig, FormationController, load_formation_config


def config(count=2, shape="column"):
    return FormationConfig(
        leader="Drone1", followers=tuple(f"Drone{i}" for i in range(2, count + 1)),
        shape=shape, spacing_m=3, max_snapshot_age_s=0.5, max_sim_skew_s=0.5,
        command_ttl_s=0.4, command_sim_lifetime_s=0.5, recovery_gain=0.5,
        max_recovery_speed_mps=1, recovery_tolerance_m=0.25,
    )


def snapshot(count=2, *, leader_yaw=0, positions=None, sequence=1):
    positions = positions or {}
    states = {
        f"Drone{i}": VehicleState(
            position_ned_m=list(positions.get(i, (0, 0, -2))),
            linear_velocity_ned_mps=[0, 0, 0], yaw_rad=leader_yaw if i == 1 else 0,
            timestamp=1_000_000_000, ready=True,
        ) for i in range(1, count + 1)
    }
    return SwarmStateSnapshot(sequence, 10, states, True, 0.01)


def plan(controller, snap, state="NORMAL", received=10, now=10.1):
    return controller.plan(snap, received_at_monotonic_s=received,
                           now_monotonic_s=now, coordination_state=state)


def test_single_vehicle_has_no_follower_command():
    assert plan(FormationController(config(1), task_id="mission"), snapshot(1)) == {}


def test_two_vehicle_column_rotates_with_leader_and_has_lifecycle():
    controller = FormationController(config(), task_id="mission", stream_id="boot")
    commands = plan(controller, snapshot(2, leader_yaw=math.pi / 2))
    command = commands["Drone2"]
    assert command.mode == "position"
    assert command.target_position_ned_m == pytest.approx((0, -3, -2))
    assert command.lifecycle.task_id == "mission"
    assert command.lifecycle.source == "formation"
    assert command.lifecycle.expires_at > command.lifecycle.issued_at
    assert plan(controller, snapshot(2, leader_yaw=math.pi / 2)) == {}


def test_five_vehicle_vee_and_switch_preserve_assignment():
    controller = FormationController(config(5, "vee"), task_id="mission")
    commands = plan(controller, snapshot(5))
    assert [commands[f"Drone{i}"].target_position_ned_m for i in range(2, 6)] == [
        (-3, 3, -2), (-3, -3, -2), (-6, 6, -2), (-6, -6, -2),
    ]
    controller.switch_shape("line")
    changed = plan(controller, snapshot(5, sequence=2))
    assert changed["Drone2"].target_position_ned_m == (0, 3, -2)
    assert changed["Drone5"].target_position_ned_m == (0, -6, -2)
    assert changed["Drone5"].lifecycle.sequence_id > commands["Drone5"].lifecycle.sequence_id


def test_rejects_stale_partial_skewed_and_unready_snapshots():
    controller = FormationController(config(), task_id="mission")
    assert plan(controller, snapshot(), now=11) == {}
    partial = replace(snapshot(), collection_complete=False)
    assert plan(controller, partial) == {}
    skewed = snapshot()
    skewed.states["Drone2"].timestamp += 600_000_000
    assert plan(controller, skewed) == {}
    unready = snapshot()
    unready.states["Drone1"].ready = False
    assert plan(controller, unready) == {}


def test_avoidance_handoff_and_recovery():
    controller = FormationController(config(), task_id="mission")
    assert plan(controller, snapshot(), "AVOIDING") == {}
    assert plan(controller, snapshot(sequence=2), "NORMAL") == {}
    recovering = plan(controller, snapshot(sequence=3), "CONFLICT_CLEARED")["Drone2"]
    assert recovering.mode == "velocity"
    assert recovering.desired_velocity_ned_mps == pytest.approx((-1, 0, 0))
    assert math.dist(recovering.desired_velocity_ned_mps, (0, 0, 0)) <= 1
    settled = plan(controller, snapshot(positions={2: (-3, 0, -2)}, sequence=4),
                   "FORMATION_RECOVERING")["Drone2"]
    assert settled.mode == "position"
    assert plan(controller, snapshot(positions={2: (-3, 0, -2)}, sequence=5))["Drone2"].mode == "position"


def test_config_file_loads():
    config_file = load_formation_config("configs/formation.yaml")
    assert config_file.vehicle_names == ("Drone1", "Drone2")

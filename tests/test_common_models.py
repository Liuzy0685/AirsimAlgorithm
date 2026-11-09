"""Phase 1 — unified data model tests (World NED).

Covers:
- FormationCommand / AvoidanceCommand / TelemetryMessage construction + validation.
- 1..5 drone data model (Drone1..Drone5) through the unified types.

Self-contained; no AirSim / numpy required.
"""

import math

import pytest

from common.models import (
    AvoidanceCommand,
    CommandLifecycle,
    FormationCommand,
    TelemetryMessage,
    VehicleState,
)


def _lifecycle(source="formation"):
    return CommandLifecycle(
        command_id="cmd-1", sequence_id=1, issued_at=1_000_000_000,
        expires_at=2_000_000_000, source=source, task_id="task-1",
        stream_id="session-1", ttl_s=0.4,
    )


def test_command_lifecycle_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="sequence_id"):
        CommandLifecycle("cmd", -1, 1, 2, "formation", "task", "session", 0.4)
    with pytest.raises(ValueError, match="issued_at/expires_at"):
        CommandLifecycle("cmd", 1, 2, 2, "formation", "task", "session", 0.4)
    with pytest.raises(ValueError, match="ttl_s"):
        CommandLifecycle("cmd", 1, 1, 2, "formation", "task", "session", 0)


def test_command_lifecycle_uses_separate_deadlines():
    meta = _lifecycle()
    assert meta.is_current(sim_time_ns=1_500_000_000,
                           received_monotonic_seconds=10.0,
                           now_monotonic_seconds=10.2)
    assert not meta.is_current(sim_time_ns=2_000_000_000,
                               received_monotonic_seconds=10.0,
                               now_monotonic_seconds=10.2)
    assert not meta.is_current(sim_time_ns=1_500_000_000,
                               received_monotonic_seconds=10.0,
                               now_monotonic_seconds=10.5)


def test_cross_process_commands_require_lifecycle():
    with pytest.raises(ValueError, match="requires lifecycle"):
        FormationCommand("Drone1").require_lifecycle()
    with pytest.raises(ValueError, match="requires lifecycle"):
        AvoidanceCommand("Drone1").require_lifecycle()
    command = FormationCommand("Drone1", lifecycle=_lifecycle())
    assert command.require_lifecycle().command_id == "cmd-1"
    avoidance = AvoidanceCommand("Drone1", source="orca", lifecycle=_lifecycle("orca"))
    assert avoidance.require_lifecycle().source == "orca"


def test_command_source_mismatch_rejected():
    with pytest.raises(ValueError, match="source"):
        FormationCommand("Drone1", lifecycle=_lifecycle("orca"))
    with pytest.raises(ValueError, match="source"):
        AvoidanceCommand("Drone1", source="orca", lifecycle=_lifecycle("formation"))


# ---------------------------------------------------------------------------
# FormationCommand
# ---------------------------------------------------------------------------


def test_formation_velocity_default_is_hover():
    cmd = FormationCommand(vehicle_name="Drone1")
    assert cmd.mode == "velocity"
    assert cmd.desired_velocity_ned_mps == (0.0, 0.0, 0.0)
    assert cmd.valid


def test_formation_position_mode_requires_target():
    with pytest.raises(ValueError, match="position"):
        FormationCommand(vehicle_name="Drone1", mode="position", target_position_ned_m=None)


def test_formation_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        FormationCommand(vehicle_name="Drone1", mode="teleport")


def test_formation_rejects_non_finite_velocity():
    with pytest.raises(ValueError, match="finite"):
        FormationCommand(
            vehicle_name="Drone1",
            desired_velocity_ned_mps=(1.0, float("nan"), 0.0),
        )


def test_formation_rejects_blank_vehicle_name():
    with pytest.raises(ValueError, match="vehicle_name"):
        FormationCommand(vehicle_name="   ")


# ---------------------------------------------------------------------------
# AvoidanceCommand
# ---------------------------------------------------------------------------


def test_avoidance_command_defaults():
    cmd = AvoidanceCommand(vehicle_name="Drone1")
    assert cmd.desired_velocity_ned_mps == (0.0, 0.0, 0.0)
    assert cmd.source == "none"
    assert cmd.valid


def test_avoidance_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        AvoidanceCommand(vehicle_name="Drone1", desired_velocity_ned_mps=(float("inf"), 0.0, 0.0))


# ---------------------------------------------------------------------------
# TelemetryMessage + 1..5 drone model
# ---------------------------------------------------------------------------


def _drone_state(index: int) -> VehicleState:
    return VehicleState(
        position_ned_m=[float(index), float(index) * 2.0, -1.0],
        linear_velocity_ned_mps=[0.1 * index, 0.0, 0.0],
        yaw_rad=0.0,
        ready=True,
        can_arm=True,
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_telemetry_supports_1_to_5_drones(count):
    names = [f"Drone{i}" for i in range(1, count + 1)]
    vehicles = {name: _drone_state(i) for i, name in enumerate(names, start=1)}
    msg = TelemetryMessage(timestamp_s=1.0, vehicles=vehicles, coordination_state="monitor")
    assert msg.vehicle_names() == tuple(names)
    for name in names:
        assert msg.vehicle(name) is vehicles[name]
    assert msg.vehicle("DroneMissing") is None


def test_telemetry_empty_by_default():
    msg = TelemetryMessage(timestamp_s=0.0)
    assert msg.vehicle_names() == ()
    assert msg.avoidance is None
    assert msg.formation is None
    assert msg.valid


def test_telemetry_carries_avoidance_and_formation():
    formation = FormationCommand(
        vehicle_name="Drone1",
        desired_velocity_ned_mps=(1.0, 0.0, 0.0),
        formation_id="line",
    )
    avoidance = AvoidanceCommand(
        vehicle_name="Drone1",
        desired_velocity_ned_mps=(0.5, 0.2, 0.0),
        source="orca",
    )
    msg = TelemetryMessage(
        timestamp_s=2.0,
        vehicles={"Drone1": _drone_state(1)},
        avoidance=avoidance,
        formation=formation,
        coordination_state="orca_filter",
    )
    assert msg.avoidance is avoidance
    assert msg.formation is formation
    assert msg.coordination_state == "orca_filter"


def test_max_drones_constant_matches_spec():
    from common.constants import MAX_DRONES

    assert MAX_DRONES == 5

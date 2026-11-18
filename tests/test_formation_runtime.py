"""B status handling with A's independently timed status/snapshot frames."""

from formation import FormationController
from formation.runtime import FormationStatusGate
from test_formation_controller import config, snapshot


def status(gate, name, state, sequence, at=10.0, task="mission"):
    return gate.ingest(dict(vehicle_name=name, state=state, task_id=task),
                       sequence_id=sequence, received_at=at)


def test_waits_for_every_fresh_status_and_rejects_other_tasks():
    gate = FormationStatusGate(("Drone1", "Drone2"), task_id="mission", max_age_s=2)
    assert gate.state_for_snapshot(now=10.1) is None
    assert not status(gate, "Drone1", "NORMAL", 1, task="other")
    assert status(gate, "Drone1", "NORMAL", 1)
    assert gate.state_for_snapshot(now=10.1) is None
    assert status(gate, "Drone2", "NORMAL", 2)
    assert gate.state_for_snapshot(now=10.1) == "NORMAL"
    assert gate.state_for_snapshot(now=12.0) is None
    gate.reset()
    assert gate.state_for_snapshot(now=10.1) is None


def test_clear_event_survives_normal_status_until_complete_snapshot():
    gate = FormationStatusGate(("Drone1", "Drone2"), task_id="mission", max_age_s=2)
    planner = FormationController(config(), task_id="mission")
    assert status(gate, "Drone1", "NORMAL", 1)
    assert status(gate, "Drone2", "AVOIDING", 2)
    assert gate.state_for_snapshot(now=10.1) == "AVOIDING"
    assert planner.plan(snapshot(sequence=1), received_at_monotonic_s=10,
                        now_monotonic_s=10.1, coordination_state="AVOIDING") == {}
    assert status(gate, "Drone2", "CONFLICT_CLEARED", 3)
    assert status(gate, "Drone2", "NORMAL", 4)
    assert gate.state_for_snapshot(now=10.1) == "CONFLICT_CLEARED"
    incomplete = snapshot(sequence=2)
    incomplete.states.pop("Drone2")
    assert planner.plan(incomplete, received_at_monotonic_s=10,
                        now_monotonic_s=10.1,
                        coordination_state=gate.state_for_snapshot(now=10.1)) == {}
    assert gate.state_for_snapshot(now=10.1) == "CONFLICT_CLEARED"
    commands = planner.plan(snapshot(sequence=3), received_at_monotonic_s=10,
                            now_monotonic_s=10.1,
                            coordination_state=gate.state_for_snapshot(now=10.1))
    assert commands["Drone2"].mode == "velocity"
    gate.commands_emitted()
    assert gate.state_for_snapshot(now=10.1) == "NORMAL"


def test_active_conflict_blocks_pending_clear_and_old_sequence():
    gate = FormationStatusGate(("Drone1", "Drone2"), task_id="mission", max_age_s=2)
    assert status(gate, "Drone1", "CONFLICT_CLEARED", 1)
    assert status(gate, "Drone2", "AVOIDING", 2)
    assert gate.state_for_snapshot(now=10.1) == "AVOIDING"
    assert not status(gate, "Drone2", "NORMAL", 2)
    assert status(gate, "Drone2", "NORMAL", 3)
    assert gate.state_for_snapshot(now=10.1) == "NORMAL"
    assert status(gate, "Drone2", "CONFLICT_CLEARED", 4)
    assert gate.state_for_snapshot(now=10.1) == "CONFLICT_CLEARED"


def test_new_avoidance_and_expiry_discard_old_clear_event():
    gate = FormationStatusGate(("Drone1", "Drone2"), task_id="mission", max_age_s=2)
    assert status(gate, "Drone1", "NORMAL", 1)
    assert status(gate, "Drone2", "CONFLICT_CLEARED", 2)
    assert status(gate, "Drone2", "AVOIDING", 3)
    assert status(gate, "Drone2", "NORMAL", 4)
    assert gate.state_for_snapshot(now=10.1) == "NORMAL"
    assert status(gate, "Drone2", "CONFLICT_CLEARED", 5)
    assert gate.state_for_snapshot(now=12.0) is None
    assert status(gate, "Drone1", "NORMAL", 6, at=12.1)
    assert status(gate, "Drone2", "NORMAL", 7, at=12.1)
    assert gate.state_for_snapshot(now=12.2) == "NORMAL"


def test_disconnect_emergency_blocks_and_malformed_status_is_ignored():
    gate = FormationStatusGate(("Drone1", "Drone2"), task_id="mission", max_age_s=2)
    assert status(gate, "Drone1", "NORMAL", 1)
    assert status(gate, "Drone2", "NORMAL", 2)
    assert not gate.ingest(dict(vehicle_name=[], state="NORMAL", task_id="mission"),
                           sequence_id=3, received_at=10)
    assert status(gate, "Drone2", "EMERGENCY", 3, task="")
    assert gate.state_for_snapshot(now=10.1) == "EMERGENCY"

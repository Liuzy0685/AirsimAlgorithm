"""VelocityArbitrator priority tests."""

from common.models import AvoidanceCommand, FormationCommand
from control.velocity_arbitrator import (
    VelocityArbitrator,
    VelocityCandidate,
    avoidance_candidate,
    formation_candidate,
)


def test_safety_hold_beats_formation_and_navigation():
    arb = VelocityArbitrator()
    winner = arb.arbitrate([
        VelocityCandidate("navigation", (1.0, 0.0, 0.0)),
        VelocityCandidate("formation", (1.5, 0.0, 0.0)),
        VelocityCandidate("safety_hold", (0.0, 0.0, 0.0)),
    ])
    assert winner.source == "safety_hold"
    assert winner.valid is True


def test_avoidance_beats_formation():
    arb = VelocityArbitrator()
    winner = arb.arbitrate([
        VelocityCandidate("formation", (1.0, 0.0, 0.0)),
        VelocityCandidate("orca", (0.5, 0.2, 0.0)),
    ])
    assert winner.source == "orca"


def test_invalid_candidates_ignored():
    arb = VelocityArbitrator()
    winner = arb.arbitrate([
        VelocityCandidate("formation", (1.0, 0.0, 0.0), valid=False),
        VelocityCandidate("navigation", (0.1, 0.0, 0.0)),
    ])
    assert winner.source == "navigation"


def test_no_valid_candidate():
    arb = VelocityArbitrator()
    winner = arb.arbitrate([VelocityCandidate("formation", (1.0, 0.0, 0.0), valid=False)])
    assert winner.valid is False
    assert winner.reason == "no_valid_candidate"


def test_candidate_helpers():
    formation = FormationCommand(
        vehicle_name="Drone1", desired_velocity_ned_mps=(1.0, 0.0, 0.0)
    )
    avoidance = AvoidanceCommand(
        vehicle_name="Drone1", desired_velocity_ned_mps=(0.5, 0.0, 0.0), source="orca"
    )
    assert formation_candidate(formation).source == "formation"
    assert avoidance_candidate(avoidance).source == "orca"
    assert formation_candidate(None) is None
    assert avoidance_candidate(AvoidanceCommand(vehicle_name="Drone1", valid=False)) is None

"""Regression for the observed Blocks RPC omitting ready/can_arm fields."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS

from sensors.state_reader import StateReader
from coordination.state_provider import StateProvider
from coordination.coordinate_transform import CoordinateTransformer, CoordinateTransformValidation, VehicleTransformCheck
from coordination.wire import build_swarm_snapshot_envelope, parse_envelope, parse_swarm_snapshot
from formation.controller import FormationController
from test_formation_controller import config, snapshot


def read_state(*, reported=None):
    class LegacyRaw:
        ready = False
        can_arm = False
    raw = LegacyRaw()
    raw.kinematics_estimated = NS(position=NS(x_val=0., y_val=0., z_val=-1.),
        linear_velocity=NS(x_val=0., y_val=0., z_val=0.),
        angular_velocity=NS(x_val=0., y_val=0., z_val=0.),
        orientation=NS(x_val=0., y_val=0., z_val=0., w_val=1.))
    raw.timestamp, raw.landed_state = 1000000000, 0
    if reported is not None: raw.ready = reported
    adapter=NS(vehicle_name="Drone1", get_raw_client=lambda: NS(getMultirotorState=lambda **kw: raw))
    return StateReader(adapter,euler_converter=lambda q: (0.,0.,0.)).read()


def collect(state, *, legacy=False):
    tf = CoordinateTransformer({"Drone1": (0.,0.,0.)})
    tf._validated = True
    tf._last_validation = CoordinateTransformValidation(validated=True,
        vehicles={"Drone1": VehicleTransformCheck("Drone1", ok=True)})
    return StateProvider({"Drone1": NS(read=lambda: state)}, tf,
        allow_legacy_readiness=legacy).get_snapshot()


def test_missing_ready_does_not_become_a_false_reported_failure():
    state = read_state()
    assert not state.ready and not state.readiness_available
    assert not collect(state).collection_complete
    snap=collect(state,legacy=True)
    assert snap.collection_complete
    env=build_swarm_snapshot_envelope(snap,configured_names=["Drone1"],producer_id="a",stream_id="s")
    decoded=parse_swarm_snapshot(parse_envelope(json.dumps(env)),received_at=10.)
    assert not decoded.states["Drone1"].ready
    assert not decoded.states["Drone1"].readiness_available


def test_explicit_ready_false_is_rejected_even_with_legacy_opt_in():
    state=read_state(reported=False)
    assert state.readiness_available
    assert not collect(state,legacy=True).collection_complete


def test_b_requires_explicit_legacy_opt_in_and_still_rejects_real_not_ready():
    snap=snapshot()
    for state in snap.states.values():
        state.ready=False
        state.readiness_available=False
    assert not FormationController(config(),task_id="m").plan(snap,received_at_monotonic_s=10,now_monotonic_s=10.1)
    b=FormationController(replace(config(),allow_legacy_readiness=True),task_id="m")
    assert b.plan(snap,received_at_monotonic_s=10,now_monotonic_s=10.1)
    snap=replace(snap,sequence_id=snap.sequence_id+1)
    snap.states["Drone1"].readiness_available=True
    assert not b.plan(snap,received_at_monotonic_s=10,now_monotonic_s=10.1)

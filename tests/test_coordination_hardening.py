import math
from types import SimpleNamespace as NS

import pytest

from common.models import VehicleState
from control.safety_filter import SafetyFilter
from control.velocity_controller import VelocityCommandRejected
from coordination.orca_2d import compute_orca_velocity


@pytest.mark.parametrize('field', ['position_ned_m', 'linear_velocity_ned_mps'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None, [0., 0.], [False, 0., 0.]])
@pytest.mark.parametrize('own', [True, False])
def test_corrupt_state_cannot_pass_cpa(field, bad, own):
    a, b = VehicleState(position_ned_m=[0., 0., -2.]), VehicleState(position_ned_m=[10., 0., -2.])
    value = bad if bad is None or isinstance(bad, list) else [bad, 0., 0.]
    setattr(a if own else b, field, value)
    decision = SafetyFilter().check_velocity((0., 0., 0.), own_state=a, neighbors=[b])
    assert not decision.allowed and decision.reason == 'invalid_vehicle_state'


@pytest.mark.parametrize('distance', [3., 4., 5., 6., 8.])
def test_measured_peer_constraint_matches_final_cpa(distance):
    own = VehicleState(position_ned_m=[0., 0., -2.])
    peer = VehicleState(position_ned_m=[distance, 0., -2.], linear_velocity_ned_mps=[-1., 0., 0.])
    candidate = compute_orca_velocity(position=(0, 0), velocity=(1, 0), preferred_velocity=(1, 0),
        neighbors=[((distance, 0), (-1, 0))], safety_distance_m=2.05, horizon_s=3.,
        time_step_s=.1, max_speed_mps=1., passing_bias_mps=.1, responsibility=1.)
    assert candidate is not None
    assert SafetyFilter(max_horizontal_speed_mps=1.).check_velocity(
        (*candidate, 0.), own_state=own, neighbors=[peer]).allowed


@pytest.mark.parametrize('responsibility', [True, .49, 1.01, float('nan')])
def test_invalid_responsibility_is_rejected(responsibility):
    with pytest.raises(ValueError):
        compute_orca_velocity(position=(0,0), velocity=(0,0), preferred_velocity=(1,0),
            neighbors=[], safety_distance_m=2., horizon_s=3., time_step_s=.1,
            max_speed_mps=1., responsibility=responsibility)


@pytest.mark.parametrize('failure', ['state_expired', 'lidar_expired', 'disconnected', 'lock_lost'])
def test_deadlines_rechecked_after_slow_clearance(monkeypatch, failure):
    from test_swarm_wiring import bridge_fixture
    b, raw, _ = bridge_fixture()
    def slow(*args):
        if failure == 'state_expired':
            from dataclasses import replace
            b.link._snapshot = replace(b.link._snapshot, timestamp=b.link._snapshot.timestamp-1.)
        if failure == 'lidar_expired': b.points_at -= 1.
        if failure == 'disconnected': b.link._connected = False
        if failure == 'lock_lost': b.session._owns_lock = False
        return math.inf
    monkeypatch.setattr('control.swarm_flight_bridge.measured_command_clearance', slow)
    with pytest.raises(VelocityCommandRejected):
        b.proxy.moveByVelocityAsync(0.,0.,0.,.1,vehicle_name=b.name)
    assert not raw.calls
    b.prepare_cleanup()
    assert b.session.state.phase.name == 'MANUAL_INTERVENTION_REQUIRED'


def test_independent_goals_require_all_vehicles_arrived():
    from test_swarm_wiring import bridge_fixture
    from coordination.navigation import NavigationConfig
    b, _, _ = bridge_fixture()
    b.navigation_cfg = NavigationConfig(enabled=True, independent_goals=True)
    b._independent_members = [dict(vehicle_name=n,goal_x=0.,goal_y=0.,target_z=-1.) for n in b.link.names]
    for n in b.link.names:
        b.link._snapshot.states[n].position_ned_m = [0.,0.,-1.]
        b.link._snapshot.states[n].linear_velocity_ned_mps = [0.,0.,0.]
    assert not b.follower and b._mission_geometry_settled()
    b.link._snapshot.states['Drone1'].position_ned_m[0] = 2.
    assert not b.mission_settled()


def test_mission_completion_requires_distinct_frames_and_stable_dwell():
    from dataclasses import replace
    from test_swarm_wiring import bridge_fixture
    b,_,_=bridge_fixture()
    clock=NS(now=10.)
    b._status_clock=lambda:clock.now
    b._mission_geometry_settled=lambda:True
    assert not b.mission_settled()
    for i in range(1,6):
        clock.now=10.+i*.4
        b.link._snapshot=replace(b.link._snapshot,sequence_id=i+1)
        assert b.mission_settled() is (i==5)
    b._mission_geometry_settled=lambda:False
    assert not b.mission_settled()


@pytest.mark.parametrize('name', ['head_on','crossing','overtaking','stationary_peer','five_way'])
@pytest.mark.parametrize('dt', [.05,.1,.15])
def test_production_solver_and_cpa_complete_encounters(name,dt):
    from scripts.validate_inter_uav_encounters import replay, scenarios
    result = replay(*scenarios()[name],dt=dt)
    assert result['passed'], {k:v for k,v in result.items() if k != 'trace'}
    assert result['minimum_segment_separation_m'] >= 2.


def test_nonconvex_fallback_finds_safe_alternate_side():
    from coordination.orca_2d import compute_gateway_velocity
    # Inside the preferred planning margin, but outside the hard distance:
    # safe retreat must remain available without reducing the hard minimum.
    inputs=dict(position=(0.,0.),velocity=(.4,0.),preferred_velocity=(1.,0.),
        neighbors=[((2.2,0.),(0.,0.))],safety_distance_m=2.3,horizon_s=3.,
        time_step_s=.1,max_speed_mps=1.,passing_bias_mps=.1)
    candidate=compute_gateway_velocity(**inputs,minimum_distance_m=2.01)
    assert candidate is not None
    assert SafetyFilter(max_horizontal_speed_mps=1.).check_velocity((*candidate,0.),
        own_state=VehicleState(position_ned_m=[0.,0.,-2.]),
        neighbors=[VehicleState(position_ned_m=[2.2,0.,-2.])]).allowed


@pytest.mark.parametrize('age', [-.1,float('nan')])
def test_invalid_peer_age_never_reaches_solver(monkeypatch,age):
    from test_inter_uav_orca import make_mode
    mode,state=make_mode()
    mode._swarm_peer_states['Drone2']['received_monotonic']=100.-age
    monkeypatch.setattr('flight_modes.automatic_mode.time.monotonic',lambda:100.)
    assert mode._apply_inter_uav_orca(state,1.,0.,0.,.1,[])[2]=='safety_hold_stale_peer'


def test_swapped_endpoint_occupied_by_stationary_peer_can_begin_checked_approach():
    from test_swarm_wiring import bridge_fixture
    from coordination.navigation import NavigationConfig,CommittedGridRouter
    import numpy as np
    b,_,_=bridge_fixture()
    b.navigation_cfg=NavigationConfig(enabled=True,independent_goals=True,resolution_m=.25,clearance_m=1.1)
    b.router=CommittedGridRouter(b.navigation_cfg)
    b.points_world=np.array([[30.,30.,-1.]])
    own=b.link._snapshot.states[b.name]
    own.position_ned_m=[10.,0.,-1.]
    peer=b.link._snapshot.states['Drone1']
    peer.position_ned_m=[0.,0.,-1.];peer.linear_velocity_ned_mps=[0.,0.,0.]
    waypoint=b.route_goal(own,(0.,0.,-1.))
    assert b._navigation_goal==(0.,0.,-1.)
    assert b.router.path_available and waypoint[0]<10.
    assert b._staging_peer=='Drone1'
    staging=b._staging_goal
    peer.linear_velocity_ned_mps=[.5,0.,0.]
    b.route_goal(own,(0.,0.,-1.))
    assert b._staging_goal==staging


def test_staging_endpoint_cannot_be_inside_a_third_parked_vehicle():
    from test_swarm_wiring import bridge_fixture
    from coordination.navigation import NavigationConfig,CommittedGridRouter
    import numpy as np
    b,_,_=bridge_fixture()
    b.navigation_cfg=NavigationConfig(enabled=True,independent_goals=True,resolution_m=.25,clearance_m=1.1)
    b.router=CommittedGridRouter(b.navigation_cfg)
    b.points_world=np.array([[30.,30.,-1.]])
    own=b.link._snapshot.states[b.name];own.position_ned_m=[0.,6.,-1.]
    peer=b.link._snapshot.states['Drone1'];peer.position_ned_m=[0.,0.,-1.]
    peer.linear_velocity_ned_mps=[0.,0.,0.]
    import copy
    third=copy.deepcopy(peer);third.position_ned_m=[0.,3.,-1.]
    b.link._snapshot.states['Drone3']=third;b.link.names.add('Drone3')
    b.route_goal(own,(0.,0.,-1.))
    assert abs(b._staging_goal[0])>2.5
    assert b.router.path_available


@pytest.mark.parametrize('guard', ['motion_guard','lifecycle_guard'])
def test_revoked_lease_during_safety_check_cannot_reach_raw_rpc(guard):
    from test_swarm_wiring import bridge_fixture
    b,raw,_=bridge_fixture()
    def revoke(api,args,kwargs):
        b.authority.release(b.name,b.controller_id)
        return args,kwargs
    setattr(b.proxy,guard,revoke)
    with pytest.raises(PermissionError):
        b.proxy.moveByVelocityAsync(0.,0.,0.,.1,vehicle_name=b.name)
    assert not raw.calls


@pytest.mark.parametrize('blocked', [False,True])
def test_latest_peer_refinement_still_requires_full_raw_cloud_clearance(blocked):
    from test_swarm_wiring import bridge_fixture
    import numpy as np
    b,raw,_=bridge_fixture();b._gateway_refine_enabled=True
    own=b.link._snapshot.states[b.name]
    own.position_ned_m=[10.,0.,-1.];own.linear_velocity_ned_mps=[-1.,0.,0.]
    peer=b.link._snapshot.states['Drone1']
    peer.position_ned_m=[5.,0.,-1.];peer.linear_velocity_ned_mps=[1.,0.,0.]
    if blocked:b.points=np.array([[.1,0.,0.]])
    if blocked:
        with pytest.raises(VelocityCommandRejected,match='lidar_clearance'):
            b.proxy.moveByVelocityAsync(-1.,0.,0.,.1,vehicle_name=b.name)
        assert not raw.calls
    else:
        args,kwargs=b.guard_motion('moveByVelocityAsync',(-1.,0.,0.,.1),{})
        assert b._gateway_refined and args[:3]!=(-1.,0.,0.)
        assert b.safety.check_velocity(args[:3],own_state=own,neighbors=[peer]).allowed
        b.report_dispatch('navigation_route')
        assert b.link._pending_status[-1]['state']=='AVOIDING'

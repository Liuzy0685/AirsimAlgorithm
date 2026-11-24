"""Analytic policy and gateway integration tests; no simulator flights."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from coordination.navigation import CommittedGridRouter, NavigationConfig, segment_clear
from coordination.navigation_hardening import (HardeningConfig, ObservedStaticMemory,
    ProgressGuard, MotionLimiter, PassageAdvisor, RuntimeDiagnostics, latency_speed_cap, stopping_speed_cap, swept_free_distance, stop_category)


def cfg(**kwargs):
    return replace(HardeningConfig(enabled=True), **kwargs)


def test_static_observation_does_not_expire_because_it_left_view():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(2.,0.)],0.,scan_id=1)
    assert (2.,0.) in map(tuple,memory.points(120.))


def test_three_distinct_clear_rays_remove_a_changed_obstacle():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(2.,0.)],0.,scan_id=1)
    for i in (2,3):
        memory.observe([(4.,0.)],float(i),origin=(0.,0.),scan_id=i)
        assert (2.,0.) in map(tuple,memory.points(i))
    memory.observe([(4.,0.)],4.,origin=(0.,0.),scan_id=4)
    assert (2.,0.) not in map(tuple,memory.points(4.))


def test_empty_missing_and_replayed_scans_cannot_clear_walls():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(2.,0.)],0.,scan_id=1)
    memory.observe([(4.,0.)],1.,origin=(0.,0.),scan_id=2)
    for _ in range(10):
        memory.observe([(4.,0.)],2.,origin=(0.,0.),scan_id=2)
    memory.observe([],3.,origin=(0.,0.),scan_id=3)
    assert (2.,0.) in map(tuple,memory.points(3.))


def test_ray_ending_in_front_of_wall_does_not_clear_behind_it():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(3.,0.)],0.,scan_id=1)
    for i in range(2,8):memory.observe([(2.,0.)],i,origin=(0.,0.),scan_id=i)
    assert (3.,0.) in map(tuple,memory.points(8.))


def test_diagonal_nonplanar_hits_are_not_clearing_evidence():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(2.,0.)],0.,scan_id=1)
    for i in range(2,8):
        memory.observe([(4.,0.)],i,origin=(0.,0.),scan_id=i,clearing_points=[])
    assert (2.,0.) in map(tuple,memory.points(8.))


def test_known_peer_returns_are_temporary_but_do_not_erase_static_memory():
    memory=ObservedStaticMemory(.25,cfg())
    memory.observe([(3.,0.)],0.,scan_id=1)
    memory.observe([(2.,0.),(3.,0.)],1.,scan_id=2,dynamic_regions=[(2.5,0.,.8)])
    points=set(map(tuple,memory.points(1.1)))
    assert (2.,0.) in points and (3.,0.) in points
    points=set(map(tuple,memory.points(2.)))
    assert (2.,0.) not in points and (3.,0.) in points


def test_memory_size_and_clear_votes_remain_bounded():
    memory=ObservedStaticMemory(.25,cfg(max_memory_cells=100))
    memory.observe([(float(i),0.) for i in range(500)],0.,scan_id=1)
    assert len(memory.cells)==100
    assert set(memory.free_votes)<=memory.cells.keys()


@pytest.mark.parametrize('reason',['inter_uav_yield','passage_wait','leader_waiting_for_formation',
    'formation_wait','lidar_unavailable_at_gateway','authoritative_snapshot_unavailable'])
def test_intentional_or_unavailable_waits_do_not_trigger_stuck_recovery(reason):
    guard=ProgressGuard(cfg())
    assert not any(guard.update(i,(0.,0.),(10.,0.),expected=True,reason=reason) for i in range(20))
    assert guard.reason.startswith('paused_')


def test_real_stagnation_requires_both_displacement_and_progress_failure():
    guard=ProgressGuard(cfg())
    result=[guard.update(i,(0.,0.),(10.,0.),expected=True,reason='navigation_route',commanded_speed=.2)
            for i in range(7)]
    assert result[-1] and guard.reason=='command_not_executed'
    guard=ProgressGuard(cfg())
    assert not any(guard.update(i,(i*.1,0.),(10.,0.),expected=True) for i in range(20))


def test_goal_change_or_pause_restarts_the_stall_window():
    guard=ProgressGuard(cfg())
    for i in range(6):assert not guard.update(i,(0.,0.),(10.,0.),expected=True)
    assert not guard.update(6,(0.,0.),(20.,0.),expected=True)
    assert not guard.update(7,(0.,0.),(20.,0.),expected=False)
    assert not guard.update(8,(0.,0.),(20.,0.),expected=True)


def test_high_rate_samples_do_not_truncate_the_configured_stall_window():
    guard=ProgressGuard(cfg(stall_window_s=10.))
    assert any(guard.update(i/200,(0.,0.),(10.,0.),expected=True) for i in range(2100))
    assert len(guard.history)<=256


def test_failed_route_memory_is_a_penalty_not_a_fake_obstacle():
    clock=SimpleNamespace(now=0.)
    router=CommittedGridRouter(NavigationConfig(enabled=True),clock=lambda:clock.now,hardening=cfg())
    router.path=[(0.,0.),(.25,0.),(1.,0.),(2.,0.)]
    router.note_stall((0.,0.,-2.),'no_route_progress')
    assert router.failed_cells and not router.path
    assert len(router.obstacles())==0
    clock.now=31.
    router.waypoint((0.,0.,-2.),(4.,0.,-2.))
    assert not router.failed_cells


def test_retreat_checks_entire_traveled_segment_against_current_points_and_peers():
    router=CommittedGridRouter(NavigationConfig(enabled=True),hardening=cfg())
    router.travel_history=[(0.,0.),(.5,0.),(1.,0.)]
    retreat=router.verified_retreat((1.,0.,-2.))
    assert retreat is not None and retreat[0]<1.
    assert router.verified_retreat((1.,0.,-2.),temporary_obstacles=[(.5,0.)]) is None
    router.observe([(.5,0.)])
    assert router.verified_retreat((1.,0.,-2.)) is None


def test_smoothing_bounds_acceleration_but_never_delays_requested_stop():
    limiter=MotionLimiter(.4)
    first=limiter.limit((1.,0.),(0.,0.),0.)
    assert math.hypot(*first)<=.4*.05+1e-12
    second=limiter.limit((0.,1.),first,.1)
    assert math.dist(first,second)<=.4*.1+1e-12
    assert limiter.limit((0.,0.),second,.11)==(0.,0.)


def test_command_ramp_reaches_requested_speed_despite_persistent_tracking_error():
    limiter=MotionLimiter(.35)
    command=(0.,0.)
    for i in range(100):
        measured=(.9*command[0],.9*command[1])
        command=limiter.limit((.35,0.),measured,i*.05)
        limiter.accept(command)
    assert command==pytest.approx((.35,0.))


def test_filtered_stop_reanchors_the_next_command_ramp():
    limiter=MotionLimiter(.35)
    for i in range(30):limiter.limit((.35,0.),(0.,0.),i*.05)
    limiter.accept((0.,0.))
    result=limiter.limit((.35,0.),(0.,0.),1.5)
    assert math.hypot(*result)<=.35*.05+1e-12


def test_long_pause_reanchors_from_measured_velocity_instead_of_replaying_old_command():
    limiter=MotionLimiter(.35)
    for i in range(30):limiter.limit((.35,0.),(0.,0.),i*.05)
    result=limiter.limit((.35,0.),(0.,0.),3.)
    assert math.hypot(*result)<=.35*.2+1e-12


def test_bridge_feeds_actual_gateway_velocity_back_to_smoother():
    from test_swarm_wiring import bridge_fixture
    bridge,_,_=bridge_fixture()
    bridge.hardening=cfg()
    bridge.motion_limiter.accept((.35,0.))
    bridge._last_guard_velocity=(0.,0.,0.)
    bridge.report_dispatch('inter_uav_orca_fallback_hold',avoiding=True)
    assert bridge.motion_limiter.last_command==(0.,0.)


def test_speed_cap_reduces_with_latency_and_respects_braking_equation():
    early=latency_speed_cap(2.,.9,.1,.35,2.)
    late=latency_speed_cap(2.,.9,.5,.35,2.)
    assert 0<late<early
    assert late*.5+late*late/(2*.35)<=1.+1e-12
    assert latency_speed_cap(.9,.9,.1,.35,2.)==0.


@pytest.mark.parametrize('smoothing',[False,True])
def test_legacy_smoothing_yaml_keeps_combined_policy(smoothing):
    assert cfg(smoothing=smoothing).latency_limit_enabled is smoothing


@pytest.mark.parametrize('smoothing',[False,True])
@pytest.mark.parametrize('latency_limit',[False,True])
def test_motion_policies_can_be_isolated_without_changing_gate(monkeypatch,smoothing,latency_limit):
    import time
    from test_swarm_wiring import bridge_fixture
    import control.swarm_flight_bridge as module
    bridge,_,_=bridge_fixture()
    bridge.hardening=cfg(passage=False,smoothing=smoothing,latency_limit=latency_limit)
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg,hardening=bridge.hardening)
    bridge.router.path_available=True
    own=bridge.link.snapshot().states[bridge.name]
    x,y,z=own.position_ned_m
    bridge._navigation_waypoint=(x+3.,y,z)
    bridge.points_world=np.array([(x,y+1.1,z)])
    bridge.points_at=time.monotonic()
    calls=[]
    def cap(*args):
        calls.append('cap')
        return .1
    def smooth(desired,*args):
        calls.append('smooth')
        return desired
    monkeypatch.setattr(module,'latency_speed_cap',cap)
    monkeypatch.setattr(bridge.motion_limiter,'limit',smooth)
    bridge._navigation_velocity(own,0.,'trajectory',bridge.link.command())
    assert ('cap' in calls) is latency_limit
    assert ('smooth' in calls) is smoothing
    assert bridge.cfg.snapshot_max_age_s==.6
    assert bridge._navigation_motion['latency_limit_enabled'] is latency_limit


@pytest.mark.parametrize('bad',['true',1,0,[],{}])
def test_invalid_latency_limit_switch_rejected(bad):
    with pytest.raises(ValueError):cfg(latency_limit=bad)


@pytest.mark.parametrize('dimensions',[2,3])
def test_swept_distance_ignores_safe_side_and_rear_but_not_forward_points(dimensions):
    point=lambda x,y:(x,y) if dimensions==2 else (x,y,0.)
    origin=point(0.,0.);heading=point(1.,0.)
    assert math.isinf(swept_free_distance([point(0.,1.1),point(-1.1,0.)],origin,heading,1.05))
    assert swept_free_distance([point(2.,0.)],origin,heading,1.05)==pytest.approx(.95)
    assert swept_free_distance([point(-1.,0.)],origin,heading,1.05)==0.


@pytest.mark.parametrize('age',[0.,.1,.5])
def test_swept_braking_distance_respects_buffered_cloud(age):
    rng=np.random.default_rng(381)
    points=rng.uniform(-5.,5.,(500,2))
    points=points[np.linalg.norm(points,axis=1)>1.2]
    direction=np.array((.6,.8))
    distance=swept_free_distance(points,(0.,0.),direction,1.05)
    speed=stopping_speed_cap(distance,age,.35,2.)
    travel=speed*age+speed*speed/(2*.35)
    assert travel<=distance+1e-10
    assert segment_clear((0.,0.),direction*max(0.,travel-1e-9),points,1.05-1e-9)


def test_swept_limiter_rechecks_heading_after_smoothing(monkeypatch):
    import time
    from test_swarm_wiring import bridge_fixture
    bridge,_,_=bridge_fixture()
    bridge.hardening=cfg(passage=False,latency_clearance_mode='swept')
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg,hardening=bridge.hardening)
    bridge.router.path_available=True
    own=bridge.link.snapshot().states[bridge.name]
    x,y,z=own.position_ned_m
    bridge._navigation_waypoint=(x+3.,y,z)
    # Route tangent is parallel to this return; the changed heading approaches it.
    bridge.points_world=np.array([(x,y+1.1,z)])
    bridge.points_at=time.monotonic()
    monkeypatch.setattr(bridge.motion_limiter,'limit',lambda *args:(0.,.35))
    result=bridge._navigation_velocity(own,0.,'trajectory',bridge.link.command())
    assert 0<math.hypot(*result[:2])<.2
    assert bridge._navigation_motion['post_smoothing_cap_mps']<.2


@pytest.mark.parametrize('points',[None,np.empty((0,3))])
def test_swept_policy_never_treats_missing_raw_cloud_as_free(points):
    from test_swarm_wiring import bridge_fixture
    bridge,_,_=bridge_fixture()
    bridge.points_world=points
    assert bridge._swept_speed_cap((0.,0.,0.),np.empty((0,2)),(.35,0.),.1,.35)==(0.,0.)


@pytest.mark.parametrize('mode',['',True,'fast',1,[]])
def test_invalid_latency_clearance_mode_rejected(mode):
    with pytest.raises(ValueError):cfg(latency_clearance_mode=mode)


def test_passage_tie_break_matches_existing_yield_policy_without_circular_priority():
    a=PassageAdvisor(cfg(),'Drone1',{})
    b=PassageAdvisor(cfg(),'Drone2',{})
    assert a.update(0.,(0.,0.),(4.,0.),{'Drone2':((3.,0.),(-.2,0.))},narrow=True,safety_distance=2.)==(False,False)
    assert b.update(0.,(3.,0.),(-1.,0.),{'Drone1':((0.,0.),(.2,0.))},narrow=True,safety_distance=2.)==(True,False)
    assert a.state in ('ENTERING','OCCUPYING') and b.state=='YIELDING'


def test_passage_timeout_requests_retreat_and_never_forces_grant():
    advisor=PassageAdvisor(cfg(),'Drone2',{})
    peers={'Drone1':((0.,0.),(.2,0.))}
    advisor.update(0.,(3.,0.),(-1.,0.),peers,narrow=True,safety_distance=2.)
    assert advisor.update(9.,(3.,0.),(-1.,0.),peers,narrow=True,safety_distance=2.)==(True,True)
    assert advisor.state=='BLOCKED_TIMEOUT'
    assert advisor.update(10.,(3.,0.),(-1.,0.),{},narrow=False,safety_distance=2.)==(False,False)
    assert advisor.state=='EXITING'


def test_stationary_leader_waiting_for_followers_is_not_an_oncoming_vehicle():
    advisor=PassageAdvisor(cfg(),'Drone2',{})
    assert advisor.update(0.,(0.,0.),(4.,0.),{'Drone1':((3.,0.),(0.,0.),(10.,0.))},
        narrow=True,safety_distance=2.)==(False,False)
    assert advisor.update(1.,(0.,0.),(4.,0.),{'Drone1':((3.,0.),(0.,0.),(-4.,0.))},
        narrow=True,safety_distance=2.)==(True,False)


def test_bounded_diagnostics_reject_nonfinite_samples_and_explain_failure():
    diagnostics=RuntimeDiagnostics(capacity=3)
    for i in range(8):diagnostics.record('lidar_unavailable_at_gateway',gateway_ms=i,sensor_age_ms=math.nan)
    summary=diagnostics.summary()
    assert summary['samples']==3 and summary['category']=='perception_unavailable'
    assert summary['timing']['gateway_ms']['maximum']==7.
    assert 'sensor_age_ms' not in summary['timing']
    assert stop_category('no_verified_cleanup_action_manual_intervention_required')=='cleanup_unverified'


def test_bridge_gateway_instrumentation_preserves_rejection_and_raw_safety():
    from test_swarm_wiring import bridge_fixture
    from control.velocity_controller import VelocityCommandRejected
    bridge,raw,_=bridge_fixture()
    bridge.hardening=cfg()
    bridge.points=np.array([[0.,0.,0.]])
    with pytest.raises(VelocityCommandRejected):
        bridge.guard_motion('moveByVelocityAsync',(.1,0.,0.,.1),{})
    assert bridge.diagnostics.summary()['reason']


def test_open_map_with_ground_returns_and_empty_horizontal_memory_can_move():
    import time
    from test_swarm_wiring import bridge_fixture
    bridge,_,_=bridge_fixture()
    bridge.hardening=cfg(passage=False)
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg,hardening=bridge.hardening,minimum_clearance_m=.5)
    bridge.router.path_available=True
    own=bridge.link.snapshot().states[bridge.name]
    x,y,z=own.position_ned_m
    bridge._navigation_waypoint=(x+2.,y,z)
    bridge.points_world=np.array([(x,y,z+2.)])
    bridge.points_at=time.monotonic()
    result=bridge._navigation_velocity(own,0.,'trajectory',bridge.link.command())
    assert math.hypot(*result[:2])>0


def test_slow_diagnostics_cannot_extend_a_previously_checked_dispatch_deadline(monkeypatch):
    from test_swarm_wiring import bridge_fixture
    from control.velocity_controller import VelocityCommandRejected
    import control.swarm_flight_bridge as module
    bridge,_,_=bridge_fixture()
    bridge.hardening=cfg()
    clock=SimpleNamespace(now=10.)
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock.now,perf_counter=lambda:clock.now))
    def checked(*args):
        bridge._gate_state_deadline=10.2
        bridge.points_at=10.
        return (.1,0.,0.,.1),{}
    def slow_log():clock.now=10.3
    monkeypatch.setattr(bridge,'_guard_motion_impl',checked)
    monkeypatch.setattr(bridge,'_emit_runtime_diagnostics',slow_log)
    with pytest.raises(VelocityCommandRejected,match='dispatch_deadline_elapsed'):
        bridge.guard_motion('moveByVelocityAsync',(.1,0.,0.,.1),{})


@pytest.mark.parametrize('overrides',[{'enabled':'true'},{'free_confirmations':1},
    {'max_memory_cells':9000},{'clearing_rays':1000},{'stall_window_s':math.inf},
    {'acceleration_mps2':0.},{'passage_wait_s':True}])
def test_invalid_hardening_limits_are_rejected(overrides):
    with pytest.raises(ValueError):cfg(**overrides)

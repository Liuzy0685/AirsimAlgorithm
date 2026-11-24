import math
from dataclasses import replace
import numpy as np
import pytest
from coordination.navigation import NavigationConfig,CommittedGridRouter,segment_clear,validate_navigation_feedback
from formation.navigation import FormationNavigation
from test_formation_controller import config,snapshot


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


def feedback(points=(),goal=(10.,0.,-2.),observation_id=1_000_000_000):
    return dict(observation_id=observation_id,obstacle_points_xy_m=list(points),goal_ned_m=list(goal),
        path_available=True,goal_blocked=False,tracking_error_m=2.,route_version=1)


def test_committed_route_goes_around_pillar_and_never_declares_blocked_goal_reached():
    clock=Clock();cfg=NavigationConfig(enabled=True)
    router=CommittedGridRouter(cfg,clock=clock)
    circle=np.array([(5.+math.cos(t),math.sin(t)) for t in np.linspace(0,2*math.pi,100)])
    router.observe(circle)
    waypoint=router.waypoint((0.,0.,-2.),(10.,0.,-2.))
    assert router.path_available and abs(waypoint[1])>.1
    assert all(segment_clear(a,b,router.obstacles(),cfg.clearance_m) for a,b in zip(router.path,router.path[1:]))
    version=router.route_version
    clock.now=.1
    assert router.waypoint((.02,0.,-2.),(10.,0.,-2.))[1]*waypoint[1]>0
    assert router.route_version==version
    router.observe([[10.,0.]])
    assert router.waypoint((.02,0.,-2.),(10.,0.,-2.))==(.02,0.,-2.)
    assert router.goal_blocked and not router.path_available


def test_new_obstacle_on_committed_segment_stops_before_replan_deadline():
    clock=Clock();router=CommittedGridRouter(NavigationConfig(enabled=True),clock=clock)
    router.observe([(5.,0.)]);router.waypoint((0.,0.,-2.),(10.,0.,-2.))
    assert router.path_available
    router.observe([(0.4,0.)]);clock.now=.1
    assert router.waypoint((0.,0.,-2.),(10.,0.,-2.))==(0.,0.,-2.)
    assert not router.path_available


def test_old_obstacle_observations_expire_and_malformed_feedback_is_rejected():
    clock=Clock();router=CommittedGridRouter(NavigationConfig(obstacle_ttl_s=1.),clock=clock)
    router.observe([(1.,0.)]);clock.now=1.1
    assert len(router.obstacles())==0
    for invalid in (dict(feedback(),tracking_error_m=float('nan')),
                    dict(feedback(),path_available='true'),dict(feedback(),obstacle_points_xy_m=[[0.,float('inf')]])):
        with pytest.raises(ValueError):validate_navigation_feedback(invalid)


def test_adaptive_slots_avoid_pillar_keep_separation_and_restore_nominal():
    cfg=NavigationConfig(enabled=True,adaptive_slots=True,trail_fallback=False)
    snap=snapshot(5)
    from formation.controller import FormationController
    planner=FormationController(config(5,'line'),task_id='mission')
    nav=FormationNavigation(cfg,task_id='mission',vehicle_names=config(5).vehicle_names,safety_distance_m=2.)
    nominal=planner.slot_targets(snap.states['Drone1'])
    assert nav.targets(snap,nominal,leader='Drone1',now=0.) is None
    for name in nav.names:
        nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=feedback([[0.,3.]])),received_at=0.)
    targets=nav.targets(snap,nominal,leader='Drone1',now=.1)
    assert targets is not None and targets['Drone2']!=nominal['Drone2']
    assert all(math.dist(p[:2],(0.,3.))>=cfg.clearance_m+.6/math.sqrt(2.) for p in targets.values())
    assert all(math.dist(a,b)>=2.3 for i,a in enumerate(targets.values()) for b in list(targets.values())[i+1:])
    commands=planner.plan(snap,received_at_monotonic_s=0.,now_monotonic_s=.1,target_overrides=targets)
    assert commands['Drone2'].target_position_ned_m==targets['Drone2']
    for name in nav.names:
        nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=feedback(observation_id=1_100_000_000)),received_at=9.)
    assert nav.targets(snap,nominal,leader='Drone1',now=9.1)==nominal


def test_repeated_or_reconnected_old_scan_never_refreshes_navigation_freshness():
    snap=snapshot(2);cfg=NavigationConfig(enabled=True,adaptive_slots=True)
    nav=FormationNavigation(cfg,task_id='mission',vehicle_names=config().vehicle_names,safety_distance_m=2.)
    for name in nav.names:
        payload=dict(vehicle_name=name,task_id='mission',navigation=feedback())
        assert nav.ingest(payload,received_at=0.)
        assert not nav.ingest(payload,received_at=1.4)
    assert nav.targets(snap,{'Drone2':(-3.,0.,-2.)},leader='Drone1',now=1.6) is None
    nav.reset()
    for name in nav.names:
        nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=feedback(observation_id=1)),received_at=10.)
    snap.states['Drone1'].timestamp=10_000_000_000
    assert nav.targets(snap,{'Drone2':(-3.,0.,-2.)},leader='Drone1',now=10.1) is None


def test_navigation_feedback_is_identity_checked_and_server_heartbeat_cannot_renew_controller_status():
    import json
    from test_swarm_wiring import fixture,connect_controller
    from coordination.wire import build_envelope
    clock,names,publisher,_,service=fixture()
    link,cid=connect_controller(service,publisher,names)
    stamp=service.latest_sim_time_ns
    payload=dict(vehicle_name='Drone2',state='FORMATION_RECOVERING',task_id='swarm-mission',
        navigation=feedback(observation_id=stamp))
    env=build_envelope(message_type='coordination.status',producer_id=link.producer_id,
        stream_id=link.stream_id,sequence_id=1,payload=payload)
    service.handle_message(cid,json.dumps(env))
    assert service._last_status['Drone2']['navigation']['observation_id']==stamp
    c=service.register_client(role='c_telemetry',producer_id='C',stream_id='c',send=lambda _:None)
    malicious=build_envelope(message_type='coordination.status',producer_id='C',stream_id='c',sequence_id=2,
        payload=dict(payload,navigation=feedback(observation_id=stamp+100)))
    service.handle_message(c,json.dumps(malicious))
    assert service._last_status['Drone2']['navigation']['observation_id']==stamp
    # Keep the authoritative state fresh while controller status stops.
    clock.now+=2.1;publisher.collect_and_publish()
    for _ in range(3):
        event=service.heartbeat_status('Drone2',2.)
        assert event['payload']['reason']=='controller_status_stale'
    assert service._controller_status_at['Drone2']<clock.now-2.


def test_route_mode_replaces_blind_recovery_with_a_checked_detour():
    from test_swarm_wiring import bridge_fixture,command,send_command
    from types import SimpleNamespace as NS
    import time
    bridge,raw,service=bridge_fixture()
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg)
    cmd=command(command_id='route',seq=2,recovery_target_ned_m=(20.,0.,-1.))
    assert send_command(service,cmd)[0]['payload']['accepted']
    own=bridge.link.snapshot().states['Drone2']
    frame=NS(sensor_pose=dict(position=dict(x=10.,y=0.,z=-1.),orientation=dict(x=0.,y=0.,z=0.,w=1.)),
        received_monotonic_seconds=time.monotonic(),raw_timestamp_ns=1_500_000_001,
        point_cloud_sensor=np.array([[2.,0.,0.]]))
    bridge.update_lidar(frame)
    goal=bridge.goal(own);bridge.route_goal(own,goal)
    vx,vy,vz,source=bridge.select(.5,0.,0.,'recovery',own)
    assert source=='navigation_route' and abs(vy)>.01
    assert math.hypot(vx,vy)<=.1+1e-9
    bridge.proxy.moveByVelocityAsync(vx,vy,vz,.15,vehicle_name='Drone2')
    assert len(raw.calls)==1


def test_safe_escape_exits_preferred_inflation_without_approaching_near_surface():
    router=CommittedGridRouter(NavigationConfig(enabled=True))
    router.observe([(1.2,.8)])
    p=(0.,0.,-2.)
    waypoint=router.waypoint(p,(6.,0.,-2.))
    assert router.escaping and router.path_available
    displacement=np.asarray(waypoint[:2])-np.asarray(p[:2])
    assert float(np.asarray((1.2,.8))@displacement)<=.01
    assert segment_clear(p,waypoint,router.obstacles(),.95+.4/math.sqrt(2.))


def test_waiting_peer_is_routed_around_without_becoming_a_persistent_static_obstacle():
    router=CommittedGridRouter(NavigationConfig(enabled=True))
    ring=[(5.+.9*math.cos(i*math.pi/16),.9*math.sin(i*math.pi/16)) for i in range(32)]
    waypoint=router.waypoint((0.,0.,-2.),(10.,0.,-2.),temporary_obstacles=ring)
    assert router.path_available and abs(waypoint[1])>.1
    assert len(router.obstacles())==0
    assert router.waypoint((0.,0.,-2.),(10.,0.,-2.))==(10.,0.,-2.)


def test_leader_waits_for_fresh_follower_progress_and_resumes_with_hysteresis():
    from test_swarm_wiring import bridge_fixture
    import time
    bridge,_,_=bridge_fixture()
    bridge.name='Drone1';bridge.navigation_cfg=NavigationConfig(enabled=True)
    bridge.router.path_available=True;bridge._navigation_waypoint=(10.,0.,-1.)
    own=bridge.link.snapshot().states['Drone1']
    def selected():return bridge.select(.3,0.,0.,'trajectory',own)
    assert selected()[3]=='leader_waiting_for_formation'
    sim=max(s.timestamp for s in bridge.link._snapshot.states.values())
    for error,moves in ((6.,False),(3.,False),(1.,True)):
        nav=feedback(observation_id=sim);nav['tracking_error_m']=error
        bridge.link._navigation_received['Drone2']=(nav,time.monotonic())
        velocity=selected()
        assert (velocity[0]>0)==moves
    nav['path_available']=False
    bridge.link._navigation_received['Drone2']=(nav,time.monotonic())
    assert selected()[3]=='leader_waiting_for_formation'


def test_trail_uses_leader_history_through_corner_and_stages_missing_history():
    from formation.controller import FormationController
    snap=snapshot(2,positions={2:(0.,3.,-2.)})
    planner=FormationController(config(2,'line'),task_id='mission')
    nav=FormationNavigation(NavigationConfig(enabled=True,adaptive_slots=True,resolution_m=.25,clearance_m=1.1),
        task_id='mission',vehicle_names=config(2).vehicle_names,safety_distance_m=2.,max_speed_mps=.1,prediction_horizon_s=1.)
    def update(x,y,stamp,points,now,at_goal=False):
        snap.states['Drone1'].position_ned_m=[x,y,-2.]
        for name in nav.names:
            snap.states[name].timestamp=stamp
            f=feedback(points,observation_id=stamp);f['obstacle_resolution_m']=.25
            if name=='Drone1' and at_goal:f['tracking_error_m']=.05
            nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=f),received_at=now)
        return nav.targets(snap,planner.slot_targets(snap.states['Drone1']),leader='Drone1',now=now+.01)
    update(0.,0.,1_000_000_000,[],0.)
    goals=update(2.,0.,1_200_000_000,[[2.,3.]],.2)
    assert nav.trailing and goals['Drone2']==(0.,3.,-2.)
    update(4.,0.,1_400_000_000,[],.4)
    goals=update(4.,4.,1_600_000_000,[[4.,7.]],.6)
    assert nav.trailing and goals['Drone2']==pytest.approx((4.,1.,-2.))
    # Once the leader is at its task goal and old obstacle observations have
    # expired, followers recover their requested nominal shape.
    goals=update(4.,4.,10_000_000_000,[],9.,at_goal=True)
    assert nav.trailing
    goals=update(4.,4.,12_000_000_000,[],11.1,at_goal=True)
    assert not nav.trailing and goals['Drone2']==(4.,7.,-2.)


@pytest.mark.parametrize('values',[dict(adaptive_slots=True),dict(clearance_m=float('nan')),
    dict(resolution_m=.001),dict(max_expansions=10000000),dict(leader_resume_error_m=10.)])
def test_invalid_navigation_profiles_fail_before_flight(values):
    with pytest.raises(ValueError):NavigationConfig(**values)


def test_B_cooperative_stop_exits_without_connecting_or_restarting(tmp_path,monkeypatch):
    import asyncio
    from scripts import formation_service
    from types import SimpleNamespace as NS
    stop=tmp_path/'formation.stop';stop.touch()
    async def forbidden(*args,**kwargs):pytest.fail('Stopped B must not reconnect')
    monkeypatch.setattr(formation_service.SwarmConnection,'connect',forbidden)
    asyncio.run(formation_service.run(NS(swarm_config='configs/acceptance/swarm_2_navigation.yaml',
        formation_config='configs/acceptance/formation_2.yaml',stop_file=str(stop))))


def test_holding_a_trail_join_point_is_not_reported_as_nominal_formation_complete():
    from test_swarm_wiring import bridge_fixture,command,send_command
    bridge,_,service=bridge_fixture()
    own=bridge.link.snapshot().states['Drone2']
    cmd=command(command_id='trail-hold',seq=2,mode='position',
        target_position_ned_m=tuple(own.position_ned_m),formation_id='line:trail')
    assert send_command(service,cmd)[0]['payload']['accepted']
    bridge.adopted_command=bridge.link.command()
    bridge.report_dispatch('navigation_route')
    assert bridge.link._pending_status[-1]['state']=='FORMATION_RECOVERING'


def test_emergency_status_is_not_lost_because_optional_navigation_observation_is_old():
    from test_swarm_wiring import fixture,connect_controller
    from coordination.wire import build_envelope
    import json
    _,names,publisher,_,service=fixture()
    link,cid=connect_controller(service,publisher,names)
    link.navigation_feedback=feedback(observation_id=1)
    link.report('EMERGENCY','lidar_unavailable')
    payload=link._pending_status[-1]
    assert 'navigation' not in payload
    env=build_envelope(message_type='coordination.status',producer_id=link.producer_id,
        stream_id=link.stream_id,sequence_id=1,payload=payload)
    service.handle_message(cid,json.dumps(env))
    assert service._last_status['Drone2']['state']=='EMERGENCY'


def test_trail_spacing_reserves_stop_go_room_without_changing_nominal_shape():
    from formation.controller import FormationController
    nav=FormationNavigation(NavigationConfig(enabled=True,adaptive_slots=True),task_id='mission',
        vehicle_names=config().vehicle_names,safety_distance_m=2.,spacing_m=3.,max_speed_mps=.35,prediction_horizon_s=3.)
    assert nav.spacing==pytest.approx(4.6)
    assert FormationController(config(2,'line'),task_id='mission').slot_targets(snapshot().states['Drone1'])['Drone2']==(0.,3.,-2.)


def test_finer_feedback_does_not_erase_quantization_uncertainty_of_remembered_coarse_points():
    snap=snapshot(2,positions={2:(0.,3.,-2.)})
    nav=FormationNavigation(NavigationConfig(enabled=True,adaptive_slots=True,trail_fallback=False,clearance_m=1.1),
        task_id='mission',vehicle_names=config().vehicle_names,safety_distance_m=2.)
    for name in nav.names:
        nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=feedback([[0.,1.7]])),received_at=0.)
        fine=feedback(observation_id=1_100_000_000);fine['obstacle_resolution_m']=.25
        nav.ingest(dict(vehicle_name=name,task_id='mission',navigation=fine),received_at=.2)
    goals=nav.targets(snap,{'Drone2':(0.,3.,-2.)},leader='Drone1',now=.3)
    assert goals is not None and goals['Drone2']!=(0.,3.,-2.)

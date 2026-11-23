from dataclasses import replace
import json
import math
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from coordination.navigation import NavigationConfig,CommittedGridRouter
from coordination.navigation_hardening import HardeningConfig
from coordination.navigation_trace import NavigationTraceRecorder,replay_snapshot


def recorded_router(*,temporary=(),last_plan=-math.inf):
    router=CommittedGridRouter(NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1),
        hardening=HardeningConfig(enabled=True),clock=lambda:10.)
    points=np.array([(1.25*math.cos(t),1.25*math.sin(t)) for t in np.linspace(0,2*math.pi,24,endpoint=False)])
    router.observe(points,scan_id=1)
    router.last_plan=last_plan;router.capture_next=True
    router.waypoint((0.,0.,-2.),(4.,0.,-2.),temporary_obstacles=temporary)
    return router


def test_replay_reproduces_blocked_start_and_escape_rejections():
    router=recorded_router()
    snapshot=router.captured_diagnostic
    replay=replay_snapshot(snapshot)
    assert not replay['path_available']
    assert replay['reason']==snapshot['outcome']['reason']=='start_blocked'
    assert replay['searches'][0]['start_blocked']
    assert replay['escape']['rejected']['inside_minimum_buffer']>0
    assert replay['waypoint']==pytest.approx(snapshot['outcome']['waypoint'])


def test_escape_diagnostic_distinguishes_approaching_surfaces():
    router=recorded_router()
    points=np.array([(0.,-1.25),(1.8,0.)])
    router._escape_path((0.,0.,-2.),(4.,0.,-2.),points)
    assert router.escape_diagnostic['rejected']['approaching_surface']>0


def test_replay_keeps_peer_obstacles_even_without_static_returns():
    router=CommittedGridRouter(NavigationConfig(enabled=True),clock=lambda:10.)
    router.capture_next=True
    router.waypoint((0.,0.,-2.),(4.,0.,-2.),temporary_obstacles=[(4.,0.)])
    snapshot=router.captured_diagnostic
    assert replay_snapshot(snapshot)['goal_blocked']
    cleared=dict(snapshot,temporary_obstacles=[])
    assert replay_snapshot(cleared)['path_available']


def test_replay_preserves_replan_interval_without_using_wall_time():
    router=recorded_router(last_plan=9.5)
    assert router.captured_diagnostic['outcome']['reason']=='waiting_to_replan'
    assert replay_snapshot(router.captured_diagnostic)['reason']=='waiting_to_replan'


def test_captured_geometry_is_not_changed_by_later_memory_updates():
    router=recorded_router()
    snapshot=router.captured_diagnostic
    before=snapshot['static_obstacles'].copy()
    router.memory.cells.clear();router.observe([(20.,20.)],scan_id=2)
    np.testing.assert_array_equal(snapshot['static_obstacles'],before)
    assert not replay_snapshot(snapshot)['path_available']


def test_capture_does_not_change_route_decision():
    cfg=NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1)
    routers=[CommittedGridRouter(cfg,clock=lambda:10.) for _ in range(2)]
    for router in routers:router.observe([(2.,0.)])
    routers[1].capture_next=True
    decisions=[r.waypoint((0.,0.,-2.),(4.,0.,-2.)) for r in routers]
    assert decisions[0]==decisions[1]
    assert routers[0].path==routers[1].path


def test_writer_is_bounded_nonblocking_and_keeps_replayable_geometry(tmp_path,monkeypatch):
    recorder=NavigationTraceRecorder(tmp_path,'Drone/2',max_records=2)
    started=threading.Event();release=threading.Event();threads=[]
    original=recorder._write
    def slow(index,payload):
        threads.append(threading.get_ident());started.set()
        assert release.wait(3.)
        original(index,payload)
    monkeypatch.setattr(recorder,'_write',slow)
    payload=recorded_router().captured_diagnostic
    assert recorder.submit(payload,10.)
    assert started.wait(1.)
    assert recorder.submit(payload,15.)
    assert not recorder.submit(payload,20.)
    release.set();assert recorder.close()
    assert recorder.written==2 and recorder.dropped==1
    assert all(t!=threading.get_ident() for t in threads)
    files=list(tmp_path.glob('*.json'));assert len(files)==2
    document=json.loads(files[0].read_text(encoding='utf-8'))
    assert document['source_file_sha256']
    assert not replay_snapshot(document)['path_available']


def test_writer_error_does_not_escape_to_control_caller(tmp_path,monkeypatch):
    recorder=NavigationTraceRecorder(tmp_path,'Drone2')
    def broken(*args):raise OSError('disk unavailable')
    monkeypatch.setattr(recorder,'_write',broken)
    assert recorder.submit(recorded_router().captured_diagnostic,10.)
    assert recorder.close() and recorder.errors==1


def test_oversized_snapshot_is_dropped_without_starting_writer(tmp_path):
    recorder=NavigationTraceRecorder(tmp_path,'Drone2')
    assert not recorder.submit(dict(static_obstacles=np.zeros((8001,2))),10.)
    assert recorder.submitted==0 and recorder._thread is None


def test_bridge_records_only_persistent_no_route(monkeypatch):
    import control.swarm_flight_bridge as module
    from test_swarm_wiring import bridge_fixture
    bridge,_,_=bridge_fixture();clock=NS(now=10.)
    monkeypatch.setattr(module,'time',NS(monotonic=lambda:clock.now,perf_counter=lambda:clock.now))
    bridge.navigation_cfg=NavigationConfig(enabled=True,independent_goals=True,resolution_m=.25,clearance_m=1.1)
    bridge.hardening=HardeningConfig(enabled=True)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg,clock=lambda:clock.now,hardening=bridge.hardening)
    own=bridge.link.snapshot().states[bridge.name]
    x,y,z=own.position_ned_m
    bridge.router.observe([(x+1.25*math.cos(t),y+1.25*math.sin(t)) for t in np.linspace(0,2*math.pi,24,endpoint=False)],scan_id=1)
    bridge.points_world=np.array([(x+20.,y+20.,z)])
    captures=[]
    bridge.trace_recorder=NS(ready=lambda now:True,submit=lambda data,now:captures.append(data))
    for dt in (0.,1.,2.1):
        clock.now=10.+dt;bridge.route_goal(own,(x+4.,y,z))
        if dt<2.:assert not captures
    assert len(captures)==1
    assert captures[0]['vehicle']==bridge.name
    assert captures[0]['outcome']['searches'][0]['start_blocked']

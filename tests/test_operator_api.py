import copy
import http.client
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from coordination.operator_common import OperatorError, RuntimeDirectoryLock, atomic_json, read_json
from coordination.operator_http import OperatorHTTPServer
from coordination.operator_manager import MissionManager
from coordination.operator_runtime import OperatorProfile, SubprocessMissionBackend
from scripts.operator_mission_runner import summarize


class Backend:
    def __init__(self):
        self.calls=[]
        self.code=None

    def launch(self, mission, directory, url):
        self.calls.append((mission,directory,url))
        return SimpleNamespace(pid=123,poll=lambda:self.code)


@pytest.fixture
def env(tmp_path):
    path=tmp_path/'map.json'
    atomic_json(path,dict(map_id='maze',frame='world_ned',units='m',bounds_xy_m=[-10,40,-20,20],
                          coverage='partial',obstacles=[]))
    p=OperatorProfile('configs/acceptance/settings_2.json','configs/acceptance/swarm_2_navigation.yaml',
                      'configs/acceptance/formation_2.yaml',path,allow_flight=True)
    states={n:dict(position_ned_m=[.5,-3.+3.*i,1.1],velocity_ned_mps=[0,0,0],
                   yaw_rad=0.,sim_timestamp_ns=1000000000,landed_state=0) for i,n in enumerate(p.names)}
    backend=Backend()
    clock=SimpleNamespace(now=1.)
    m=MissionManager(p,tmp_path/'jobs',lambda:copy.deepcopy(states),backend,clock=lambda:clock.now)
    body=dict(request_id='create-1',map_id='maze',map_revision=p.map.revision,vehicle_names=list(p.names),
              leader=p.service.leader,leader_goal_ned_m=[15.5,-3.,-2.],formation=dict(shape='line',spacing_m=3.))
    return SimpleNamespace(p=p,m=m,body=body,backend=backend,clock=clock,states=states)


def create(e):
    return e.m.create(e.body)['mission']['mission_id']


def start(e,mid):
    return e.m.start(mid,dict(request_id='start-1',expected_revision=1))


def test_create_preview_uses_common_coordinates_and_idempotent_request(env):
    result=env.m.validate(env.body)
    assert result['terminal_slots_preview']['Drone2']==(15.5,0.,-2.)
    first=env.m.create(env.body)
    assert env.m.create(env.body)==first
    assert len(env.m.data['missions'])==1
    assert '_b_token' not in json.dumps(first)
    with pytest.raises(OperatorError,match='reused'):
        env.m.create(dict(env.body,leader_goal_ned_m=[16.,-3.,-2.]))


@pytest.mark.parametrize('patch',[
    {'map_revision':'old'}, {'leader':'Drone2'}, {'vehicle_names':['Drone1']},
    {'leader_goal_ned_m':[15.5,-3.,-5.]}, {'leader_goal_ned_m':[float('nan'),0.,-2.]},
    {'leader_goal_ned_m':[15.5,-3.,True]}, {'formation':{'shape':'circle','spacing_m':3.}},
    {'formation':{'shape':'line','spacing_m':2.}}, {'spawn':[0,0,0]},
    {'leader_goal_ned_m':[45.,-3.,-2.]}
])
def test_invalid_or_unsupported_task_fails_before_launch(env,patch):
    with pytest.raises(OperatorError):
        env.m.create(dict(env.body,**patch))
    assert not env.backend.calls


def test_known_obstacle_endpoint_is_rejected_instead_of_labelled_safe(env):
    env.p.map.value['obstacles']=[dict(id='pillar',shape='aabb',min_ned_m=[15.,-3.5,-5.],max_ned_m=[16.,-2.5,0.])]
    with pytest.raises(OperatorError,match='obstacle'):
        env.m.validate(env.body)


def test_geofence_is_centered_at_actual_takeoff_after_a_previous_task(env):
    for state in env.states.values():state['position_ned_m'][0]=-10.
    with pytest.raises(OperatorError,match='actual start'):
        env.m.validate(dict(env.body,leader_goal_ned_m=[38.,-3.,-2.]))


def test_start_is_rechecked_deduplicated_and_refuses_concurrent_flight(env):
    mid=create(env)
    result=start(env,mid)
    assert start(env,mid)==result
    assert len(env.backend.calls)==1
    other=env.m.create(dict(env.body,request_id='create-2'))['mission']['mission_id']
    with pytest.raises(OperatorError,match='Another mission'):
        env.m.start(other,dict(request_id='start-2',expected_revision=1))


def test_physical_preflight_failure_is_reported_as_failed_operation(env):
    mid=create(env)
    def unavailable():raise OperatorError('offline','simulator offline',503)
    env.m.probe=unavailable
    result=start(env,mid)
    assert result['accepted'] and env.m.operation('start-1')['state']=='FAILED'
    assert env.m.get(mid)['state']=='FAILED'
    assert not env.backend.calls


def test_start_disabled_and_old_revision_do_not_launch(env):
    mid=create(env)
    with pytest.raises(OperatorError,match='Reload'):
        env.m.start(mid,dict(request_id='stale',expected_revision=0))
    env.p.allow_flight=False
    with pytest.raises(OperatorError,match='allow-flight'):
        start(env,mid)


def test_stop_uses_request_file_and_waits_for_verified_cleanup(env):
    mid=create(env);start(env,mid)
    reply=env.m.stop(mid,dict(request_id='stop-1'))
    assert not reply['completed']
    assert (env.m.directory/mid/'stop.requested').exists()
    assert env.m.get(mid)['state']=='STOPPING'
    env.backend.code=0
    env.m.tick()
    assert env.m.get(mid)['state']=='FAILED'
    assert env.m.get(mid)['manual_intervention_required']


def test_cancel_during_start_preflight_does_not_takeoff(env):
    mid=create(env)
    def stop_during_probe():
        env.m.stop(mid,dict(request_id='stop-before-launch'))
        return env.states
    env.m.probe=stop_during_probe
    start(env,mid)
    assert env.m.get(mid)['state']=='STOPPED' and not env.backend.calls


def test_draft_shape_change_and_missing_B_runtime_are_distinct(env):
    mid=create(env)
    reply=env.m.configure(mid,dict(request_id='shape-draft',expected_revision=1,formation=dict(shape='vee',spacing_m=3.)))
    assert reply['operation']['state']=='APPLIED_TO_DRAFT'
    env.m.start(mid,dict(request_id='start-1',expected_revision=2))
    env.m.data['missions'][mid]['state']='RUNNING'
    with pytest.raises(OperatorError,match='B has not connected'):
        env.m.configure(mid,dict(request_id='shape-live',expected_revision=2,formation=dict(shape='column',spacing_m=3.)))


def test_B_credentials_scope_deferred_apply_and_no_false_physical_completion(env):
    mid=create(env);start(env,mid)
    mission=env.m.data['missions'][mid];mission['state']='RUNNING'
    token=mission['_b_token']
    with pytest.raises(OperatorError):env.m.b_heartbeat(mid,'wrong',{})
    env.m.b_heartbeat(mid,token,dict(protocol_version=1,capabilities=['formation.configure']))
    formation=dict(shape='column',spacing_m=3.)
    reply=env.m.configure(mid,dict(request_id='shape-live',expected_revision=1,formation=formation))
    assert reply['operation']['state']=='PENDING'
    request=env.m.b_request(mid,token)['request']
    assert request['revision']==2
    result=dict(request_id='shape-live',revision=2,state='DEFERRED',formation=formation,reason='avoiding')
    env.m.b_result(mid,token,result)
    assert env.m.get(mid)['applied_formation']['shape']=='line'
    with pytest.raises(OperatorError):env.m.b_result(mid,token,dict(result,revision=1))
    accepted=env.m.b_result(mid,token,dict(result,state='APPLIED',reason=''))
    assert accepted['physical_formation_settled'] is False
    assert env.m.get(mid)['applied_formation']['shape']=='column'
    assert env.m.b_request(mid,token)['request'] is None
    assert env.m.operation('shape-live')['state']=='APPLIED'


def test_B_stale_channel_expires_pending_configuration(env):
    mid=create(env);start(env,mid)
    mission=env.m.data['missions'][mid];mission['state']='RUNNING'
    env.m.b_heartbeat(mid,mission['_b_token'],dict(protocol_version=1,capabilities=['formation.configure']))
    env.m.configure(mid,dict(request_id='shape-live',expected_revision=1,formation=dict(shape='column',spacing_m=3.)))
    env.clock.now+=2.1
    env.m.tick()
    assert env.m.operation('shape-live')['state']=='FAILED'


def test_B_actual_mode_requires_current_revision_fresh_snapshot_and_nonreplayed_time(env):
    mid=create(env);start(env,mid)
    m=env.m.data['missions'][mid];m['state']='RUNNING'
    token=m['_b_token']
    stamp=1_000_000_000
    runtime=dict(timestamp_unix_ms=int(time.time()*1000),collection_complete=True,
        vehicles={n:dict(timestamp=stamp) for n in env.p.names})
    atomic_json(env.m.directory/mid/'runtime.json',runtime)
    body=dict(revision=1,nominal_shape='line',effective_mode='trail',
        slot_targets_ned_m={'Drone2':[1.,-3.,-2.]},tracking_error_m={'Drone2':1.},
        sim_timestamp_ns=stamp,reason='obstacle')
    with pytest.raises(OperatorError):env.m.b_status(mid,token,dict(body,revision=2))
    with pytest.raises(OperatorError):env.m.b_status(mid,token,dict(body,sim_timestamp_ns=1))
    env.m.b_status(mid,token,body)
    assert env.m.get(mid)['formation_runtime']['effective_mode']=='trail'
    with pytest.raises(OperatorError):env.m.b_status(mid,token,body)


def test_stop_immediately_after_takeoff_collects_perception_before_cleanup(tmp_path):
    from unittest.mock import MagicMock,patch
    from test_automatic_mode import _make_mock_session,_setup_patches,_cfg_mocks
    from flight_modes.automatic_mode import AutomaticMode,AutomaticModeParams
    session=_make_mock_session()
    session.stop_file=tmp_path/'stop';session.stop_file.touch()
    bridge=MagicMock();session.swarm_bridge=bridge
    bridge.goal.return_value=None
    bridge.route_goal.side_effect=lambda state,goal:goal
    bridge.expects_navigation_progress.return_value=False
    stack,mocks=_setup_patches()
    with stack:
        _cfg_mocks(mocks)
        mode=AutomaticMode(session,params=AutomaticModeParams(max_flight_duration_s=2.))
        with patch.object(session,'takeoff_and_climb'):
            result=mode.run()
    assert result.termination_reason=='coordinated_stop'
    bridge.update_lidar.assert_called_once()


def test_cleanup_caches_only_a_real_fresh_scan_without_masking_emergency():
    from test_swarm_wiring import bridge_fixture
    from types import SimpleNamespace as NS
    import numpy as np
    b,raw,_=bridge_fixture()
    stamp=max(s.timestamp for s in b.link.snapshot().states.values())
    f=NS(frame_valid=True,received_monotonic_seconds=time.monotonic(),raw_timestamp_ns=stamp,
        sensor_pose=dict(position=dict(x=10.,y=0.,z=-1.),orientation=dict(x=0.,y=0.,z=0.,w=1.)),
        point_cloud_sensor=np.array([[20.,20.,0.]]))
    b.points_at=0.
    assert b.cache_cleanup_lidar(f)
    assert b.points_at==f.received_monotonic_seconds
    old=b.points_at;f.received_monotonic_seconds-=10.
    assert not b.cache_cleanup_lidar(f) and b.points_at==old
    f.received_monotonic_seconds=time.monotonic();b._unverified_emergency=True
    assert not b.cache_cleanup_lidar(f) and not raw.calls


def test_stopping_cancels_pending_B_reconfiguration(env):
    mid=create(env);start(env,mid)
    m=env.m.data['missions'][mid];m['state']='RUNNING';token=m['_b_token']
    env.m.b_heartbeat(mid,token,dict(protocol_version=1,capabilities=['formation.configure']))
    env.m.configure(mid,dict(request_id='shape-live',expected_revision=1,formation=dict(shape='column',spacing_m=3.)))
    env.m.stop(mid,dict(request_id='stop-1'))
    assert env.m.operation('shape-live')['state']=='FAILED'
    assert env.m.b_request(mid,token)['request'] is None


def test_user_stop_does_not_hide_an_unrelated_failed_termination(env):
    mid=create(env);start(env,mid)
    directory=env.m.directory/mid
    atomic_json(directory/'mission.json',env.m.data['missions'][mid])
    results=good_results();results['Drone2']['termination_reason']='time_limit'
    for name,result in results.items():atomic_json(directory/'runtime'/f'{name}.result.json',result)
    atomic_json(directory/'runner.result.json',summarize(directory,0))
    env.m.stop(mid,dict(request_id='stop-1'));env.backend.code=0;env.m.tick()
    assert env.m.get(mid)['state']=='FAILED'


def good_results():
    return {n:dict(termination_reason='mission_complete' if n=='Drone1' else 'coordinated_stop',
                  landing_confirmed=True,disarmed=True,api_control_released=True,
                  cleanup=dict(control_released=True)) for n in ['Drone1','Drone2']}


def test_real_result_files_determine_completion_not_launcher_exit_alone(env):
    mid=create(env);start(env,mid)
    directory=env.m.directory/mid
    atomic_json(directory/'mission.json',env.m.data['missions'][mid])
    for name,result in good_results().items():
        atomic_json(directory/'runtime'/f'{name}.result.json',result)
    report=summarize(directory,0)
    assert report['mission_completed'] and report['cleanup_verified']
    atomic_json(directory/'runner.result.json',report)
    env.backend.code=0;env.m.tick()
    assert env.m.get(mid)['state']=='COMPLETED'
    assert env.m.operation('start-1')['state']=='COMPLETED'


def test_api_restart_never_replays_active_flight_and_preserves_receipts(env):
    mid=create(env);start(env,mid)
    restored=MissionManager(env.p,env.m.directory,lambda:env.states,env.backend)
    assert restored.get(mid)['state']=='INTERRUPTED'
    assert restored.get(mid)['manual_intervention_required']
    assert restored.start(mid,dict(request_id='start-1',expected_revision=1))['accepted']
    assert len(env.backend.calls)==1


def test_backend_translates_world_goal_once_and_passes_task_binding(env,monkeypatch):
    mid=create(env)
    mission=env.m.data['missions'][mid]
    directory=env.m.directory/mid;directory.mkdir()
    captured=[]
    monkeypatch.setattr('coordination.operator_runtime.subprocess.Popen',lambda *a,**kw:captured.append((a,kw)))
    SubprocessMissionBackend(env.p).launch(mission,directory,'http://127.0.0.1:8790/api/v1')
    import yaml
    cfg=yaml.safe_load((directory/'swarm.yaml').read_text())
    assert cfg['state_service']['task_id']==mid
    assert cfg['drones'][0]['goal_x']==15.
    assert cfg['drones'][0]['goal_y']==0.
    assert cfg['drones'][1]['goal_y']==-3.
    assert cfg['operator_api']['mission_id']==mid
    assert 'shell' not in captured[0][1]


def test_runtime_lock_prevents_second_supervisor(tmp_path):
    first=RuntimeDirectoryLock(tmp_path)
    try:
        with pytest.raises(OperatorError):RuntimeDirectoryLock(tmp_path)
    finally:first.close()


def test_mesh_export_ignores_unreferenced_vertex_garbage(tmp_path,monkeypatch):
    from scripts import export_operator_map
    import hashlib
    project=tmp_path/'Blocks.uproject';(tmp_path/'Content').mkdir()
    (tmp_path/'Content/maze.umap').write_bytes(b'asset')
    vec=lambda x,y,z:SimpleNamespace(x_val=x,y_val=y,z_val=z)
    pose=SimpleNamespace(position=vec(5.,2.,-3.),orientation=SimpleNamespace(x_val=0.,y_val=0.,z_val=0.,w_val=1.))
    mesh=SimpleNamespace(name='cylinder',position=vec(500.,200.,300.),
        vertices=[400.,100.,200.,600.,100.,400.,500.,300.,300.,100000.,100000.,100000.],indices=[0,1,2])
    client=SimpleNamespace(listVehicles=lambda:[],simListSceneObjects=lambda:['Cylinder'],
        simGetMeshPositionVertexBuffers=lambda:[mesh],simGetObjectPose=lambda _:pose,simGetObjectScale=lambda _:vec(2.,2.,2.))
    monkeypatch.setattr(export_operator_map,'active_map',lambda _: '/Game/maze')
    value=export_operator_map.export(client,project,'maze',[-5,40,-16,16])
    assert value['obstacles'][0]['min_ned_m']==[4.,1.,-4.]
    assert value['obstacles'][0]['max_ned_m']==[6.,3.,-2.]


def test_real_http_auth_origin_json_validation_and_roundtrip(env):
    token='x'*48
    server=OperatorHTTPServer(('127.0.0.1',0),env.m,token,['http://127.0.0.1:5173'])
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(method,path,body=None,**headers):
        c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        headers={'Content-Type':'application/json',**headers}
        c.request(method,'/api/v1'+path,body=json.dumps(body) if body is not None else None,headers=headers)
        response=c.getresponse();data=json.loads(response.read());code=response.status;c.close()
        return code,data
    try:
        assert request('GET','/capabilities')[0]==401
        auth={'Authorization':'Bearer '+token}
        code,data=request('GET','/capabilities',**auth)
        assert code==200 and data['supports']['pause_resume'] is False
        assert request('GET','/capabilities',Origin='https://untrusted.example',**auth)[0]==403
        assert request('POST','/missions',{'request_id':'bad'},**auth)[0]==400
        code,data=request('POST','/missions',env.body,**auth)
        assert code==201
        mid=data['mission']['mission_id']
        assert request('GET','/missions/'+mid,**auth)[1]['state']=='READY'
        assert request('GET','/internal/missions/'+mid+'/formation/request',**auth)[0]==401
        assert request('PUT','/missions/'+mid+'/formation',dict(request_id='cfg',expected_revision=1,
            formation=dict(shape='vee',spacing_m=3.)),**auth)[0]==202
    finally:
        env.m.closed.set();server.shutdown();server.server_close();thread.join()

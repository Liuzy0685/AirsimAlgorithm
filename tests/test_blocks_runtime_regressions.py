from types import SimpleNamespace as NS
from unittest.mock import patch
from pathlib import Path
import pytest
from flight_modes.shared_flight_session import SharedFlightSession
from test_swarm_wiring import bridge_fixture
from test_swarm_wiring import command,send_command


def test_conflict_clearance_requires_multiple_frames_and_elapsed_time():
    b,_,_=bridge_fixture()
    now=[0.]
    b._status_clock=lambda: now[0]
    b.report_dispatch('inter_uav_orca',avoiding=True)
    for t in (.1,.2,.3):
        now[0]=t;b.report_dispatch('inter_uav_orca')
        assert b.link._pending_status[-1]['state']=='AVOIDING'
    now[0]=.7;b.report_dispatch('inter_uav_orca')
    assert b.link._pending_status[-1]['state']=='CONFLICT_CLEARED'


@pytest.mark.parametrize('z,timestamp,normal,expected',[(1.1,200,(0.,0.,-1.),True),
    (1.1,100,(0.,0.,-1.),False),(-2.,200,(0.,0.,-1.),False),(1.1,200,(1.,0.,0.),False)])
def test_legacy_landing_needs_new_verified_ground_event(z,timestamp,normal,expected):
    s=SharedFlightSession(settings_json='unused')
    s._startup_ground_z=1.1;s._startup_floor_ts=100
    raw=NS(landed_state=1,kinematics_estimated=NS(position=NS(z_val=z),
        linear_velocity=NS(x_val=0.,y_val=0.,z_val=0.)))
    col=NS(has_collided=False,object_name='(null)',time_stamp=timestamp,
        normal=NS(x_val=normal[0],y_val=normal[1],z_val=normal[2]))
    s._client=NS(getMultirotorState=lambda **kw:raw,simGetCollisionInfo=lambda **kw:col)
    now=[0.]
    def clock(): now[0]+=.1;return now[0]
    with patch('flight_modes.shared_flight_session.time.monotonic',side_effect=clock),patch('flight_modes.shared_flight_session.time.sleep'):
        assert s._poll_landed_detailed('Drone1',max_wait_s=5,interval_s=.1) is expected


def test_controller_cooperative_stop_never_interrupts_landing(tmp_path):
    from scripts.launch_swarm import _stop_group
    class Process:
        pid=1
        stop_file=tmp_path/'stop'
        def poll(self): return None
        def send_signal(self,signal): raise AssertionError('Must not inject KeyboardInterrupt during cleanup')
        def wait(self,timeout): assert self.stop_file.exists()
    _stop_group([('Drone1',Process())])


def test_new_formation_update_does_not_expire_a_still_live_pinned_frame():
    bridge,raw,service=bridge_fixture()
    pinned=bridge.link.command()
    bridge.adopted_command=pinned
    ack,_=send_command(service,command(command_id='new-target',seq=2))
    assert ack['payload']['accepted']
    assert bridge.link.command().lifecycle.command_id=='new-target'
    bridge.proxy.moveByVelocityAsync(.1,0.,0.,.15,vehicle_name='Drone2')
    assert len(raw.calls)==1
    cmd,at=bridge.link._permitted_commands[pinned.lifecycle.command_id]
    bridge.link._permitted_commands[pinned.lifecycle.command_id]=(cmd,at-1.)
    assert bridge.link.permitted_command(pinned.lifecycle.command_id) is None


def test_recovery_velocity_keeps_the_local_static_avoidance_direction():
    bridge,_,_=bridge_fixture()
    state=bridge.link.snapshot().states['Drone2']
    # B asks forward; A's local trajectory chooses a lateral bypass.
    vx,vy,vz,source=bridge.select(0.,.1,0.,'trajectory',state)
    assert source=='formation_velocity'
    assert vx==0. and vy==pytest.approx(.1)


def test_leader_waits_for_current_settled_follower_status_and_motion():
    from test_swarm_wiring import fixture,connect_controller
    from coordination.wire import build_envelope,parse_envelope
    import json
    _,names,publisher,_,service=fixture()
    link,_=connect_controller(service,publisher,names,name='Drone1')
    now=[link.clock()]
    link.clock=lambda:now[0]
    assert not link.followers_settled()
    payload=dict(vehicle_name='Drone2',state='NORMAL',task_id=link.cfg.task_id,
        active_command_id='live-command')
    def report():
        link.ingest(parse_envelope(json.dumps(build_envelope(message_type='coordination.status',
            producer_id='a-state-service',stream_id='a-boot',sequence_id=10,payload=payload))))
    report()
    assert link.followers_settled()
    link._snapshot.states['Drone2'].linear_velocity_ned_mps=[.3,0.,0.]
    assert not link.followers_settled()
    link._snapshot.states['Drone2'].linear_velocity_ned_mps=[0.,0.,0.]
    payload['task_id']='previous-task';report()
    assert not link.followers_settled()
    payload['task_id']=link.cfg.task_id;report()
    assert link.followers_settled()
    now[0]+=link.cfg.disconnect_timeout_s
    assert not link.followers_settled()


def test_expired_target_is_not_dispatched_but_separately_verified_hold_can_be():
    from control.swarm_flight_bridge import FormationPermitExpired
    bridge,raw,_=bridge_fixture()
    bridge.adopted_command=bridge.link.command()
    bridge.link._received_command_at-=1.
    with pytest.raises(FormationPermitExpired):
        bridge.proxy.moveByVelocityAsync(.1,0.,0.,.15,vehicle_name='Drone2')
    assert not raw.calls and not bridge._unverified_emergency
    bridge.adopted_command=None
    bridge.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
    assert raw.calls[0][0][:3]==(0.,0.,0.)
    # A hold still cannot bypass the final inter-vehicle check.
    bridge.link._snapshot.states['Drone1'].position_ned_m=[10.1,0.,-1.]
    with pytest.raises(Exception,match='inter_vehicle_conflict'):
        bridge.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
    assert len(raw.calls)==1


@pytest.mark.parametrize('yaw,point,allowed',[(0.,.4,True),(0.,-.6,False),
    (1.5707963267948966,.4,True),(1.5707963267948966,-.6,False)])
def test_gateway_uses_measured_sensor_mount_pose(yaw,point,allowed):
    import math,time
    import numpy as np
    from control.velocity_controller import VelocityCommandRejected
    bridge,raw,_=bridge_fixture()
    own=bridge.link._snapshot.states['Drone2']
    own.yaw_rad=yaw
    frame=NS(sensor_pose=dict(position=dict(x=10.+.2*math.cos(yaw),y=.2*math.sin(yaw),z=-1.),
        orientation=dict(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))),received_monotonic_seconds=time.monotonic())
    bridge.update_lidar(frame,np.array([[point,0.,0.]]))
    if allowed:
        bridge.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
        assert len(raw.calls)==1
    else:
        with pytest.raises(VelocityCommandRejected,match='lidar_clearance'):
            bridge.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
        assert not raw.calls


def test_recovery_navigation_uses_the_distant_slot_instead_of_a_goal_inside_a_pillar():
    bridge,_,service=bridge_fixture()
    cmd=command(command_id='recover-slot',seq=2,recovery_target_ned_m=(30.,3.,-1.))
    ack,_=send_command(service,cmd)
    assert ack['payload']['accepted']
    own=bridge.link.snapshot().states['Drone2']
    assert bridge.goal(own)==(30.,3.,-1.)
    # The distant goal does not authorize a speed larger than B requested.
    vx,vy,_,_=bridge.select(.5,0.,0.,'trajectory',own)
    assert vx==pytest.approx(.1) and vy==0.


def test_invalid_recovery_target_is_rejected_on_wire():
    from dataclasses import asdict
    from coordination.wire import parse_formation_command_payload,WireError
    payload=asdict(command())
    payload['recovery_target_ned_m']=[0.,float('nan'),-1.]
    with pytest.raises(WireError,match='finite'):
        parse_formation_command_payload(payload)


def test_final_gate_keeps_raw_return_inside_navigation_self_exclusion_box():
    import numpy as np,time
    from control.velocity_controller import VelocityCommandRejected
    bridge,raw,_=bridge_fixture()
    frame=NS(sensor_pose=dict(position=dict(x=10.2,y=0.,z=-1.),
        orientation=dict(x=0.,y=0.,z=0.,w=1.)),received_monotonic_seconds=time.monotonic(),
        point_cloud_sensor=np.array([[.2,0.,0.],[20.,20.,0.]]))
    bridge.update_lidar(frame)
    with pytest.raises(VelocityCommandRejected,match='lidar_clearance'):
        bridge.proxy.moveByVelocityAsync(.1,0.,0.,.15,vehicle_name='Drone2')
    assert not raw.calls


def test_local_recovery_does_not_report_follower_settled_without_adopted_slot():
    bridge,_,_=bridge_fixture()
    bridge.report_dispatch('recovery')
    assert bridge.link._pending_status[-1]['state']=='FORMATION_RECOVERING'


def test_launcher_refuses_reusing_a_directory_with_old_flight_markers(tmp_path,monkeypatch):
    from scripts import launch_swarm
    import yaml
    config=tmp_path/'swarm.yaml'
    config.write_text(yaml.safe_dump(dict(drones=[dict(vehicle_name='Drone1'),dict(vehicle_name='Drone2')])),encoding='utf-8')
    old_run=tmp_path/'old_run';old_run.mkdir();(old_run/'mission.complete').touch()
    monkeypatch.setattr(launch_swarm,'_parse_args',lambda:NS(confirm_simulation_clearance=True,
        config=str(config),log_dir=str(old_run)))
    monkeypatch.setattr(launch_swarm.subprocess,'Popen',lambda *a,**kw:pytest.fail('Must reject before launching any controller'))
    with pytest.raises(FileExistsError):launch_swarm.main()

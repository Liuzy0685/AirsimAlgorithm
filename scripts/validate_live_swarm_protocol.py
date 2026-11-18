"""Exercise A's real WebSocket ingress against landed AirSim state; NO actuation.

Controller connections here are protocol test peers, never flight controllers.
The simulator is briefly paused to distinguish wall TTL from simulation time.
"""
import argparse
import asyncio
from dataclasses import asdict,replace
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from common.models import FormationCommand,CommandLifecycle
from coordination.client import SwarmConnection
from coordination.runtime_config import load_config
from coordination.wire import build_envelope,parse_swarm_snapshot


async def until(connection,kind):
    async def receive():
        while True:
            e=await connection.receive()
            if e.message_type==kind:return e
    return await asyncio.wait_for(receive(),3)


async def run(args):
    config,cfg=load_config(args.swarm_config)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    try:
        with socket.create_connection((cfg.host,cfg.port),timeout=.2):
            raise RuntimeError('Existing A service detected; stop all flight controllers first')
    except OSError:pass
    adapter=AirSimClientAdapter(readonly=True);adapter.connect();rpc=adapter.get_raw_client()
    names=[m['vehicle_name'] for m in config['drones']]
    if set(rpc.listVehicles())!=set(names):raise RuntimeError('Runtime vehicles differ')
    expected=json.loads(Path(args.settings_json).read_text(encoding='utf-8'))
    if json.loads(rpc.getSettingsString()).get('Vehicles')!=expected.get('Vehicles'):
        raise RuntimeError('Runtime vehicle settings differ')
    for n in names:
        s=rpc.getMultirotorState(vehicle_name=n)
        v=s.kinematics_estimated.linear_velocity
        if rpc.isApiControlEnabled(vehicle_name=n) or s.landed_state!=0 or sum(x*x for x in (v.x_val,v.y_val,v.z_val))>.05**2:
            raise RuntimeError('Protocol experiment requires every vehicle landed and control released')
    log=(output/'service.log').open('w',encoding='utf-8')
    ready=output/'ready'
    process=subprocess.Popen([sys.executable,str(ROOT/'scripts/swarm_state_service.py'),
        '--settings-json',args.settings_json,'--swarm-config',args.swarm_config,'--ready-file',str(ready.resolve())],
        cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=='win32' else 0)
    connections=[];checks=[];paused=False
    def check(name,actual,expected):
        checks.append(dict(name=name,actual=actual,expected=expected,passed=actual==expected))
    stream=str(uuid.uuid4());target=next(n for n in names if n!=cfg.leader)
    try:
        deadline=time.monotonic()+cfg.startup_timeout_s
        while not ready.exists() and process.poll() is None and time.monotonic()<deadline:await asyncio.sleep(.1)
        if not ready.exists():raise RuntimeError('A failed to start; inspect service.log')
        async def connect(role,producer,**kw):
            c=await SwarmConnection.connect(cfg.url,role=role,producer_id=producer,stream_id=stream,**kw)
            connections.append(c);return c
        observer=await connect('c_telemetry','protocol-observer')
        controller=await connect('a_controller','protocol-peer',vehicle_name=target)
        b=await connect('b_formation','protocol-B')
        async def fresh():
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                snap=parse_swarm_snapshot(await until(observer,'swarm.snapshot'),received_at=time.monotonic())
                if snap.collection_complete:return max(s.timestamp for s in snap.states.values())
            raise RuntimeError('No complete real state within three seconds')
        async def command(seq,**kw):
            stamp=await fresh()
            life=CommandLifecycle(command_id=str(uuid.uuid4()),sequence_id=seq,issued_at=stamp,
                expires_at=stamp+5_000_000_000,source='formation',task_id=cfg.task_id,stream_id=stream,ttl_s=.8)
            return FormationCommand(vehicle_name=target,desired_velocity_ned_mps=(0.,0.,0.),lifecycle=life,**kw)
        async def send(c,cmd,producer='protocol-B'):
            await c.send(build_envelope(message_type='formation.command',producer_id=producer,stream_id=stream,
                sequence_id=cmd.lifecycle.sequence_id,payload=asdict(cmd)))
        async def ack(cmd):
            await send(b,cmd)
            return (await until(b,'command.ack')).payload['reason']
        async def permit():
            rid=str(uuid.uuid4())
            await controller.send(build_envelope(message_type='command.request',producer_id='protocol-peer',
                stream_id=stream,sequence_id=100,payload=dict(request_id=rid)))
            return (await until(controller,'command.permit')).payload['command']
        first=await command(10)
        check('valid_command_ack',await ack(first),'accepted')
        check('permit_for_accepted_command',(await permit())['lifecycle']['command_id'],first.lifecycle.command_id)
        check('duplicate_id',await ack(first),'duplicate_command_id')
        old=await command(9);check('old_sequence',await ack(old),'stale_sequence')
        unknown=await command(11);unknown.vehicle_name='Missing'
        check('unknown_vehicle',await ack(unknown),'unknown_vehicle')
        wrong=await command(12);wrong.lifecycle=replace(wrong.lifecycle,task_id='wrong-task')
        check('wrong_task',await ack(wrong),'task_id_mismatch')
        expired=await command(13);expired.lifecycle=replace(expired.lifecycle,issued_at=1,expires_at=2)
        check('expired_sim_window',await ack(expired),'expired')
        rpc.simPause(True);paused=True
        frozen=rpc.getMultirotorState(vehicle_name=target).timestamp
        cmd=await command(20);cmd.lifecycle=replace(cmd.lifecycle,ttl_s=.25)
        check('paused_command_ack',await ack(cmd),'accepted')
        await asyncio.sleep(.4)
        check('sim_clock_stayed_paused',rpc.getMultirotorState(vehicle_name=target).timestamp,frozen)
        check('paused_command_unavailable_after_wall_ttl',await permit(),None)
        rpc.simPause(False);paused=False
        ttl=await command(25);ttl.lifecycle=replace(ttl.lifecycle,ttl_s=.1)
        check('short_wall_ttl_command_ack',await ack(ttl),'accepted')
        await asyncio.sleep(.25)
        check('sim_window_still_open_after_wall_ttl',rpc.getMultirotorState(vehicle_name=target).timestamp<ttl.lifecycle.expires_at,True)
        check('wall_ttl_expires_with_fresh_state',await permit(),None)
        cmd=await command(30);check('before_disconnect_ack',await ack(cmd),'accepted')
        await b.close();await asyncio.sleep(.1)
        check('B_disconnect_revokes_permit',await permit(),None)
        b=await connect('b_formation','protocol-B')
        check('B_reconnect_does_not_reset_replay_history',await ack(first),'duplicate_command_id')
        cmd=await command(40)
        await send(observer,cmd,producer='protocol-observer')
        await asyncio.sleep(.1)
        check('C_cannot_inject_command',await permit(),None)
        check('all_API_control_still_released',all(not rpc.isApiControlEnabled(vehicle_name=n) for n in names),True)
    finally:
        if paused:rpc.simPause(False)
        for c in connections:await c.close()
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform=='win32' else signal.SIGINT)
            await asyncio.to_thread(process.wait,15)
        log.close();adapter.close()
        report=dict(evidence_type='live_airsim_protocol_only_no_actuation',checks=checks,
            passed=len(checks)==18 and all(c['passed'] for c in checks),
            limitation='No airborne failure recovery was tested. Pause also invalidates nonadvancing state timestamps; TTL-only expiry is checked separately with the simulation running.')
        (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings-json',required=True)
    p.add_argument('--swarm-config',required=True)
    p.add_argument('--output',required=True)
    return asyncio.run(run(p.parse_args()))


if __name__=='__main__':raise SystemExit(main())

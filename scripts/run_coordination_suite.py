"""Serial owned-simulator encounter / formation-switch / cooperative-fault suite.

Only starts fresh local UE game processes, never resets an existing instance.
Any unverified cleanup stops the suite and leaves that simulation for inspection.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_common import atomic_json
from scripts.build_encounter_configs import build
from scripts.run_swarm_acceptance import source_manifest


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.2):return True
    except OSError:return False


def hidden():
    if sys.platform!='win32':return {}
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    return dict(startupinfo=startup,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def extend_test_floor(client, settings, output):
    """Ephemeral test fixture, before any controller/API acquisition.

    Verify the mesh and an actual disarmed vehicle resting on its extension;
    restore the original spawn before A starts. Never use during a flight.
    """
    import airsim
    import numpy as np
    names=list(settings['Vehicles'])
    if any(client.isApiControlEnabled(vehicle_name=n) for n in names):
        raise RuntimeError('Cannot prepare a fixture while API control is owned')
    floor=client.simSpawnObject('Floor_3','EditorCube',
        airsim.Pose(airsim.Vector3r(8.,-6.,1.55),airsim.to_quaternion(0.,0.,0.)),
        airsim.Vector3r(60.,60.,.5),False)
    if not floor:raise RuntimeError('Test floor asset could not be spawned')
    if floor!='Floor_3':raise RuntimeError('Reserved verified-floor name is already occupied')
    def bounds():
        mesh=next((m for m in client.simGetMeshPositionVertexBuffers() if m.name.lower()==floor.lower()),None)
        if mesh is None:raise RuntimeError('Cannot inspect test floor mesh')
        pose=client.simGetObjectPose(floor)
        origin=np.array([mesh.position.x_val,mesh.position.y_val,mesh.position.z_val]) - np.array(
            [pose.position.x_val,pose.position.y_val,pose.position.z_val])*[100.,100.,-100.]
        v=np.asarray(mesh.vertices).reshape(-1,3)[np.unique(mesh.indices)]
        ned=(v-origin)*[.01,.01,-.01]
        return ned.min(0),ned.max(0)
    lo,hi=bounds();size=hi-lo
    scale=client.simGetObjectScale(floor)
    if not client.simSetObjectScale(floor,airsim.Vector3r(scale.x_val*60/size[0],
            scale.y_val*60/size[1],scale.z_val*.5/size[2])):
        raise RuntimeError('Test floor scale was not applied')
    lo,hi=bounds();pose=client.simGetObjectPose(floor)
    pose.position.x_val+=8.-(lo[0]+hi[0])/2
    pose.position.y_val+=-6.-(lo[1]+hi[1])/2
    pose.position.z_val+=1.30-lo[2]
    if not client.simSetObjectPose(floor,pose,True):raise RuntimeError('Test floor pose was not applied')
    lo,hi=bounds()
    if abs(lo[2]-1.30)>.03 or min((hi-lo)[:2])<59.9:raise RuntimeError('Test floor measured geometry mismatch')
    probe=names[0];probe_rows=[];verified=False
    spawn=settings['Vehicles'][probe]
    probe_x=-15.-spawn.get('X',0.)
    probe_y=-25.-spawn.get('Y',0.)
    try:
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(probe_x,probe_y,-.5),airsim.to_quaternion(0.,0.,0.)),
                                True,vehicle_name=probe)
        deadline=time.monotonic()+12;stable=0
        while time.monotonic()<deadline:
            state=client.getMultirotorState(vehicle_name=probe)
            p=state.kinematics_estimated.position;v=state.kinematics_estimated.linear_velocity
            contact=client.simGetCollisionInfo(vehicle_name=probe)
            good=(abs(p.x_val-probe_x)<.1 and abs(p.y_val-probe_y)<.1 and .8<p.z_val<1.5
                  and v.x_val*v.x_val+v.y_val*v.y_val+v.z_val*v.z_val<.05**2
                  and contact.object_name==floor and contact.normal.z_val<=-.9)
            stable=stable+1 if good else 0
            probe_rows.append(dict(x=p.x_val,y=p.y_val,z=p.z_val,vz=v.z_val,contact_object=contact.object_name,
                                   api=client.isApiControlEnabled(vehicle_name=probe)))
            if probe_rows[-1]['api']:raise RuntimeError('API was acquired during fixture preparation')
            if stable>=3:verified=True;break
            time.sleep(.2)
    finally:
        # Resizing the temporary mesh can leave unarmed SimpleFlight pawns
        # asleep at its earlier height. Reset every own spawn, then the caller
        # measures a new stable baseline for each vehicle before A starts.
        for name in names:
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0.,0.,0.),airsim.to_quaternion(0.,0.,0.)),
                                    True,vehicle_name=name)
        atomic_json(output/'fixture_setup.json',dict(type='ephemeral_ground_extension',asset='EditorCube',name=floor,
            bounds_min_ned_m=lo.tolist(),bounds_max_ned_m=hi.tolist(),physical_ground_probe_verified=verified,
            probe_vehicle=probe,probe_rows=probe_rows,probe_restored_before_controllers=True,
            all_vehicle_spawns_restored_before_controllers=names,
            note='Setup-only disarmed teleport; no flight controller/API acquisition. Original map assets unchanged.'))
    if not verified:raise RuntimeError('Test floor physical ground probe did not settle')


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    for port in (41451,8765,8790):
        if port_open(port):raise RuntimeError(f'Existing service on {port}; owned suite refused')
    profiles=build(output/'profiles')
    AirSimClientAdapter._ensure_airsim_importable()
    import airsim
    report=dict(evidence_type='live_airsim_owned_serial_suite',source_manifest=source_manifest(),cases=[])
    python=[sys.executable,'-X','utf8']
    cases=args.cases or ['formation_switch','head_on','crossing','overtaking','b_disconnect','controller_stop','narrow_corridor','formation_five']
    for case in cases:
        folder=output/case;folder.mkdir()
        profile=profiles.get(case,{})
        map_label=profile.get('map_label','maze')
        use_test_floor=bool(args.extend_test_floor and profile and map_label=='maze')
        settings=Path(profile.get('settings_json',ROOT/('configs/acceptance/settings_5.json' if case=='formation_five' else 'configs/acceptance/settings_2.json')))
        config=Path(profile.get('swarm_config',ROOT/('configs/acceptance/swarm_5_navigation.yaml' if case=='formation_five' else 'configs/acceptance/swarm_2_navigation.yaml')))
        ue=None;operator=None;client=None;clean=False;result=None
        handles=[]
        try:
            ue=subprocess.Popen([str(Path(args.ue_editor).resolve()),str(Path(args.blocks_project).resolve()),
                '/Game/'+map_label,'-game','-settings='+str(settings),'-windowed','-ResX=960','-ResY=540',
                '-NoSound','-unattended','-abslog='+str(folder/'ue.log')],cwd=ROOT,**hidden())
            atomic_json(folder/'owned_processes.json',dict(ue_pid=ue.pid,ue_executable=str(args.ue_editor),
                settings=str(settings),map='/Game/'+map_label))
            deadline=time.monotonic()+60
            while not port_open(41451):
                if ue.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned UE did not start RPC')
                time.sleep(.2)
            client=airsim.MultirotorClient(timeout_value=5);client.ping()
            expected=json.loads(settings.read_text(encoding='utf-8'))
            if json.loads(client.getSettingsString())['Vehicles']!=expected['Vehicles']:
                raise RuntimeError('Fresh UE settings mismatch')
            if use_test_floor:
                extend_test_floor(client,expected,folder)
            ground_frames=0;deadline=time.monotonic()+15
            while ground_frames<3:
                stable=True
                for name in client.listVehicles():
                    state=client.getMultirotorState(vehicle_name=name)
                    p=state.kinematics_estimated.position;v=state.kinematics_estimated.linear_velocity
                    if (client.isApiControlEnabled(vehicle_name=name) or state.landed_state!=0
                            or abs(p.z_val)>5. or v.x_val*v.x_val+v.y_val*v.y_val+v.z_val*v.z_val>.05**2):
                        stable=False
                ground_frames=ground_frames+1 if stable else 0
                if time.monotonic()>deadline:raise RuntimeError('Spawn did not become stationary on valid ground')
                time.sleep(.2)
            if case=='formation_switch':
                runtime=folder/'operator'
                handles=[(folder/n).open('w',encoding='utf-8') for n in ('operator.stdout.log','operator.stderr.log')]
                operator=subprocess.Popen([*python,'scripts/operator_api.py','--settings-json',str(settings),
                    '--swarm-config',str(config),'--formation-config','configs/acceptance/formation_2.yaml',
                    '--map-manifest','configs/operator/maze.json','--blocks-project',args.blocks_project,
                    '--runtime-dir',str(runtime),'--allow-flight','--stop-file',str(folder/'api.stop')],
                    cwd=ROOT,stdout=handles[0],stderr=handles[1],**hidden())
                deadline=time.monotonic()+15
                while not port_open(8790):
                    if operator.poll() is not None or time.monotonic()>deadline:raise RuntimeError('API startup failed')
                    time.sleep(.1)
                command=[*python,'scripts/validate_formation_switch_live.py','--token-file',str(runtime/'operator.token'),
                    '--output',str(folder/'flight'),'--duration','300','--confirm-simulation-clearance']
            else:
                command=[*python,'scripts/run_swarm_acceptance.py','--settings-json',str(settings),
                    '--swarm-config',str(config),'--formation-config','configs/acceptance/formation_5.yaml' if case in ('five_way','formation_five')
                    else 'configs/acceptance/formation_2.yaml','--blocks-project',args.blocks_project,
                    '--map-label',map_label,'--engine-log',str(folder/'ue.log'),
                    '--output',str(folder/'flight'),'--duration','540' if case=='formation_five' else '260','--confirm-simulation-clearance']
                if case in ('b_disconnect','controller_stop'):command+=['--fault',case]
                if use_test_floor:command+=['--test-fixture-json',str(folder/'fixture_setup.json')]
            with (folder/'observer.log').open('w',encoding='utf-8') as log:
                # A timeout intentionally leaves the controller/UE processes
                # available for diagnosis. Never terminate flight children.
                observer=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,**hidden())
                code=observer.wait(timeout=720 if case=='formation_five' else 450)
            path=folder/'flight'/('report.json' if case=='formation_switch' else 'summary.json')
            result=json.loads(path.read_text(encoding='utf-8'))
            passed=result.get('passed') if case=='formation_switch' else result.get(
                'fault_acceptance_passed' if case in ('b_disconnect','controller_stop') else 'flight_acceptance_passed')
            report['cases'].append(dict(case=case,returncode=code,passed=bool(passed),report=str(path.relative_to(output))))
            if operator is not None:
                (folder/'api.stop').touch();operator.wait(timeout=30)
            states={n:dict(landed=client.getMultirotorState(vehicle_name=n).landed_state,
                api_control=client.isApiControlEnabled(vehicle_name=n)) for n in client.listVehicles()}
            clean=all(s['landed']==0 and not s['api_control'] for s in states.values())
            atomic_json(folder/'final_ground_check.json',dict(verified_ground_released=clean,states=states))
            if not clean:raise RuntimeError('Unverified cleanup; suite stopped without resetting this simulator')
        except Exception as exc:
            report['failure']=dict(case=case,reason=str(exc),owned_ue_pid=ue.pid if ue else None)
            break
        finally:
            if clean and client is not None:
                try:client.simRunConsoleCommand('quit')
                except Exception:pass
                if ue is not None:
                    try:ue.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        report['failure']=dict(case=case,reason='Owned UE did not exit after verified ground cleanup')
            if client is not None:client.client.close()
            for handle in handles:handle.close()
            atomic_json(output/'suite.json',report)
        if 'failure' in report:break
        print(json.dumps(report['cases'][-1]),flush=True)
    report['passed']=len(report['cases'])==len(cases) and all(c['passed'] for c in report['cases']) and 'failure' not in report
    atomic_json(output/'suite.json',report)
    print(json.dumps(dict(passed=report['passed'],cases=report['cases'],failure=report.get('failure'))),flush=True)
    return 0 if report['passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ue-editor',required=True);p.add_argument('--blocks-project',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--extend-test-floor',action='store_true',help='Verified ephemeral ground extension for independent scenarios only')
    p.add_argument('--cases',nargs='+',choices=['formation_switch','head_on','crossing','overtaking','five_way','formation_five','narrow_corridor','b_disconnect','controller_stop'])
    p.add_argument('--confirm-simulation-clearance',action='store_true')
    args=p.parse_args()
    if not args.confirm_simulation_clearance:p.error('--confirm-simulation-clearance required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())

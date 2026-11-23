"""Owned, serial AirSim A/B pilot. Stops on unverified cleanup; no resets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('NAV_AB_OUTPUT_DIR',str(Path(__file__).resolve().parent))).resolve()
sys.path.insert(0,str(ROOT))
from scripts.prepare_navigation_trial import prepare
from scripts.run_coordination_suite import hidden,port_open
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_common import atomic_json
from scripts.run_swarm_acceptance import source_manifest

UE=Path(r'D:\30817\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe')
PROJECT=Path(r'D:\30817\Adrone\blocks\Blocks.uproject')


def ground_check(client,names):
    states={}
    for name in names:
        state=client.getMultirotorState(vehicle_name=name)
        v=state.kinematics_estimated.linear_velocity
        p=state.kinematics_estimated.position
        collision=client.simGetCollisionInfo(vehicle_name=name)
        states[name]=dict(landed=int(state.landed_state),api_control=client.isApiControlEnabled(vehicle_name=name),
            speed_mps=math.sqrt(v.x_val*v.x_val+v.y_val*v.y_val+v.z_val*v.z_val),
            local_position=[p.x_val,p.y_val,p.z_val],collision_object=collision.object_name,
            collision_timestamp=collision.time_stamp)
    good=all(s['landed']==0 and not s['api_control'] and s['speed_mps']<.05 for s in states.values())
    return dict(verified_ground_released=good,states=states)


def run_case(count,variant,cameras):
    label=f'{count}_{variant}_cam{cameras}'
    folder=OUT/label
    folder.mkdir(parents=True,exist_ok=False)
    for port in (41451,8765,8780):
        if port_open(port):raise RuntimeError(f'existing service on {port}; refusing new owned case')
    names=[f'Drone{i+1}' for i in range(count)]
    settings=ROOT/f'configs/acceptance/settings_{count}.json'
    formation=ROOT/f'configs/acceptance/formation_{count}.yaml'
    base=ROOT/f'configs/acceptance/swarm_{count}_navigation.yaml'
    config=folder/'swarm.yaml'
    features=[] if variant=='baseline' else os.environ.get('NAV_AB_CANDIDATE_FEATURES','memory,recovery,smoothing,latency_limit,passage').split(',')
    geometry=os.environ.get('NAV_AB_LATENCY_CLEARANCE_MODE','radial')
    prepare(base,config,features,latency_clearance_mode=geometry)
    handles=[];ue=None;camera=None;observer=None;client=None;clean=False
    summary=None
    ue_command=[str(UE),str(PROJECT),'/Game/maze','-game','-settings='+str(settings),
        '-windowed','-ResX=960','-ResY=540','-NoSound','-unattended','-abslog='+str(folder/'ue.log')]
    meta=dict(label=label,vehicle_count=count,variant=variant,extra_camera_count=cameras,
        map='/Game/maze',settings=str(settings),formation=str(formation),swarm=str(config),
        ue_command=ue_command,source_manifest=source_manifest(),started_at_unix_ms=int(time.time()*1000))
    meta['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['enabled_features']=features
    meta['latency_clearance_mode']=geometry
    atomic_json(folder/'ownership.json',meta)
    try:
        with (folder/'ue_stdout.log').open('w',encoding='utf-8') as log:
            ue=subprocess.Popen(ue_command,cwd=ROOT,stdout=log,stderr=log,**hidden())
        meta['ue_pid']=ue.pid;atomic_json(folder/'ownership.json',meta)
        deadline=time.monotonic()+90
        while not port_open(41451):
            if ue.poll() is not None:raise RuntimeError(f'owned UE exited: {ue.returncode}')
            if time.monotonic()>deadline:raise RuntimeError('owned AirSim startup timed out')
            time.sleep(.25)
        AirSimClientAdapter._ensure_airsim_importable()
        import airsim
        client=airsim.MultirotorClient(timeout_value=5)
        client.confirmConnection()
        actual=json.loads(client.getSettingsString())
        expected=json.loads(settings.read_text(encoding='utf-8'))
        if actual.get('Vehicles')!=expected.get('Vehicles') or set(client.listVehicles())!=set(names):
            raise RuntimeError('running settings/vehicles mismatch')
        # Require repeated stationary ground checks before any flight controller.
        stable=0;checks=[];deadline=time.monotonic()+20
        while stable<3 and time.monotonic()<deadline:
            check=ground_check(client,names)
            checks.append(check)
            stable=stable+1 if check['verified_ground_released'] else 0
            time.sleep(.3)
        atomic_json(folder/'preflight_ground_samples.json',checks)
        if stable<3:raise RuntimeError('preflight did not settle landed/unowned/stationary')
        atomic_json(folder/'preflight_ground.json',check)
        if cameras:
            command=[sys.executable,'-X','utf8',str(ROOT/'scripts/camera_service.py'),'--vehicles',*names,
                '--max-total-fps','10','--capture-utilization','.5']
            log=(folder/'camera.log').open('w',encoding='utf-8');handles.append(log)
            camera=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,**hidden())
            meta['camera_pid']=camera.pid;atomic_json(folder/'ownership.json',meta)
            deadline=time.monotonic()+20
            while not port_open(8780):
                if camera.poll() is not None or time.monotonic()>deadline:raise RuntimeError('camera startup failed')
                time.sleep(.1)
            selection=[dict(vehicle=n,view=v) for n in names for v in ('first','third')][:cameras]
            request=urllib.request.Request('http://127.0.0.1:8780/api/cameras/active',
                data=json.dumps(dict(mode='custom',cameras=selection,fps=5)).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=20) as response:
                atomic_json(folder/'camera_selection.json',json.load(response))
        command=[sys.executable,'-X','utf8',str(ROOT/'scripts/run_swarm_acceptance.py'),
            '--settings-json',str(settings),'--swarm-config',str(config),'--formation-config',str(formation),
            '--map-label','maze','--blocks-project',str(PROJECT),'--engine-log',str(folder/'ue.log'),
            '--output',str(folder/'flight'),'--duration','540' if count==5 else '260',
            '--confirm-simulation-clearance']
        log=(folder/'observer.log').open('w',encoding='utf-8');handles.append(log)
        observer=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,**hidden())
        meta['observer_pid']=observer.pid;atomic_json(folder/'ownership.json',meta)
        print(json.dumps(dict(event='flight_started',label=label,ue_pid=ue.pid,observer_pid=observer.pid)),flush=True)
        # Bound the observer, but do not kill controllers or UE if it times out.
        code=observer.wait(timeout=750 if count==5 else 450)
        path=folder/'flight/summary.json'
        if path.exists():summary=json.loads(path.read_text(encoding='utf-8'))
        if summary is None:raise RuntimeError('observer did not produce a summary')
        if cameras:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8780/api/cameras',timeout=5) as response:
                    atomic_json(folder/'camera_final_status.json',json.load(response))
            except Exception as exc:meta['camera_status_error']=str(exc)
        for _ in range(3):
            check=ground_check(client,names)
            if not check['verified_ground_released']:break
            time.sleep(.3)
        atomic_json(folder/'postflight_ground.json',check)
        clean=check['verified_ground_released']
        results=summary.get('vehicle_results',{})
        # Never infer verified cleanup just from a landed snapshot.
        reported_clean=bool(results) and all(v and v.get('cleanup',{}).get('control_released')
            and v.get('landing_confirmed') and v.get('disarmed') and v.get('api_control_released') for v in results.values())
        meta.update(observer_returncode=code,flight_acceptance_passed=summary.get('flight_acceptance_passed'),
            actual_ground_released=clean,reported_cleanup_verified=reported_clean)
        if not clean or not reported_clean:
            clean=False
            raise RuntimeError('cleanup unverified; stopped without resetting or killing owned UE/controllers')
        return dict(label=label,passed=bool(summary.get('flight_acceptance_passed')),summary=str(path),
            minimum_piecewise_linear_separation_m=summary.get('minimum_piecewise_linear_separation_m'),
            elapsed_s=summary.get('elapsed_s'),cleanup_verified=True)
    finally:
        if camera is not None:
            camera.terminate()
            try:camera.wait(timeout=8)
            except subprocess.TimeoutExpired:meta['camera_shutdown_unverified']=True
        if clean and client is not None:
            try:client.simRunConsoleCommand('quit')
            except Exception:pass
            if ue is not None:
                try:ue.wait(timeout=25)
                except subprocess.TimeoutExpired:meta['ue_shutdown_unverified']=True
        if client is not None:
            try:client.client.close()
            except Exception:pass
        for handle in handles:handle.close()
        meta['finished_at_unix_ms']=int(time.time()*1000)
        atomic_json(folder/'ownership.json',meta)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vehicles',type=int,choices=(2,5),default=2)
    parser.add_argument('--cameras',type=int,default=0)
    parser.add_argument('--variants',nargs='+',choices=('baseline','candidate'),default=['baseline','candidate'])
    args=parser.parse_args()
    if not 0<=args.cameras<=2*args.vehicles:parser.error('camera count out of range')
    report=dict(evidence_type='live_airsim_serial_ab_pilot',cases=[])
    try:
        for variant in args.variants:
            result=run_case(args.vehicles,variant,args.cameras)
            report['cases'].append(result)
            atomic_json(OUT/f'report_{args.vehicles}_cam{args.cameras}.json',report)
            print(json.dumps(dict(event='case_completed',**result)),flush=True)
    except Exception as exc:
        report['failure']=str(exc)
        print(json.dumps(dict(event='suite_stopped',reason=str(exc))),flush=True)
    atomic_json(OUT/f'report_{args.vehicles}_cam{args.cameras}.json',report)
    return 2 if 'failure' in report else 0


if __name__=='__main__':raise SystemExit(main())

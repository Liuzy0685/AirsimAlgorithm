"""Run the real A/B flight stack and record measured evidence, never mock flight."""
import argparse
import asyncio
import hashlib
import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.acceptance_metrics import AcceptanceMetrics
from coordination.client import SwarmConnection
from coordination.runtime_config import load_config
from coordination.wire import parse_swarm_snapshot
from formation.controller import FormationController,load_formation_config


def source_manifest():
    """Hash the actual Python sources, including uncommitted development changes."""
    files={}
    for directory in ('adapters','common','control','coordination','flight_modes','formation',
                      'models','planners','perception','sensors','transforms','utils','scripts'):
        for path in sorted((ROOT/directory).rglob('*.py')):
            files[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest,files=files)


async def run(args):
    config,cfg=load_config(args.swarm_config)
    names=[m['vehicle_name'] for m in config['drones']]
    formation_config=load_formation_config(args.formation_config)
    scorer=FormationController(formation_config,task_id=cfg.task_id)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    adapter=AirSimClientAdapter(readonly=True)
    adapter.connect()
    rpc=adapter.get_raw_client()
    settings=json.loads(rpc.getSettingsString())
    expected=json.loads(Path(args.settings_json).read_text(encoding='utf-8'))
    if settings.get('Vehicles')!=expected.get('Vehicles'):
        raise RuntimeError('Running simulator vehicle settings differ from --settings-json')
    actual=rpc.listVehicles()
    if set(names)!=set(actual):
        raise RuntimeError(f'Unexpected running vehicles: {actual}')
    baseline={}
    ground_z={}
    for name in names:
        state=rpc.getMultirotorState(vehicle_name=name)
        p=state.kinematics_estimated.position
        v=state.kinematics_estimated.linear_velocity
        if state.landed_state!=0 or sum(float(getattr(v,k))**2 for k in ('x_val','y_val','z_val'))>.05**2:
            raise RuntimeError(f'{name} must be stationary and landed before this experiment')
        if abs(p.z_val)>5:
            raise RuntimeError(f'{name} appears outside the valid spawn/ground envelope')
        baseline[name]=rpc.simGetCollisionInfo(vehicle_name=name).time_stamp
        ground_z[name]=float(p.z_val)+expected['Vehicles'][name].get('Z',0.)
    metadata=dict(map_label=args.map_label,vehicle_names=names,algorithm=config['coordination']['algorithm'],
        settings_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
        server_version=rpc.getServerVersion(),started_at_unix_ms=int(time.time()*1000),
        evidence_type='live_airsim',scene_objects=rpc.simListSceneObjects())
    metadata['source_manifest']=source_manifest()
    metadata['config_sha256']={str(Path(p).relative_to(ROOT) if Path(p).is_absolute() and Path(p).is_relative_to(ROOT) else p):
        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        (args.settings_json,args.swarm_config,args.formation_config,
         *{m['flight_config'] for m in config['drones']},
         *{m.get('perception_config','configs/perception.yaml') for m in config['drones']},
         *{m.get('trajectory_config','configs/trajectory_planner.yaml') for m in config['drones']})}
    if args.blocks_project:
        from scripts.validate_blocks_maps import active_map
        metadata['active_map']=active_map(args.blocks_project)
        if metadata['active_map']!='/Game/'+args.map_label:
            raise RuntimeError('Blocks active map differs from --map-label')
        map_asset=Path(args.blocks_project).parent/'Content'/f'{args.map_label}.umap'
        metadata['map_asset_sha256']=hashlib.sha256(map_asset.read_bytes()).hexdigest()
    geometry=[]
    for name in metadata['scene_objects']:
        if name.lower().startswith('cylinder'):
            pose=rpc.simGetObjectPose(name)
            item=dict(name=name,position_ned_m=[pose.position.x_val,pose.position.y_val,pose.position.z_val])
            try:
                scale=rpc.simGetObjectScale(name)
                item['scale']=[scale.x_val,scale.y_val,scale.z_val]
            except Exception: item['scale']=None
            geometry.append(item)
    metadata['cylinders']=geometry
    (output/'environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    command=[sys.executable,str(ROOT/'scripts/launch_swarm.py'),'--settings-json',str(Path(args.settings_json).resolve()),
        '--config',str(Path(args.swarm_config).resolve()),'--formation-config',str(Path(args.formation_config).resolve()),
        '--with-formation','--confirm-simulation-clearance','--log-dir',str(output/'runtime'),
        '--stop-file',str(output/'stop.requested')]
    metrics=AcceptanceMetrics(names,safety_distance_m=config['coordination']['safety_distance_m'],
        formation_tolerance_m=formation_config.recovery_tolerance_m)
    collisions=[]
    launch=None
    connection=None
    start=time.monotonic()
    log=(output/'launcher.log').open('w',encoding='utf-8')
    trace=(output/'snapshots.jsonl').open('w',encoding='utf-8')
    navigation_trace=(output/'navigation.jsonl').open('w',encoding='utf-8')
    last_navigation={};navigation_counts={name:0 for name in names};blocked_counts={name:0 for name in names}
    active_errors=[]
    try:
        launch=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=='win32' else 0)
        deadline=start+cfg.startup_timeout_s
        while time.monotonic()<deadline and launch.poll() is None:
            try:
                connection=await SwarmConnection.connect(cfg.url,role='c_telemetry',
                    producer_id='acceptance-observer',stream_id=str(uuid.uuid4()))
                break
            except (OSError,ConnectionError,asyncio.TimeoutError):
                await asyncio.sleep(.1)
        if connection is None:
            raise RuntimeError('A state service did not become available; see launcher.log')
        stopping=False
        stop_deadline=None
        while launch.poll() is None:
            now=time.monotonic()
            if not stopping and now-start>=args.duration:
                (output/'stop.requested').touch()
                stopping=True
                stop_deadline=now+120
            if stopping and now>stop_deadline:
                raise RuntimeError('Flight cleanup did not complete; processes left running for diagnosis')
            try:
                env=await asyncio.wait_for(connection.receive(),1.)
            except asyncio.TimeoutError:
                continue
            except (ConnectionError,OSError):
                break
            now=time.monotonic()
            if env.message_type=='swarm.snapshot':
                snap=parse_swarm_snapshot(env,received_at=now)
                targets=None
                if snap.collection_complete and all(abs(snap.states[m['vehicle_name']].position_ned_m[2]-
                    (expected['Vehicles'][m['vehicle_name']].get('Z',0.)+m['target_z']))<=.3 for m in config['drones']):
                    targets=scorer.slot_targets(snap.states[formation_config.leader])
                if metrics.observe(snap,received_at=now,stream_id=env.stream_id,formation_targets=targets):
                    trace.write(json.dumps(dict(t=now-start,snapshot_id=snap.sequence_id,
                        complete=snap.collection_complete,errors=snap.errors,states={n:dict(
                            position=s.position_ned_m,velocity=s.linear_velocity_ned_mps,
                            sim_timestamp_ns=s.timestamp,landed_state=s.landed_state) for n,s in snap.states.items()}))+'\n')
                for name in names:
                    col=rpc.simGetCollisionInfo(vehicle_name=name)
                    if col.has_collided and col.time_stamp!=baseline[name]:
                        baseline[name]=col.time_stamp
                        state=snap.states.get(name)
                        ground=(state is not None and abs(state.position_ned_m[2]-ground_z[name])<=.3
                            and col.normal.z_val<=-.9 and col.penetration_depth<=.02)
                        collisions.append(dict(t=now-start,vehicle=name,object=col.object_name,
                            timestamp=col.time_stamp,classification='verified_ground_contact' if ground else 'collision',
                            normal=[col.normal.x_val,col.normal.y_val,col.normal.z_val]))
            elif env.message_type=='coordination.status':
                metrics.status(env.payload,now)
                nav=env.payload.get('navigation');name=env.payload.get('vehicle_name')
                if nav is not None and name in navigation_counts and nav['observation_id']!=last_navigation.get(name):
                    last_navigation[name]=nav['observation_id'];navigation_counts[name]+=1
                    blocked_counts[name]+=not nav['path_available']
                    if name!=formation_config.leader and env.payload.get('active_command_id'):
                        active_errors.append(nav['tracking_error_m'])
                    item={k:v for k,v in nav.items() if k!='obstacle_points_xy_m'}
                    item.update(t=now-start,vehicle=name,obstacle_points_count=len(nav['obstacle_points_xy_m']),
                        obstacle_points_sha256=hashlib.sha256(json.dumps(nav['obstacle_points_xy_m']).encode()).hexdigest())
                    navigation_trace.write(json.dumps(item)+'\n')
        if launch.poll() is None:
            (output/'stop.requested').touch()
            await asyncio.to_thread(launch.wait,120)
    finally:
        if launch is not None and launch.poll() is None:
            (output/'stop.requested').touch()
            try: await asyncio.to_thread(launch.wait,120)
            except subprocess.TimeoutExpired: pass
        if connection: await connection.close()
        log.close();trace.close();navigation_trace.close();adapter.close()
        results={}
        for name in names:
            path=output/'runtime'/f'{name}.result.json'
            results[name]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        summary=dict(metrics.summary(),environment=metadata,elapsed_s=time.monotonic()-start,
            launcher_returncode=launch.poll() if launch else None,vehicle_results=results,
            collision_events=collisions,collision_events_excluding_landed_ground=sum(c['classification']=='collision' for c in collisions),
            shutdown_complete=launch is not None and launch.poll() is not None)
        summary['navigation_observed_frames']=navigation_counts
        summary['navigation_blocked_frames']=blocked_counts
        summary['feedback_active_target_error_rmse_m']=(sum(e*e for e in active_errors)/len(active_errors))**.5 if active_errors else None
        summary['feedback_active_target_error_max_m']=max(active_errors,default=None)
        formation_log=output/'runtime'/'formation.log'
        summary['formation_navigation_modes']=[line.split('mode=',1)[1].strip() for line in
            formation_log.read_text(encoding='utf-8',errors='replace').splitlines() if line.startswith('formation_navigation mode=')] if formation_log.exists() else []
        summary['flight_acceptance_passed']=(summary['shutdown_complete'] and summary['launcher_returncode']==0
            and summary['separation_observed_ok'] is True and summary['collision_events_excluding_landed_ground']==0
            and summary['formation_settled_frames']>0
            and all(r and r.get('cleanup',{}).get('control_released',False) for r in results.values())
            and any(r and r.get('termination_reason')=='mission_complete' for r in results.values()))
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('flight_acceptance_passed','frames','minimum_sampled_separation_m','launcher_returncode','vehicle_results')},ensure_ascii=False),flush=True)
    return 0 if summary['flight_acceptance_passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings-json',required=True)
    p.add_argument('--swarm-config',required=True)
    p.add_argument('--formation-config',required=True)
    p.add_argument('--map-label',required=True)
    p.add_argument('--blocks-project',help='Verify active Blocks map from its engine log and hash its asset')
    p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=90.)
    p.add_argument('--confirm-simulation-clearance',action='store_true')
    args=p.parse_args()
    if not args.confirm_simulation_clearance: p.error('--confirm-simulation-clearance required')
    if args.duration<=0: p.error('--duration must be positive')
    return asyncio.run(run(args))


if __name__=='__main__':
    if hasattr(signal,'SIGBREAK'): signal.signal(signal.SIGBREAK,signal.default_int_handler)
    raise SystemExit(main())

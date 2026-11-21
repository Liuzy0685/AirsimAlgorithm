"""Read-only state acceptance across the existing five Blocks maps.

Map changes reset this owned simulation; active A control service/API control
prevents map changes. No map assets or user settings are edited.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.coordinate_transform import CoordinateTransformer
from coordination.runtime_config import load_config
from coordination.state_provider import StateProvider

MAPS=('maze','regularMaze','Corridor','oneBarrierCorridor','uShapedBarrier')


def server_client():
    AirSimClientAdapter._ensure_airsim_importable()
    import airsim
    return airsim.MultirotorClient(timeout_value=5)


def active_map(project):
    path=Path(project).parent/'Saved/Logs/Blocks.log'
    text=path.read_text(encoding='utf-8',errors='replace')
    # UE4 writes different map-load records depending on whether a level was
    # opened from the editor, with the console command, or as a PIE world.
    # Keep the source position so the last record wins across mixed formats.
    patterns=(
        re.compile(r'LoadMap\(\s*(/Game/[^)\s]+)'),
        re.compile(r'MAP\s+LOAD\s+FILE\s*=\s*"[^"]*?[\\/]Content[\\/](.+?)\.umap"', re.I),
        re.compile(r'Created PIE world by copying editor world from\s+(/Game/[^.\s]+)', re.I),
    )
    matches=[]
    for pattern in patterns:
        for match in pattern.finditer(text):
            value=match.group(1)
            if value.startswith('/Game/'):
                map_path=value
            else:
                map_path='/Game/'+value.replace('\\', '/')
            matches.append((match.start(), map_path))
    return max(matches, key=lambda item:item[0])[1] if matches else None


def collect_map(name,args,config,cfg):
    try:
        with socket.create_connection(('127.0.0.1',cfg.port),timeout=.2):
            raise RuntimeError('Stop A controllers/service before resetting a simulation map')
    except (ConnectionRefusedError,TimeoutError,OSError):
        pass
    c=server_client()
    names=[m['vehicle_name'] for m in config['drones']]
    for vehicle in c.listVehicles():
        if c.isApiControlEnabled(vehicle_name=vehicle):
            raise RuntimeError(f'{vehicle} still has API control; map reset refused')
    accepted=c.simRunConsoleCommand('open /Game/'+name)
    if not accepted: raise RuntimeError(f'Engine rejected map {name}')
    time.sleep(2)
    deadline=time.monotonic()+45
    adapter=None
    while time.monotonic()<deadline:
        try:
            if active_map(args.blocks_project)!='/Game/'+name:
                time.sleep(.25);continue
            adapter=AirSimClientAdapter(readonly=True)
            adapter.connect()
            rpc=adapter.get_raw_client()
            if set(rpc.listVehicles())!=set(names):raise RuntimeError('configured vehicles do not match runtime')
            expected=json.loads(Path(args.settings_json).read_text(encoding='utf-8'))
            if json.loads(rpc.getSettingsString()).get('Vehicles')!=expected.get('Vehicles'):
                raise RuntimeError('runtime vehicle settings differ from validation settings')
            for n in names: rpc.getMultirotorState(vehicle_name=n)
            break
        except (ConnectionError,OSError,RuntimeError):
            if adapter: adapter.close()
            adapter=None
            time.sleep(.25)
    if adapter is None:raise RuntimeError(f'map {name} did not become ready')
    rpc=adapter.get_raw_client()
    time.sleep(1.)
    tf=CoordinateTransformer.from_settings_json(args.settings_json,names)
    report=tf.validate(rpc)
    result=dict(map=name,active_map=active_map(args.blocks_project),evidence_type='live_airsim',
        transform=asdict(report),snapshots=[],spawn=[],passed=False)
    try:
        for n in names:
            raw=rpc.getMultirotorState(vehicle_name=n)
            p=raw.kinematics_estimated.position
            v=raw.kinematics_estimated.linear_velocity
            values=[float(v.x_val),float(v.y_val),float(v.z_val)]
            result['spawn'].append(dict(vehicle=n,landed_state=raw.landed_state,
                local_position_m=[p.x_val,p.y_val,p.z_val],speed_mps=math.sqrt(sum(x*x for x in values)),
                stationary_on_map=raw.landed_state==0 and math.sqrt(sum(x*x for x in values))<.05 and abs(p.z_val)<5))
        if report.validated:
            provider=StateProvider.from_adapter(adapter,names,tf,max_age_s=cfg.snapshot_max_age_s,
                max_skew_s=config['coordination']['max_state_timestamp_skew_s'],allow_legacy_readiness=cfg.allow_legacy_readiness)
            for _ in range(args.samples):
                snap=provider.get_snapshot()
                result['snapshots'].append(dict(sequence_id=snap.sequence_id,complete=snap.collection_complete,
                    collection_duration_s=snap.collection_duration,max_receive_skew_s=snap.max_time_skew,
                    max_sim_skew_s=snap.max_sim_timestamp_skew,read_duration_s=snap.read_duration_s,errors=snap.errors))
                time.sleep(config['coordination']['peer_poll_interval_s'])
            result['passed']=(all(s['complete'] for s in result['snapshots']) and all(s['stationary_on_map'] for s in result['spawn']))
        return result
    finally:
        adapter.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks-project',required=True)
    p.add_argument('--settings-json',required=True)
    p.add_argument('--swarm-config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--maps',nargs='+',choices=MAPS,default=list(MAPS))
    p.add_argument('--samples',type=int,default=25)
    args=p.parse_args()
    if args.samples<1:p.error('--samples must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config,cfg=load_config(args.swarm_config)
    results=[]
    for name in args.maps:
        try:
            result=collect_map(name,args,config,cfg)
        except Exception as exc:
            result=dict(map=name,passed=False,error=f'{type(exc).__name__}: {exc}',evidence_type='live_airsim')
        (output/f'{name}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        results.append(result)
        print(json.dumps(dict(map=name,passed=result['passed'],error=result.get('error'),samples=len(result.get('snapshots',[])))),flush=True)
    (output/'summary.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in results),results=results),indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if all(x['passed'] for x in results) else 2


if __name__=='__main__':raise SystemExit(main())

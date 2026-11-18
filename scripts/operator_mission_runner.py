"""Run the existing launcher unchanged, recording actual final lifecycle evidence."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coordination.operator_common import atomic_json, read_json
from coordination.runtime_config import load_config


async def observe(directory, cfg, stopped):
    from coordination.client import SwarmConnection
    from coordination.wire import parse_swarm_snapshot
    statuses = {}
    while not stopped.is_set():
        connection = None
        try:
            connection = await SwarmConnection.connect(cfg.url, role='c_telemetry',
                producer_id='operator-observer', stream_id=uuid.uuid4().hex)
            while not stopped.is_set():
                env = await asyncio.wait_for(connection.receive(), 1.)
                if env.message_type == 'coordination.status':
                    payload = {k:v for k,v in env.payload.items() if k!='navigation'}
                    statuses[payload['vehicle_name']] = payload
                elif env.message_type == 'swarm.snapshot':
                    snap = parse_swarm_snapshot(env, received_at=time.monotonic())
                    atomic_json(directory / 'runtime.json', dict(timestamp_unix_ms=int(time.time()*1000),
                        snapshot_id=snap.sequence_id, stream_id=env.stream_id,
                        collection_complete=snap.collection_complete, errors=snap.errors,
                        vehicles={n:asdict(s) for n,s in snap.states.items()}, coordination=statuses))
        except (OSError, ConnectionError, asyncio.TimeoutError, ValueError):
            await asyncio.sleep(.2)
        finally:
            if connection:
                await connection.close()


def summarize(directory, code):
    mission = read_json(directory/'mission.json')
    results = {}
    for name in mission['config']['vehicle_names']:
        try:
            results[name] = read_json(directory/'runtime'/f'{name}.result.json')
        except (OSError, ValueError):
            results[name] = None
    cleanup = all(r and r.get('cleanup',{}).get('control_released') is True
                  and r.get('landing_confirmed') is True and r.get('disarmed') is True
                  and r.get('api_control_released') is True for r in results.values())
    leader = results[mission['config']['leader']]
    completed = bool(code==0 and cleanup and leader and leader.get('termination_reason')=='mission_complete'
                     and all(r['termination_reason'] in {'mission_complete','coordinated_stop'} for r in results.values()))
    stopped = bool(code==0 and cleanup and all(r['termination_reason']=='coordinated_stop' for r in results.values()))
    reason = 'mission_complete' if completed else ('launcher_returned_without_mission_completion' if cleanup else 'cleanup_unverified')
    return dict(launcher_returncode=code, vehicle_results=results,
                cleanup_verified=bool(cleanup), mission_completed=completed, coordinated_stop_verified=stopped,reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--settings-json', required=True)
    args = parser.parse_args()
    directory = Path(args.job_dir).resolve()
    _, cfg = load_config(directory/'swarm.yaml')
    stopped = threading.Event()
    observer = threading.Thread(target=lambda: asyncio.run(observe(directory,cfg,stopped)), daemon=True)
    observer.start()
    from scripts import launch_swarm
    original = sys.argv
    code = 1
    try:
        sys.argv = ['launch_swarm.py','--config',str(directory/'swarm.yaml'),
            '--settings-json',args.settings_json,'--with-formation','--formation-config',str(directory/'formation.yaml'),
            '--log-dir',str(directory/'runtime'),'--stop-file',str(directory/'stop.requested'),
            '--confirm-simulation-clearance']
        code = launch_swarm.main()
    finally:
        sys.argv = original
        stopped.set()
        observer.join(3.)
        atomic_json(directory/'runner.result.json', summarize(directory, code))
    return code


if __name__=='__main__':
    raise SystemExit(main())

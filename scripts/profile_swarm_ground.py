"""Read-only ground RPC/processing benchmark. Never acquires API control or arms.

Run single UE / dual UE / single UE again with identical primary settings.
These measurements do not constitute flight acceptance or prove B's historical setup.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_common import atomic_json
from perception.pointcloud_filter import filter_pointcloud
from perception.perception_config import load_perception_config
from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
from sensors.lidar_reader import LidarReader


def stats(values):
    values=np.asarray(values,float)
    return dict(count=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),max_ms=float(values.max()))


def reference_filter(revision):
    if not re.fullmatch(r'[a-f0-9]{7,40}',revision):raise ValueError('Need an explicit baseline commit SHA')
    source=subprocess.check_output(['git','show',f'{revision}:perception/pointcloud_filter.py'],cwd=ROOT)
    module=types.ModuleType('_ground_reference_filter');sys.modules[module.__name__]=module
    exec(compile(source,'<trusted git baseline filter>','exec'),module.__dict__)
    return module.filter_pointcloud,hashlib.sha256(source).hexdigest()


def measure_vehicle(member,args,reference):
    import airsim
    c=airsim.MultirotorClient(port=args.port,timeout_value=5)
    n=member['vehicle_name']
    timings={k:[] for k in ('state_rpc','collision_rpc','lidar_read','reference_filter','optimized_filter','sectorization')}
    counts=[];same=[];trace=[];last=None
    cfg=load_perception_config(member.get('perception_config','configs/perception.yaml'))
    pc,sz=cfg.pointcloud,cfg.sectorization
    reader=LidarReader(types.SimpleNamespace(get_raw_client=lambda:c),vehicle_name=n,lidar_name=member['lidar_name'])
    kwargs=dict(min_range_m=pc.min_range_m,max_range_m=pc.max_range_m,
                self_exclusion=vars(pc.self_exclusion),voxel_downsample=pc.voxel_downsample.enabled,
                voxel_size_m=pc.voxel_downsample.voxel_size_m)
    try:
        for index in range(args.warmup+args.samples):
            start=time.monotonic()
            state=c.getMultirotorState(vehicle_name=n)
            state_ms=(time.monotonic()-start)*1000
            if state.landed_state!=0 or c.isApiControlEnabled(vehicle_name=n):
                raise RuntimeError('Ground benchmark stopped: vehicle is airborne or API-owned')
            t=time.perf_counter();c.simGetCollisionInfo(vehicle_name=n);collision_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();frame=reader.read();lidar_ms=(time.perf_counter()-t)*1000
            if not frame.frame_valid:raise RuntimeError('Invalid ground LiDAR frame: '+str(frame.invalid_reason))
            measured={}
            funcs=[('reference_filter',reference),('optimized_filter',filter_pointcloud)]
            if index%2:funcs.reverse()
            for key,fn in funcs:
                t=time.perf_counter();result=fn(frame.point_cloud_sensor,**kwargs);measured[key]=(time.perf_counter()-t)*1000
                if key=='reference_filter':old=result
                else:new=result
            np.testing.assert_array_equal(old.filtered_points_sensor,new.filtered_points_sensor)
            t=time.perf_counter()
            pointcloud_to_directional_distances(new.filtered_points_sensor,sz.sectors,
                default_max_range_m=sz.default_max_range_m,default_min_points=sz.default_min_points,
                distance_strategy=sz.default_distance_strategy,nearest_k=sz.nearest_k,percentile=sz.percentile)
            sector_ms=(time.perf_counter()-t)*1000
            if index>=args.warmup:
                row=dict(state_rpc=state_ms,collision_rpc=collision_ms,lidar_read=lidar_ms,
                         sectorization=sector_ms,**measured)
                for key,value in row.items():timings[key].append(value)
                counts.append(frame.point_count);same.append(last==frame.raw_timestamp_ns)
                trace.append(dict(**row,raw_timestamp_ns=frame.raw_timestamp_ns,point_count=frame.point_count))
            last=frame.raw_timestamp_ns
            time.sleep(max(0.,.1-(time.monotonic()-start)))
        if c.isApiControlEnabled(vehicle_name=n) or c.getMultirotorState(vehicle_name=n).landed_state!=0:
            raise RuntimeError('Vehicle was not left landed and released')
        return dict(vehicle=n,timing={k:stats(v) for k,v in timings.items()},
            point_count_min=min(counts),point_count_max=max(counts),repeated_scan_reads=sum(same),
            trace=trace,landed_and_api_released=True)
    finally:c.client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings-json',required=True)
    parser.add_argument('--swarm-config',required=True)
    parser.add_argument('--baseline',default='cc2f976')
    parser.add_argument('--case',required=True)
    parser.add_argument('--port',type=int,default=41451)
    parser.add_argument('--samples',type=int,default=40)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--wait-ready-s',type=float,default=45.)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 5<=args.samples<=200 or not 0<=args.warmup<=20:parser.error('Invalid sample count')
    import yaml
    members=yaml.safe_load(Path(args.swarm_config).read_text(encoding='utf-8'))['drones']
    expected=json.loads(Path(args.settings_json).read_text(encoding='utf-8'))
    AirSimClientAdapter._ensure_airsim_importable()
    import airsim
    deadline=time.monotonic()+args.wait_ready_s
    while True:
        client=airsim.MultirotorClient(port=args.port,timeout_value=3)
        try:
            client.ping()
            break
        except Exception:
            client.client.close()
            if time.monotonic()>=deadline:raise
            time.sleep(.5)
    try:
        if json.loads(client.getSettingsString())['Vehicles']!=expected['Vehicles']:
            raise RuntimeError('Primary vehicle settings differ')
        if set(client.listVehicles())!={m['vehicle_name'] for m in members}:
            raise RuntimeError('Fleet mismatch')
        if client.simIsPause():raise RuntimeError('Simulator must not be paused')
    finally:client.client.close()
    baseline,source_hash=reference_filter(args.baseline)
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        futures=[pool.submit(measure_vehicle,m,args,baseline) for m in members]
        results=[future.result() for future in futures]
    report=dict(evidence_type='read_only_ground_no_actuation',case=args.case,
        baseline=args.baseline,baseline_filter_sha256=source_hash,
        optimized_filter_sha256=hashlib.sha256((ROOT/'perception/pointcloud_filter.py').read_bytes()).hexdigest(),
        python=sys.version,numpy=np.__version__,platform=platform.platform(),results=results,
        settings_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
        limitation='This host and these ground scenes only. No flight, controller-rate, collision-rate or historical dual-UE attribution claims.')
    atomic_json(args.output,report)
    print(json.dumps(dict(case=args.case,results=[dict(vehicle=r['vehicle'],timing=r['timing']) for r in results])))


if __name__=='__main__':main()

"""Read completed cases only; navigation_motion records are sampled decisions."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def analyze(folder):
    folder=Path(folder)
    raw=json.loads((folder/'flight/summary.json').read_text(encoding='utf-8'))
    ground=json.loads((folder/'postflight_ground.json').read_text(encoding='utf-8'))
    ownership=json.loads((folder/'ownership.json').read_text(encoding='utf-8'))
    per_vehicle={}
    for name in raw['vehicle_results']:
        path=folder/f'flight/runtime/{name}.log'
        samples=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            if 'navigation_runtime {' not in line:continue
            record=json.loads(line.split('navigation_runtime ',1)[1])
            motion=record.get('navigation_motion')
            if motion is not None:samples.append(dict(log_time=line.split(' [',1)[0],reason=record['reason'],**motion))
        active=[s for s in samples if s['requested_mps']>=.1]
        capped=[s for s in active if s['latency_cap_mps']<s['requested_mps']-.001]
        smoothed=[s for s in active if s['output_mps']<s['latency_cap_mps']-.001]
        per_vehicle[name]=dict(sampled_motion_records=len(samples),active_request_records=len(active),
            latency_limited_records=len(capped),output_below_cap_records=len(smoothed),
            median_cap_ratio=statistics.median(s['latency_cap_mps']/s['requested_mps'] for s in active) if active else None,
            samples=samples,log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(features=ownership['enabled_features'],summary_sha256=hashlib.sha256((folder/'flight/summary.json').read_bytes()).hexdigest(),
        passed=raw['flight_acceptance_passed'],flight_duration_s=max(v['flight_duration_s'] for v in raw['vehicle_results'].values()),
        collision_count=raw['collision_events_excluding_landed_ground'],minimum_pair_distance_m=raw['minimum_piecewise_linear_separation_m'],
        actual_ground_verified=ground['verified_ground_released'],reported_cleanup_verified=ownership['reported_cleanup_verified'],
        termination={n:v['termination_reason'] for n,v in raw['vehicle_results'].items()},motion=per_vehicle)


if __name__=='__main__':
    report={name:analyze(ROOT/f'runs/live_navigation_ab_20261123_{suffix}/2_candidate_cam0')
        for suffix,name in [('06','smoothing_only'),('07','latency_only')]}
    (ROOT/'runs/live_navigation_ab_20261123_06/analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for name,row in report.items():
        print(name,json.dumps({k:v for k,v in row.items() if k!='motion'}))
        for vehicle,values in row['motion'].items():print(vehicle,json.dumps({k:v for k,v in values.items() if k!='samples'}))

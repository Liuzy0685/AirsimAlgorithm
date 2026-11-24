"""Conservative paired A/B screening. Missing evidence is never 'improved'.

Input: JSON list of records, each with case_id, seed, environment and metrics.
The explicit schema prevents unrelated maps/settings/camera loads being mixed.
This is a screening decision, not a universal proof or statistical significance.
"""
import argparse
import json
import math
from pathlib import Path

ENVIRONMENT=('map_sha256','settings_sha256','formation_sha256','vehicle_count',
             'camera_count','requested_fps','speed_limit_mps','safety_distance_m',
             'hard_clearance_m','max_scan_age_s')
METRICS=('success','collision_count','cleanup_verified','minimum_peer_distance_m',
         'minimum_obstacle_clearance_m','planner_p95_ms','sensor_age_p95_ms',
         'control_p95_ms','task_time_s','path_length_m','unresolved_deadlocks')


def compare(baseline,candidate,minimum_repeats=5):
    def indexed(rows):
        result={}
        for row in rows:
            key=(row['case_id'],row['seed'])
            if key in result:raise ValueError('duplicate case/seed')
            if any(k not in row.get('environment',{}) for k in ENVIRONMENT):
                raise ValueError('missing environment evidence')
            metrics=row.get('metrics',{})
            if any(k not in metrics or metrics[k] is None for k in METRICS):
                raise ValueError('missing outcome/timing evidence')
            if type(metrics['success']) is not bool or type(metrics['cleanup_verified']) is not bool:
                raise ValueError('outcome flags must be boolean')
            for name in METRICS:
                if name in ('success','cleanup_verified'):continue
                value=metrics[name]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('invalid numeric metric')
            env=row['environment']
            if (type(env['vehicle_count']) is not int or not 1<=env['vehicle_count']<=5
                    or type(env['camera_count']) is not int or not 0<=env['camera_count']<=env['vehicle_count']*2):
                raise ValueError('invalid vehicle/camera counts')
            for name in ('requested_fps','speed_limit_mps'):
                if type(env[name]) not in (int,float) or not math.isfinite(env[name]) or env[name]<0:
                    raise ValueError('invalid load/speed evidence')
            for name in ('safety_distance_m','hard_clearance_m','max_scan_age_s'):
                if type(env[name]) not in (int,float) or not math.isfinite(env[name]) or env[name]<=0:
                    raise ValueError('invalid safety threshold')
            result[key]=row
        return result
    try:
        old,new=indexed(baseline),indexed(candidate)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(verdict='inconclusive',reason=str(exc))
    if not old or set(old)!=set(new):
        return dict(verdict='inconclusive',reason='unmatched case/seed sets')
    case_counts={}
    regressions=[];improvements=[];tradeoffs=[]
    for key,prior in old.items():
        current=new[key]
        if any(prior['environment'][k]!=current['environment'][k] for k in ENVIRONMENT):
            return dict(verdict='inconclusive',reason='environment or safety thresholds differ',case=list(key))
        case_counts[key[0]]=case_counts.get(key[0],0)+1
        a,b=prior['metrics'],current['metrics'];env=current['environment']
        for condition,reason in (
            (b['collision_count']>0,'collision'),
            (not b['cleanup_verified'],'cleanup_unverified'),
            (env['vehicle_count']>1 and b['minimum_peer_distance_m']<env['safety_distance_m'],'peer_clearance'),
            (b['minimum_obstacle_clearance_m']<env['hard_clearance_m'],'obstacle_clearance'),
            (b['sensor_age_p95_ms']>env['max_scan_age_s']*1000,'sensor_freshness'),
            (a['success'] and not b['success'],'lost_success'),
            (b['unresolved_deadlocks']>a['unresolved_deadlocks'],'new_deadlock'),
        ):
            if condition:regressions.append(dict(case=list(key),metric=reason))
        for metric in ('planner_p95_ms','sensor_age_p95_ms','control_p95_ms','task_time_s','path_length_m'):
            if not (a['success'] and b['success']) and metric in ('task_time_s','path_length_m'):
                continue  # A fast failed task is not a speed improvement.
            delta=b[metric]-a[metric]
            ratio=delta/max(a[metric],1e-9)
            if ratio>.10 and delta>.01:
                regressions.append(dict(case=list(key),metric=metric,relative_change=ratio))
            elif ratio>0:
                tradeoffs.append(dict(case=list(key),metric=metric,relative_change=ratio))
            elif ratio<-.05:
                improvements.append(dict(case=list(key),metric=metric,relative_change=ratio))
        if not a['success'] and b['success']:improvements.append(dict(case=list(key),metric='recovered_success'))
        if a['collision_count']>b['collision_count']:improvements.append(dict(case=list(key),metric='fewer_collisions'))
        if a['unresolved_deadlocks']>b['unresolved_deadlocks']:improvements.append(dict(case=list(key),metric='fewer_deadlocks'))
    if regressions:
        verdict='regression'
    elif any(n<minimum_repeats for n in case_counts.values()):
        verdict='inconclusive'
    elif not all(row['metrics']['success'] for row in new.values()):
        verdict='inconclusive'
    elif improvements:
        verdict='promising_with_tradeoffs' if tradeoffs else 'promising'
    else:verdict='no_demonstrated_gain'
    return dict(verdict=verdict,paired_runs=len(old),case_repeats=case_counts,
                regressions=regressions,improvements=improvements,tradeoffs=tradeoffs,
                limitation='Finite matched trials only; no universal superiority or statistical-significance claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(json.loads(args.baseline.read_text(encoding='utf-8')),
                   json.loads(args.candidate.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

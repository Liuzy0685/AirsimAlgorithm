"""Archive diagnostics validation without relabeling the prior timeout as fixed."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.export_swarm_acceptance import export
from coordination.navigation import CommittedGridRouter,NavigationConfig
from coordination.navigation_hardening import HardeningConfig
from coordination.navigation_trace import replay_snapshot,source_verification,NavigationTraceRecorder,TRACE_SOURCES

spec=importlib.util.spec_from_file_location('geometry_archive',ROOT/'runs/live_navigation_ab_20261123_08/archive_geometry.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=NavigationTraceRecorder._encode,allow_nan=False)+'\n',encoding='utf-8')


def check_replay(path,output):
    snapshot=json.loads(path.read_text(encoding='utf-8'))
    checks=source_verification(snapshot)
    if not all(checks.values()):raise RuntimeError('Source mismatch for '+str(path))
    result=replay_snapshot(snapshot);recorded=snapshot['outcome']
    match=dict(path_available=result['path_available']==recorded['path_available'],
        goal_blocked=result['goal_blocked']==recorded['goal_blocked'],reason=result['reason']==recorded['reason'],
        waypoint=math.dist(result['waypoint'],recorded['waypoint'])<1e-8)
    if not recorded['search_timed_out'] and not all(match.values()):raise RuntimeError('Unexpected geometry mismatch')
    report=dict(input_sha256=sha(path),trigger=snapshot.get('capture_trigger'),source_checks=checks,
        recorded_outcome=recorded,replay=result,geometry_matches=match,
        runtime_timeout_not_reproduced=recorded['search_timed_out'])
    save(output,report)
    return dict(file=path.name,trigger=snapshot.get('capture_trigger'),matches=match,recorded_timeout=recorded['search_timed_out'])


def main():
    out=ROOT/'artifacts/navigation_trace/2026-11-23';out.mkdir(parents=True,exist_ok=False)
    (out/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
    rows=[];replays=[]
    for index,ref in [(1,'7f37cea'),(2,'7f37cea'),(3,'d7ba2f6')]:
        folder=ROOT/f'runs/live_navigation_trace_20261123_{index:02d}/2_candidate_cam0'
        row=helpers.analysis.analyze(folder)
        helpers.runtime.ROOT=folder.parent;full=helpers.runtime.case('candidate')
        row['source_file_checks']=helpers.source_checks(ref,full['summary']['environment']['source_manifest'])
        row.update(source_commit=ref,performance=full['public']['performance_full_run_summaries'])
        destination=out/f'live_{index:02d}';export(folder/'flight',destination)
        for src,dst in [('flight/summary.json','raw_summary.json'),('preflight_ground.json','preflight_ground.json'),
                ('postflight_ground.json','postflight_ground.json'),('swarm.yaml','swarm.yaml'),
                ('swarm.manifest.json','swarm.manifest.json'),('ownership.json','ownership.json')]:
            shutil.copy2(folder/src,destination/dst)
        save(destination/'decision_analysis.json',row)
        trace_dir=destination/'navigation_traces';trace_dir.mkdir()
        traces=list((folder/'flight/runtime/navigation_traces').glob('*.json'))
        for path in traces:
            target=trace_dir/path.name;shutil.copy2(path,target)
            replays.append(check_replay(target,target.with_suffix('.replay.json')))
        rows.append(dict(case=index,source_commit=ref,passed=row['passed'],flight_duration_s=row['flight_duration_s'],
            collision_count=row['collision_count'],minimum_pair_distance_m=row['minimum_pair_distance_m'],
            actual_ground_verified=row['actual_ground_verified'],reported_cleanup_verified=row['reported_cleanup_verified'],
            diagnostic_files=len(traces)))
    synthetic=out/'synthetic';synthetic.mkdir()
    for kind in ('start_blocked','temporary_peer','cpu_budget'):
        clock=[0.]
        def advancing():clock[0]+=.1;return clock[0]
        router=CommittedGridRouter(NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1),
            clock=advancing if kind=='cpu_budget' else lambda:10.,hardening=HardeningConfig(enabled=True))
        points=[(2.,0.)] if kind=='cpu_budget' else ([(1.25*math.cos(t),1.25*math.sin(t)) for t in np.linspace(0,2*math.pi,24,endpoint=False)] if kind=='start_blocked' else [])
        router.observe(points,scan_id=1);router.capture_next=True
        router.waypoint((0.,0.,-2.),(4.,0.,-2.),temporary_obstacles=[(4.,0.)] if kind=='temporary_peer' else [])
        payload=router.captured_diagnostic
        payload.update(capture_trigger='synthetic_fixture',source_file_sha256={name:sha(ROOT/name) for name in TRACE_SOURCES})
        path=synthetic/(kind+'.json');save(path,payload)
        check_replay(path,path.with_suffix('.replay.json'))
    report=dict(evidence_type='diagnostic_instrumentation_validation',live_cases=rows,live_replays=replays,
        persistent_no_route_observed=any(r['trigger']=='persistent_no_route' for r in replays),
        prior_failed_case_reproduced=False,
        prior_failure_fixed=False,
        scope='Initial-route replay validates the diagnostic chain; synthetic cases validate classifications, not the prior AirSim failure. No behavior fix or performance gain claim.')
    save(out/'validation.json',report)
    shutil.copy2(Path(__file__),out/'archived_export.py')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()

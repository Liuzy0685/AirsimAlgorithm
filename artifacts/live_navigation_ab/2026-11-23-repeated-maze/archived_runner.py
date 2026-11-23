"""Three additional serial matched pairs; no mutation of flight source/configs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owned_ab',ROOT/'runs/live_navigation_ab_20261123_01/run_live_ab.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng=random.Random(20261123)
    orders=[['radial','swept'],['swept','radial'],rng.sample(['radial','swept'],2)]
    rng.shuffle(orders)
    plan=dict(order_random_seed=20261123,seed_scope='Run order only; not an AirSim physics/randomization seed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_manifest=harness.source_manifest(),settings_sha256=sha(ROOT/'configs/acceptance/settings_2.json'),
        formation_sha256=sha(ROOT/'configs/acceptance/formation_2.yaml'),
        map_sha256=sha(Path(r'D:\30817\Adrone\blocks\Content\maze.umap')),
        harness_sha256=sha(ROOT/'runs/live_navigation_ab_20261123_01/run_live_ab.py'),
        runner_sha256=sha(Path(__file__)),pair_orders=orders,vehicle_count=2,extra_cameras=0,
        note='Three additional exploratory matched pairs, no safety thresholds or shared settings edited')
    if (OUT/'plan.json').exists():raise RuntimeError('Existing plan: refusing to overwrite repeated trials')
    harness.atomic_json(OUT/'plan.json',plan)
    report=dict(started_unix_ms=int(time.time()*1000),cases=[],planned_pairs=3)
    harness.atomic_json(OUT/'report.json',report)
    try:
        for pair,order in enumerate(orders,1):
            for geometry in order:
                if (OUT/'STOP_AFTER_CURRENT').exists():
                    report['stopped_at_safe_checkpoint']=True;return 0
                if harness.source_manifest()!=plan['source_manifest']:
                    raise RuntimeError('Flight source changed; stopping before next launch')
                if sha(ROOT/'configs/acceptance/settings_2.json')!=plan['settings_sha256'] or sha(ROOT/'configs/acceptance/formation_2.yaml')!=plan['formation_sha256']:
                    raise RuntimeError('Shared configuration changed')
                harness.OUT=OUT/f'pair_{pair:02d}_{geometry}'
                os.environ['NAV_AB_CANDIDATE_FEATURES']='memory,recovery,smoothing,latency_limit,passage'
                os.environ['NAV_AB_LATENCY_CLEARANCE_MODE']=geometry
                print(json.dumps(dict(event='case_starting',pair=pair,geometry=geometry,output=str(harness.OUT))),flush=True)
                result=harness.run_case(2,'candidate',0)
                folder=harness.OUT/'2_candidate_cam0'
                raw=json.loads((folder/'flight/summary.json').read_text(encoding='utf-8'))
                owner=json.loads((folder/'ownership.json').read_text(encoding='utf-8'))
                result.update(pair=pair,geometry=geometry,
                    flight_duration_s=max(v['flight_duration_s'] for v in raw['vehicle_results'].values()),
                    collision_count=raw['collision_events_excluding_landed_ground'])
                report['cases'].append(result);harness.atomic_json(OUT/'report.json',report)
                print(json.dumps(dict(event='case_completed',**result)),flush=True)
                if any(owner.get(key) for key in ('ue_shutdown_unverified','camera_shutdown_unverified')):
                    raise RuntimeError('Owned process shutdown unverified')
                if any(harness.port_open(port) for port in (41451,8765,8780)):
                    raise RuntimeError('A flight service remained after verified cleanup')
                if not result['passed'] or result['collision_count']:
                    raise RuntimeError('Acceptance/safety failure; no further automatic launch')
        report['all_planned_cases_completed']=True
        return 0
    except Exception as exc:
        report['failure']=str(exc)
        print(json.dumps(dict(event='repeated_suite_stopped',reason=str(exc))),flush=True)
        return 2
    finally:
        report['finished_unix_ms']=int(time.time()*1000)
        harness.atomic_json(OUT/'report.json',report)


if __name__=='__main__':raise SystemExit(main())

"""Read raw live pilot evidence without adding RPC or changing flight outcomes."""
from pathlib import Path
import hashlib
import json
import re

import yaml

ROOT=Path(__file__).resolve().parent


def case(variant):
    folder=ROOT/f'2_{variant}_cam0'
    path=folder/'flight/summary.json'
    if not path.exists():return None
    summary=json.loads(path.read_text(encoding='utf-8'))
    ownership=json.loads((folder/'ownership.json').read_text(encoding='utf-8'))
    ground=json.loads((folder/'postflight_ground.json').read_text(encoding='utf-8'))
    performance={};window_p95={};navigation_final={}
    for name in summary['vehicle_results']:
        lines=(folder/f'flight/runtime/{name}.log').read_text(encoding='utf-8',errors='replace').splitlines()
        for line in lines:
            if 'realtime_architecture_summary ' in line:
                values={}
                for key,value in re.findall(r'(\w+)=([^\s]+)',line.split('realtime_architecture_summary ',1)[1]):
                    try:values[key]=float(value)
                    except ValueError:values[key]=value
                performance[name]=values
            if 'navigation_runtime {' in line:
                row=json.loads(line.split('navigation_runtime ',1)[1])
                stats=window_p95.setdefault(name,{})
                for key,value in row['timing'].items():stats[key]=max(stats.get(key,0),value['p95'])
    nav=folder/'flight/navigation.jsonl'
    if nav.exists():
        for line in nav.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);navigation_final[row['vehicle']]={k:row[k] for k in
                ('tracking_error_m','path_available','goal_blocked','route_version')}
    results=summary['vehicle_results']
    return dict(summary=summary,ownership=ownership,
        public=dict(variant=variant,acceptance_passed=summary['flight_acceptance_passed'],
            mission_completed=any(r['termination_reason']=='mission_complete' for r in results.values()),
            elapsed_including_startup_cleanup_s=summary['elapsed_s'],
            flight_duration_s={n:r['flight_duration_s'] for n,r in results.items()},
            termination_reason={n:r['termination_reason'] for n,r in results.items()},
            collisions=summary['collision_events_excluding_landed_ground'],
            minimum_sampled_separation_m=summary['minimum_sampled_separation_m'],
            minimum_piecewise_linear_separation_m=summary['minimum_piecewise_linear_separation_m'],
            configured_safety_distance_m=summary['configured_safety_distance_m'],
            path_length_m=summary['path_length_m'],
            verified_ground_api_released=ground['verified_ground_released'],
            performance_full_run_summaries=performance,
            observed_window_max_p95=window_p95,
            final_navigation_feedback=navigation_final))


def main():
    old,new=case('baseline'),case('candidate')
    report=dict(evidence_type='live_airsim_single_paired_pilot',
        limitations=['One trial per variant: no statistical-superiority claim',
            'No additional camera streams; built-in AirSim subwindows unchanged',
            'Pairwise separation is sampled/piecewise-linear, not an exact continuous-time proof',
            'Window P95 maxima are not global frame-level percentiles'],
        baseline=old['public'] if old else None,candidate=new['public'] if new else None)
    if old and new:
        env1,env2=old['summary']['environment'],new['summary']['environment']
        checks={k:env1.get(k)==env2.get(k) for k in ('map_asset_sha256','settings_sha256','source_manifest')}
        config1=yaml.safe_load((ROOT/'2_baseline_cam0/swarm.yaml').read_text(encoding='utf-8'))
        config2=yaml.safe_load((ROOT/'2_candidate_cam0/swarm.yaml').read_text(encoding='utf-8'))
        flags1=config1.pop('local_navigation_hardening');flags2=config2.pop('local_navigation_hardening')
        checks['non_treatment_config_identical']=config1==config2
        form_key='configs/acceptance/formation_2.yaml'
        def formation_hash(environment):
            hashes={key.replace('\\','/'):value for key,value in environment['config_sha256'].items()}
            return hashes.get(form_key)
        h1,h2=formation_hash(env1),formation_hash(env2)
        checks['formation_hash_identical']=bool(h1 and h1==h2)
        report.update(comparable_conditions=checks,treatment=dict(baseline=flags1,candidate=flags2),
            both_completed_and_clean=all(r['public']['acceptance_passed'] and r['public']['mission_completed']
                and r['public']['verified_ground_api_released'] for r in (old,new)))
        if all(checks.values()) and report['both_completed_and_clean']:
            a=max(old['public']['flight_duration_s'].values());b=max(new['public']['flight_duration_s'].values())
            report['flight_duration_relative_change']=(b-a)/a
        report['raw_summary_sha256']={v:hashlib.sha256((ROOT/f'2_{v}_cam0/flight/summary.json').read_bytes()).hexdigest()
                                     for v in ('baseline','candidate')}
    (ROOT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','candidate','treatment')},ensure_ascii=False))


if __name__=='__main__':main()

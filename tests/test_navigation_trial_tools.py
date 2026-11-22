"""A/B evidence screening: fail closed on mismatch, missing data or regression."""
import copy
import json
from pathlib import Path

import yaml

from scripts.compare_navigation_trials import compare
from scripts.prepare_navigation_trial import prepare
from scripts.summarize_navigation_runtime import summarize


def record(seed=0):
    return dict(case_id='maze_2_no_camera',seed=seed,
        environment=dict(map_sha256='a'*64,settings_sha256='b'*64,formation_sha256='c'*64,
            vehicle_count=2,camera_count=0,requested_fps=5,speed_limit_mps=.35,
            safety_distance_m=2.,hard_clearance_m=.9,max_scan_age_s=.6),
        metrics=dict(success=True,collision_count=0,cleanup_verified=True,
            minimum_peer_distance_m=2.3,minimum_obstacle_clearance_m=1.,
            planner_p95_ms=10.,sensor_age_p95_ms=300.,control_p95_ms=100.,
            task_time_s=50.,path_length_m=20.,unresolved_deadlocks=0))


def test_missing_evidence_and_unmatched_runs_are_inconclusive():
    old=[record()];new=copy.deepcopy(old)
    del new[0]['metrics']['sensor_age_p95_ms']
    assert compare(old,new)['verdict']=='inconclusive'
    assert compare(old,[])['verdict']=='inconclusive'


def test_different_map_camera_load_or_safety_thresholds_are_not_comparable():
    for key,value in [('map_sha256','d'*64),('camera_count',10),('max_scan_age_s',1.)]:
        old=[record()];new=copy.deepcopy(old);new[0]['environment'][key]=value
        assert compare(old,new)['verdict']=='inconclusive'


def test_faster_failure_collision_or_unverified_cleanup_never_counts_as_improvement():
    for metric,value in [('success',False),('collision_count',1),('cleanup_verified',False),
                          ('minimum_peer_distance_m',1.9),('minimum_obstacle_clearance_m',.8),
                          ('sensor_age_p95_ms',601.),('unresolved_deadlocks',1)]:
        old=[record()];new=copy.deepcopy(old)
        new[0]['metrics'][metric]=value;new[0]['metrics']['task_time_s']=10.
        assert compare(old,new)['verdict']=='regression'


def test_too_few_runs_never_proves_better_even_if_time_improves():
    old=[record()];new=copy.deepcopy(old);new[0]['metrics']['task_time_s']=40.
    assert compare(old,new)['verdict']=='inconclusive'


def test_two_failing_variants_cannot_prove_an_improvement_from_lower_latency():
    old=[record(i) for i in range(5)];new=copy.deepcopy(old)
    for a,b in zip(old,new):
        a['metrics']['success']=b['metrics']['success']=False
        b['metrics']['planner_p95_ms']=1.
    assert compare(old,new)['verdict']=='inconclusive'


def test_matched_repeats_report_benefit_and_tradeoffs_explicitly():
    old=[record(i) for i in range(5)];new=copy.deepcopy(old)
    for row in new:
        row['metrics']['task_time_s']=40.
        row['metrics']['planner_p95_ms']=10.5
    result=compare(old,new)
    assert result['verdict']=='promising_with_tradeoffs'
    assert result['tradeoffs'] and result['improvements']


def test_large_latency_regression_blocks_a_shorter_path_claim():
    old=[record(i) for i in range(5)];new=copy.deepcopy(old)
    for row in new:
        row['metrics']['path_length_m']=15.
        row['metrics']['control_p95_ms']=130.
    assert compare(old,new)['verdict']=='regression'


def test_trial_generator_changes_only_policy_not_shared_settings_or_thresholds(tmp_path):
    base=Path(__file__).resolve().parents[1]/'configs/acceptance/swarm_2_navigation.yaml'
    before=base.read_bytes();output=tmp_path/'swarm.yaml'
    prepare(base,output,['memory','smoothing'])
    result=yaml.safe_load(output.read_text(encoding='utf-8'))
    policy=result.pop('local_navigation_hardening')
    assert result==yaml.safe_load(before)
    assert policy['memory'] and policy['smoothing'] and not policy['passage']
    assert base.read_bytes()==before


def test_log_summary_keeps_window_percentiles_distinct_from_global_percentiles(tmp_path):
    log=tmp_path/'Drone1.log'
    row=dict(vehicle='Drone1',category='traffic_wait',reason='passage_wait',
             passage_state='YIELDING',timing={'sensor_age_ms':dict(p50=100,p95=300,maximum=400)})
    log.write_text('INFO navigation_runtime '+json.dumps(row)+'\n',encoding='utf-8')
    result=summarize([log])
    assert result['evidence_complete']
    assert result['vehicles']['Drone1']['observed_window_max_p95']['sensor_age_ms']==300
    assert 'global' in result['scope']

from dataclasses import replace
import pytest
from coordination.acceptance_metrics import AcceptanceMetrics,segment_min_distance
from test_swarm_wiring import fixture


def test_crossing_between_samples_is_not_missed():
    assert segment_min_distance((-1,0,0),(1,0,0),(1,0,0),(-1,0,0)) == 0
    assert segment_min_distance((0,0,0),(0,0,0),(3,0,0),(3,0,0)) == 3


def test_metrics_do_not_invent_distance_when_no_complete_state():
    metrics=AcceptanceMetrics(["a","b"],safety_distance_m=2)
    assert metrics.summary()["separation_observed_ok"] is None


def test_incomplete_snapshot_breaks_trajectory_interpolation():
    _,names,publisher,_,_=fixture()
    snap=publisher.latest_snapshot
    m=AcceptanceMetrics(names,safety_distance_m=2)
    assert m.observe(snap,received_at=1,stream_id="s")
    assert not m.observe(snap,received_at=1.1,stream_id="s")
    m.observe(replace(snap,sequence_id=2,collection_complete=False),received_at=2,stream_id="s")
    m.observe(replace(snap,sequence_id=3),received_at=3,stream_id="s")
    summary=m.summary()
    assert summary["complete_frames"]==2 and summary["incomplete_frames"]==1
    assert summary["receive_intervals"]["count"]==0
    assert summary["minimum_sampled_separation_m"]==pytest.approx(10)


def test_status_transitions_are_deduplicated_per_vehicle():
    m=AcceptanceMetrics(['a','b'],safety_distance_m=2)
    for t in (1.,2.,3.):
        for n in ('a','b'):m.status(dict(vehicle_name=n,state='NORMAL',reason='clear'),t)
    assert len(m.status_events)==2
    m.status(dict(vehicle_name='a',state='AVOIDING',reason='CPA'),4.)
    assert len(m.status_events)==3


def test_simulation_progress_is_reported_separately_from_wall_time():
    _,names,publisher,_,_=fixture()
    snap=publisher.latest_snapshot
    m=AcceptanceMetrics(names,safety_distance_m=2)
    m.observe(snap,received_at=10.,stream_id='a')
    for state in snap.states.values():state.timestamp+=2_000_000_000
    m.observe(replace(snap,sequence_id=2),received_at=14.,stream_id='a')
    assert m.summary()['sim_to_wall_time_ratio']==pytest.approx(.5)

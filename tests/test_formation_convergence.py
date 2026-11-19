import copy
from coordination.formation_convergence import FormationConvergence


def evidence():
    m = dict(state='RUNNING', applied_revision=2,
             config=dict(vehicle_names=['Drone1','Drone2'],leader='Drone1'),
             formation_runtime=dict(revision=2,effective_mode='nominal',received_at_unix_ms=1000,
                 sim_timestamp_ns=1_000_000_000,slot_targets_ned_m={'Drone2':[0.,3.,-2.]},
                 tracking_error_m={'Drone2':0.}))
    r = dict(stale=False,collection_complete=True,timestamp_unix_ms=1000,stream_id='run-1',snapshot_id=1,
             vehicles={name:dict(position_ned_m=[0.,3.*i,-2.],linear_velocity_ned_mps=[.3,0.,0.],
                                timestamp=1_000_000_000) for i,name in enumerate(['Drone1','Drone2'])})
    return m,r


def observe(c,m,r,t=1.):
    return c.observe(m,r,now=t,now_ms=int(t*1000))


def test_applied_or_reported_zero_error_does_not_prove_physical_convergence():
    c=FormationConvergence();m,r=evidence()
    m['formation_runtime']=None
    assert not observe(c,m,r)['physical_settled']
    m,r=evidence();r['vehicles']['Drone2']['position_ned_m'][0]=5.
    result=observe(c,m,r)
    assert not result['physical_settled'] and result['tracking_error_m']['Drone2']==5.


def test_only_distinct_fresh_frames_and_dwell_prove_settlement():
    c=FormationConvergence();m,r=evidence()
    for i in range(4):
        t=1.+i*.4
        m['formation_runtime']['received_at_unix_ms']=r['timestamp_unix_ms']=int(t*1000)
        r['snapshot_id']=i+1
        result=observe(c,m,r,t)
        assert result['physical_settled'] is (i==3)
    # API polling/replayed snapshots cannot manufacture new evidence.
    for i in range(20):
        assert observe(c,m,r,2.2)['distinct_frames']==4
    assert not observe(c,m,r,3.)['physical_settled']


def test_stream_restart_revision_change_temporary_mode_and_missing_peer_reset():
    for change in ('stream','revision','mode','missing','nan','simulation_time'):
        c=FormationConvergence(dwell_s=0.,minimum_frames=1);m,r=evidence()
        assert observe(c,m,r)['physical_settled']
        if change=='stream':r['stream_id']='new'
        if change=='revision':m['applied_revision']=3
        if change=='mode':m['formation_runtime']['effective_mode']='trail'
        if change=='missing':del r['vehicles']['Drone2']
        if change=='nan':r['vehicles']['Drone2']['position_ned_m'][0]=float('nan')
        if change=='simulation_time':m['formation_runtime']['sim_timestamp_ns']=10_000_000_000
        result=observe(c,m,r)
        assert not result['physical_settled'] or change=='stream' and result['distinct_frames']==1

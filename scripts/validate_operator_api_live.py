"""Exercise the real operator HTTP API, including one complete flight and one stop.

Requires an explicitly flight-enabled local API and a clear simulation scene.
Credentials and generated private configurations are never exported.
"""
import argparse
import hashlib
import http.client
import itertools
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.operator_common import atomic_json,read_json
from scripts.run_swarm_acceptance import source_manifest


def run(args):
    url=urlsplit(args.url)
    if url.scheme!='http' or url.hostname!='127.0.0.1':
        raise ValueError('Only a loopback API may be tested')
    token=Path(args.token_file).read_text().strip()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    checks=[];missions=[];mid=None;trace=[];minimum=math.inf
    manifest=source_manifest()
    def request(method,path,body=None,authorized=True):
        c=http.client.HTTPConnection(url.hostname,url.port,timeout=30)
        headers={'Content-Type':'application/json'}
        if authorized:headers['Authorization']='Bearer '+token
        c.request(method,url.path.rstrip('/')+path,body=json.dumps(body) if body is not None else None,headers=headers)
        response=c.getresponse();value=json.loads(response.read());status=response.status;c.close()
        return status,value
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise RuntimeError(name)
    def rid():return uuid.uuid4().hex
    def wait_task(mission_id,stop_when_airborne=False):
        nonlocal minimum
        deadline=time.monotonic()+300
        seen=set();stop_sent=False
        while time.monotonic()<deadline:
            code,m=request('GET','/missions/'+mission_id)
            check_ok=code==200
            if not check_ok:raise RuntimeError('Task query failed')
            runtime=m['runtime'];states=runtime.get('vehicles',{})
            identity=(runtime.get('stream_id'),runtime.get('snapshot_id'))
            if not runtime['stale'] and runtime.get('collection_complete') and identity not in seen:
                seen.add(identity)
                trace.append(dict(mission_id=mission_id,state=m['state'],runtime=runtime))
                for a,b in itertools.combinations(states,2):
                    minimum=min(minimum,math.dist(states[a]['position_ned_m'],states[b]['position_ned_m']))
            if stop_when_airborne and not stop_sent and any(s.get('landed_state')==1 for s in states.values()):
                status,reply=request('POST','/missions/'+mission_id+'/stop',dict(request_id=rid()))
                check('stop_accepted_before_cleanup',status==202 and reply.get('completed') is False)
                stop_sent=True
            if m['state'] in {'COMPLETED','STOPPED','FAILED','INTERRUPTED'}:
                return m,stop_sent
            if m['state']=='RUNNING' and not stop_when_airborne and not any(c['name']=='B_unwired_is_explicit' for c in checks):
                status,reply=request('PUT','/missions/'+mission_id+'/formation',dict(request_id=rid(),
                    expected_revision=m['revision'],formation=dict(shape='column',spacing_m=3.)))
                check('B_unwired_is_explicit',status==503 and reply.get('error')=='b_control_unavailable')
            time.sleep(.25)
        raise RuntimeError('Task did not complete within live observer limit')
    try:
        status,caps=request('GET','/capabilities')
        check('capabilities_and_flight_enabled',status==200 and caps['flight_enabled'])
        check('operator_auth_required',request('GET','/capabilities',authorized=False)[0]==401)
        status,map_data=request('GET','/maps/current')
        check('measured_map',status==200 and len(map_data['obstacles'])==19 and map_data['collision_certified'] is False)
        body=dict(request_id=rid(),map_id=map_data['map_id'],map_revision=map_data['map_revision'],
            vehicle_names=caps['vehicle_names'],leader=caps['leader'],leader_goal_ned_m=args.goal,
            formation=dict(shape='line',spacing_m=3.))
        check('old_map_rejected',request('POST','/missions/validate',dict(body,map_revision='old'))[0]==409)
        status,report=request('POST','/missions/validate',body)
        check('live_preflight',status==200 and report['valid'])
        status,created=request('POST','/missions',body)
        check('create_task',status==201)
        mid=created['mission']['mission_id']
        check('create_replay_same_task',request('POST','/missions',body)[1]['mission']['mission_id']==mid)
        for revision,shape in [(1,'vee'),(2,'line')]:
            status,reply=request('PUT','/missions/'+mid+'/formation',dict(request_id=rid(),expected_revision=revision,
                formation=dict(shape=shape,spacing_m=3.)))
            check('initial_shape_'+shape,status==202 and reply['operation']['state']=='APPLIED_TO_DRAFT')
        start=dict(request_id=rid(),expected_revision=3)
        status,reply=request('POST','/missions/'+mid+'/start',start)
        check('start_accepted',status==202)
        check('start_replay_same_receipt',request('POST','/missions/'+mid+'/start',start)[1]==reply)
        m,_=wait_task(mid);missions.append(m)
        check('complete_task_and_release',m['state']=='COMPLETED' and all(r['cleanup']['control_released'] for r in m['vehicle_results'].values()))
        mid=None
        # At the actual terminal ground positions, start a new nearby task and
        # request a cooperative stop while takeoff/flight is still in progress.
        body.update(request_id=rid(),leader_goal_ned_m=[args.goal[0]+1.,args.goal[1],args.goal[2]])
        status,created=request('POST','/missions',body)
        check('second_task_after_verified_release',status==201)
        mid=created['mission']['mission_id']
        request('POST','/missions/'+mid+'/start',dict(request_id=rid(),expected_revision=1))
        m,sent=wait_task(mid,True);missions.append(m)
        check('stop_and_release',sent and m['state']=='STOPPED' and all(r['cleanup']['control_released'] for r in m['vehicle_results'].values()))
        check('sampled_separation',minimum>=caps['safety_distance_m'])
        mid=None
    finally:
        if mid:
            status,m=request('GET','/missions/'+mid)
            if status==200 and m['state'] in {'STARTING','TAKING_OFF','RUNNING','STOPPING','LANDING'}:
                request('POST','/missions/'+mid+'/stop',dict(request_id=rid()))
                try:
                    m,_=wait_task(mid);missions.append(m)
                except Exception:pass
        atomic_json(output/'report.json',dict(evidence_type='live_airsim_operator_http',checks=checks,
            passed=bool(checks) and all(c['passed'] for c in checks) and len(missions)==2
                   and missions[0]['state']=='COMPLETED' and missions[1]['state']=='STOPPED',
            missions=missions,minimum_sampled_separation_m=minimum if math.isfinite(minimum) else None,
            map_revision=map_data['map_revision'] if 'map_data' in locals() else None,source_manifest=manifest,
            limitation='B runtime configuration was not implemented/tested in flight. Sampled API states do not prove continuous-time separation.'))
        with (output/'snapshots.jsonl').open('w',encoding='utf-8') as f:
            for row in trace:f.write(json.dumps(row)+'\n')
    report=read_json(output/'report.json')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),
                         minimum_sampled_separation_m=report['minimum_sampled_separation_m'])))
    return 0 if report['passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8790/api/v1')
    parser.add_argument('--token-file',required=True)
    parser.add_argument('--goal',type=float,nargs=3,default=[15.5,-3.,-2.])
    parser.add_argument('--output',required=True)
    parser.add_argument('--confirm-simulation-clearance',action='store_true')
    args=parser.parse_args()
    if not args.confirm_simulation_clearance:parser.error('--confirm-simulation-clearance required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())

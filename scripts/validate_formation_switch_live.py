"""Real operator -> B -> A flight test, distinguishing APPLIED from convergence.

Requires an owned, clear simulation and an explicitly flight-enabled operator
API. Requests a cooperative stop in finally; never resets or forces landing.
"""
import argparse
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
from coordination.operator_common import atomic_json
from scripts.run_swarm_acceptance import source_manifest


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    parsed=urlsplit(args.url)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost'):
        raise ValueError('Local operator API required')
    token=Path(args.token_file).read_text(encoding='utf-8').strip()
    def request(method,path,body=None):
        c=http.client.HTTPConnection(parsed.hostname,parsed.port,timeout=10)
        try:
            c.request(method,parsed.path.rstrip('/')+path,
                body=json.dumps(body) if body is not None else None,
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();value=json.loads(r.read())
            return r.status,value
        finally:c.close()
    def rid():return uuid.uuid4().hex
    mid=None;final=None;operations=[];trace=[];failure=None;minimum=math.inf
    start=time.monotonic();last_identity=None;settled_after=[];stop_sent=False
    try:
        status,caps=request('GET','/capabilities')
        if status!=200 or not caps['flight_enabled']:raise RuntimeError('API is not flight-enabled')
        status,geometry=request('GET','/maps/current')
        if status!=200:raise RuntimeError('Map unavailable')
        body=dict(request_id=rid(),map_id=geometry['map_id'],map_revision=geometry['map_revision'],
            vehicle_names=caps['vehicle_names'],leader=caps['leader'],leader_goal_ned_m=args.goal,
            formation=dict(shape='line',spacing_m=3.))
        status,created=request('POST','/missions',body)
        if status!=201:raise RuntimeError('Mission rejected: '+json.dumps(created))
        mid=created['mission']['mission_id']
        status,reply=request('POST','/missions/'+mid+'/start',dict(request_id=rid(),expected_revision=1))
        if status!=202:raise RuntimeError('Start rejected: '+json.dumps(reply))
        shapes=iter(args.shapes);next_shape=next(shapes,None);pending=None
        running_at=None
        while time.monotonic()-start<args.duration:
            status,m=request('GET','/missions/'+mid)
            if status!=200:raise RuntimeError('Mission query failed')
            runtime=m['runtime'];identity=(runtime.get('stream_id'),runtime.get('snapshot_id'))
            if not runtime.get('stale',True) and runtime.get('collection_complete') and identity!=last_identity:
                last_identity=identity
                trace.append(dict(t=time.monotonic()-start,state=m['state'],revision=m['revision'],
                    applied_revision=m['applied_revision'],formation_convergence=m.get('formation_convergence'),
                    formation_runtime=m.get('formation_runtime'),runtime=runtime))
                states=runtime['vehicles']
                for a,b in itertools.combinations(states,2):
                    minimum=min(minimum,math.dist(states[a]['position_ned_m'],states[b]['position_ned_m']))
            if m['state'] in ('COMPLETED','STOPPED','FAILED','INTERRUPTED'):
                final=m;break
            if m['state']=='RUNNING':
                if running_at is None:running_at=time.monotonic()
                if pending is not None:
                    code,op=request('GET','/operations/'+pending)
                    if code!=200:raise RuntimeError('Operation query failed')
                    if not operations or operations[-1]!=op:operations.append(op)
                    if op['state'] in ('FAILED','REJECTED'):raise RuntimeError('Formation change failed: '+json.dumps(op))
                    convergence=m.get('formation_convergence',{})
                    if (op['state']=='APPLIED' and convergence.get('physical_settled')
                            and convergence.get('revision')==op['revision']):
                        settled_after.append(dict(request_id=pending,revision=op['revision'],evidence=convergence))
                        pending=None;next_shape=next(shapes,None)
                if (pending is None and next_shape is not None and m.get('b_control_online')
                        and time.monotonic()-running_at>=args.switch_after):
                    request_id=rid()
                    code,reply=request('PUT','/missions/'+mid+'/formation',dict(request_id=request_id,
                        expected_revision=m['revision'],formation=dict(shape=next_shape,spacing_m=3.)))
                    if code!=202:raise RuntimeError('Formation request rejected: '+json.dumps(reply))
                    pending=request_id;operations.append(reply['operation'])
            time.sleep(.2)
        if final is None:raise RuntimeError('Flight observer time limit')
    except Exception as exc:
        failure=str(exc)
    finally:
        if mid and (final is None or final['state'] not in ('COMPLETED','STOPPED','FAILED','INTERRUPTED')):
            try:
                request('POST','/missions/'+mid+'/stop',dict(request_id=rid()));stop_sent=True
                deadline=time.monotonic()+120
                while time.monotonic()<deadline:
                    _,m=request('GET','/missions/'+mid)
                    if m['state'] in ('COMPLETED','STOPPED','FAILED','INTERRUPTED'):
                        final=m;break
                    time.sleep(.5)
            except Exception as exc:failure=(failure or '')+'; cleanup observer: '+str(exc)
        released=bool(final and final['vehicle_results'] and all(
            r.get('cleanup',{}).get('control_released') and r.get('landing_confirmed')
            and r.get('disarmed') and r.get('api_control_released') for r in final['vehicle_results'].values()))
        passed=bool(not failure and final and final['state']=='COMPLETED' and released
                    and len(settled_after)==len(args.shapes) and minimum>=2.)
        atomic_json(output/'report.json',dict(evidence_type='live_airsim_runtime_formation_switch',
            passed=passed,failure=failure,minimum_sampled_separation_m=minimum if math.isfinite(minimum) else None,
            requested_shapes=args.shapes,operations=operations,physical_settlements=settled_after,
            cleanup_verified=released,cooperative_stop_requested=stop_sent,final_mission=final,
            source_manifest=source_manifest(),limitation='Sampled separation; no continuous-time or general convergence proof'))
        atomic_json(output/'trace.json',trace)
    print(json.dumps(dict(passed=passed,failure=failure,settled_changes=len(settled_after),
                         cleanup_verified=released,final_state=final['state'] if final else None)))
    return 0 if passed else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8790/api/v1')
    parser.add_argument('--token-file',required=True)
    parser.add_argument('--goal',type=float,nargs=3,default=[15.5,-3.,-2.])
    parser.add_argument('--shapes',nargs='+',choices=['line','vee','column'],default=['vee','line'])
    parser.add_argument('--switch-after',type=float,default=8.)
    parser.add_argument('--duration',type=float,default=300.)
    parser.add_argument('--output',required=True)
    parser.add_argument('--confirm-simulation-clearance',action='store_true')
    args=parser.parse_args()
    if not args.confirm_simulation_clearance:parser.error('--confirm-simulation-clearance required')
    if not math.isfinite(args.duration) or not 30<=args.duration<=600:parser.error('duration must be 30..600 s')
    if not math.isfinite(args.switch_after) or not 0<=args.switch_after<=60:parser.error('switch-after must be 0..60 s')
    return run(args)


if __name__=='__main__':raise SystemExit(main())

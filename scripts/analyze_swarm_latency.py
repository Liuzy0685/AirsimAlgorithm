"""Summarize provided failure logs without touching the source evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.operator_common import atomic_json


def fields(line):
    result={}
    for key,value in re.findall(r'(\w+)=([^\s]+)',line):
        try:result[key]=float(value)
        except ValueError:result[key]=value
    return result


def analyze(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8-sig'))
    report=dict(evidence_type='provided_failure_logs_not_a_rerun',
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()},
        flight_acceptance_passed=summary['flight_acceptance_passed'],vehicles={})
    for name in ('Drone1','Drone2'):
        text=(directory/f'{name}.log').read_text(encoding='utf-8-sig')
        lines=text.splitlines()
        result=json.loads((directory/f'{name}.result.json').read_text(encoding='utf-8-sig'))
        timing=[fields(line) for line in lines if 'loop_exclusive_timing ' in line]
        perception=[fields(line) for line in lines if 'perception_worker_timing ' in line]
        overview=next((fields(line) for line in reversed(lines) if 'realtime_architecture_summary ' in line),{})
        rejects=[dict(line=i+1,text=line) for i,line in enumerate(lines) if 'gateway_dispatch_rejected' in line]
        report['vehicles'][name]=dict(result=result,architecture_summary=overview,
            sampled_loop_timing=timing,sampled_perception_timing=perception,gateway_rejections=rejects)
    formation=(directory/'formation.log').read_text(encoding='utf-8-sig')
    report['B_command_ACKs']=dict(accepted=formation.count("'accepted': True"),rejected=formation.count("'accepted': False"))
    report['interpretation']=dict(first_failure='A final dispatch rejected aged LiDAR; B ACK acceptance does not bypass final A checks',
        source_of_latency='Mixed RPC, Python processing, diagnostic workers and potential engine load; historical dual-UE attribution is not established',
        cleanup='Drone1 cleanup unverified/manual intervention; Drone2 landed and API released')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=analyze(args.logs)
    atomic_json(args.output,report)
    print(json.dumps(dict(vehicles={n:v['result']['termination_reason'] for n,v in report['vehicles'].items()},
                          ACKs=report['B_command_ACKs']),ensure_ascii=False))


if __name__=='__main__':main()

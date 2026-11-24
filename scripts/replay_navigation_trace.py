"""Replay a bounded navigation diagnostic locally; never imports an AirSim client."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.navigation_trace import replay_snapshot,source_verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-source-mismatch',action='store_true',help='Explicit counterfactual replay after a source change; not an exact-version replay.')
    args=parser.parse_args()
    if args.snapshot.stat().st_size>8*1024*1024:parser.error('diagnostic exceeds 8 MiB')
    if args.output and args.output.exists():parser.error('output exists; use a new path')
    snapshot=json.loads(args.snapshot.read_text(encoding='utf-8'))
    checks=source_verification(snapshot)
    if not all(checks.values()) and not args.allow_source_mismatch:parser.error('source hashes differ or are missing; use matching code or explicitly allow a counterfactual replay')
    result=replay_snapshot(snapshot)
    result['source_file_checks']=checks
    result['recorded_outcome']=snapshot['outcome']
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()

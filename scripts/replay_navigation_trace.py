"""Replay a bounded navigation diagnostic locally; never imports an AirSim client."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.navigation_trace import replay_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.snapshot.stat().st_size>8*1024*1024:parser.error('diagnostic exceeds 8 MiB')
    if args.output and args.output.exists():parser.error('output exists; use a new path')
    snapshot=json.loads(args.snapshot.read_text(encoding='utf-8'))
    result=replay_snapshot(snapshot)
    result['recorded_outcome']=snapshot['outcome']
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()

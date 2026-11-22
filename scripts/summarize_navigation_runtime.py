"""Read bounded navigation_runtime JSON logs; no RPC, no flight commands.

Logs contain overlapping rolling windows. Report maximum window P95, never
pretend that aggregating those percentiles yields a global frame-level P95.
"""
import argparse
import json
from pathlib import Path


def summarize(paths):
    vehicles={};malformed=0
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8',errors='replace').splitlines():
            if 'navigation_runtime {' not in line:continue
            try:
                row=json.loads(line.split('navigation_runtime ',1)[1])
                name=row['vehicle']
                item=vehicles.setdefault(name,dict(windows=0,categories={},observed_window_max_p95={},maximum={}))
                item['windows']+=1
                category=row['category'];item['categories'][category]=item['categories'].get(category,0)+1
                for key,value in row['timing'].items():
                    item['observed_window_max_p95'][key]=max(item['observed_window_max_p95'].get(key,0),value['p95'])
                    item['maximum'][key]=max(item['maximum'].get(key,0),value['maximum'])
                item['last_passage_state']=row.get('passage_state')
                item['last_reason']=row.get('reason')
            except (ValueError,KeyError,TypeError):malformed+=1
    return dict(scope='rolling-window observations, not a flight acceptance or global P95',
                vehicles=vehicles,malformed_windows=malformed,
                evidence_complete=bool(vehicles) and malformed==0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.logs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

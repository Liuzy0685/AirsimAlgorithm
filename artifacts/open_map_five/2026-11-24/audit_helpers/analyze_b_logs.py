import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
LOGS=ROOT/'imported/formation_live_20261123_v_open'
rows=[]
for path in sorted(LOGS.glob('Drone[1-5].log')):
    text=path.read_text(encoding='utf-8')
    routes=re.findall(r'route vehicle=\S+ version=(\d+) available=(True|False) blocked_goal=(True|False).*goal=(\([^\n]+?\)) waypoint=(\([^\n]+?\))',text)
    positions=[tuple(map(float,m)) for m in re.findall(r'flight_state\s+pos=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\)',text)]
    stats={}
    for line in text.splitlines():
        if 'realtime_architecture_summary ' in line:
            stats={k:float(v) for k,v in re.findall(r'(\w+)=([-0-9.]+)',line.split('realtime_architecture_summary ',1)[1])}
    result=json.loads(path.with_suffix('.result.json').read_text(encoding='utf-8'))
    rows.append(dict(vehicle=path.stem,route_record_count=len(routes),unavailable_records=sum(r[1]=='False' for r in routes),
        goal_blocked_records=sum(r[2]=='True' for r in routes),last_route=routes[-1] if routes else None,
        maximum_logged_xy_displacement_m=max((math.dist(p[:2],positions[0][:2]) for p in positions),default=None),
        performance=stats,result=result))
formation=(LOGS/'formation.log').read_text(encoding='utf-8')
report=dict(vehicles=rows,accepted_ack_count=formation.count("'accepted': True"),rejected_ack_count=formation.count("'accepted': False"),
    scope='Provided logs only; archive has no source commit/config contents/map hash. Hover/trajectory source counters are not physical-motion classifications.')
(ROOT/'b_log_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ('performance','result')},ensure_ascii=False))
print('ACK',report['accepted_ack_count'],report['rejected_ack_count'])

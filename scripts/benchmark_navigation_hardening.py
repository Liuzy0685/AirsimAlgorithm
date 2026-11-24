"""Offline memory/router A/B fixtures. No flight, RPC or camera simulation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from coordination.navigation import NavigationConfig,CommittedGridRouter,segment_clear
from coordination.navigation_hardening import HardeningConfig


class Clock:
    def __init__(self):self.offset=0.;self.start=time.perf_counter()
    def __call__(self):return self.offset+time.perf_counter()-self.start


def run(task):
    worker,iterations=task
    angle=np.linspace(0.,math.tau,8000,endpoint=False)
    pillar=np.column_stack((4.+.5*np.cos(angle),.5*np.sin(angle)))
    rows=[]
    for scene in ('visible_8000_points','previously_seen_wall_outside_view'):
        for enabled in (False,True):
            clock=Clock()
            policy=HardeningConfig(enabled=enabled)
            router=CommittedGridRouter(NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1),
                clock=clock,hardening=policy,minimum_clearance_m=.9)
            durations=[];available=misses=timeouts=0
            for i in range(iterations):
                clock.offset=i*.75
                cloud=pillar if scene=='visible_8000_points' or i==0 else np.array([(0.,15.)])
                started=time.perf_counter()
                router.observe(cloud,origin=(0.,0.),scan_id=i+1)
                waypoint=router.waypoint((0.,0.,-2.),(8.,0.,-2.))
                durations.append((time.perf_counter()-started)*1000)
                available+=int(router.path_available)
                timeouts+=int(router.search_timed_out)
                # Offline oracle only checks the chosen first segment against
                # the static pillar seen earlier. This is not a flight collision.
                misses+=int(router.path_available and not segment_clear((0.,0.),waypoint,pillar,.9))
            rows.append(dict(worker=worker,scene=scene,hardening=enabled,frames=iterations,
                step_p50_ms=float(np.percentile(durations,50)),step_p95_ms=float(np.percentile(durations,95)),
                maximum_ms=max(durations),available_frames=available,search_timeouts=timeouts,
                previously_seen_static_segment_misses=misses))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=(1,2,5),default=1)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 20<=args.iterations<=1000:parser.error('iterations must be 20..1000')
    tasks=[(i+1,args.iterations) for i in range(args.workers)]
    if args.workers==1:rows=run(tasks[0])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=[row for batch in pool.map(run,tasks) for row in batch]
    sources=('coordination/navigation_hardening.py','coordination/navigation.py',
        'control/swarm_flight_bridge.py','scripts/camera_service.py','scripts/benchmark_navigation_hardening.py')
    report=dict(scope='offline observed-memory/router fixtures; CPU workers are not cooperative flights',
        recorded_utc=datetime.now(timezone.utc).isoformat(),workers=args.workers,iterations=args.iterations,
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256_lf={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in sources},rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for enabled in (False,True):
        subset=[row for row in rows if row['hardening']==enabled]
        print(json.dumps(dict(hardening=enabled,max_step_p95_ms=max(row['step_p95_ms'] for row in subset),
            static_segment_misses=sum(row['previously_seen_static_segment_misses'] for row in subset),
            available_frames=sum(row['available_frames'] for row in subset))))


if __name__=='__main__':main()

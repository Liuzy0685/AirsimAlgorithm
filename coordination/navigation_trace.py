"""Bounded, best-effort diagnostics. Never owns a command or safety authority."""
import hashlib
import json
import logging
from pathlib import Path
import queue
import re
import threading
import uuid

import numpy as np


class NavigationTraceRecorder:
    def __init__(self,directory,vehicle,*,max_records=6,interval_s=5.):
        self.directory=Path(directory)
        safe=re.sub(r'[^A-Za-z0-9_-]','_',vehicle)[:40]
        self.prefix=f'{safe}-{uuid.uuid4().hex[:8]}'
        self.max_records=max_records;self.interval_s=interval_s
        self.submitted=0;self.written=0;self.dropped=0;self.errors=0
        self.last_at=-float('inf');self._queue=queue.Queue(maxsize=1)
        self._stop=threading.Event();self._thread=None

    def ready(self,now):
        return not self._stop.is_set() and self.submitted<self.max_records and now-self.last_at>=self.interval_s and not self._queue.full()

    def submit(self,payload,now):
        if not self.ready(now):self.dropped+=1;return False
        # Fail closed for diagnostics overflow; do not truncate inputs then claim
        # exact replay. The flight planner and safety gateway are untouched.
        if (len(payload.get('static_obstacles',()))>8000 or len(payload.get('temporary_obstacles',()))>16384
                or len(payload.get('path_before',()))>50000 or len(payload.get('dynamic_memory',()))>8000):
            self.dropped+=1;return False
        try:self._queue.put_nowait((self.submitted,payload))
        except queue.Full:self.dropped+=1;return False
        self.submitted+=1;self.last_at=now
        if self._thread is None:
            self._thread=threading.Thread(target=self._run,name='navigation-trace',daemon=True)
            self._thread.start()
        return True

    @staticmethod
    def _encode(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)

    def _write(self,index,payload):
        sources={}
        root=Path(__file__).resolve().parents[1]
        for name in ('coordination/navigation.py','coordination/navigation_hardening.py','control/swarm_flight_bridge.py'):
            sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        document=dict(payload,source_file_sha256=sources)
        data=json.dumps(document,ensure_ascii=False,allow_nan=False,default=self._encode,separators=(',',':'))
        self.directory.mkdir(parents=True,exist_ok=True)
        target=self.directory/f'{self.prefix}-{index:02d}.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(data+'\n',encoding='utf-8')
        temporary.replace(target)

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:index,payload=self._queue.get(timeout=.1)
            except queue.Empty:continue
            try:self._write(index,payload);self.written+=1
            except Exception:
                self.errors+=1
                logging.getLogger('navigation_trace').exception('navigation diagnostic write failed')
            finally:self._queue.task_done()

    def close(self,timeout=2.):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()


def replay_snapshot(snapshot):
    """Re-run frozen geometry without RPC, workers or a wall-clock budget."""
    from coordination.navigation import NavigationConfig,CommittedGridRouter
    from coordination.navigation_hardening import HardeningConfig
    if snapshot.get('schema_version')!=1:raise ValueError('unsupported navigation snapshot')
    points=np.asarray(snapshot['static_obstacles'],float).reshape(-1,2)
    temporary=np.asarray(snapshot['temporary_obstacles'],float).reshape(-1,2)
    if len(points)>8000 or len(temporary)>16384 or not np.all(np.isfinite(points)) or not np.all(np.isfinite(temporary)):
        raise ValueError('invalid or oversized diagnostic geometry')
    now=float(snapshot['decision_monotonic_s'])
    router=CommittedGridRouter(NavigationConfig(**snapshot['navigation_config']),clock=lambda:now,
        hardening=HardeningConfig(**snapshot['hardening_config']),minimum_clearance_m=snapshot['minimum_clearance_m'])
    router.path=[tuple(p) for p in snapshot['path_before']]
    router.goal=snapshot['goal_before'];router.escaping=snapshot['escaping_before']
    router.route_version=snapshot['route_version_before']
    router.last_plan=-float('inf') if snapshot['last_plan_monotonic_s'] is None else snapshot['last_plan_monotonic_s']
    router.failed_cells={(x,y):expiry for x,y,expiry in snapshot['failed_cells']}
    router.travel_history=[tuple(p) for p in snapshot['travel_history']]
    router.obstacles=lambda position=None:points.copy()
    waypoint=router.waypoint(snapshot['position'],snapshot['goal'],temporary_obstacles=temporary)
    return dict(waypoint=waypoint,path_available=router.path_available,goal_blocked=router.goal_blocked,
        reason=router.last_route_reason,searches=router.search_diagnostics,escape=router.escape_diagnostic,
        search_timed_out=router.search_timed_out,
        limitation='Frozen geometry only; actual CPU deadlines and later observations are not replayed')

"""Bounded 2-D observed-obstacle routing and navigation feedback contract.

An engineering adaptation of multi-route/subgoal planning, not a reproduction
of RAPTOR or a complete-map/continuous-time convergence guarantee.
"""
from dataclasses import dataclass, asdict
import heapq
import math
import time
import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    enabled: bool = False
    run_legacy_diagnostics: bool = False
    independent_goals: bool = False
    resolution_m: float = .4
    clearance_m: float = 1.3
    obstacle_ttl_s: float = 8.
    replan_interval_s: float = .8
    waypoint_distance_m: float = 2.
    max_range_m: float = 40.
    max_expansions: int = 12000
    adaptive_slots: bool = False
    maintain_heading: bool = True
    trail_fallback: bool = True
    max_slot_shift_m: float = 4.
    leader_wait_error_m: float = 5.
    leader_resume_error_m: float = 2.
    feedback_max_age_s: float = 1.5

    def __post_init__(self):
        if any(type(getattr(self,k)) is not bool for k in ('enabled','adaptive_slots','maintain_heading','trail_fallback','run_legacy_diagnostics','independent_goals')):
            raise ValueError('navigation enable flags must be boolean')
        for k in ('resolution_m','clearance_m','obstacle_ttl_s','replan_interval_s',
                  'waypoint_distance_m','max_range_m','max_slot_shift_m',
                  'leader_wait_error_m','leader_resume_error_m','feedback_max_age_s'):
            v=getattr(self,k)
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{k} must be finite and positive')
        if (not .2<=self.resolution_m<=1. or self.max_range_m>60.
                or self.clearance_m>5. or self.max_slot_shift_m>8. or self.waypoint_distance_m>5.
                or self.obstacle_ttl_s>60. or self.replan_interval_s<.2 or self.feedback_max_age_s>2.
                or (self.adaptive_slots and not self.enabled)
                or (self.independent_goals and (not self.enabled or self.adaptive_slots))
                or self.leader_resume_error_m>=self.leader_wait_error_m
                or type(self.max_expansions) is not int or not 100<=self.max_expansions<=50000):
            raise ValueError('invalid navigation bounds')


def validate_navigation_feedback(value):
    if value is None:return None
    keys={'observation_id','obstacle_points_xy_m','goal_ned_m','path_available',
          'goal_blocked','tracking_error_m','route_version'}
    if not isinstance(value,dict) or set(value)-{'obstacle_resolution_m'}!=keys:
        raise ValueError('invalid navigation feedback fields')
    resolution=value.get('obstacle_resolution_m',.6)
    if type(resolution) not in (int,float) or not math.isfinite(resolution) or not .2<=resolution<=1.:
        raise ValueError('invalid obstacle resolution')
    for k in ('observation_id','route_version'):
        if type(value[k]) is not int or value[k]<0:raise ValueError('invalid navigation identity')
    for k in ('path_available','goal_blocked'):
        if type(value[k]) is not bool:raise ValueError('invalid navigation flags')
    error=value['tracking_error_m']
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1000:
        raise ValueError('invalid tracking error')
    points=value['obstacle_points_xy_m']
    if not isinstance(points,(list,tuple)) or len(points)>512:raise ValueError('too many navigation obstacles')
    for vector,size in [(value['goal_ned_m'],3),*((p,2) for p in points)]:
        if (not isinstance(vector,(list,tuple)) or len(vector)!=size
                or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>10000 for v in vector)):
            raise ValueError('invalid World NED navigation vector')
    return dict(value)


def segment_clear(a,b,points,clearance):
    if len(points)==0:return True
    a,b=np.asarray(a[:2],float),np.asarray(b[:2],float)
    p=np.asarray(points,float).reshape(-1,2)
    d=b-a;den=float(d@d)
    t=np.clip((p-a)@d/den,0.,1.) if den>1e-12 else np.zeros(len(p))
    return bool(np.all(np.linalg.norm(p-a-t[:,None]*d,axis=1)>=clearance))


class CommittedGridRouter:
    """Persist observed static surfaces, commit a detour and recheck new scans."""
    def __init__(self,cfg:NavigationConfig,*,clock=time.monotonic,minimum_clearance_m=.95,hardening=None):
        from coordination.navigation_hardening import HardeningConfig, ObservedStaticMemory
        self.cfg,self.clock=cfg,clock
        self.hardening=hardening or HardeningConfig()
        self.memory=ObservedStaticMemory(cfg.resolution_m,self.hardening)
        self.travel_history=[];self.failed_cells={};self.last_failure_reason=''
        self.search_timed_out=False;self._search_deadline=math.inf
        self.minimum_clearance_m=minimum_clearance_m
        self.cells={};self.path=[];self.goal=None
        self.last_plan=-math.inf;self.route_version=0
        self.path_available=False;self.goal_blocked=False
        self.escaping=False
        self.capture_next=False;self.captured_diagnostic=None
        self.last_route_reason='not_planned';self.search_diagnostics=[];self.escape_diagnostic={}

    def _escape_path(self,position,goal,points):
        """Exit a preferred-margin pocket without moving toward its surfaces.

        The raw-point final gateway remains authoritative. Cell half-diagonal
        accounts for quantization; the minimum is set above its hard clearance.
        """
        self.escape_diagnostic=dict(rejected={},accepted=0)
        def reject(reason):
            counts=self.escape_diagnostic['rejected'];counts[reason]=counts.get(reason,0)+1
        if not len(points):reject('no_points');return []
        current=float(np.linalg.norm(points-np.asarray(position[:2]),axis=1).min())
        minimum=self.minimum_clearance_m+self.cfg.resolution_m/math.sqrt(2.)
        self.escape_diagnostic.update(nearest_m=current,minimum_m=minimum)
        if current<minimum:reject('inside_minimum_buffer');return []
        if current>self.cfg.clearance_m+self.cfg.resolution_m*2:reject('outside_escape_band');return []
        candidates=[]
        for radius in (.8,1.2,1.6):
            for i in range(24):
                angle=i*math.pi/12
                end=(position[0]+radius*math.cos(angle),position[1]+radius*math.sin(angle))
                clearance=float(np.linalg.norm(points-np.asarray(end),axis=1).min())
                if clearance<max(current+.15,self.cfg.clearance_m+self.cfg.resolution_m/math.sqrt(2.)):
                    reject('endpoint_clearance');continue
                # All nearby surfaces must be nonapproaching on this straight
                # escape, not just clear at the two endpoints.
                near=points[np.linalg.norm(points-np.asarray(position[:2]),axis=1)<self.cfg.clearance_m+.8]
                if len(near) and np.any((near-np.asarray(position[:2]))@(np.asarray(end)-position[:2])>.01):
                    reject('approaching_surface');continue
                if segment_clear(position,end,points,minimum):
                    candidates.append((math.dist(end,goal[:2])+.2*radius,end))
                else:reject('segment_clearance')
        self.escape_diagnostic['accepted']=len(candidates)
        if not candidates:return []
        return [tuple(position[:2]),min(candidates,key=lambda x:x[0])[1]]

    def observe(self,points_xy,*,origin=None,scan_id=None,dynamic_regions=(),clearing_points=None):
        now=self.clock();h=self.cfg.resolution_m
        if self.hardening.enabled and self.hardening.memory:
            self.memory.observe(points_xy,now,origin=origin,scan_id=scan_id,dynamic_regions=dynamic_regions,
                                clearing_points=clearing_points)
            return
        points=np.asarray(points_xy,float).reshape(-1,2)
        if not np.all(np.isfinite(points)):raise ValueError('nonfinite observed obstacle')
        for x,y in np.unique(np.rint(points/h).astype(int),axis=0):self.cells[(int(x),int(y))]=now
        self.cells={k:t for k,t in self.cells.items() if 0<=now-t<self.cfg.obstacle_ttl_s}
        if len(self.cells)>8000:
            self.cells=dict(sorted(self.cells.items(),key=lambda kv:kv[1],reverse=True)[:8000])

    def obstacles(self,position=None):
        now=self.clock();h=self.cfg.resolution_m
        if self.hardening.enabled and self.hardening.memory:
            p=self.memory.points(now)
            if position is not None and len(p):p=p[np.linalg.norm(p-np.asarray(position[:2]),axis=1)<=self.cfg.max_range_m]
            return p
        self.cells={k:t for k,t in self.cells.items() if 0<=now-t<self.cfg.obstacle_ttl_s}
        p=np.asarray([(x*h,y*h) for x,y in self.cells],float).reshape(-1,2)
        if position is not None and len(p):p=p[np.linalg.norm(p-np.asarray(position[:2]),axis=1)<=self.cfg.max_range_m]
        return p

    def note_stall(self,position,reason):
        if not self.hardening.enabled or not self.hardening.recovery:return
        now=self.clock();h=self.cfg.resolution_m
        self.failed_cells={k:t for k,t in self.failed_cells.items() if now<t}
        for point in self.path[2:]:
            if math.dist(position[:2],point)<.5:continue
            self.failed_cells[tuple(round(v/h) for v in point)]=now+self.hardening.failure_ttl_s
        if len(self.failed_cells)>2048:
            self.failed_cells=dict(sorted(self.failed_cells.items(),key=lambda item:item[1],reverse=True)[:2048])
        self.last_failure_reason=reason
        self.path=[];self.last_plan=-math.inf

    def verified_retreat(self,position,*,temporary_obstacles=()):
        """Return a short previously traveled segment, rechecked against NOW."""
        points=self.obstacles(position)
        temporary=np.asarray(temporary_obstacles,float).reshape(-1,2)
        if len(temporary):points=np.vstack((points,temporary))
        margin=max(self.minimum_clearance_m+self.cfg.resolution_m/math.sqrt(2.),self.cfg.clearance_m)
        previous=tuple(position[:2]);distance=0.
        for point in reversed(self.travel_history):
            step=math.dist(previous,point)
            if step<.05:continue
            distance+=step
            if distance>2. or not segment_clear(previous,point,points,margin):break
            if distance>=.4 and segment_clear(position,point,points,margin):
                return point[0],point[1],position[2]
            previous=point
        return None

    def _search(self,start,goal,points,penalty=None):
        h=self.cfg.resolution_m
        s=tuple(round(x/h) for x in start[:2]);g=tuple(round(x/h) for x in goal[:2])
        report=dict(start_cell=s,goal_cell=g,penalized=bool(penalty),expanded=0)
        self.search_diagnostics=self.search_diagnostics[-1:]
        self.search_diagnostics.append(report)
        def finish(reason,path=None):
            report['reason']=reason
            return [] if path is None else path
        margin=round(5./h)
        lo=tuple(min(a,b)-margin for a,b in zip(s,g));hi=tuple(max(a,b)+margin for a,b in zip(s,g))
        radius=self.cfg.clearance_m+h/math.sqrt(2.)
        n=math.ceil(radius/h)
        stencil=[(i,j) for i in range(-n,n+1) for j in range(-n,n+1) if math.hypot(i,j)*h<=radius]
        blocked=set()
        for index,(x,y) in enumerate(points):
            if self.hardening.enabled and self.hardening.recovery and index%128==0 and self.clock()>self._search_deadline:
                self.search_timed_out=True;return finish('inflation_budget')
            ix,iy=round(x/h),round(y/h)
            if not lo[0]-n<=ix<=hi[0]+n or not lo[1]-n<=iy<=hi[1]+n:continue
            blocked.update((ix+i,iy+j) for i,j in stencil)
        report.update(start_blocked=s in blocked,goal_blocked=g in blocked,blocked_cells=len(blocked))
        if s in blocked:return finish('start_blocked')
        if g in blocked:return finish('goal_cell_blocked')
        opened=[(math.dist(s,g),0.,s)];cost={s:0.};parent={};expanded=0
        deadline=min(self.clock()+.06,self._search_deadline)
        while opened and expanded<self.cfg.max_expansions:
            _,old,u=heapq.heappop(opened)
            if old!=cost.get(u):continue
            if u==g:
                path=[u]
                while u in parent:u=parent[u];path.append(u)
                path=[(x*h,y*h) for x,y in reversed(path)]
                path[0]=tuple(start[:2]);path[-1]=tuple(goal[:2])
                return finish('found',path)
            expanded+=1
            report['expanded']=expanded
            if expanded%100==0 and self.clock()>deadline:
                self.search_timed_out=True;return finish('search_budget')
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                v=(u[0]+dx,u[1]+dy)
                if (not lo[0]<=v[0]<=hi[0] or not lo[1]<=v[1]<=hi[1] or v in blocked
                        or (dx and dy and ((u[0]+dx,u[1]) in blocked or (u[0],u[1]+dy) in blocked))):continue
                c=old+math.hypot(dx,dy)+(1.5 if penalty and v in penalty else 0.)
                if self.hardening.enabled and self.hardening.recovery and self.failed_cells and self.failed_cells.get(v,-math.inf)>self.clock():c+=3.
                if c<cost.get(v,math.inf):
                    cost[v]=c;parent[v]=u;heapq.heappush(opened,(c+math.dist(v,g),c,v))
        return finish('expansion_limit' if opened else 'no_grid_path')

    def waypoint(self,position,goal,*,temporary_obstacles=()):
        if not self.capture_next:
            return self._waypoint_impl(position,goal,temporary_obstacles=temporary_obstacles)
        self.capture_next=False
        points=self.obstacles(position)
        now=self.clock()
        # Own arrays/copies: the background writer must never see later updates.
        snapshot=dict(schema_version=1,recorded_monotonic_s=now,navigation_config=asdict(self.cfg),
            hardening_config=asdict(self.hardening),minimum_clearance_m=self.minimum_clearance_m,
            position=list(position),goal=list(goal),static_obstacles=points.copy(),
            temporary_obstacles=np.asarray(temporary_obstacles,float).reshape(-1,2).copy(),
            path_before=list(self.path),goal_before=self.goal,
            last_plan_age_s=now-self.last_plan if math.isfinite(self.last_plan) else None,
            last_plan_monotonic_s=self.last_plan if math.isfinite(self.last_plan) else None,
            escaping_before=self.escaping,route_version_before=self.route_version,
            failed_cells=[(x,y,t) for (x,y),t in self.failed_cells.items() if t>now],
            travel_history=list(self.travel_history),
            static_memory=[(x,y,now-t,self.memory.free_votes.get((x,y),0)) for (x,y),t in self.memory.cells.items()],
            dynamic_memory=[(x,y,now-t) for (x,y),t in self.memory.dynamic.items()],
            legacy_memory=[(x,y,now-t) for (x,y),t in self.cells.items()],last_scan_id=self.memory.last_scan,
            replay_scope='Frozen observed geometry and router state; not wall-clock scheduling or flight control')
        result=self._waypoint_impl(position,goal,temporary_obstacles=temporary_obstacles,_points=points)
        snapshot['decision_monotonic_s']=self._decision_monotonic
        snapshot['outcome']=dict(waypoint=result,path_available=self.path_available,goal_blocked=self.goal_blocked,
            route_version=self.route_version,escaping=self.escaping,reason=self.last_route_reason,
            search_timed_out=self.search_timed_out,searches=[dict(row) for row in self.search_diagnostics],
            escape=dict(self.escape_diagnostic))
        self.captured_diagnostic=snapshot
        return result

    def _waypoint_impl(self,position,goal,*,temporary_obstacles=(),_points=None):
        if any(not math.isfinite(v) for v in (*position,*goal)):raise ValueError('invalid route endpoints')
        points=self.obstacles(position) if _points is None else _points
        now=self.clock();self.search_diagnostics=[];self.escape_diagnostic={}
        self._decision_monotonic=now
        self.last_route_reason='cached_route'
        if self.hardening.enabled:
            self.search_timed_out=False
            self._search_deadline=now+.08 if self.hardening.recovery else math.inf
            self.failed_cells={k:t for k,t in self.failed_cells.items() if now<t}
            if ((self.hardening.recovery or self.hardening.passage)
                    and (not self.travel_history or math.dist(position[:2],self.travel_history[-1])>.15)):
                self.travel_history.append(tuple(position[:2]));self.travel_history=self.travel_history[-128:]
        if (math.dist(position,goal)<.15 and
                segment_clear(position,goal,points,self.minimum_clearance_m+self.cfg.resolution_m/math.sqrt(2.))):
            self.path=[];self.path_available=True;self.goal_blocked=False;self.escaping=False
            self.last_route_reason='at_goal'
            return tuple(goal)
        transient=np.asarray(temporary_obstacles,float).reshape(-1,2)
        if not np.all(np.isfinite(transient)):raise ValueError('invalid temporary obstacles')
        if len(transient):points=np.vstack((points,transient))
        self.goal_blocked=not segment_clear(goal,goal,points,self.cfg.clearance_m)
        if math.dist(position[:2],goal[:2])>self.cfg.max_range_m or self.goal_blocked:
            self.last_route_reason='goal_blocked' if self.goal_blocked else 'goal_out_of_range'
            self.path_available=False;return tuple(position)
        if segment_clear(position,goal,points,self.cfg.clearance_m):
            self.path=[];self.goal=tuple(goal);self.path_available=True
            self.escaping=False
            self.last_route_reason='direct'
            return tuple(goal)
        if self.escaping and self.path and math.dist(position[:2],self.path[-1])<.25:
            self.path=[];self.escaping=False;self.last_plan=-math.inf
        # Keep the selected homotopy instead of changing sides every frame.
        if self.path:
            nearest=min(range(len(self.path)),key=lambda i:math.dist(position[:2],self.path[i]))
            self.path=self.path[nearest:]
        route_margin=(self.minimum_clearance_m+self.cfg.resolution_m/math.sqrt(2.)) if self.escaping else self.cfg.clearance_m
        usable=(self.path and self.goal is not None and math.dist(self.goal[:2],goal[:2])<1.
            and all(segment_clear(a,b,points,route_margin)
                    for a,b in zip([position[:2],*self.path[:7]],self.path[:8])))
        if not usable and now-self.last_plan>=self.cfg.replan_interval_s:
            self.last_plan=now;self.goal=tuple(goal)
            path=self._search(position,goal,points)
            if path:
                # An alternative avoids the first route's middle cells. Select
                # by length; temporal commitment above prevents side chatter.
                h=self.cfg.resolution_m
                penalty={tuple(round(v/h) for v in p) for p in path[len(path)//4:3*len(path)//4]}
                alternative=self._search(position,goal,points,penalty)
                candidates=[p for p in (path,alternative) if p and all(segment_clear(a,b,points,self.cfg.clearance_m) for a,b in zip(p,p[1:]))]
                def route_cost(p):
                    length=sum(math.dist(a,b) for a,b in zip(p,p[1:]))
                    return length+(3*h*sum(tuple(round(v/h) for v in q) in self.failed_cells for q in p)
                                   if self.hardening.enabled else 0.)
                self.path=min(candidates,key=route_cost) if candidates else []
                self.escaping=False
            else:self.path=[]
            if not self.path:
                self.path=self._escape_path(position,goal,points)
                self.escaping=bool(self.path)
            self.route_version+=1
            usable=bool(self.path)
            self.last_route_reason=('escape' if self.escaping else 'grid_route') if usable else (
                self.search_diagnostics[-1]['reason'] if self.search_diagnostics else 'no_route')
        self.path_available=bool(usable)
        if not usable:
            if self.last_route_reason=='cached_route':self.last_route_reason='waiting_to_replan'
            return tuple(position)
        chosen=self.path[0]
        for point in self.path[1:]:
            if math.dist(position[:2],point)>self.cfg.waypoint_distance_m:break
            if not segment_clear(position,point,points,route_margin):break
            chosen=point
        return (chosen[0],chosen[1],goal[2])

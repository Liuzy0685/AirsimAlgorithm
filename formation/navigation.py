"""Obstacle-aware goal deformation around B's existing three nominal shapes.

Consumes A feedback only. This module has no simulator or actuator access.
"""
import math
import numpy as np
from coordination.navigation import validate_navigation_feedback
from coordination.navigation import segment_clear


class FormationNavigation:
    def __init__(self,config,*,task_id,vehicle_names,safety_distance_m,spacing_m=3.,
                 max_speed_mps=.35,prediction_horizon_s=3.):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (safety_distance_m,spacing_m,max_speed_mps,prediction_horizon_s)):
            raise ValueError('formation navigation limits must be finite and positive')
        self.cfg=config;self.task_id=task_id;self.names=frozenset(vehicle_names)
        # Leave room for A's temporary stopped-peer routing buffer and grid
        # quantization, instead of handing A goals inside that planning buffer.
        self.separation=safety_distance_m+.7
        self.feedback={};self.previous={};self.obstacles={}
        # A stopped leader at 3 m and a 0.35 m/s follower are already close to
        # the 3 s CPA envelope (2+0.35*3=3.05 m). Leave additional trail room
        # for stop-and-go and corners, while retaining nominal shape spacing.
        self.spacing=max(spacing_m,self.separation+.3,
            safety_distance_m+2*max_speed_mps*prediction_horizon_s+.5)
        self.history=[];self.trailing=False;self.clear_since=None;self.join_holds={}
        self.mode='nominal'

    def reset(self):
        self.feedback.clear();self.previous.clear();self.obstacles.clear()
        self.history.clear();self.trailing=False;self.clear_since=None;self.join_holds.clear();self.mode='nominal'

    def _trail_goals(self,snapshot,nominal,leader,points,clearance,now):
        position=tuple(snapshot.states[leader].position_ned_m)
        if not self.history or math.dist(position[:2],self.history[-1][:2])>=.08:
            self.history.append(position)
            self.history=self.history[-2500:]
        end_clear=all(segment_clear(p,p,points,clearance) for p in nominal.values())
        direct_clear=end_clear and all(segment_clear(snapshot.states[n].position_ned_m,p,points,clearance)
            for n,p in nominal.items())
        blocked=any((nav['goal_blocked'] or (not nav['path_available'] and nav['tracking_error_m']>1.))
            for nav,_ in self.feedback.values())
        at_goal=(self.feedback[leader][0]['tracking_error_m']<.35 and
            sum(v*v for v in snapshot.states[leader].linear_velocity_ned_mps)<.1**2)
        if (not direct_clear or blocked) and not at_goal:
            if not self.trailing:
                self.join_holds={name:tuple(snapshot.states[name].position_ned_m) for name in nominal}
            self.trailing=True;self.clear_since=None
        elif self.trailing:
            if direct_clear or (at_goal and end_clear):
                if self.clear_since is None:self.clear_since=now
                if at_goal or now-self.clear_since>=2.:
                    self.trailing=False;self.join_holds.clear()
            else:self.clear_since=None
        self.mode='trail' if self.trailing else 'nominal'
        if not self.trailing:return nominal
        # Follow arc length along actual leader history, not a straight offset
        # behind its current yaw. Unavailable history stages later joins at
        # their own verified starting positions instead of stacking goals.
        goals={}
        for rank,name in enumerate(nominal,1):
            distance=rank*self.spacing;accumulated=0.;target=None
            for a,b in zip(reversed(self.history[1:]),reversed(self.history[:-1])):
                length=math.dist(a[:2],b[:2])
                if accumulated+length>=distance:
                    t=(distance-accumulated)/length if length>1e-9 else 0.
                    target=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),position[2]);break
                accumulated+=length
            goals[name]=target or self.join_holds[name]
        return goals

    def ingest(self,payload,*,received_at):
        name=payload.get('vehicle_name')
        if name not in self.names or payload.get('task_id')!=self.task_id:return False
        nav=validate_navigation_feedback(payload.get('navigation'))
        if nav is None:return False
        previous=self.feedback.get(name)
        # Repeated status frames must never refresh the age of an old scan.
        if previous is not None and nav['observation_id']<=previous[0]['observation_id']:return False
        self.feedback[name]=(nav,received_at)
        h=nav.get('obstacle_resolution_m',.6)
        for p in nav['obstacle_points_xy_m']:self.obstacles[(*p,h)]=received_at
        self.obstacles={p:at for p,at in self.obstacles.items() if 0<=received_at-at<self.cfg.obstacle_ttl_s}
        if len(self.obstacles)>12000:self.obstacles=dict(sorted(self.obstacles.items(),key=lambda kv:kv[1],reverse=True)[:12000])
        return True

    def targets(self,snapshot,nominal,*,leader,now):
        if not self.cfg.adaptive_slots:return nominal
        if (set(self.feedback)!=self.names or
                any(not 0<=now-at<self.cfg.feedback_max_age_s for _,at in self.feedback.values())):
            return None
        sim_time=max(s.timestamp for s in snapshot.states.values())
        if any(abs(nav['observation_id']-sim_time)>int(self.cfg.feedback_max_age_s*1e9)
               for nav,_ in self.feedback.values()):return None
        self.obstacles={p:at for p,at in self.obstacles.items() if 0<=now-at<self.cfg.obstacle_ttl_s}
        points=np.asarray([(p[0],p[1]) for p in self.obstacles],float).reshape(-1,2)
        # Include the maximum reported cell displacement, not an exact-surface
        # displacement instead of presenting quantized surfaces as exact ones.
        resolution=max((p[2] for p in self.obstacles),default=max(nav.get('obstacle_resolution_m',.6) for nav,_ in self.feedback.values()))
        clearance=self.cfg.clearance_m+resolution/math.sqrt(2.)
        if self.cfg.trail_fallback:
            nominal=self._trail_goals(snapshot,nominal,leader,points,clearance,now)
        leader_xy=tuple(snapshot.states[leader].position_ned_m[:2])
        def free(xy):return not len(points) or bool(np.all(np.linalg.norm(points-np.asarray(xy),axis=1)>=clearance))
        values=list(nominal.values())
        if (all(free(p[:2]) and math.dist(p[:2],leader_xy)>=self.separation for p in values)
                and all(math.dist(a,b)>=self.separation for i,a in enumerate(values) for b in values[i+1:])):
            self.previous=dict(nominal)
            return dict(nominal)
        candidates={}
        for name,target in nominal.items():
            if free(target[:2]):
                # Nominal is preferred, while alternatives permit a joint
                # assignment when another slot must move around an obstacle.
                locations=[tuple(target)]
            else:locations=[]
            for radius in np.arange(.5,self.cfg.max_slot_shift_m+.01,.5):
                radius=float(radius)
                for i in range(16):
                    a=i*math.pi/8
                    p=(target[0]+radius*math.cos(a),target[1]+radius*math.sin(a),target[2])
                    if free(p[:2]) and math.dist(p[:2],leader_xy)>=self.separation:locations.append(p)
            own=snapshot.states[name].position_ned_m
            previous=self.previous.get(name,target)
            ranked=sorted(locations,key=lambda p:math.dist(p,target)**2+.08*math.dist(p,own)+.15*math.dist(p,previous))
            candidates[name]=ranked[:16]
            if not candidates[name]:return None
        # A small bounded beam prevents two independently projected slots from
        # collapsing into one safe cell. All four follower assignments are joint.
        beam=[(0.,{})]
        for name,target in nominal.items():
            options=[]
            for score,chosen in beam:
                for p in candidates[name]:
                    if (math.dist(p[:2],leader_xy)<self.separation or
                            any(math.dist(p,q)<self.separation for q in chosen.values())):continue
                    own=snapshot.states[name].position_ned_m
                    previous=self.previous.get(name,target)
                    cost=math.dist(p,target)**2+.08*math.dist(p,own)+.15*math.dist(p,previous)
                    options.append((score+cost,dict(chosen,**{name:p})))
            if not options:return None
            beam=sorted(options,key=lambda x:x[0])[:32]
        self.previous=beam[0][1] if beam else {}
        return dict(self.previous)

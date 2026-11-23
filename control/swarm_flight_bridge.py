"""Connect the existing LiDAR/trajectory loop to A's state channel and gateway."""
import math
import time
import numpy as np
import logging
import json
from coordination.navigation import NavigationConfig,CommittedGridRouter
from coordination.navigation_hardening import (HardeningConfig, RuntimeDiagnostics,
    ProgressGuard, MotionLimiter, PassageAdvisor, latency_speed_cap, stop_category)

from control.command_authority import CommandAuthority
from control.safety_filter import SafetyFilter
from control.vehicle_command_gateway import VehicleCommandGateway
from control.velocity_controller import VelocityController, VelocityCommandRejected
from coordination.controller_link import ControllerLink
from control.swept_clearance import measured_command_clearance


class FormationPermitExpired(VelocityCommandRejected):
    """The B target expired; A may propose a separately verified bounded hold."""


class SwarmFlightBridge:
    def __init__(self, session, config, cfg, params):
        if not session._owns_lock:
            raise RuntimeError("swarm gateway requires the per-vehicle OS control lock")
        self.session, self.cfg, self.params = session, cfg, params
        self.name = session.vehicle_name
        self.link = ControllerLink(cfg, self.name, [m["vehicle_name"] for m in config["drones"]])
        self.authority = CommandAuthority(lease_ttl_s=cfg.lease_ttl_s)
        self.controller_id = self.link.stream_id
        self.safety = SafetyFilter(max_horizontal_speed_mps=params.forward_speed_mps,
            max_vertical_speed_mps=params.max_vertical_speed_mps,
            safety_distance_m=config["coordination"]["safety_distance_m"],
            prediction_horizon_s=config["coordination"]["prediction_horizon_s"])
        dispatcher = VelocityController(session.adapter,
            max_horizontal_speed_mps=params.forward_speed_mps,
            max_vertical_speed_mps=params.max_vertical_speed_mps,
            command_duration_seconds=params.command_duration_s)
        self.gateway = VehicleCommandGateway(self.authority, dispatcher=dispatcher, safety_filter=self.safety)
        self.proxy = self.gateway.bind_raw_client(session.client, vehicle_name=self.name, controller_id=self.controller_id)
        self.mode = None
        self.points = None
        self.points_world = None
        self.points_at = 0.0
        self.points_received_at = 0.0
        self._gateway_scan_id = None
        self.raw_lidar_supplier = None
        self.adopted_command = None
        self._goal_command_id = None
        self._conflict_active = False
        self._unverified_emergency = False
        self.last_rejection_reason = None
        self._clear_frames = 0
        self._clear_since = None
        self._waiting_for_command = False
        self.leader_goal_hold = False
        self._status_clock = time.monotonic
        self.navigation_cfg=NavigationConfig(**config.get('navigation',{}))
        self._independent_members = config['drones']
        if self.navigation_cfg.independent_goals:
            for member in self._independent_members:
                if (member.get('goal_actor') or member.get('start_actor')
                        or any(type(member.get(k)) not in (int, float) or not math.isfinite(member[k])
                               for k in ('goal_x','goal_y','target_z'))):
                    raise ValueError('independent goals require finite settings-relative XYZ targets')
        self.hardening=HardeningConfig(**config.get('local_navigation_hardening',{}))
        if self.hardening.enabled and not self.navigation_cfg.enabled:
            raise ValueError('local navigation hardening requires navigation.enabled')
        self.router=CommittedGridRouter(self.navigation_cfg,hardening=self.hardening)
        self.diagnostics=RuntimeDiagnostics()
        self.progress_guard=ProgressGuard(self.hardening)
        self.motion_limiter=MotionLimiter(self.hardening.acceleration_mps2)
        self.passage=PassageAdvisor(self.hardening,self.name,
            {m['vehicle_name']:m.get('yield_priority',0.) for m in config['drones']})
        self._last_dispatch_source='';self._accepted_speed=0.;self._last_guard_velocity=(0.,0.,0.)
        self._last_gateway_at=None;self._diagnostic_at=-math.inf
        self._navigation_motion=None
        self._last_guard_start=None
        self._route_transients=[]
        self._navigation_waypoint=None
        self._navigation_goal=None
        self._navigation_command_available=False
        self._leader_waiting=False
        self._last_lidar_id=-1
        self._feedback_points=[]
        self._last_route_log=None
        self._navigation_source=None
        self._gateway_refine_enabled = config['coordination'].get('algorithm') == 'orca'
        self._passing_bias = config['coordination'].get('passing_bias_mps', .1)
        self._gateway_refined = False
        self._settle_since = self._settle_last = self._settle_identity = None
        self._settle_frames = 0

    @property
    def follower(self):
        return not self.navigation_cfg.independent_goals and self.name != self.cfg.leader

    def mission_settled(self):
        settled = self._mission_geometry_settled()
        snap = self.link.snapshot()
        now = self._status_clock()
        if not settled or snap is None:
            self._settle_since = self._settle_last = self._settle_identity = None
            self._settle_frames = 0
            return False
        identity = (self.link._state_stream, snap.sequence_id)
        if (self._settle_last is not None and now-self._settle_last >= self.cfg.snapshot_max_age_s
                or self._settle_identity is not None and
                (identity[0] != self._settle_identity[0] or identity[1] < self._settle_identity[1])):
            self._settle_since = None
            self._settle_frames = 0
        if identity != self._settle_identity:
            if self._settle_since is None:self._settle_since = now
            self._settle_frames += 1
            self._settle_identity, self._settle_last = identity, now
        return (self._settle_since is not None and self._settle_frames >= self.cfg.formation_settle_frames
                and now-self._settle_since >= self.cfg.formation_settle_duration_s)

    def _mission_geometry_settled(self):
        if not self.navigation_cfg.independent_goals:
            return self.link.followers_settled()
        snap = self.link.snapshot()
        if snap is None or self.mode is None:
            return False
        for member in self._independent_members:
            name = member['vehicle_name']
            offset = self.mode._swarm_spawn_offsets[name]
            target = tuple(member[k] + offset[i] for i, k in enumerate(('goal_x','goal_y','target_z')))
            state = snap.states[name]
            if (math.dist(state.position_ned_m, target) > self.cfg.formation_tolerance_m
                    or math.sqrt(sum(v*v for v in state.linear_velocity_ned_mps)) > .2):
                return False
        return True

    def start(self):
        self.link.start()
        self.authority.acquire(self.name, self.controller_id, ttl_s=self.cfg.lifecycle_lease_s)
        self.session._client = self.session.adapter._client = self.proxy
        self.proxy.lifecycle_guard = self.guard_lifecycle
        self.session.swarm_bridge = self

    def guard_lifecycle(self, api, args, kwargs):
        if not self.session._owns_lock:
            self.fail("vehicle_lock_lost")
        if self.link.snapshot() is None:
            self.fail("authoritative_snapshot_unavailable")

    def begin_cruise(self, mode):
        self.mode = mode
        if self.navigation_cfg.enabled and self.navigation_cfg.clearance_m<mode._traj_params.hard_clearance_m:
            raise ValueError('navigation clearance cannot be below the final hard clearance')
        self.router.minimum_clearance_m=mode._traj_params.hard_clearance_m
        self.proxy.motion_guard = self.guard_motion

    def update_lidar(self, frame, points_sensor=None, *, navigation=True):
        """Use the measured sensor pose, including mounting offset/rotation."""
        processing_started=time.perf_counter()
        if type(getattr(frame,'raw_timestamp_ns',None)) is not int or frame.raw_timestamp_ns<0:
            self.fail('invalid_lidar_timestamp_at_gateway')
        from transforms.lidar_to_local_ned import sensor_to_local_ned
        pose=frame.sensor_pose
        p,q=pose['position'],pose['orientation']
        quaternion=[q[k] for k in ('x','y','z','w')]
        if (not all(math.isfinite(v) for v in quaternion)
                or abs(sum(v*v for v in quaternion)-1.)>.01):
            self.fail('invalid_lidar_pose_at_gateway')
        # The navigation self-exclusion box can also remove nearby obstacles.
        # Final safety must retain those raw returns rather than mistake a
        # disappearing filtered obstacle for newly free space.
        if points_sensor is None:points_sensor=frame.point_cloud_sensor
        local=sensor_to_local_ned(points_sensor,[p[k] for k in ('x','y','z')],quaternion)
        world=local+np.asarray(self.mode._swarm_spawn_offsets[self.name])
        # A later processed snapshot must not replace a newer raw safety scan.
        # Re-reading the same simulation scan never renews its wall-clock age.
        if self._gateway_scan_id is None or frame.raw_timestamp_ns>self._gateway_scan_id:
            received=frame.received_monotonic_seconds
            started=getattr(frame,'request_started_monotonic_seconds',None)
            if started is None:started=received
            if (type(started) not in (int,float) or not math.isfinite(started)
                    or type(received) not in (int,float) or not math.isfinite(received)
                    or not 0<=started<=received<=time.monotonic()):
                self.fail('invalid_lidar_timing_at_gateway')
            self.points_world=world
            self.points=points_sensor
            self.points_received_at=received
            self.points_at=started
            self._gateway_scan_id=frame.raw_timestamp_ns
        if navigation and self.navigation_cfg.enabled and frame.raw_timestamp_ns>self._last_lidar_id:
            self._last_lidar_id=frame.raw_timestamp_ns
            snap=self.link.snapshot()
            if snap is not None:
                own=np.asarray(snap.states[self.name].position_ned_m)
                p=world
                p=p[(np.abs(p[:,2]-own[2])<.65)&(np.linalg.norm(p[:,:2]-own[:2],axis=1)<self.navigation_cfg.max_range_m)]
                observed=p.copy();dynamic_regions=[]
                for name,state in snap.states.items():
                    if name!=self.name:
                        skew=abs(frame.raw_timestamp_ns-state.timestamp)/1e9
                        radius=.8+min(.7,math.sqrt(sum(v*v for v in state.linear_velocity_ned_mps))*skew)
                        if skew<=self.cfg.snapshot_max_age_s:
                            dynamic_regions.append((*state.position_ned_m[:2],radius))
                        p=p[np.linalg.norm(p-np.asarray(state.position_ned_m),axis=1)>.8]
                if self.hardening.enabled and self.hardening.memory:
                    origin=np.array([pose['position'][k] for k in ('x','y','z')])+np.asarray(self.mode._swarm_spawn_offsets[self.name])
                    planar=observed[np.abs(observed[:,2]-origin[2])<.10]
                    self.router.observe(observed[:,:2],origin=origin[:2],scan_id=frame.raw_timestamp_ns,
                        dynamic_regions=dynamic_regions,clearing_points=planar[:,:2])
                else:self.router.observe(p[:,:2])
                h=self.navigation_cfg.resolution_m
                coarse=np.unique(np.rint(p[:,:2]/h)*h,axis=0)
                if len(coarse)>512:
                    coarse=coarse[np.argsort(np.linalg.norm(coarse-own[:2],axis=1))[:512]]
                self._feedback_points=coarse.tolist()
        if self.hardening.enabled:
            self.diagnostics.record(**{'navigation_memory_update_ms' if navigation else 'lidar_transform_ms':
                (time.perf_counter()-processing_started)*1000})

    def _refresh_gateway_scan(self):
        if self.raw_lidar_supplier is None:return
        frame=self.raw_lidar_supplier()
        if frame is None:self.fail('lidar_unavailable_at_gateway')
        if not frame.frame_valid:self.fail('invalid_lidar_frame_at_gateway')
        if type(frame.raw_timestamp_ns) is not int or frame.raw_timestamp_ns<0:
            self.fail('invalid_lidar_timestamp_at_gateway')
        self.update_lidar(frame,navigation=False)

    def _log_gateway_age(self, reason, snap):
        now=time.monotonic()
        logging.getLogger('swarm_gateway').error(
            'gateway_freshness vehicle=%s reason=%s scan_id=%s request_start_age_s=%.6f '
            'receive_age_s=%.6f snapshot_age_s=%.6f limit_s=%.3f source=%s',
            self.name,reason,self._gateway_scan_id,now-self.points_at,
            now-self.points_received_at,now-snap.timestamp,self.cfg.snapshot_max_age_s,
            'latest_raw' if self.raw_lidar_supplier is not None else 'control_snapshot')

    def route_goal(self,own_state,requested_goal):
        if not self.navigation_cfg.enabled:return requested_goal
        route_started=time.perf_counter()
        offset=self.mode._swarm_spawn_offsets[self.name]
        position=tuple(p+o for p,o in zip(own_state.position_ned_m,offset))
        goal=tuple(p+o for p,o in zip(requested_goal,offset))
        self._navigation_goal=goal
        self._navigation_command_available=not self.follower or self.link.command() is not None
        transient=[]
        planning_goal=goal
        staging_peer=None
        stand_off=self.safety.safety_distance_m+.2+2*self.navigation_cfg.resolution_m
        snap=self.link.snapshot()
        if snap is not None:
            radius=max(0.,self.safety.safety_distance_m+.2-self.navigation_cfg.clearance_m)
            for name,state in snap.states.items():
                if name!=self.name and sum(v*v for v in state.linear_velocity_ned_mps)<.12**2:
                    x,y=state.position_ned_m[:2]
                    transient.append((x,y))
                    transient.extend((x+radius*math.cos(i*math.pi/16),y+radius*math.sin(i*math.pi/16)) for i in range(32))
                    # In a goal swap, both peers initially occupy the other's
                    # endpoint. A blocked endpoint must not suppress every
                    # approach command. Route only to an outside staging point;
                    # preserve the real mission goal and every final safety gate.
                    if math.hypot(goal[0]-x,goal[1]-y)<self.safety.safety_distance_m+.2:
                        distance=math.hypot(position[0]-goal[0],position[1]-goal[1])
                        fraction=min(1.,stand_off/max(distance,1e-12))
                        planning_goal=(goal[0]+(position[0]-goal[0])*fraction,
                                       goal[1]+(position[1]-goal[1])*fraction,goal[2])
                        staging_peer=name
        from coordination.navigation import segment_clear
        points=self.router.obstacles(position)
        if transient:points=np.vstack((points,np.asarray(transient,float)))
        margin=self.navigation_cfg.clearance_m+self.navigation_cfg.resolution_m/math.sqrt(2.)
        cached=getattr(self,'_staging_goal',None)
        previous_goal=getattr(self,'_staging_mission_goal',None)
        if (cached is not None and previous_goal is not None and math.dist(previous_goal,goal)<.5
                and math.dist(position,cached)>.4 and segment_clear(cached,cached,points,margin)):
            # Commit to the selected passing side until reached. Releasing it
            # as soon as the peer moves makes five vehicles merge back into
            # the same center line before they have passed one another.
            planning_goal=cached
            staging_peer=getattr(self,'_staging_peer',staging_peer)
        if staging_peer is not None:
            if (cached is not None and staging_peer==getattr(self,'_staging_peer',None)
                    and previous_goal is not None and math.dist(previous_goal,goal)<.5
                    and segment_clear(cached,cached,points,margin)):
                planning_goal=cached
            else:
                angle=math.atan2(position[1]-goal[1],position[0]-goal[0])
                # The direct stand-off can itself be occupied by a third
                # parked vehicle in a five-way swap. Choose a checked lateral
                # endpoint rather than relaxing any obstacle/separation margin.
                for turn in (0.,math.pi/2,-math.pi/2,math.pi/4,-math.pi/4,math.pi):
                    candidate=(goal[0]+stand_off*math.cos(angle+turn),
                               goal[1]+stand_off*math.sin(angle+turn),goal[2])
                    if segment_clear(candidate,candidate,points,margin):
                        planning_goal=candidate
                        break
            self._staging_goal=planning_goal
            self._staging_mission_goal=goal
        else:self._staging_goal=None
        if staging_peer!=getattr(self,'_staging_peer',None):
            logging.getLogger('swarm_navigation').info('peer_endpoint_staging vehicle=%s peer=%s mission_goal=%s planning_goal=%s',
                self.name,staging_peer,goal,planning_goal)
        self._staging_peer=staging_peer
        waypoint=self.router.waypoint(position,planning_goal,temporary_obstacles=transient) if self.points_world is not None else position
        self._route_transients=transient
        if self.hardening.enabled and self.hardening.recovery:
            expected=(self._navigation_command_available and not self.leader_goal_hold
                      and not self._leader_waiting and math.dist(position,goal)>self.cfg.formation_tolerance_m)
            if self.progress_guard.update(time.monotonic(),position,goal,expected=expected,
                    reason=self._last_dispatch_source,commanded_speed=self._accepted_speed):
                self.router.note_stall(position,self.progress_guard.reason)
                self.router.path_available=False
                waypoint=position
        if self.hardening.enabled:
            self.diagnostics.record(route_ms=(time.perf_counter()-route_started)*1000)
        self._navigation_waypoint=waypoint
        event=(self.router.route_version,self.router.path_available,self.router.goal_blocked,self.router.escaping)
        if event!=self._last_route_log:
            self._last_route_log=event
            logging.getLogger('swarm_navigation').info('route vehicle=%s version=%d available=%s blocked_goal=%s escape=%s goal=%s waypoint=%s',
                self.name,*event,goal,waypoint)
        self.link.navigation_feedback=dict(observation_id=max(0,self._last_lidar_id),
            obstacle_points_xy_m=self._feedback_points,obstacle_resolution_m=self.navigation_cfg.resolution_m,goal_ned_m=list(goal),
            path_available=bool(self.router.path_available and self._navigation_command_available),
            goal_blocked=self.router.goal_blocked,tracking_error_m=math.dist(position,goal),
            route_version=self.router.route_version)
        return tuple(p-o for p,o in zip(waypoint,offset))

    def _navigation_velocity(self,own_state,vz,source,command=None):
        self._navigation_motion=None
        # A complete, checked grid route owns local recovery in this opt-in
        # mode. Never launch a blind forward probe when the route is blocked.
        if source=='vertical_escape':return None
        if not self.router.path_available or self._navigation_waypoint is None:
            return 0.,0.,vz,'navigation_blocked'
        if self.follower and command is None:return 0.,0.,vz,'formation_wait'
        if not self.follower and not self.navigation_cfg.independent_goals:
            peers=self.link.peer_navigation(self.navigation_cfg.feedback_max_age_s)
            follower_names=self.link.names-{self.name}
            all_fresh=follower_names<=set(peers)
            worst=max((peers[n]['tracking_error_m'] for n in follower_names if n in peers),default=0.)
            blocked=not all_fresh or any(not peers[n]['path_available'] for n in follower_names if n in peers)
            if blocked or worst>self.navigation_cfg.leader_wait_error_m:self._leader_waiting=True
            elif worst<self.navigation_cfg.leader_resume_error_m:self._leader_waiting=False
            if self.leader_goal_hold or self._leader_waiting:return 0.,0.,vz,'leader_waiting_for_formation'
        own=tuple(p+o for p,o in zip(own_state.position_ned_m,self.mode._swarm_spawn_offsets[self.name]))
        waypoint=self._navigation_waypoint
        points=None
        if self.hardening.enabled:
            points=self.router.obstacles(own)
            if self.hardening.passage:
                direction=np.asarray(waypoint[:2])-own[:2]
                length=float(np.linalg.norm(direction))
                narrow=False
                if length>1e-6 and len(points):
                    direction/=length;relative=points-np.asarray(own[:2])
                    ahead=relative@direction
                    side=relative[:,0]*direction[1]-relative[:,1]*direction[0]
                    near=(ahead>=0)&(ahead<=3.)&(np.abs(side)<=self.safety.safety_distance_m+1.)
                    narrow=bool(np.any(near&(side>.5)) and np.any(near&(side<-.5)))
                snapshot=self.link.snapshot()
                intents=self.link.peer_navigation(self.navigation_cfg.feedback_max_age_s)
                peers={n:(s.position_ned_m,s.linear_velocity_ned_mps,intents.get(n,{}).get('goal_ned_m')) for n,s in snapshot.states.items()
                       if n!=self.name} if snapshot else {}
                wait,retreat=self.passage.update(time.monotonic(),own,waypoint,peers,narrow=narrow,
                    safety_distance=self.safety.safety_distance_m)
                if wait:
                    back=self.router.verified_retreat(own,temporary_obstacles=self._route_transients) if retreat else None
                    if back is None:return 0.,0.,vz,'passage_timeout_no_verified_retreat' if retreat else 'passage_wait'
                    waypoint=back
        self._navigation_waypoint=waypoint
        dx,dy=(self._navigation_waypoint[i]-own[i] for i in (0,1))
        speed=min(self.params.forward_speed_mps,.7*math.hypot(dx,dy))
        if command is not None and command.mode=='velocity':speed=min(speed,math.hypot(*command.desired_velocity_ned_mps[:2]))
        length=math.hypot(dx,dy)
        wx,wy=(speed*dx/length,speed*dy/length) if length>1e-8 else (0.,0.)
        source='passage_retreat' if self.hardening.enabled and self.passage.state=='BLOCKED_TIMEOUT' else 'navigation_route'
        cap=speed
        clearance=None
        if self.hardening.enabled and self.hardening.latency_limit_enabled:
            age=max(0.,time.monotonic()-self.points_at)
            clearance=float(np.linalg.norm(points-np.asarray(own[:2]),axis=1).min()) if len(points) else math.inf
            # An open map may have no returns in the horizontal planning slab
            # while a valid raw cloud still measures the ground. Empty 2-D
            # memory is not itself a zero-clearance obstacle or a free-space
            # certificate. Use the current raw 3-D distance as another cap;
            # missing raw data stays a stop and is rejected by the final gate.
            if self.points_world is not None and len(self.points_world):
                clearance=min(clearance,float(np.linalg.norm(self.points_world-np.asarray(own),axis=1).min()))
            else:clearance=0.
            cap=latency_speed_cap(clearance,self.router.minimum_clearance_m,age,
                self.hardening.braking_mps2,speed)
            if speed>cap and speed>0:wx,wy=wx*cap/speed,wy*cap/speed
        if self.hardening.enabled and self.hardening.smoothing:
            wx,wy=self.motion_limiter.limit((wx,wy),own_state.linear_velocity_ned_mps[:2],time.monotonic())
        if self.hardening.enabled and (self.hardening.smoothing or self.hardening.latency_limit_enabled):
            from coordination.navigation import segment_clear
            end=(own[0]+wx*.5,own[1]+wy*.5)
            if not segment_clear(own,end,points,self.router.minimum_clearance_m):
                wx=wy=0.;source='navigation_smooth_blocked'
        if self.hardening.enabled:
            self._navigation_motion=dict(requested_mps=speed,latency_cap_mps=cap,
                output_mps=math.hypot(wx,wy),clearance_m=clearance if clearance is not None and math.isfinite(clearance) else None,
                latency_limit_enabled=self.hardening.latency_limit_enabled,smoothing_enabled=self.hardening.smoothing)
        c,s=math.cos(own_state.yaw_rad),math.sin(own_state.yaw_rad)
        hold_vz=max(-self.params.max_vertical_speed_mps,min(self.params.max_vertical_speed_mps,
            self.params.target_z_ned-own_state.position_ned_m[2]))
        self.adopted_command=command
        return wx*c+wy*s,-wx*s+wy*c,hold_vz,source

    def tick(self):
        self.adopted_command = None
        if not self.authority.renew(self.name, self.controller_id):
            self.fail("controller_lease_expired")
        snap = self.link.snapshot()
        if snap is None:
            self.fail("authoritative_snapshot_unavailable")
        return snap

    def goal(self, own_state):
        self._goal_command_id = None
        if not self.follower:
            return None
        command = self.link.command()
        if command is not None and command.mode == "position":
            offset = self.mode._swarm_spawn_offsets[self.name]
            goal = tuple(p-o for p, o in zip(command.target_position_ned_m, offset))
            if abs(goal[2] - self.params.target_z_ned) > 0.3:
                self.link.report("EMERGENCY", "formation_altitude_outside_cruise_band", command)
                return tuple(own_state.position_ned_m)
            self._goal_command_id = command.lifecycle.command_id
            return goal
        if command is not None and command.mode == "velocity":
            # Recovery velocity is guidance for A's existing static planner,
            # not a post-planner override through a pillar.
            self._goal_command_id = command.lifecycle.command_id
            if command.recovery_target_ned_m is not None:
                offset=self.mode._swarm_spawn_offsets[self.name]
                goal=tuple(p-o for p,o in zip(command.recovery_target_ned_m,offset))
                if abs(goal[2]-self.params.target_z_ned)>.3:
                    self.link.report("EMERGENCY","formation_altitude_outside_cruise_band",command)
                    return tuple(own_state.position_ned_m)
                return goal
            velocity=command.desired_velocity_ned_mps
            return (own_state.position_ned_m[0]+velocity[0]*self.safety.prediction_horizon_s,
                    own_state.position_ned_m[1]+velocity[1]*self.safety.prediction_horizon_s,
                    self.params.target_z_ned)
        return tuple(own_state.position_ned_m)

    def select(self, vx, vy, vz, source, own_state):
        self.adopted_command = None
        if self.navigation_cfg.enabled:
            command=self.link.command() if self.follower else None
            self._waiting_for_command=self.follower and command is None
            if command is not None and command.lifecycle.command_id!=self._goal_command_id:
                return 0.,0.,vz,'formation_goal_updated_during_planning'
            selected=self._navigation_velocity(own_state,vz,source,command)
            if selected is not None:
                self._navigation_source=selected[3]
                return selected
        if not self.follower:
            if self.leader_goal_hold:
                return 0.,0.,vz,"leader_waiting_for_formation"
            return vx, vy, vz, source
        command = self.link.command()
        self._waiting_for_command = command is None
        # Existing local safety/recovery/vertical escape always owns priority.
        if source not in {"trajectory", "guided_apf", "apf", "reactive"}:
            return vx, vy, vz, source
        if command is None:
            self.link.report("FORMATION_RECOVERING", "waiting_for_current_formation_command")
            return 0.0, 0.0, vz, "formation_wait"
        if command.mode == "position":
            if command.lifecycle.command_id != self._goal_command_id:
                return 0.0, 0.0, vz, "formation_goal_updated_during_planning"
            local_z = command.target_position_ned_m[2] - self.mode._swarm_spawn_offsets[self.name][2]
            if abs(local_z - self.params.target_z_ned) > 0.3:
                return 0.0, 0.0, vz, "formation_altitude_rejected"
            self.adopted_command = command
            # The moving slot has already passed through the trajectory/APF planner.
            return vx, vy, vz, "formation_position"
        requested=math.hypot(*command.desired_velocity_ned_mps[:2])
        selected=math.hypot(vx,vy)
        limit=min(requested,self.params.forward_speed_mps)
        if selected>limit and selected>1e-9:
            vx,vy=vx*limit/selected,vy*limit/selected
        self.adopted_command = command
        return vx,vy,vz,"formation_velocity"

    def expects_navigation_progress(self, own_state):
        if self.hardening.enabled and stop_category(self._last_dispatch_source) in ('traffic_wait','formation_wait','command_unavailable'):
            return False
        if self.navigation_cfg.enabled and not self.follower and self._leader_waiting:return False
        if not self.follower: return not self.leader_goal_hold
        command=self.link.command()
        if command is None: return False
        if command.mode=="velocity":
            return math.sqrt(sum(v*v for v in command.desired_velocity_ned_mps))>.05
        offset=self.mode._swarm_spawn_offsets[self.name]
        world=tuple(p+o for p,o in zip(own_state.position_ned_m,offset))
        return math.dist(world,command.target_position_ned_m)>self.cfg.formation_tolerance_m

    def report_dispatch(self, source, *, avoiding=False):
        self._accepted_speed=math.hypot(*self._last_guard_velocity[:2])
        if self.hardening.enabled and self.hardening.smoothing:
            self.motion_limiter.accept(self._last_guard_velocity[:2])
        if self.hardening.enabled and self._last_gateway_at is not None:
            self.diagnostics.record(source,dispatch_return_ms=(time.monotonic()-self._last_gateway_at)*1000)
        if self._gateway_refined:
            avoiding=True
            source='gateway_peer_velocity_refinement'
        if self.navigation_cfg.enabled and source=='inter_uav_orca' and not avoiding:
            source=self._navigation_source or source
        self._last_dispatch_source=source
        if source=='gateway_peer_velocity_refinement' and self._accepted_speed<.05:
            self._last_dispatch_source='inter_uav_yield'
        conflict = avoiding or source in {"inter_uav_yield", "inter_uav_orca_fallback_hold", "formation_rejected"}
        if conflict:
            self._conflict_active = True
            self._clear_frames = 0
            self._clear_since = None
            self.link.report("AVOIDING", source, self.adopted_command)
        elif self._conflict_active:
            from coordination.collision_predictor import predict_pairwise_conflict
            snap=self.link.snapshot()
            own=snap.states[self.name] if snap else None
            clear=(snap is not None and all(not predict_pairwise_conflict(
                own.position_ned_m,own.linear_velocity_ned_mps,other.position_ned_m,other.linear_velocity_ned_mps,
                safety_distance_m=self.safety.safety_distance_m,horizon_s=self.safety.prediction_horizon_s).conflict
                for name,other in snap.states.items() if name!=self.name))
            now=self._status_clock()
            if clear:
                if self._clear_since is None: self._clear_since=now
                self._clear_frames+=1
            else:
                self._clear_frames,self._clear_since=0,None
            if (self._clear_since is not None and self._clear_frames >= self.cfg.conflict_clear_frames
                    and now-self._clear_since >= self.cfg.conflict_clear_duration_s):
                self._conflict_active=False
                self.link.report("CONFLICT_CLEARED",source,self.adopted_command)
            else:
                self.link.report("AVOIDING","verifying_conflict_clearance",self.adopted_command)
        else:
            command=self.adopted_command
            snap=self.link.snapshot()
            settled=(command is not None and command.mode=="position" and snap is not None
                and not command.formation_id.endswith(':trail')
                and math.dist(snap.states[self.name].position_ned_m,command.target_position_ned_m)<=self.cfg.formation_tolerance_m
                and sum(v*v for v in snap.states[self.name].linear_velocity_ned_mps)<=.2**2)
            recovering=self.follower and not settled
            self.link.report("FORMATION_RECOVERING" if recovering else "NORMAL",source,command)

    def fail(self, reason):
        if self.hardening.enabled:self.diagnostics.record(reason)
        self._unverified_emergency = True
        self.last_rejection_reason = reason
        self.link.report("EMERGENCY", reason, self.adopted_command)
        raise VelocityCommandRejected(reason)

    def guard_motion(self, api, args, kwargs):
        if not self.hardening.enabled:return self._guard_motion_impl(api,args,kwargs)
        started=time.perf_counter()
        now=time.monotonic()
        if self._last_guard_start is not None:
            self.diagnostics.record(dispatch_interval_ms=(now-self._last_guard_start)*1000)
        self._last_guard_start=now
        try:
            result=self._guard_motion_impl(api,args,kwargs)
            self._last_gateway_at=time.monotonic()
            snap=self.link.snapshot()
            worker=getattr(self.mode,'_perception_worker',None)
            self.diagnostics.record(gateway_ms=(time.perf_counter()-started)*1000,
                sensor_age_ms=(self._last_gateway_at-self.points_at)*1000,
                state_age_ms=(self._last_gateway_at-snap.timestamp)*1000 if snap else None,
                lidar_rpc_ms=getattr(worker,'lidar_rpc_ms',None),
                perception_ms=getattr(worker,'processing_ms',None))
            self._emit_runtime_diagnostics()
            # Instrumentation is real work too. It must not extend the exact
            # snapshot/scan deadlines checked by the original safety decision.
            if self.link.snapshot() is None:self.fail('authoritative_snapshot_unavailable')
            remaining=min(self._gate_state_deadline-time.monotonic(),
                          self.points_at+self.cfg.snapshot_max_age_s-time.monotonic())
            if self.adopted_command is not None:
                permit=self.link.permitted_command(self.adopted_command.lifecycle.command_id)
                if permit is None:raise FormationPermitExpired('formation_expired_before_dispatch')
                remaining=min(remaining,permit[1])
            values,keywords=result;values=list(values);keywords=dict(keywords)
            duration=values[3] if len(values)>3 else keywords['duration']
            duration=min(duration,remaining)
            if duration<=0 or not self.session._owns_lock:self.fail('dispatch_deadline_elapsed')
            if len(values)>3:values[3]=duration
            else:keywords['duration']=duration
            self._last_gateway_at=time.monotonic()
            return tuple(values),keywords
        except VelocityCommandRejected as exc:
            self.diagnostics.record(str(exc),gateway_ms=(time.perf_counter()-started)*1000)
            self._emit_runtime_diagnostics()
            raise

    def _emit_runtime_diagnostics(self):
        now=time.monotonic()
        if self.hardening.diagnostics and now-self._diagnostic_at>=1.:
            self._diagnostic_at=now
            payload=dict(vehicle=self.name,**self.diagnostics.summary(),passage_state=self.passage.state,
                passage_peer=self.passage.peer,progress_reason=self.progress_guard.reason,
                search_timed_out=self.router.search_timed_out,navigation_motion=self._navigation_motion)
            logging.getLogger('navigation_runtime').info('navigation_runtime %s',json.dumps(payload,allow_nan=False))

    def _guard_motion_impl(self, api, args, kwargs):
        self._gateway_refined = False
        if api != "moveByVelocityAsync":
            self.fail(f"unverified_cruise_api:{api}")
        if not self.session._owns_lock:
            self.fail("vehicle_lock_lost")
        snap = self.link.snapshot()
        if snap is None:
            self.fail("authoritative_snapshot_unavailable")
        values = list(args)
        def value(i, key):
            return values[i] if len(values) > i else kwargs[key]
        velocity = tuple(float(value(i, key)) for i, key in enumerate(("vx", "vy", "vz")))
        if (not all(math.isfinite(v) for v in velocity)
                or math.hypot(*velocity[:2])>self.params.forward_speed_mps+1e-8
                or abs(velocity[2])>self.params.max_vertical_speed_mps+1e-8):
            self.fail('velocity_exceeds_gateway_limits')
        decision = self.safety.check_velocity(velocity, own_state=snap.states[self.name],
            neighbors=[s for n, s in snap.states.items() if n != self.name])
        if (not decision.allowed and self._gateway_refine_enabled
                and decision.reason.startswith('inter_vehicle_conflict')):
            from coordination.orca_2d import compute_gateway_velocity
            own=snap.states[self.name]
            refined=compute_gateway_velocity(position=own.position_ned_m[:2],
                velocity=own.linear_velocity_ned_mps[:2],preferred_velocity=velocity[:2],
                neighbors=[(s.position_ned_m[:2],s.linear_velocity_ned_mps[:2])
                           for n,s in sorted(snap.states.items()) if n!=self.name],
                safety_distance_m=self.safety.safety_distance_m+2*self.params.forward_speed_mps*self.params.command_duration_s+.05,
                minimum_distance_m=self.safety.safety_distance_m+.01,
                horizon_s=self.safety.prediction_horizon_s,time_step_s=self.params.command_duration_s,
                max_speed_mps=self.params.forward_speed_mps,passing_bias_mps=min(self._passing_bias,self.params.forward_speed_mps))
            if refined is not None:
                proposed=(*refined,velocity[2])
                checked=self.safety.check_velocity(proposed,own_state=own,
                    neighbors=[s for n,s in snap.states.items() if n!=self.name])
                if checked.allowed:
                    logging.getLogger('swarm_gateway').info('gateway_velocity_refined vehicle=%s snapshot=%s original=%s refined=%s',
                        self.name,snap.sequence_id,velocity,proposed)
                    velocity,decision=proposed,checked
                    self._gateway_refined=True
                    kwargs=dict(kwargs)
                    for i,key in enumerate(('vx','vy','vz')):
                        if len(values)>i:values[i]=velocity[i]
                        else:kwargs[key]=velocity[i]
        if not decision.allowed:
            self.fail(decision.reason)
        if any(abs(a-b) > 1e-8 for a, b in zip(velocity, decision.desired_velocity_ned_mps)):
            self.fail("velocity_exceeds_gateway_limits")
        remaining = self.cfg.snapshot_max_age_s - (time.monotonic() - snap.timestamp)
        if self.adopted_command is not None:
            permitted = self.link.permitted_command(self.adopted_command.lifecycle.command_id)
            if permitted is None:
                self.link.report("FORMATION_RECOVERING", "formation_expired_before_dispatch")
                raise FormationPermitExpired("formation_expired_before_dispatch")
            current,permit_remaining = permitted
            remaining = min(remaining,permit_remaining)
        duration = min(float(value(3, "duration")), self.params.command_duration_s, remaining)
        if not math.isfinite(duration) or duration <= 0:
            self.fail("dispatch_deadline_elapsed")
        # Recheck the final ORCA velocity against the actual 3D LiDAR cloud.
        self._refresh_gateway_scan()
        if (self._gateway_scan_id is not None and abs(self._gateway_scan_id-
                max(s.timestamp for s in snap.states.values()))>int(self.cfg.snapshot_max_age_s*1e9)):
            self._log_gateway_age('lidar_sim_time_skew_at_gateway',snap)
            self.fail('lidar_sim_time_skew_at_gateway')
        if self.points is None or time.monotonic() - self.points_at > self.cfg.snapshot_max_age_s:
            self._log_gateway_age('lidar_unavailable_at_gateway',snap)
            self.fail("lidar_unavailable_at_gateway")
        duration=min(duration,self.cfg.snapshot_max_age_s-(time.monotonic()-self.points_at))
        if duration<=0:self.fail('lidar_unavailable_at_gateway')
        yaw = snap.states[self.name].yaw_rad
        c, s = math.cos(yaw), math.sin(yaw)
        body_v = np.array((velocity[0]*c+velocity[1]*s, -velocity[0]*s+velocity[1]*c, velocity[2]))
        points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        if self.points_world is not None:
            relative=self.points_world-np.asarray(snap.states[self.name].position_ned_m)
            points=relative@np.array(((c,-s,0.),(s,c,0.),(0.,0.,1.)))
        if not len(points) or not np.all(np.isfinite(points)):
            self.fail("invalid_lidar_cloud_at_gateway")
        measured_world=snap.states[self.name].linear_velocity_ned_mps
        measured_body=(measured_world[0]*c+measured_world[1]*s,
            -measured_world[0]*s+measured_world[1]*c,measured_world[2])
        age=max(time.monotonic()-snap.timestamp,time.monotonic()-self.points_at)
        # Three seconds is the inter-vehicle CPA horizon, not a request to fly
        # the current tangent straight for three seconds through a curved plan.
        clearance=measured_command_clearance(points,measured_body,body_v,duration,age)
        if clearance < self.mode._traj_params.hard_clearance_m:
            self.fail("final_velocity_lidar_clearance")
        # Cloud transforms/clearance can be slow. Their completion must not
        # extend a state, LiDAR or B permit deadline checked at entry.
        if self.link.snapshot() is None:
            self.fail('authoritative_snapshot_unavailable')
        remaining = min(self.cfg.snapshot_max_age_s - (time.monotonic()-snap.timestamp),
                        self.cfg.snapshot_max_age_s - (time.monotonic()-self.points_at))
        if self.adopted_command is not None:
            permitted = self.link.permitted_command(self.adopted_command.lifecycle.command_id)
            if permitted is None:
                raise FormationPermitExpired('formation_expired_before_dispatch')
            remaining = min(remaining, permitted[1])
        duration = min(duration, remaining)
        if duration <= 0 or not self.session._owns_lock:
            self.fail('dispatch_deadline_elapsed')
        self._gate_state_deadline=snap.timestamp+self.cfg.snapshot_max_age_s
        if len(values) > 3:
            values[3] = duration
        else:
            kwargs = dict(kwargs, duration=duration)
        self._last_guard_velocity=velocity
        return tuple(values), kwargs

    def cache_cleanup_lidar(self, frame):
        """Use the worker's last real scan after planner shutdown, without renewing it."""
        snap=self.link.snapshot()
        if self._unverified_emergency or snap is None or frame is None or not frame.frame_valid:
            return False
        if (not 0<=time.monotonic()-frame.received_monotonic_seconds<self.cfg.snapshot_max_age_s
                or abs(frame.raw_timestamp_ns-max(s.timestamp for s in snap.states.values()))>
                   int(self.cfg.snapshot_max_age_s*1e9)):
            return False
        self.update_lidar(frame)
        return True

    def prepare_cleanup(self):
        if (self._unverified_emergency or self.link.snapshot() is None
                or (self.mode is not None and (self.points is None
                    or time.monotonic() - self.points_at > self.cfg.snapshot_max_age_s))):
            from flight_modes.shared_flight_session import SessionPhase
            self.session.state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            self.link.report("EMERGENCY", "no_verified_cleanup_action_manual_intervention_required")
        # Explicit bounded lifecycle lease while this process still owns the OS lock.
        if self.session._owns_lock:
            self.authority.acquire(self.name, self.controller_id, ttl_s=self.cfg.lifecycle_lease_s)
        self.proxy.motion_guard = None

    def close(self):
        self.authority.release(self.name, self.controller_id)
        self.link.close()

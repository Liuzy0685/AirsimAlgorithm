"""Connect the existing LiDAR/trajectory loop to A's state channel and gateway."""
import math
import time
import numpy as np

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

    @property
    def follower(self):
        return self.name != self.cfg.leader

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
        self.proxy.motion_guard = self.guard_motion

    def update_lidar(self, frame, points_sensor=None):
        """Use the measured sensor pose, including mounting offset/rotation."""
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
        self.points_world=local+np.asarray(self.mode._swarm_spawn_offsets[self.name])
        self.points=points_sensor
        self.points_at=frame.received_monotonic_seconds

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
        if not self.follower: return not self.leader_goal_hold
        command=self.link.command()
        if command is None: return False
        if command.mode=="velocity":
            return math.sqrt(sum(v*v for v in command.desired_velocity_ned_mps))>.05
        offset=self.mode._swarm_spawn_offsets[self.name]
        world=tuple(p+o for p,o in zip(own_state.position_ned_m,offset))
        return math.dist(world,command.target_position_ned_m)>self.cfg.formation_tolerance_m

    def report_dispatch(self, source, *, avoiding=False):
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
                and math.dist(snap.states[self.name].position_ned_m,command.target_position_ned_m)<=self.cfg.formation_tolerance_m
                and sum(v*v for v in snap.states[self.name].linear_velocity_ned_mps)<=.2**2)
            recovering=self.follower and not settled
            self.link.report("FORMATION_RECOVERING" if recovering else "NORMAL",source,command)

    def fail(self, reason):
        self._unverified_emergency = True
        self.last_rejection_reason = reason
        self.link.report("EMERGENCY", reason, self.adopted_command)
        raise VelocityCommandRejected(reason)

    def guard_motion(self, api, args, kwargs):
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
        decision = self.safety.check_velocity(velocity, own_state=snap.states[self.name],
            neighbors=[s for n, s in snap.states.items() if n != self.name])
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
        if self.points is None or time.monotonic() - self.points_at > self.cfg.snapshot_max_age_s:
            self.fail("lidar_unavailable_at_gateway")
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
        if len(values) > 3:
            values[3] = duration
        else:
            kwargs = dict(kwargs, duration=duration)
        return tuple(values), kwargs

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

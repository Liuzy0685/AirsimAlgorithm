"""Explicit orphan-flight landing. Never take off, arm, teleport or seize a live lock.

This is a bounded simulation lifecycle operation, not a new route planner. It
requires exclusive fleet locks and a stationary, separated fleet. New collision
events remain fatal unless confirmed as stationary landed ground contact.
"""
import itertools
import json
import math
import socket
import time
import uuid

from control.command_authority import CommandAuthority
from control.ground_contact import is_stationary_ground_contact
from control.vehicle_command_gateway import VehicleCommandGateway
from coordination.operator_common import require
from models.collision_state import CollisionState
from models.vehicle_state import VehicleState


def _vector(value):
    out = [value.x_val, value.y_val, value.z_val]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in out),
            'invalid_recovery_state', 'Nonfinite recovery state', 409)
    return out


class _NoVelocityDispatcher:
    def send_velocity_world_ned(self, *args, **kwargs):
        raise PermissionError('Recovery is restricted to lifecycle landing')


class RecoveryLanding:
    def __init__(self, profile, client, *, clock=time.monotonic, sleep=time.sleep,
                 lock_acquire=None, lock_release=None, timeout_s=30., ground_reference=None):
        from flight_modes.shared_flight_session import SharedFlightSession
        self.profile, self.raw, self.clock, self.sleep = profile, client, clock, sleep
        self.acquire = lock_acquire or SharedFlightSession.acquire_lock
        self.release = lock_release or SharedFlightSession.release_lock
        self.timeout_s = timeout_s
        self.locks, self.proxies, self.baselines, self.stamps = {}, {}, {}, {}
        self.advanced_at = {}
        self.ground_reference=ground_reference or {}
        self.ground_history={}
        self.observation_trace=[]
        self.confirmed = set()
        self.in_flight_land = set()
        self.authority = CommandAuthority(lease_ttl_s=timeout_s*len(profile.names)+30, clock=clock)
        self.controller_id = 'explicit-recovery-' + uuid.uuid4().hex

    def observe(self, *, initial=False):
        started = self.clock()
        states = {}
        for name in self.profile.names:
            raw = self.raw.getMultirotorState(vehicle_name=name)
            kin = raw.kinematics_estimated
            pos, vel = _vector(kin.position), _vector(kin.linear_velocity)
            require(type(raw.timestamp) is int and raw.timestamp > 0
                    and raw.timestamp >= self.stamps.get(name, -1),
                    'recovery_state_not_advancing', 'Need new simulator state for every observation', 409)
            advanced=raw.timestamp > self.stamps.get(name,-1)
            if advanced:self.advanced_at[name]=self.clock()
            require(0 <= self.clock()-self.advanced_at[name] <= self.profile.service.snapshot_max_age_s,
                    'recovery_state_stale','Simulator state stopped advancing',409)
            self.stamps[name] = raw.timestamp
            require(type(raw.landed_state) is int and raw.landed_state in (0, 1),
                    'invalid_recovery_state', 'Unknown landed state', 409)
            require(math.hypot(vel[0], vel[1]) <= .2 and (not initial or math.dist(vel, [0, 0, 0]) <= .2),
                    'recovery_vehicle_moving', 'Recovery requires a settled fleet, without horizontal flight', 409)
            col = self.raw.simGetCollisionInfo(vehicle_name=name)
            require(type(col.has_collided) is bool and type(col.time_stamp) is int
                    and col.time_stamp >= 0 and (not col.has_collided or col.time_stamp > 0),
                    'invalid_recovery_collision', 'Invalid collision timestamp', 409)
            collision = CollisionState(has_collided=bool(col.has_collided), object_name=col.object_name,
                normal_ned=_vector(col.normal), penetration_depth=col.penetration_depth,
                raw_timestamp=col.time_stamp)
            state = VehicleState(position_ned_m=pos, linear_velocity_ned_mps=vel, landed_state=raw.landed_state)
            ground = is_stationary_ground_contact(collision, state)
            spawn = self.profile.settings['Vehicles'][name]
            common = [pos[i]+spawn.get(k, 0.) for i, k in enumerate(('X','Y','Z'))]
            ref=self.ground_reference.get(name,{})
            ref_position=ref.get('position_ned_m',[])
            previous = self.baselines.setdefault(name, col.time_stamp)
            from control.ground_contact import GROUND_OBJECT_NAMES
            # A new floor hit may precede the SDK's landed-state transition.
            # It is observed, not ignored: require the original A-verified
            # takeoff ground height, normal/depth, then a stable dwell below.
            contact_candidate=(collision.raw_timestamp > previous and collision.object_name in GROUND_OBJECT_NAMES
                and ref.get('landed_state')==0 and len(ref_position)==3
                and all(type(v) in (int,float) and math.isfinite(v) for v in ref_position)
                and abs(common[2]-ref_position[2])<=.15
                and math.hypot(*collision.normal_ned[:2])<=.2 and collision.normal_ned[2]<=-.9
                and type(collision.penetration_depth) in (int,float)
                and math.isfinite(collision.penetration_depth) and 0<=collision.penetration_depth<=.02)
            fallback=False
            if contact_candidate and math.dist(vel,[0,0,0])<=.05:
                key=(collision.object_name,collision.raw_timestamp)
                history=self.ground_history.get(name)
                if history is None or history['key']!=key:
                    history=dict(key=key,since=self.clock(),frames=0);self.ground_history[name]=history
                if advanced:history['frames']+=1
                fallback=self.clock()-history['since']>=2. and history['frames']>=5
            else:self.ground_history.pop(name,None)
            if collision.has_collided or collision.raw_timestamp != previous:
                # Existing wall contact is never a permissible takeover baseline.
                require(collision.object_name in GROUND_OBJECT_NAMES and (
                    ground or contact_candidate or (raw.landed_state == 1 and col.time_stamp == previous
                               and math.hypot(*collision.normal_ned[:2]) <= .2
                               and collision.normal_ned[2] <= -.9
                               and type(collision.penetration_depth) in (int, float)
                               and math.isfinite(collision.penetration_depth)
                               and 0 <= collision.penetration_depth <= .02)),
                    'recovery_collision', 'Wall/unknown/new airborne collision; manual intervention required', 409)
                require(col.time_stamp == previous or ground or contact_candidate, 'recovery_new_collision',
                        'New collision is not a confirmed stationary landing', 409)
            states[name] = dict(raw=raw, position=common, velocity=vel, ground=ground, advanced=advanced,
                               reported_landed_state=raw.landed_state,
                               confirmation='sdk_landed_state' if raw.landed_state==0 else 'verified_ground_contact',
                               landed=(raw.landed_state == 0 or fallback) and math.dist(vel, [0,0,0]) <= .05)
            history=self.ground_history.get(name,{})
            self.observation_trace.append(dict(vehicle=name,time=self.clock(),position=common,velocity=vel,
                reported_landed_state=raw.landed_state,collision_object=collision.object_name,
                collision_timestamp=collision.raw_timestamp,collision_pulse=collision.has_collided,
                contact_normal=collision.normal_ned,penetration_depth=collision.penetration_depth,
                contact_candidate=contact_candidate,
                reference_position=ref_position,stable_frames=history.get('frames',0),
                stable_dwell_s=max(0.,self.clock()-history['since']) if history else 0.,confirmed=states[name]['landed']))
            del self.observation_trace[:-250]
        for a, b in itertools.combinations(states, 2):
            if not (states[a]['landed'] and states[b]['landed']):
                require(math.dist(states[a]['position'][:2], states[b]['position'][:2]) >= self.profile.safety_distance,
                        'recovery_separation', 'Vertical landing requires separated horizontal columns', 409)
        require(0 <= self.clock()-started <= self.profile.service.snapshot_max_age_s,
                'recovery_state_stale', 'Recovery observation exceeded the state deadline', 409)
        return states

    def guard(self, name, api, args, kwargs):
        require(name in self.locks and not getattr(self.locks[name], 'closed', False),
                'recovery_lock_lost', 'Exclusive recovery lock lost', 409)
        require(api in {'landAsync','armDisarm','enableApiControl','cancelLastTask'},
                'recovery_api_denied', 'Recovery cannot take off or send movement commands', 403)
        if api == 'cancelLastTask':
            return  # Cancel a landing after a failed check; never disarm/release in air.
        if api == 'armDisarm':
            require(args == (False,) and name in self.confirmed,
                    'recovery_not_landed', 'Disarm only after confirmed landing', 409)
        if api == 'enableApiControl' and args == (False,):
            require(name in self.confirmed, 'recovery_not_landed', 'Do not release an airborne vehicle', 409)
        current=self.observe()
        if api=='armDisarm' or (api=='enableApiControl' and args==(False,)):
            require(current[name]['landed'],'recovery_not_landed','Landing changed before the final write',409)

    def run(self):
        results = {n: dict(landing_confirmed=False, disarmed=False, api_control_released=False,
                          cleanup={'control_released': False}, reason='not_attempted') for n in self.profile.names}
        report = dict(cleanup_verified=False, vehicle_results=results, reason='recovery_not_started',
                      observation_trace=self.observation_trace)
        try:
            for name in sorted(self.profile.names):
                lock = self.acquire('operator-recovery', name)
                require(lock is not None, 'recovery_controller_busy', f'{name} has a live controller; no takeover', 409)
                self.locks[name] = lock
            require(set(self.raw.listVehicles()) == set(self.profile.names)
                    and json.loads(self.raw.getSettingsString())['Vehicles'] == self.profile.settings['Vehicles'],
                    'runtime_settings_mismatch', 'Recovery fleet/settings differ', 409)
            require(not self.raw.simIsPause(), 'simulation_paused', 'Resume before recovery', 409)
            try:
                with socket.create_connection((self.profile.service.host,self.profile.service.port),timeout=.2):
                    require(False,'recovery_service_busy','Old A service is still running; request its cooperative stop',409)
            except (ConnectionRefusedError,TimeoutError,OSError):
                pass
            from coordination.operator_map_runtime import verify_map
            verify_map(self.profile,self.raw)
            # The shared transform validates the configured origin against actual vehicle poses.
            from coordination.coordinate_transform import CoordinateTransformer
            tf=CoordinateTransformer.from_settings_json(str(self.profile.settings_path),self.profile.names)
            require(tf.validate(self.raw).validated,'coordinate_validation_failed','Recovery coordinate transform failed',409)
            self.observe(initial=True)
            gateway = VehicleCommandGateway(self.authority, dispatcher=_NoVelocityDispatcher())
            for name in self.profile.names:
                self.authority.acquire(name,self.controller_id)
                proxy=gateway.bind_raw_client(self.raw,vehicle_name=name,controller_id=self.controller_id)
                proxy.lifecycle_guard=lambda api,args,kwargs,n=name:self.guard(n,api,args,kwargs)
                self.proxies[name]=proxy
            for name in self.profile.names:
                proxy=self.proxies[name]
                states=self.observe()
                proxy.enableApiControl(True,vehicle_name=name)
                if not states[name]['landed']:
                    proxy.landAsync(timeout_sec=self.timeout_s,vehicle_name=name)
                    self.in_flight_land.add(name)
                deadline=self.clock()+self.timeout_s
                consecutive=0
                while self.clock()<deadline:
                    states=self.observe()
                    if not states[name]['landed']:consecutive=0
                    elif states[name]['advanced']:consecutive+=1
                    if consecutive>=3:break
                    self.sleep(.2)
                require(consecutive>=3,'recovery_landing_unconfirmed',f'{name} landing was not confirmed',409)
                self.in_flight_land.discard(name)
                self.confirmed.add(name)
                results[name]['landing_confirmed']=True
                results[name]['landing_confirmation']=states[name]['confirmation']
                results[name]['reported_landed_state']=states[name]['reported_landed_state']
                require(proxy.armDisarm(False,vehicle_name=name) is True,
                        'recovery_disarm_unconfirmed',f'{name} disarm rejected',409)
                results[name]['disarmed']=True
                proxy.enableApiControl(False,vehicle_name=name)
                require(self.raw.isApiControlEnabled(vehicle_name=name) is False,
                        'recovery_release_unconfirmed',f'{name} API release not confirmed',409)
                require(self.observe()[name]['landed'],'recovery_not_landed','Landing state changed after release',409)
                results[name].update(api_control_released=True,cleanup={'control_released':True},reason='recovery_landed')
            report.update(cleanup_verified=True,reason='recovery_landed')
        except Exception as exc:
            report['reason']=getattr(exc,'code',type(exc).__name__)+': '+str(exc)
            for name in self.in_flight_land:
                try:self.proxies[name].cancelLastTask(vehicle_name=name)
                except Exception:pass
            for result in results.values():
                if not result['cleanup']['control_released']:result['reason']=report['reason']
        finally:
            for name,lock in self.locks.items():
                self.authority.release(name,self.controller_id)
                self.release(lock)
            self.locks.clear()
        return report

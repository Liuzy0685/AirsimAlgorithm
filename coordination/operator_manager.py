"""Durable task supervisor. All motion stays in the existing A flight stack."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import secrets
import threading
import time
import uuid

from coordination.operator_common import OperatorError, atomic_json, canonical, identifier, number, read_json, require

TERMINAL = {'COMPLETED', 'STOPPED', 'FAILED', 'INTERRUPTED'}
ACTIVE = {'STARTING', 'TAKING_OFF', 'RUNNING', 'STOPPING', 'LANDING'}


class MissionManager:
    def __init__(self, profile, directory, probe, backend, *, operator_url='', clock=time.monotonic):
        self.profile, self.probe, self.backend = profile, probe, backend
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.url, self.clock = operator_url, clock
        self.lock = threading.RLock()
        self.processes = {}
        self.b_seen = {}
        self.b_status_stamps = {}
        self._convergence = {}
        self.closed = threading.Event()
        path = self.directory / 'state.json'
        self.data = read_json(path) if path.exists() else dict(missions={}, operations={}, receipts={}, sequence=0, events=[])
        # A supervisor restart must never relaunch or replay a flight.
        for mission in self.data['missions'].values():
            if mission['state'] in ACTIVE or mission['state']=='INTERRUPTED':
                self._finish_from_disk(mission, interrupted=True)
        self._save()

    def _save(self):
        atomic_json(self.directory / 'state.json', self.data)

    def _mission(self, mid):
        identifier(mid, 'mission_id')
        require(mid in self.data['missions'], 'mission_not_found', 'Unknown mission', 404)
        return self.data['missions'][mid]

    def public(self, mission):
        result = {k: copy.deepcopy(v) for k, v in mission.items() if not k.startswith('_')}
        path = self.directory / mission['mission_id'] / 'runtime.json'
        try:
            runtime = read_json(path)
            runtime['stale'] = not 0 <= int(time.time()*1000) - runtime['timestamp_unix_ms'] < 2000
            result['runtime'] = runtime
        except (OSError, ValueError, KeyError):
            result['runtime'] = dict(stale=True, vehicles={}, coordination={})
        result['b_control_online'] = self._b_online(mission)
        feedback=result.get('formation_runtime')
        result['formation_runtime_stale'] = (feedback is None or feedback['revision']!=mission.get('applied_revision')
            or not 0 <= int(time.time()*1000)-feedback['received_at_unix_ms'] < 2000)
        from coordination.formation_convergence import FormationConvergence
        convergence = self._convergence.setdefault(mission['mission_id'], FormationConvergence(
            tolerance_m=self.profile.formation.recovery_tolerance_m))
        result['formation_convergence'] = convergence.observe(mission, result['runtime'],
            now=self.clock(), now_ms=int(time.time()*1000), max_age_s=self.profile.service.snapshot_max_age_s)
        return result

    def get(self, mid):
        with self.lock:
            return self.public(self._mission(mid))

    def operation(self, rid):
        with self.lock:
            identifier(rid)
            require(rid in self.data['operations'], 'operation_not_found', 'Unknown operation', 404)
            return copy.deepcopy(self.data['operations'][rid])

    def _event(self, mission):
        self.data['sequence'] += 1
        self.data['events'].append(dict(sequence=self.data['sequence'], timestamp_unix_ms=int(time.time()*1000),
            type='mission.changed', mission=self.public(mission)))
        self.data['events'] = self.data['events'][-256:]
        self._save()

    def events_after(self, after):
        with self.lock:
            events = self.data['events']
            return dict(sequence=self.data['sequence'], reset_required=bool(events and after < events[0]['sequence']-1),
                        events=copy.deepcopy([e for e in events if e['sequence'] > after]))

    def _replay(self, route, body):
        require(isinstance(body, dict), 'invalid_body', 'JSON object required')
        rid = identifier(body.get('request_id'))
        try:
            digest = hashlib.sha256((route + canonical(body)).encode()).hexdigest()
        except (ValueError,TypeError) as exc:
            raise OperatorError('invalid_body','Request must contain finite JSON values') from exc
        previous = self.data['receipts'].get(rid)
        if previous:
            require(previous['digest'] == digest, 'request_id_conflict', 'Request ID reused for different content', 409)
            return rid, digest, copy.deepcopy(previous['response'])
        require(len(self.data['receipts']) < 2048, 'receipt_limit', 'Archive runtime before accepting more operations', 429)
        return rid, digest, None

    def _receipt(self, rid, digest, response):
        self.data['receipts'][rid] = dict(digest=digest, response=copy.deepcopy(response))
        self._save()
        return response

    def validate(self, body):
        config, preview = self.profile.validate_config(body)
        states = self.probe()
        # Formation footprint must also be meaningful at the actual initial heading.
        from dataclasses import replace
        from formation.controller import FormationController
        from models.vehicle_state import VehicleState
        leader = states[config['leader']]
        planner = FormationController(replace(self.profile.formation, **config['formation']), task_id='preview')
        initial_slots = planner.slot_targets(VehicleState(position_ned_m=list(leader['position_ned_m'][:2])+[config['leader_goal_ned_m'][2]],
                                                        yaw_rad=leader['yaw_rad']))
        preview = dict(planner.slot_targets(VehicleState(position_ned_m=config['leader_goal_ned_m'],
                                                         yaw_rad=leader['yaw_rad'])),
                       **{config['leader']:tuple(config['leader_goal_ned_m'])})
        for point in preview.values():
            self.profile.map.check_point(point,self.profile.clearance)
        for name, state in states.items():
            require(all(math.isfinite(v) for v in state['position_ned_m']), 'invalid_state', 'Nonfinite position', 409)
            target = preview[name]
            flight = self.profile.flight_limits[name]
            # The controller centers its geofence at this run's actual takeoff
            # position, which can differ from settings after a previous task.
            actual_start=state['position_ned_m']
            require(math.hypot(target[0]-actual_start[0], target[1]-actual_start[1]) < flight['geofence_radius_m'],
                    'goal_outside_geofence', f'{name} terminal slot exceeds geofence from actual start', 422)
        return dict(valid=True, config=config, actual_start=states, terminal_slots_preview=preview,
                    initial_slots_preview=initial_slots, preview_semantics='nominal_only_not_a_collision_free_route',
                    map_coverage=self.profile.map.value.get('coverage', 'partial'),
                    warnings=['Partial map and endpoint checks do not prove a route; A rechecks real LiDAR and peers during flight.'])

    def create(self, body):
        with self.lock:
            rid, digest, previous = self._replay('create', body)
            if previous:
                return previous
            require(len(self.data['missions']) < 128, 'mission_limit', 'Archive runtime before creating more tasks', 429)
        checked = self.validate(body)
        with self.lock:
            rid, digest, previous = self._replay('create', body)
            if previous:
                return previous
            mid = uuid.uuid4().hex
            mission = dict(mission_id=mid, state='READY', revision=1, config=checked['config'],
                requested_formation=checked['config']['formation'], applied_formation=None, applied_revision=None,
                formation_runtime=None,
                b_status=None, pending_formation=None, vehicle_results={}, reason='',
                created_at_unix_ms=int(time.time()*1000), _b_token=secrets.token_urlsafe(32))
            self.data['missions'][mid] = mission
            self._event(mission)
            return self._receipt(rid, digest, dict(accepted=True, mission=self.public(mission)))

    def _idle_for_start(self):
        require(not any(m['state'] in ACTIVE for m in self.data['missions'].values()),
                'mission_active', 'Another mission owns the flight profile', 409)
        # After supervisor interruption/failed cleanup, operator must resolve and restart
        # with a new runtime directory only after independent control-release verification.
        require(not any(m.get('manual_intervention_required') for m in self.data['missions'].values()),
                'manual_intervention_required', 'Previous flight cleanup is unverified; resolve before restarting', 409)

    def start(self, mid, body):
        with self.lock:
            rid, digest, previous = self._replay('start/'+mid, body)
            if previous:
                return previous
            require(set(body) == {'request_id', 'expected_revision'}, 'invalid_fields', 'Need expected_revision')
            m = self._mission(mid)
            require(type(body['expected_revision']) is int and body['expected_revision'] == m['revision'],
                    'revision_conflict', 'Reload mission before starting', 409)
            require(self.profile.allow_flight, 'flight_disabled', 'Server was not started with --allow-flight', 403)
            require(m['state'] == 'READY', 'invalid_mission_state', 'Only a READY task may start', 409)
            self._idle_for_start()
            m['state'] = 'STARTING'
            self.data['operations'][rid] = dict(request_id=rid, mission_id=mid, kind='start', state='RUNNING')
            response = dict(accepted=True, request_id=rid, mission_id=mid, state='STARTING')
            self._receipt(rid, digest, response)
            self._event(m)
            config = copy.deepcopy(m['config'])
        try:
            self.validate(dict(config, request_id=rid))  # fresh physical checks, never trust an earlier preflight
            with self.lock:
                if m['state'] == 'STOPPING':
                    m['state'], m['reason'] = 'STOPPED', 'cancelled_before_launch'
                    self._close_operations(m)
                    self._event(m)
                    return response
                directory = self.directory / mid
                directory.mkdir(exist_ok=False)
                proc = self.backend.launch(copy.deepcopy(m), directory, self.url)
                self.processes[mid] = proc
                m['process_id'] = proc.pid
                m['applied_formation'] = copy.deepcopy(config['formation'])
                m['applied_revision'] = m['revision']
                self._event(m)
        except Exception as exc:
            with self.lock:
                m['state'], m['reason'] = 'FAILED', str(exc)
                self._close_operations(m)
                self._event(m)
        return response

    def stop(self, mid, body):
        with self.lock:
            rid, digest, previous = self._replay('stop/'+mid, body)
            if previous:
                return previous
            require(set(body) == {'request_id'}, 'invalid_fields', 'Only request_id is accepted')
            m = self._mission(mid)
            require(m['state'] in ACTIVE or m['state'] == 'INTERRUPTED', 'invalid_mission_state', 'Task is not active', 409)
            (self.directory / mid).mkdir(exist_ok=True)
            (self.directory / mid / 'stop.requested').touch()
            if m['state'] != 'INTERRUPTED':
                m['state'] = 'STOPPING'
            m['stop_requested'] = True
            self._fail_pending(m,'mission_stopping')
            self.data['operations'][rid] = dict(request_id=rid, mission_id=mid, kind='stop', state='RUNNING')
            self._event(m)
            return self._receipt(rid, digest, dict(accepted=True, request_id=rid, mission_id=mid,
                                                   state=m['state'], completed=False))

    def _b_online(self, m):
        at = self.b_seen.get(m['mission_id'])
        return at is not None and 0 <= self.clock()-at < 2. and m['state'] in ACTIVE

    def configure(self, mid, body):
        with self.lock:
            rid, digest, previous = self._replay('formation/'+mid, body)
            if previous:
                return previous
            require(set(body) == {'request_id', 'expected_revision', 'formation'}, 'invalid_fields', 'Invalid formation request')
            m = self._mission(mid)
            require(type(body['expected_revision']) is int and body['expected_revision'] == m['revision'],
                    'revision_conflict', 'Reload mission before changing formation', 409)
            require(m['state'] in {'READY', 'RUNNING'}, 'invalid_mission_state', 'Formation cannot change in this phase', 409)
            require(not m['pending_formation'], 'formation_change_pending', 'Wait for previous change result', 409)
            if m['state'] == 'RUNNING':
                require(self._b_online(m), 'b_control_unavailable', 'B has not connected its runtime configuration interface', 503)
            config = dict(m['config'], formation=body['formation'], request_id=rid)
            normalized, preview = self.profile.validate_config(config)
            m['revision'] += 1
            m['config'] = normalized
            m['requested_formation'] = normalized['formation']
            op = dict(request_id=rid, mission_id=mid, kind='formation', state='APPLIED_TO_DRAFT' if m['state']=='READY' else 'PENDING',
                      revision=m['revision'])
            if m['state'] == 'RUNNING':
                m['pending_formation'] = dict(request_id=rid, mission_id=mid, revision=m['revision'],
                    formation=normalized['formation'], expires_at_unix_ms=int(time.time()*1000)+30000)
            self.data['operations'][rid] = op
            self._event(m)
            return self._receipt(rid, digest, dict(accepted=True, operation=op, terminal_slots_preview=preview))

    def b_mission(self, mid, token):
        m = self._mission(mid)
        require(isinstance(token, str) and secrets.compare_digest(token, m['_b_token']),
                'unauthorized', 'B task credential required', 401)
        require(m['state'] in ACTIVE, 'task_not_active', 'B task is no longer active', 409)
        return m

    def b_heartbeat(self, mid, token, body):
        with self.lock:
            m = self.b_mission(mid, token)
            require(isinstance(body, dict) and set(body)=={'protocol_version','capabilities'}
                    and type(body['protocol_version']) is int and body['protocol_version']==1
                    and body['capabilities']==['formation.configure'], 'invalid_b_capabilities', 'Unsupported B control contract')
            self.b_seen[mid] = self.clock()
            return dict(online=True, mission_id=mid, revision=m['revision'])

    def b_request(self, mid, token):
        with self.lock:
            m = self.b_mission(mid, token)
            request = m['pending_formation']
            if request and request['expires_at_unix_ms'] <= int(time.time()*1000):
                self._fail_pending(m, 'configuration_expired')
                request = None
            return dict(request=copy.deepcopy(request))

    def b_result(self, mid, token, body):
        with self.lock:
            m = self.b_mission(mid, token)
            require(isinstance(body, dict) and set(body) == {'request_id','revision','state','formation','reason'},
                    'invalid_b_result', 'Invalid B configuration result')
            request = m['pending_formation']
            require(request is not None and request['expires_at_unix_ms'] > int(time.time()*1000),
                    'configuration_expired', 'No live configuration request', 409)
            require(body['request_id']==request['request_id'] and type(body['revision']) is int
                    and body['revision']==request['revision'] and body['formation']==request['formation'],
                    'configuration_mismatch', 'B result differs from requested configuration', 409)
            require(body['state'] in {'APPLIED','DEFERRED','REJECTED'} and isinstance(body['reason'],str)
                    and len(body['reason'])<=256, 'invalid_b_result', 'Invalid B result state/reason')
            self.data['operations'][request['request_id']].update(state=body['state'], reason=body['reason'])
            m['b_status'] = copy.deepcopy(body)
            if body['state'] == 'APPLIED':
                m['applied_formation'] = copy.deepcopy(request['formation'])
                m['applied_revision'] = request['revision']
            if body['state'] != 'DEFERRED':
                m['pending_formation'] = None
            self._event(m)
            return dict(accepted=True, physical_formation_settled=False)

    def b_status(self, mid, token, body):
        with self.lock:
            m=self.b_mission(mid,token)
            fields={'revision','nominal_shape','effective_mode','slot_targets_ned_m',
                    'tracking_error_m','sim_timestamp_ns','reason'}
            require(isinstance(body,dict) and set(body)==fields,'invalid_b_status','Invalid B status fields')
            require(type(body['revision']) is int and body['revision']==m['applied_revision']
                    and body['nominal_shape']==m['applied_formation']['shape'],
                    'configuration_mismatch','Status must describe the actually applied configuration',409)
            require(body['effective_mode'] in {'nominal','trail','recovering','holding'}
                    and isinstance(body['reason'],str) and len(body['reason'])<=256,
                    'invalid_b_status','Invalid mode/reason')
            followers=set(m['config']['vehicle_names'])-{m['config']['leader']}
            goals,errors=body['slot_targets_ned_m'],body['tracking_error_m']
            require(isinstance(goals,dict) and isinstance(errors,dict)
                    and set(goals)==followers and set(errors)==followers,
                    'invalid_b_status','Status must cover every follower exactly')
            for name in followers:
                require(isinstance(goals[name],list) and len(goals[name])==3,'invalid_b_status','Invalid slot target')
                for v in goals[name]:number(v,'slot target')
                number(errors[name],'tracking error',minimum=0,maximum=1000)
            runtime=self.public(m)['runtime']
            require(not runtime['stale'] and runtime.get('collection_complete') and runtime.get('vehicles'),
                    'authoritative_state_unavailable','Need fresh complete authoritative state',503)
            stamp=body['sim_timestamp_ns']
            latest=max(s['timestamp'] for s in runtime['vehicles'].values())
            require(type(stamp) is int and abs(stamp-latest)<=int(self.profile.service.snapshot_max_age_s*1e9),
                    'stale_b_status','B status is outside the authoritative simulation window',409)
            require(stamp>self.b_status_stamps.get(mid,-1),'replayed_b_status','Old status does not refresh freshness',409)
            self.b_status_stamps[mid]=stamp
            old=m.get('formation_runtime')
            m['formation_runtime']=dict(copy.deepcopy(body),received_at_unix_ms=int(time.time()*1000))
            if old is None or (old['revision'],old['effective_mode'])!=(body['revision'],body['effective_mode']):
                self._event(m)
            else:self._save()
            return dict(accepted=True)

    def _fail_pending(self, m, reason):
        if m.get('pending_formation'):
            op = self.data['operations'][m['pending_formation']['request_id']]
            op.update(state='FAILED', reason=reason)
            m['pending_formation'] = None
            self._event(m)

    def _close_operations(self, m):
        self._fail_pending(m, 'mission_ended')
        for op in self.data['operations'].values():
            if op['mission_id']==m['mission_id'] and op['state'] == 'RUNNING':
                op.update(state=m['state'], reason=m['reason'])

    def _finish_from_disk(self, m, *, interrupted=False):
        directory = self.directory / m['mission_id']
        try:
            report = read_json(directory / 'runner.result.json')
        except (OSError, ValueError):
            report = None
        if report is not None:
            m['vehicle_results'] = report['vehicle_results']
            if report['cleanup_verified']:
                m['state'] = 'COMPLETED' if report['mission_completed'] else (
                    'STOPPED' if m.get('stop_requested') and report.get('coordinated_stop_verified') else 'FAILED')
                m['reason'] = report['reason']
                m['manual_intervention_required'] = False
            else:
                m['state'], m['reason'] = 'FAILED', 'cleanup_unverified'
                m['manual_intervention_required'] = True
        else:
            m['state'], m['reason'] = ('INTERRUPTED' if interrupted else 'FAILED'), 'runner_result_missing'
            m['manual_intervention_required'] = True
        self._close_operations(m)

    def tick(self):
        with self.lock:
            for mid, proc in list(self.processes.items()):
                m = self._mission(mid)
                if proc.poll() is not None:
                    self._finish_from_disk(m)
                    del self.processes[mid]
                    self._event(m)
                    continue
                pending = m.get('pending_formation')
                if pending and (not self._b_online(m) or pending['expires_at_unix_ms'] <= int(time.time()*1000)):
                    self._fail_pending(m, 'b_control_stale_or_expired')
                if m['state'] in {'STARTING','TAKING_OFF','RUNNING'}:
                    runtime = self.public(m)['runtime']
                    vehicles = runtime.get('vehicles', {})
                    phase = m['state']
                    if not runtime['stale'] and vehicles:
                        phase = 'RUNNING' if all(v.get('landed_state')==1 for v in vehicles.values()) else 'TAKING_OFF'
                    if (self.directory / mid / 'runtime' / 'mission.complete').exists():
                        phase = 'LANDING'
                    if phase != m['state']:
                        m['state'] = phase
                        self._event(m)
            for m in self.data['missions'].values():
                if m['state']=='INTERRUPTED' and (self.directory/m['mission_id']/'runner.result.json').exists():
                    self._finish_from_disk(m)
                    self._event(m)

    def monitor(self):
        while not self.closed.wait(.2):
            self.tick()

    def shutdown(self, timeout=120.):
        with self.lock:
            for mid in list(self.processes):
                self.stop(mid, {'request_id': 'shutdown-'+uuid.uuid4().hex})
        deadline = time.monotonic()+timeout
        while self.processes and time.monotonic()<deadline:
            self.tick()
            time.sleep(.2)
        self.closed.set()
        return not self.processes

"""Trusted profile, read-only preflight and subprocess lifecycle for the operator API."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

import yaml

from coordination.operator_common import OperatorError, OperatorMap, atomic_json, read_json, require
from coordination.runtime_config import load_config
from formation.controller import FormationController, load_formation_config
from models.vehicle_state import VehicleState

ROOT = Path(__file__).resolve().parents[1]


class OperatorProfile:
    def __init__(self, settings_json, swarm_config, formation_config, map_manifest,
                 *, blocks_project=None, allow_flight=False):
        self.settings_path = Path(settings_json).resolve()
        self.settings = read_json(self.settings_path)
        self.swarm, self.service = load_config(swarm_config)
        self.swarm = copy.deepcopy(self.swarm)
        require(not self.swarm.get('navigation', {}).get('independent_goals', False),
                'unsupported_profile', 'Operator formation tasks cannot use independent-goal profiles')
        self.formation = load_formation_config(formation_config)
        self.map = OperatorMap(read_json(map_manifest))
        self.blocks_project = Path(blocks_project).resolve() if blocks_project else None
        self.allow_flight = allow_flight
        self.names = tuple(m['vehicle_name'] for m in self.swarm['drones'])
        require(self.service.enabled and len(self.names) >= 2, 'invalid_profile', 'Need A service and 2..5 vehicles')
        require(set(self.names) == set(self.formation.vehicle_names)
                and self.service.leader == self.formation.leader, 'invalid_profile', 'Fleet/leader mismatch')
        require(set(self.names) == set(self.settings['Vehicles']), 'invalid_profile', 'Settings fleet mismatch')
        require(not any(m.get('start_actor') for m in self.swarm['drones']),
                'unsupported_profile', 'Operator v1 requires settings-based coordinate origins')
        self.clearance = 0.
        self.flight_limits = {}
        for member in self.swarm['drones']:
            for key, default in (('flight_config', 'configs/trajectory_flight.yaml'),
                                 ('perception_config', 'configs/perception.yaml'),
                                 ('trajectory_config', 'configs/trajectory_planner.yaml')):
                path = Path(member.get(key, default))
                member[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
            trajectory = yaml.safe_load(Path(member['trajectory_config']).read_text(encoding='utf-8'))
            self.clearance = max(self.clearance, find_hard_clearance(trajectory))
            self.flight_limits[member['vehicle_name']] = yaml.safe_load(
                Path(member['flight_config']).read_text(encoding='utf-8'))['minimal_flight']
        require(self.clearance > 0, 'invalid_profile', 'Missing trajectory hard clearance')
        self.safety_distance = self.swarm['coordination']['safety_distance_m']

    def validate_config(self, body):
        from coordination.operator_common import identifier, number
        required = {'request_id', 'map_id', 'map_revision', 'vehicle_names', 'leader',
                    'leader_goal_ned_m', 'formation'}
        require(isinstance(body, dict) and set(body) == required, 'invalid_fields', 'Unexpected mission fields')
        identifier(body['request_id'])
        require(body['map_id'] == self.map.value['map_id'] and body['map_revision'] == self.map.revision,
                'map_revision_mismatch', 'Reload the current map before submitting', 409)
        require(body['vehicle_names'] == list(self.names) and body['leader'] == self.service.leader,
                'unsupported_fleet', 'V1 uses exactly the configured fleet and leader', 422)
        goal = body['leader_goal_ned_m']
        require(isinstance(goal, list) and len(goal) == 3, 'invalid_goal', 'Need World NED XYZ')
        goal = [number(v, 'goal') for v in goal]
        formation = body['formation']
        require(isinstance(formation, dict) and set(formation) == {'shape', 'spacing_m'},
                'invalid_formation', 'Need shape and spacing_m')
        spacing = number(formation['spacing_m'], 'spacing_m', minimum=self.safety_distance + .7, maximum=8.)
        try:
            fc = replace(self.formation, shape=formation['shape'], spacing_m=spacing)
        except ValueError as exc:
            raise OperatorError('invalid_formation', str(exc), 422) from exc
        for member in self.swarm['drones']:
            z = self.settings['Vehicles'][member['vehicle_name']].get('Z', 0.) + member['target_z']
            require(abs(goal[2] - z) < 1e-6, 'unsupported_altitude',
                    'V1 keeps the configured common cruise altitude', 422)
        planner = FormationController(fc, task_id='operator-preview')
        terminal = dict(planner.slot_targets(VehicleState(position_ned_m=goal, yaw_rad=0.)),
                        **{fc.leader: tuple(goal)})
        for point in terminal.values():
            self.map.check_point(point, self.clearance)
        return dict(map_id=body['map_id'], map_revision=self.map.revision,
                    vehicle_names=list(self.names), leader=fc.leader,
                    leader_goal_ned_m=goal, formation={'shape': fc.shape, 'spacing_m': spacing}), terminal


def find_hard_clearance(value):
    if isinstance(value, dict):
        if 'hard_clearance_m' in value:
            return float(value['hard_clearance_m'])
        for item in value.values():
            found = find_hard_clearance(item)
            if found:
                return found
    return 0.


class AirSimPreflight:
    """Create/use/close RPC on the caller thread; never acquire flight control."""
    def __init__(self, profile):
        self.profile = profile

    def __call__(self):
        from adapters.airsim_client import AirSimClientAdapter
        from coordination.coordinate_transform import CoordinateTransformer
        from scripts.validate_blocks_maps import active_map
        p = self.profile
        AirSimClientAdapter._ensure_airsim_importable()
        import airsim
        client = airsim.MultirotorClient(timeout_value=3)
        try:
            require(not client.simIsPause(), 'simulation_paused', 'Resume simulation before starting', 409)
            require(set(client.listVehicles()) == set(p.names), 'runtime_fleet_mismatch', 'Runtime fleet differs', 409)
            require(json.loads(client.getSettingsString()).get('Vehicles') == p.settings['Vehicles'],
                    'runtime_settings_mismatch', 'Runtime vehicle settings differ', 409)
            require(p.blocks_project is not None, 'map_verification_unavailable', 'Need --blocks-project', 503)
            require(active_map(p.blocks_project) == '/Game/' + p.map.value['map_id'],
                    'runtime_map_mismatch', 'Active Blocks map differs', 409)
            asset = p.blocks_project.parent / 'Content' / (p.map.value['map_id'] + '.umap')
            require(hashlib.sha256(asset.read_bytes()).hexdigest() == p.map.value.get('asset_sha256'),
                    'map_asset_changed', 'Re-export map after asset changes', 409)
            for item in p.map.value.get('obstacles', []):
                pose = client.simGetObjectPose(item['id'])
                scale = client.simGetObjectScale(item['id'])
                actual = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
                require(all(math.isfinite(v) for v in actual)
                        and math.dist(actual, item['actor_position_ned_m']) < .03,
                        'map_geometry_changed', 'Obstacle moved; re-export map', 409)
                require(math.dist([scale.x_val, scale.y_val, scale.z_val], item['actor_scale']) < .001,
                        'map_geometry_changed', 'Obstacle scale changed; re-export map', 409)
                q = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]
                require(abs(sum(a*b for a, b in zip(q, item['actor_orientation_xyzw']))) > .99999,
                        'map_geometry_changed', 'Obstacle rotated; re-export map', 409)
            tf = CoordinateTransformer.from_settings_json(str(p.settings_path), p.names)
            require(tf.validate(client).validated, 'coordinate_validation_failed', 'Coordinate validation failed', 409)
            first = {n: client.getMultirotorState(vehicle_name=n).timestamp for n in p.names}
            time.sleep(.12)
            states = {}
            for n in p.names:
                raw = client.getMultirotorState(vehicle_name=n)
                kin = raw.kinematics_estimated
                pos = tf.to_common_ned(n, [kin.position.x_val, kin.position.y_val, kin.position.z_val])
                vel = [kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val]
                require(raw.timestamp > first[n], 'state_not_advancing', 'Simulator timestamps are not advancing', 409)
                require(raw.landed_state == 0 and math.dist(vel, [0, 0, 0]) < .05
                        and not client.isApiControlEnabled(vehicle_name=n),
                        'vehicle_not_released', f'{n} must be landed, stationary and API released', 409)
                require(p.service.allow_legacy_readiness or (vars(raw).get('ready') is True
                        and vars(raw).get('can_arm') is True), 'vehicle_not_ready', f'{n} is not ready', 409)
                require(all(math.isfinite(v) for v in (*pos, *vel)), 'invalid_state', 'Nonfinite state', 409)
                states[n] = dict(position_ned_m=pos, velocity_ned_mps=vel,
                                 yaw_rad=airsim.to_eularian_angles(kin.orientation)[2],
                                 sim_timestamp_ns=raw.timestamp, landed_state=raw.landed_state)
            for a, b in itertools.combinations(p.names, 2):
                require(math.dist(states[a]['position_ned_m'], states[b]['position_ned_m']) > p.safety_distance,
                        'initial_separation', 'Initial fleet separation is insufficient', 409)
            # Existing A/controller processes must not be displaced by a new job.
            try:
                with socket.create_connection((p.service.host, p.service.port), timeout=.2):
                    raise OperatorError('state_service_in_use', 'An A service already owns the flight profile', 409)
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
            return states
        except OperatorError:
            raise
        except Exception as exc:
            raise OperatorError('simulator_unavailable', f'Read-only preflight failed: {type(exc).__name__}', 503) from exc
        finally:
            client.client.close()


class SubprocessMissionBackend:
    def __init__(self, profile):
        self.profile = profile

    def launch(self, mission, directory, operator_url):
        directory = Path(directory)
        p = self.profile
        config = copy.deepcopy(p.swarm)
        config['state_service']['task_id'] = mission['mission_id']
        config['operator_api'] = dict(version=1, base_url=operator_url,
            mission_id=mission['mission_id'], initial_revision=mission['revision'],bearer_token=mission['_b_token'])
        goal = mission['config']['leader_goal_ned_m']
        for member in config['drones']:
            spawn = p.settings['Vehicles'][member['vehicle_name']]
            member.pop('goal_actor', None)
            member['goal_x'], member['goal_y'] = goal[0] - spawn.get('X', 0.), goal[1] - spawn.get('Y', 0.)
        fc = replace(p.formation, **mission['config']['formation'])
        (directory / 'swarm.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        form = asdict(fc)
        form['followers'] = list(fc.followers)
        (directory / 'formation.yaml').write_text(yaml.safe_dump({'formation': form}), encoding='utf-8')
        atomic_json(directory / 'mission.json', mission)
        command = [sys.executable, str(ROOT / 'scripts/operator_mission_runner.py'),
                   '--job-dir', str(directory), '--settings-json', str(p.settings_path)]
        handle = (directory / 'supervisor.log').open('w', encoding='utf-8')
        try:
            return subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0)
        finally:
            handle.close()

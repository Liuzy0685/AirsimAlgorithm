"""Read-only fleet check before launching any controller; no API acquisition."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import yaml
from sensors.lidar_reader import LidarReader
from control.swept_clearance import raw_frame_takeoff_column_clearance

ROOT=Path(__file__).resolve().parents[1]


def verify_fleet_takeoff(members,settings_json,*,client=None):
    settings=json.loads(Path(settings_json).read_text(encoding='utf-8'))
    if settings.get('ApiServerPort',41451)!=41451:
        raise ValueError('Authoritative flight launcher requires primary RPC port 41451')
    owned=client is None
    if owned:
        from adapters.airsim_client import AirSimClientAdapter
        AirSimClientAdapter._ensure_airsim_importable()
        import airsim
        client=airsim.MultirotorClient(timeout_value=5)
    report={}
    try:
        names={m['vehicle_name'] for m in members}
        if set(client.listVehicles())!=names or json.loads(client.getSettingsString())['Vehicles']!=settings['Vehicles']:
            raise ValueError('Fleet/settings mismatch before takeoff')
        if client.simIsPause():raise ValueError('Simulation is paused')
        states={}
        # Check every owner before doing perception, so no vehicle is launched
        # while another still belongs to an unverified previous flight.
        for member in members:
            n=member['vehicle_name']
            if client.isApiControlEnabled(vehicle_name=n) is not False:
                raise ValueError('vehicle_api_already_enabled:'+n)
            state=client.getMultirotorState(vehicle_name=n)
            p=state.kinematics_estimated.position;v=state.kinematics_estimated.linear_velocity
            position=tuple(float(getattr(p,k)) for k in ('x_val','y_val','z_val'))
            velocity=tuple(float(getattr(v,k)) for k in ('x_val','y_val','z_val'))
            if (state.landed_state!=0 or not all(math.isfinite(x) for x in (*position,*velocity))
                    or abs(position[2])>5 or math.sqrt(sum(x*x for x in velocity))>.05):
                raise ValueError('vehicle_not_stationary_on_ground:'+n)
            states[n]=position
        for member in members:
            n=member['vehicle_name']
            trajectory=Path(member.get('trajectory_config',ROOT/'configs/trajectory_planner.yaml'))
            config=yaml.safe_load(trajectory.read_text(encoding='utf-8'))
            hard=float(config['trajectory_planner']['hard_clearance_m'])
            if not math.isfinite(hard) or hard<=0:raise ValueError('Invalid hard clearance')
            target=member.get('target_z')
            if target is None:
                flight=yaml.safe_load(Path(member.get('flight_config',ROOT/'configs/trajectory_flight.yaml')).read_text(encoding='utf-8'))
                target=flight['minimal_flight']['target_z_ned']
            reader=LidarReader(SimpleNamespace(get_raw_client=lambda:client),vehicle_name=n,lidar_name=member['lidar_name'])
            frame=reader.read()
            clearance=raw_frame_takeoff_column_clearance(frame,states[n],float(target))
            if clearance<hard:raise ValueError(f'preflight_takeoff_column_blocked:{n}: clearance={clearance:.3f}, required={hard:.3f}')
            report[n]=dict(observed_column_clearance_m=clearance if math.isfinite(clearance) else None,
                           hard_clearance_m=hard,raw_timestamp_ns=frame.raw_timestamp_ns)
        return report
    finally:
        if owned:client.client.close()

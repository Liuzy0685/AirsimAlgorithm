"""Finish API release using prior disarm acknowledgements / never-armed audit.
No takeoff/arm/movement commands. Keep full existing recovery collision checks.
"""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_runtime import OperatorProfile
from coordination.operator_map_runtime import verify_map
from control.recovery_landing import RecoveryLanding,_NoVelocityDispatcher
from control.vehicle_command_gateway import VehicleCommandGateway

OUT=Path(__file__).resolve().parent/'5_baseline_cam0'
AirSimClientAdapter._ensure_airsim_importable()
import airsim
raw=airsim.MultirotorClient(timeout_value=5)
profile=OperatorProfile(ROOT/'configs/acceptance/settings_5.json',ROOT/'configs/acceptance/swarm_5_navigation.yaml',
    ROOT/'configs/acceptance/formation_5.yaml',ROOT/'configs/operator/flying_example.json',
    blocks_project=Path(r'D:\30817\Adrone\blocks\Blocks.uproject'),engine_log=OUT/'ue.log',allow_flight=True)
baseline=json.loads((OUT/'preflight_ground.json').read_text())['states']
refs={n:dict(landed_state=0,position_ned_m=v['local_position']) for n,v in baseline.items()}
recovery=RecoveryLanding(profile,raw,ground_reference=refs)
result=dict(cleanup_verified=False,proof={},states={},reason='not_started')
try:
    for name in profile.names:
        lock=recovery.acquire('operator-recovery',name)
        if lock is None:raise RuntimeError('live controller lock: '+name)
        recovery.locks[name]=lock
    actual=json.loads(raw.getSettingsString())
    assert actual['Vehicles']==profile.settings['Vehicles'] and set(raw.listVehicles())==set(profile.names)
    assert not raw.simIsPause();verify_map(profile,raw)
    acknowledged=set()
    for filename in ('explicit_recovery_02.json','explicit_recovery_03.json'):
        report=json.loads((OUT/filename).read_text())
        acknowledged.update(n for n,v in report['vehicle_results'].items() if v.get('disarmed'))
    for name in profile.names:
        if name in acknowledged:result['proof'][name]='prior explicit recovery disarm acknowledgement; no subsequent arm operations'
        else:
            previous=json.loads((OUT/f'flight/runtime/{name}.result.json').read_text())
            text=(OUT/f'flight/runtime/{name}.log').read_text(encoding='utf-8')
            assert 'enableApiControl failed' in previous['termination_reason'] and 'Vehicle armed.' not in text
            assert not previous['armed'] and profile.settings['Vehicles'][name]['DefaultVehicleState']=='Disarmed'
            result['proof'][name]='initially disarmed; original controller failed before arm; recovery never arms'
    recovery.observe(initial=True)
    for _ in range(10):
        states=recovery.observe()
        assert all(s['landed'] and s['advanced'] for s in states.values())
        time.sleep(.2)
    recovery.confirmed.update(profile.names)
    gateway=VehicleCommandGateway(recovery.authority,dispatcher=_NoVelocityDispatcher())
    for name in profile.names:
        recovery.authority.acquire(name,recovery.controller_id)
        proxy=gateway.bind_raw_client(raw,vehicle_name=name,controller_id=recovery.controller_id)
        proxy.lifecycle_guard=lambda api,args,kwargs,n=name:recovery.guard(n,api,args,kwargs)
        recovery.proxies[name]=proxy
        if raw.isApiControlEnabled(vehicle_name=name):proxy.enableApiControl(False,vehicle_name=name)
        assert raw.isApiControlEnabled(vehicle_name=name) is False
        assert recovery.observe()[name]['landed']
        result['states'][name]=dict(landed_confirmed=True,api_control_released=True,disarm_proof=result['proof'][name])
    result.update(cleanup_verified=True,reason='verified_ground_and_api_released')
except Exception as exc:result['reason']=str(exc)
finally:
    for name,lock in recovery.locks.items():
        recovery.authority.release(name,recovery.controller_id);recovery.release(lock)
    (OUT/'final_ground_release.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    if result['cleanup_verified']:raw.simRunConsoleCommand('quit')
    raw.client.close()

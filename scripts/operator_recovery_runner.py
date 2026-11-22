"""Bounded child process for an authenticated, explicit recovery-land request."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.operator_common import atomic_json,read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    args=parser.parse_args()
    directory=Path(args.directory).resolve()
    request=read_json(directory/'request.json')
    client=None
    try:
        from coordination.operator_runtime import OperatorProfile
        profile=OperatorProfile(request['settings_json'],request['swarm_config'],request['formation_config'],
            request['map_manifest'],blocks_project=request['blocks_project'],
            engine_log=request['engine_log'],allow_flight=True)
        from adapters.airsim_client import AirSimClientAdapter
        AirSimClientAdapter._ensure_airsim_importable()
        import airsim
        client=airsim.MultirotorClient(timeout_value=3)
        from control.recovery_landing import RecoveryLanding
        result=RecoveryLanding(profile,client,ground_reference=request.get('ground_reference',{})).run()
    except Exception as exc:
        result=dict(cleanup_verified=False,vehicle_results={},reason=type(exc).__name__+': '+str(exc))
    finally:
        if client is not None:client.client.close()
    result.update(mission_id=request['mission_id'],request_id=request['request_id'],revision=request['revision'])
    atomic_json(directory/'result.json',result)
    return 0 if result['cleanup_verified'] else 2


if __name__=='__main__':raise SystemExit(main())

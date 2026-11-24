"""Offline check of configured formation endpoints; never connects to AirSim."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coordination.operator_runtime import OperatorProfile
from coordination.operator_common import OperatorError,number


def check(settings_json,swarm_config,formation_config,map_manifest):
    profile=OperatorProfile(settings_json,swarm_config,formation_config,map_manifest)
    report=dict(valid=False,scope='offline_known_map_endpoint_check_only',flight_authorized=False,
        runtime_map_verified=False,map_id=profile.map.value['map_id'],map_revision=profile.map.revision,
        expected_map_asset_sha256=profile.map.value.get('asset_sha256'))
    try:
        leader=profile.formation.leader
        member=next(m for m in profile.swarm['drones'] if m['vehicle_name']==leader)
        spawn=profile.settings['Vehicles'][leader]
        goal=[number(member.get(k),k)+spawn.get(axis,0.) for k,axis in (('goal_x','X'),('goal_y','Y'),('target_z','Z'))]
        report['leader_goal_ned_m']=goal
        # Validate the leader first, then all terminal slots under the existing
        # operator contract (yaw=0). Do not alter a goal to make it pass.
        profile.map.check_point(goal,profile.clearance)
        body=dict(request_id='offline-endpoint-check',map_id=profile.map.value['map_id'],map_revision=profile.map.revision,
            vehicle_names=list(profile.names),leader=leader,leader_goal_ned_m=goal,
            formation=dict(shape=profile.formation.shape,spacing_m=profile.formation.spacing_m))
        _,terminal=profile.validate_config(body)
        report.update(valid=True,terminal_targets=terminal,reason='known_endpoint_checks_passed_not_flight_clearance')
    except OperatorError as exc:
        report.update(reason=exc.code,message=str(exc))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('settings-json','swarm-config','formation-config','map-manifest'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error('output exists')
    report=check(args.settings_json,args.swarm_config,args.formation_config,args.map_manifest)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)
    return 0 if report['valid'] else 2


if __name__=='__main__':raise SystemExit(main())

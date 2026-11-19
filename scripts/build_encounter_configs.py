"""Generate isolated low-speed maze encounter profiles; never launches a simulator.

These fixtures use the western open region of the same maze map to isolate
inter-vehicle encounters; ground stabilization and runtime LiDAR remain mandatory.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    settings=json.loads((ROOT/'configs/acceptance/settings_5.json').read_text(encoding='utf-8'))
    swarm=yaml.safe_load((ROOT/'configs/acceptance/swarm_5_navigation.yaml').read_text(encoding='utf-8'))
    flight=yaml.safe_load((ROOT/'configs/acceptance/flight.yaml').read_text(encoding='utf-8'))
    five_starts=[(.5,y) for y in (-6.,-3.,-9.,0.,-12.)]
    cases={
        'head_on': ([(-2.,-8.),(-2.,0.)],[(-2.,0.),(-2.,-8.)],[.5,.5]),
        'crossing': ([(-2.,-6.),(.5,-6.)],[(.5,0.),(-2.,0.)],[.5,.5]),
        'overtaking': ([(-2.,-11.),(-2.,-6.)],[(-2.,0.),(-2.,-3.)],[.5,.25]),
        'five_way': (five_starts,[(.5,-12-y) for _,y in five_starts],[.35]*5),
        'narrow_corridor': ([(.5,-2.),(24.,2.)],[(24.,2.),(.5,-2.)],[.35,.35]),
    }
    result={}
    for name,(starts,goals,speeds) in cases.items():
        folder=output/name;folder.mkdir()
        fleet=copy.deepcopy(settings);cfg=copy.deepcopy(swarm)
        names=[f'Drone{i+1}' for i in range(len(starts))]
        fleet['Vehicles']={n:fleet['Vehicles'][n] for n in names}
        fleet['SubWindows']=[]
        cfg['drones']=cfg['drones'][:len(names)]
        cfg['navigation']['independent_goals']=True
        cfg['navigation']['adaptive_slots']=False
        cfg['state_service']['task_id']='encounter-'+name
        cfg['state_service']['formation_tolerance_m']=.35
        for member,start,goal,speed in zip(cfg['drones'],starts,goals,speeds):
            n=member['vehicle_name'];vehicle=fleet['Vehicles'][n]
            vehicle.update(X=start[0],Y=start[1],Z=0.)
            member.update(goal_x=goal[0]-start[0],goal_y=goal[1]-start[1],target_z=-2.)
            f=copy.deepcopy(flight)
            f['minimal_flight'].update(forward_speed_mps=speed,side_speed_mps=min(.4,speed),max_flight_duration_s=240.)
            fp=folder/(n+'_flight.yaml');fp.write_text(yaml.safe_dump(f),encoding='utf-8')
            member['flight_config']=str(fp)
        sp=folder/'settings.json';cp=folder/'swarm.yaml'
        sp.write_text(json.dumps(fleet,indent=2),encoding='utf-8')
        cp.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        result[name]=dict(settings_json=str(sp),swarm_config=str(cp),starts=starts,goals=goals,speeds=speeds,
                          map_label='Corridor' if name=='narrow_corridor' else 'maze')
    (output/'profiles.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    for name,profile in build(p.parse_args().output).items():print(name,profile['swarm_config'])

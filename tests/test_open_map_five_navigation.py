"""Open-floor geometry with the shared five-aircraft layout, no simulator."""
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np

from common.models import VehicleState
from coordination.navigation import NavigationConfig,CommittedGridRouter
from test_swarm_wiring import bridge_fixture


def test_shared_five_layout_floor_and_parked_peers_do_not_block_leader_route():
    root=Path(__file__).resolve().parents[1]
    settings=json.loads((root/'configs/acceptance/settings_5.json').read_text(encoding='utf-8'))
    origins={n:(s['X'],s['Y'],s['Z']) for n,s in settings['Vehicles'].items()}
    bridge,_,_=bridge_fixture()
    bridge.name='Drone1'
    bridge.navigation_cfg=NavigationConfig(enabled=True,adaptive_slots=True,resolution_m=.25,clearance_m=1.1)
    bridge.router=CommittedGridRouter(bridge.navigation_cfg,minimum_clearance_m=.9)
    bridge.params.target_z_ned=-2.;bridge.params.forward_speed_mps=.35
    bridge.mode._swarm_spawn_offsets=origins
    states={n:VehicleState(position_ned_m=[origin[0],origin[1],-2.],linear_velocity_ned_mps=[0.,0.,0.],
        timestamp=1_500_000_000) for n,origin in origins.items()}
    bridge.link=NS(names=set(origins),snapshot=lambda:NS(states=states),command=lambda:None,
        peer_navigation=lambda max_age:{n:dict(tracking_error_m=0.,path_available=True) for n in origins if n!='Drone1'})
    # Real floor geometry is below the cruise slab. Peer body returns are
    # present in the raw cloud, and must only leave temporary peer constraints.
    raw=[[float(x),float(y),3.02] for x in range(-5,36,2) for y in range(-8,9,2)]
    raw.extend([[-.2,o[1]-origins['Drone1'][1],0.] for n,o in origins.items() if n!='Drone1'])
    frame=NS(raw_timestamp_ns=1_500_000_001,received_monotonic_seconds=time.monotonic(),
        sensor_pose=dict(position=dict(x=.2,y=0.,z=-2.),orientation=dict(x=0.,y=0.,z=0.,w=1.)),
        point_cloud_sensor=np.asarray(raw))
    bridge.update_lidar(frame)
    assert len(bridge.router.obstacles())==0
    local=VehicleState(position_ned_m=[0.,0.,-2.],linear_velocity_ned_mps=[0.,0.,0.])
    waypoint=bridge.route_goal(local,(30.,0.,-2.))
    assert bridge.router.path_available
    assert waypoint==(30.,0.,-2.)
    selected=bridge._navigation_velocity(local,0.,'trajectory')
    assert selected[0]==.35 and selected[1]==0.
    assert len(bridge._route_transients)==4*33

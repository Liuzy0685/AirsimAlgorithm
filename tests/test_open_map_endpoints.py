import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from coordination.operator_common import OperatorMap,OperatorError
from scripts.check_mission_endpoints import check
from scripts import export_operator_map

ROOT=Path(__file__).resolve().parents[1]


def test_shared_five_open_map_goal_is_rejected_by_measured_cube():
    report=check('configs/acceptance/settings_5.json','configs/acceptance/swarm_5_navigation.yaml',
        'configs/acceptance/formation_5.yaml','configs/operator/flying_example.json')
    assert report['leader_goal_ned_m']==[30.5,-6.,-2.]
    assert not report['valid'] and report['reason']=='goal_near_known_obstacle'
    assert 'TemplateCube_Rounded_75' in report['message']
    assert not report['flight_authorized'] and not report['runtime_map_verified']


def test_endpoint_checks_passing_does_not_authorize_flight(tmp_path,monkeypatch):
    import socket
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('offline check must not use RPC'))
    value=json.loads((ROOT/'configs/operator/flying_example.json').read_text(encoding='utf-8'))
    value['obstacles']=[]
    manifest=tmp_path/'display.json';manifest.write_text(json.dumps(value),encoding='utf-8')
    report=check('configs/acceptance/settings_5.json','configs/acceptance/swarm_5_navigation.yaml',
        'configs/acceptance/formation_5.yaml',manifest)
    assert report['valid'] and not report['flight_authorized'] and not report['runtime_map_verified']


def test_selected_cube_export_includes_measured_goal_obstacle(tmp_path,monkeypatch):
    asset=tmp_path/'Content/FlyingCPP/Maps/FlyingExampleMap.umap';asset.parent.mkdir(parents=True);asset.write_bytes(b'map')
    vec=lambda x,y,z:NS(x_val=x,y_val=y,z_val=z)
    pose=NS(position=vec(28.1,-6.5,-1.48),orientation=NS(x_val=0,y_val=0,z_val=0,w_val=1))
    vertices=[(x,y,z) for x in (2310.,3310.) for y in (-1150.,-150.) for z in (-102.,398.)]
    mesh=NS(name='templatecube_rounded_75',position=vec(2810.,-650.,148.),
        vertices=[v for p in vertices for v in p],indices=[0,1,2,0,2,3,4,5,6,4,6,7])
    client=NS(listVehicles=lambda:[],simListSceneObjects=lambda:['TemplateCube_Rounded_75'],
        simGetMeshPositionVertexBuffers=lambda:[mesh],simGetObjectPose=lambda _:pose,simGetObjectScale=lambda _:vec(10,10,5))
    monkeypatch.setattr(export_operator_map,'active_map',lambda _: '/Game/FlyingCPP/Maps/FlyingExampleMap')
    value=export_operator_map.export(client,tmp_path/'Blocks.uproject','flying_example',[-5,45,-16,16],
        asset_path='/Game/FlyingCPP/Maps/FlyingExampleMap',actor_prefixes=['Cylinder','TemplateCube'],clip_to_bounds=True)
    assert len(value['obstacles'])==1 and value['geometry_selection']['intersect_editing_extent']
    m=OperatorMap(value)
    assert m.public()['collision_certified'] is False
    with pytest.raises(OperatorError,match='TemplateCube_Rounded_75'):m.check_point((30.5,-6.,-2.),.9)


def test_observed_front_wall_can_leave_goal_unmarked_but_no_route():
    from coordination.navigation import NavigationConfig,CommittedGridRouter
    router=CommittedGridRouter(NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1),clock=lambda:10.)
    router.observe([(23.1,-11.5+i*.25) for i in range(41)])
    assert router.waypoint((.5,-6.,-2.),(30.5,-6.,-2.))==(.5,-6.,-2.)
    assert not router.path_available and not router.goal_blocked
    assert router.last_route_reason=='no_grid_path'


def test_invalid_endpoint_stops_launcher_before_any_aircraft_rpc(tmp_path,monkeypatch):
    import sys
    from scripts import launch_swarm
    import control.takeoff_preflight as takeoff
    directory=tmp_path/'rejected'
    monkeypatch.setattr(sys,'argv',['launch_swarm.py','--settings-json','configs/acceptance/settings_5.json',
        '--config','configs/acceptance/swarm_5_navigation.yaml','--formation-config','configs/acceptance/formation_5.yaml',
        '--map-manifest','configs/operator/flying_example.json','--log-dir',str(directory),
        '--with-formation','--confirm-simulation-clearance'])
    monkeypatch.setattr(launch_swarm.subprocess,'Popen',lambda *a,**kw:pytest.fail('No control process may start'))
    monkeypatch.setattr(takeoff,'verify_fleet_takeoff',lambda *a,**kw:pytest.fail('Must reject before aircraft RPC'))
    assert launch_swarm.main()==2
    report=json.loads((directory/'endpoint_preflight.json').read_text(encoding='utf-8'))
    assert not report['valid'] and 'TemplateCube_Rounded_75' in report['message']

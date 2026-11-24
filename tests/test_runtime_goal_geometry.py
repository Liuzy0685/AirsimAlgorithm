import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from coordination.operator_common import OperatorError
from scripts import check_runtime_goal_geometry as geometry


def cube():
    vertices=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[np.nan,np.nan,np.nan]])
    indices=[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5]
    return vertices,indices


@pytest.mark.parametrize('point,inside',[([0,0,0],True),([3,0,0],False)])
def test_closed_render_cube_containment_and_unreferenced_garbage(point,inside):
    vertices,indices=cube();result=geometry.containment(vertices,indices,point)
    assert result['closed_after_seam_welding']
    assert result['inside_render_mesh'] is inside


def test_open_surface_is_not_claimed_to_be_a_solid():
    vertices,indices=cube();result=geometry.containment(vertices,indices[:-6],[0,0,0])
    assert not result['closed_after_seam_welding'] and result['inside_render_mesh'] is None


def test_invalid_triangle_indices_rejected():
    vertices,_=cube()
    with pytest.raises(OperatorError):geometry.containment(vertices,[0,1,100],[0,0,0])


@pytest.mark.parametrize('api_control',[True,None])
def test_runtime_inspection_refuses_api_owned_vehicle_before_mesh_read(monkeypatch,api_control):
    profile=NS(names=('Drone1',),settings=dict(Vehicles={'Drone1':{}}))
    client=NS(listVehicles=lambda:['Drone1'],getSettingsString=lambda:json.dumps(profile.settings),
        getMultirotorState=lambda **kw:NS(landed_state=0,kinematics_estimated=NS(linear_velocity=NS(x_val=0,y_val=0,z_val=0))),
        isApiControlEnabled=lambda **kw:api_control)
    monkeypatch.setattr(geometry,'verify_map',lambda *args:pytest.fail('Ground check must come first'))
    with pytest.raises(OperatorError) as exc:geometry.inspect(profile,client,[0,0,0],'Cube')
    assert exc.value.code=='vehicle_not_released'

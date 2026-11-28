import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from coordination.operator_common import OperatorMap,OperatorError
from scripts import export_operator_map


def descriptor(**kw):
    return dict(map_id='flying_example',asset_path='/Game/FlyingCPP/Maps/FlyingExampleMap',
                frame='world_ned',units='m',bounds_xy_m=[-5,45,-16,16],obstacles=[],**kw)


def test_nested_asset_path_is_separate_from_public_map_identifier(tmp_path):
    m=OperatorMap(descriptor())
    assert m.asset_file(tmp_path/'Blocks.uproject')==tmp_path/'Content/FlyingCPP/Maps/FlyingExampleMap.umap'
    assert m.public()['asset_path']=='/Game/FlyingCPP/Maps/FlyingExampleMap'
    assert m.public()['collision_certified'] is False


@pytest.mark.parametrize('path',['/Game/../maze','/Game/FlyingCPP/../../maze','/Engine/maze','C:/maze',
                                 '/Game/FlyingCPP\\Maps/FlyingExampleMap','/Game/maze.umap'])
def test_map_asset_path_rejects_traversal_or_nonproject_paths(path):
    value=descriptor();value['asset_path']=path
    with pytest.raises(OperatorError):OperatorMap(value)


def test_empty_geometry_export_is_explicit_and_has_identity_reference(tmp_path,monkeypatch):
    path=tmp_path/'Content/FlyingCPP/Maps/FlyingExampleMap.umap'
    path.parent.mkdir(parents=True);path.write_bytes(b'actual map bytes')
    vector=lambda x,y,z:NS(x_val=x,y_val=y,z_val=z)
    pose=NS(position=vector(0,0,1),orientation=NS(x_val=0,y_val=0,z_val=0,w_val=1))
    c=NS(listVehicles=lambda:[],simListSceneObjects=lambda:[],simGetMeshPositionVertexBuffers=lambda:[],
         simGetObjectPose=lambda _:pose,simGetObjectScale=lambda _:vector(1,1,1))
    monkeypatch.setattr(export_operator_map,'active_map',lambda _:'/Game/FlyingCPP/Maps/FlyingExampleMap')
    args=(c,tmp_path/'Blocks.uproject','flying_example',[-5,45,-16,16])
    with pytest.raises(OperatorError):export_operator_map.export(*args,asset_path='/Game/FlyingCPP/Maps/FlyingExampleMap')
    result=export_operator_map.export(*args,asset_path='/Game/FlyingCPP/Maps/FlyingExampleMap',
                                    allow_no_cylinders=True,reference_objects=['Ground_5'])
    assert result['coverage']=='no_cylinder_geometry_exported_not_free_space'
    assert result['asset_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['reference_objects'][0]['id']=='Ground_5'
    assert result['obstacles']==[]


def test_five_drone_operator_profile_uses_same_shared_settings():
    from coordination.operator_runtime import OperatorProfile
    p=OperatorProfile('configs/acceptance/settings_5.json','configs/acceptance/swarm_5_navigation.yaml',
                      'configs/acceptance/formation_5.yaml','configs/operator/flying_example.json')
    assert p.names==('Drone1','Drone2','Drone3','Drone4','Drone5')
    assert set(p.settings['ExternalCameras'])=={n+'Third' for n in p.names}

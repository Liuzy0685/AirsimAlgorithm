import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import pytest
from control.takeoff_preflight import verify_fleet_takeoff
from test_lidar_parsing import _make_mock_lidar_data


@pytest.mark.parametrize('failure',[None,'owned','wall','airborne','settings'])
def test_all_fleet_preflight_is_read_only_and_blocks_a_bad_member(tmp_path,failure):
    settings=dict(Vehicles={'Drone1':{},'Drone2':{}})
    path=tmp_path/'settings.json';path.write_text(json.dumps(settings),encoding='utf-8')
    raw=MagicMock();raw.listVehicles.return_value=['Drone1','Drone2'];raw.getSettingsString.return_value=json.dumps(settings)
    raw.simIsPause.return_value=False
    raw.isApiControlEnabled.side_effect=lambda **kw: failure=='owned' and kw['vehicle_name']=='Drone2'
    def state(**kw):
        vector=lambda *xyz:NS(**dict(zip(('x_val','y_val','z_val'),xyz)))
        return NS(landed_state=1 if failure=='airborne' and kw['vehicle_name']=='Drone2' else 0,
                  kinematics_estimated=NS(position=vector(0.,0.,0.),linear_velocity=vector(0.,0.,0.)))
    raw.getMultirotorState.side_effect=state
    raw.getLidarData.side_effect=lambda **kw:_make_mock_lidar_data(
        [.5,.5,-.2] if failure=='wall' and kw['vehicle_name']=='Drone2' else [10.,0.,-.2])
    if failure=='settings':raw.getSettingsString.return_value=json.dumps(dict(Vehicles={}))
    members=[dict(vehicle_name=n,lidar_name='LidarSensor'+n[-1],target_z=-2.) for n in settings['Vehicles']]
    if failure:
        with pytest.raises(ValueError):verify_fleet_takeoff(members,path,client=raw)
    else:assert set(verify_fleet_takeoff(members,path,client=raw))==set(settings['Vehicles'])
    raw.enableApiControl.assert_not_called();raw.armDisarm.assert_not_called();raw.takeoffAsync.assert_not_called()

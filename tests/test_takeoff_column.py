import math
import numpy as np
import pytest
from control.swept_clearance import observed_takeoff_column_clearance


def test_known_close_wall_blocks_takeoff_column_but_ground_is_not_an_overhead_obstacle():
    cloud=np.array([[0.,0.,.34],[.586,.586,-.2],[2.,0.,-1.]])
    assert observed_takeoff_column_clearance(cloud,(0.,0.,0.),-2.)==pytest.approx(math.hypot(.586,.586))
    assert observed_takeoff_column_clearance(cloud[[0]],(0.,0.,0.),-2.)==math.inf


@pytest.mark.parametrize('bad',[[],[[float('nan'),0.,0.]],[[1.,2.]]])
def test_missing_or_invalid_raw_cloud_never_means_takeoff_is_clear(bad):
    with pytest.raises(ValueError):observed_takeoff_column_clearance(bad,(0.,0.,0.),-2.)


def test_blocked_raw_takeoff_column_never_acquires_control():
    from unittest.mock import MagicMock
    from test_automatic_mode import _make_mock_session,_setup_patches,_cfg_mocks
    from test_gateway_freshness import scan
    from coordination.navigation import NavigationConfig
    from flight_modes.automatic_mode import AutomaticMode
    session=_make_mock_session()
    bridge=MagicMock();bridge.navigation_cfg=NavigationConfig(enabled=True);session.swarm_bridge=bridge
    stack,mocks=_setup_patches()
    with stack:
        _cfg_mocks(mocks)
        frame=scan(2_000_000_000)
        frame.sensor_pose['position']=dict(x=0.,y=0.,z=-1.)
        frame.point_cloud_sensor=np.array([[.586,.586,0.]])
        mocks[0].return_value.read.return_value=frame
        mode=AutomaticMode(session)
        result=mode.run()
    assert result.termination_reason=='preflight_takeoff_column_blocked'
    session.takeoff_and_climb.assert_not_called()

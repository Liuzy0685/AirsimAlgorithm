"""Regress the delayed-but-nonempty scans reported in B's maze_2_mytest_03."""
import copy
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from control.velocity_controller import VelocityCommandRejected
from models.lidar_frame import LidarFrame


def scan(stamp, *, age=.01, rpc_age=.03):
    now=time.monotonic()
    return LidarFrame(frame_valid=True,point_count=3,raw_timestamp_ns=stamp,
        received_monotonic_seconds=now-age,request_started_monotonic_seconds=now-rpc_age,
        sensor_pose=dict(position=dict(x=10.,y=0.,z=-1.),orientation=dict(x=0.,y=0.,z=0.,w=1.)),
        point_cloud_sensor=np.array([[20.,20.,0.],[21.,20.,0.],[20.,21.,0.]]))


def test_gateway_uses_latest_raw_scan_when_planning_snapshot_has_aged():
    from test_swarm_wiring import bridge_fixture
    b,raw,_=bridge_fixture()
    b.points_at=time.monotonic()-1.
    frame=scan(2_000_000_000,rpc_age=.3)
    b.raw_lidar_supplier=lambda:frame
    b.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
    assert len(raw.calls)==1 and b.points_at==frame.request_started_monotonic_seconds


@pytest.mark.parametrize('kind',['slow_rpc','replayed_scan','invalid_scan','simulation_age','missing_timestamp','future_wall_time'])
def test_latest_scan_never_means_renewed_old_or_invalid_data(kind):
    from test_swarm_wiring import bridge_fixture
    b,raw,_=bridge_fixture()
    frame=scan(2_000_000_000)
    if kind=='slow_rpc':frame.request_started_monotonic_seconds-=1.
    if kind=='replayed_scan':
        b._gateway_scan_id=frame.raw_timestamp_ns
        b.points_at=time.monotonic()-1.
    if kind=='invalid_scan':frame.frame_valid=False
    if kind=='simulation_age':frame.raw_timestamp_ns=10_000_000_000
    if kind=='missing_timestamp':frame.raw_timestamp_ns=None
    if kind=='future_wall_time':
        frame.received_monotonic_seconds=time.monotonic()+1.
        frame.request_started_monotonic_seconds=time.monotonic()+.5
    b.raw_lidar_supplier=lambda:frame
    with pytest.raises(VelocityCommandRejected):
        b.proxy.moveByVelocityAsync(0.,0.,0.,.15,vehicle_name='Drone2')
    assert not raw.calls and b._unverified_emergency


def test_scan_freshness_also_bounds_the_motion_duration():
    from test_swarm_wiring import bridge_fixture
    b,raw,_=bridge_fixture()
    b.raw_lidar_supplier=lambda:scan(2_000_000_000,rpc_age=.55)
    args,kwargs=b.guard_motion('moveByVelocityAsync',(0.,0.,0.,.2),{})
    assert 0<args[3]<=.05


def test_replayed_processed_frame_cannot_replace_newer_gateway_scan():
    from test_swarm_wiring import bridge_fixture
    b,_,_=bridge_fixture()
    newer=scan(2_000_000_000)
    b.update_lidar(newer)
    previous=b.points_at
    b.update_lidar(scan(1_900_000_000))
    assert b._gateway_scan_id==2_000_000_000 and b.points_at==previous


def test_reader_records_rpc_start_instead_of_filter_finish():
    from test_lidar_parsing import _make_reader,_make_mock_lidar_data
    clock=NS(now=10.)
    reader=_make_reader(monotonic_clock=lambda:clock.now)
    def delayed(**kwargs):
        clock.now+=.3
        return _make_mock_lidar_data([1.,0.,0.])
    reader._adapter.get_raw_client().getLidarData.side_effect=delayed
    frame=reader.read()
    assert frame.request_started_monotonic_seconds==10.
    assert frame.received_monotonic_seconds==pytest.approx(10.3)


def test_raw_frame_is_visible_before_slow_filter_and_duplicates_do_not_renew(monkeypatch):
    from flight_modes.automatic_mode import PerceptionWorker
    first=scan(1)
    repeated=copy.deepcopy(first)
    repeated.received_monotonic_seconds+=.2
    frames=iter([first,repeated]);seen=[]
    lidar=NS(read=lambda:next(frames))
    def perceive(frame):
        seen.append(worker.get_latest_raw_frame())
        if len(seen)==1:assert worker.get_latest_snapshot() is None
        else:worker._running=False
        return NS(filtered_points_sensor=np.ones((3,3))),NS(frame_valid=True),{}
    worker=PerceptionWorker(lidar,perceive)
    worker._running=True
    monkeypatch.setattr('flight_modes.automatic_mode.time.sleep',lambda _:None)
    worker._run()
    assert seen==[first,first]


def test_voxel_vectorization_preserves_nearest_and_original_tie_order():
    from perception.pointcloud_filter import filter_pointcloud
    rng=np.random.default_rng(43)
    points=np.concatenate([rng.uniform(-10,10,(8000,3)),[[1.1,1.2,.1],[1.2,1.1,.1]]]).astype(np.float32)
    h=.4
    expected={}
    for i,p in enumerate(points):
        key=tuple(np.floor(p/h).astype(np.int64))
        distance=np.linalg.norm(p)
        if key not in expected or distance<expected[key][0]:expected[key]=(distance,i)
    result=filter_pointcloud(points,min_range_m=0.,max_range_m=40.,voxel_downsample=True,voxel_size_m=h)
    indices=sorted(i for _,i in expected.values())
    np.testing.assert_array_equal(result.filtered_points_sensor,points[indices])


def test_grid_runtime_avoids_diagnostic_workers_but_legacy_option_remains():
    from test_swarm_wiring import bridge_fixture
    from test_automatic_mode import _make_mock_session
    from coordination.navigation import NavigationConfig
    from flight_modes.automatic_mode import AutomaticMode
    bridge,_,_=bridge_fixture()
    session=_make_mock_session();session.swarm_bridge=bridge
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    mode=AutomaticMode(session,cli_overrides={'local_navigation_mode':'trajectory'})
    assert mode._swarm_lightweight and not mode._cbmba_enabled
    assert mode._global_planner_worker is mode._local_traj_worker is mode._mapping_worker is None
    bridge.navigation_cfg=NavigationConfig(enabled=True,run_legacy_diagnostics=True)
    legacy=AutomaticMode(session,cli_overrides={'local_navigation_mode':'trajectory'})
    assert not legacy._swarm_lightweight and legacy._global_planner_worker is not None


def test_authoritative_reader_removes_spawn_without_any_rpc():
    from test_swarm_wiring import bridge_fixture
    from coordination.controller_link import SnapshotPeerReader
    b,raw,_=bridge_fixture()
    snapshot=b.link.snapshot()
    position=list(snapshot.states[b.name].position_ned_m)
    mode=NS(_authoritative_tick_snapshot=snapshot,_swarm_spawn_offsets={b.name:(5.,-2.,0.)})
    state=SnapshotPeerReader(mode,b.name).read()
    assert state.position_ned_m==[position[0]-5.,position[1]+2.,position[2]]
    assert snapshot.states[b.name].position_ned_m==position
    assert not raw.calls


def test_lightweight_control_loop_uses_snapshot_and_never_requests_legacy_work():
    from unittest.mock import MagicMock,patch
    from pathlib import Path
    from test_automatic_mode import _make_mock_session,_setup_patches,_cfg_mocks
    from test_swarm_wiring import bridge_fixture
    from coordination.navigation import NavigationConfig
    from flight_modes.automatic_mode import AutomaticMode,AutomaticModeParams
    actual,_,_=bridge_fixture()
    session=_make_mock_session()
    root=Path(__file__).resolve().parents[1]
    session.settings_json=str(root/'configs/acceptance/settings_2.json')
    bridge=MagicMock();session.swarm_bridge=bridge
    bridge.navigation_cfg=NavigationConfig(enabled=True)
    snapshot=actual.link.snapshot()
    snapshot.states['Drone1'].position_ned_m=[.5,-3.,-1.]
    snapshot.states['Drone2'].position_ned_m=[10.5,0.,-1.]
    bridge.tick.return_value=snapshot
    bridge.link.snapshot.return_value=snapshot
    bridge.goal.return_value=None
    bridge.route_goal.side_effect=lambda state,goal:goal
    bridge.expects_navigation_progress.return_value=False
    bridge.select.return_value=(0.,0.,0.,'navigation_route')
    counts=[]
    stack,mocks=_setup_patches()
    with stack:
        _cfg_mocks(mocks)
        bridge.begin_cruise.side_effect=lambda _:counts.append(mocks[1].return_value.read.call_count)
        mode=AutomaticMode(session,params=AutomaticModeParams(max_flight_duration_s=.15),
            cli_overrides=dict(local_navigation_mode='trajectory',goal_x=15.,goal_y=0.),
            swarm_config_path=str(root/'configs/acceptance/swarm_2_navigation.yaml'))
        mode._tick_global_replan=MagicMock(side_effect=AssertionError('legacy global work'))
        mode._request_local_plan=MagicMock(side_effect=AssertionError('legacy local work'))
        mode._request_map_update=MagicMock(side_effect=AssertionError('legacy map work'))
        with patch.object(session,'takeoff_and_climb'):
            result=mode.run()
    assert result.termination_reason=='time_limit'
    assert bridge.select.call_count>0
    assert mocks[1].return_value.read.call_count==counts[0]
    mode._tick_global_replan.assert_not_called()
    mode._request_local_plan.assert_not_called()
    mode._request_map_update.assert_not_called()

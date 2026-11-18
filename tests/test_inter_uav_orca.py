"""Analytic solver regressions and synchronous ideal-velocity swarm replays."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coordination import orca_2d
from coordination.orca_2d import compute_orca_velocity
from flight_modes.automatic_mode import AutomaticMode


def solve(**overrides):
    inputs = dict(position=(0, 0), velocity=(0, 0), preferred_velocity=(1, 0),
                  neighbors=[], safety_distance_m=2, horizon_s=3,
                  time_step_s=.1, max_speed_mps=2)
    inputs.update(overrides)
    return compute_orca_velocity(**inputs)


def test_boundary_projection_is_closest_feasible_velocity(monkeypatch):
    monkeypatch.setattr(orca_2d, '_orca_line', lambda *args: ((1, 1), (0, 1)))
    assert solve(preferred_velocity=(2, .3), neighbors=[((5, 0), (0, 0))]) == pytest.approx((1, .3))


def test_speed_circle_boundary_intersection(monkeypatch):
    monkeypatch.setattr(orca_2d, '_orca_line', lambda *args: ((.8, .6), (0, -1)))
    assert solve(preferred_velocity=(0, 3), max_speed_mps=1,
                 neighbors=[((5, 0), (0, 0))]) == pytest.approx((.8, .6))


def test_infeasible_constraints_are_reported(monkeypatch):
    monkeypatch.setattr(orca_2d, '_orca_line', lambda *args: ((2, 0), (0, -1)))
    assert solve(max_speed_mps=1, neighbors=[((5, 0), (0, 0))]) is None


def test_unconstrained_speed_limit_and_invalid_neighbor():
    assert solve(preferred_velocity=(3, 4), max_speed_mps=2) == pytest.approx((1.2, 1.6))
    with pytest.raises(ValueError, match='finite'):
        solve(neighbors=[((float('nan'), 0), (0, 0))])


@pytest.mark.parametrize('count', [2, 4, 8])
@pytest.mark.parametrize('bias', [0, .1])
def test_antipodal_swarm_preserves_segment_separation(count, bias):
    # This models exact, synchronous velocity tracking, NOT quadrotor dynamics.
    positions = [(10 * math.cos(i * 2 * math.pi / count),
                  10 * math.sin(i * 2 * math.pi / count)) for i in range(count)]
    goals = [(-p[0], -p[1]) for p in positions]
    velocities = [(0., 0.) for _ in positions]
    for _ in range(700):
        commands = []
        for i, p in enumerate(positions):
            delta = (goals[i][0] - p[0], goals[i][1] - p[1])
            norm = math.hypot(*delta)
            preferred = tuple(v * min(1, 1 / max(norm, 1e-12)) for v in delta)
            command = solve(position=p, velocity=velocities[i], preferred_velocity=preferred,
                            max_speed_mps=1, passing_bias_mps=bias,
                            neighbors=[(q, velocities[j]) for j, q in enumerate(positions) if j != i])
            assert command is not None
            commands.append(command)
        for i in range(count):
            for j in range(i + 1, count):
                p = tuple(positions[i][k] - positions[j][k] for k in range(2))
                v = tuple(commands[i][k] - commands[j][k] for k in range(2))
                v2 = sum(x*x for x in v)
                t = min(.1, max(0, -sum(p[k]*v[k] for k in range(2))/v2)) if v2 > 1e-12 else 0
                assert math.hypot(*(p[k] + v[k]*t for k in range(2))) >= 2 - 1e-5
        positions = [(p[0] + v[0]*.1, p[1] + v[1]*.1) for p, v in zip(positions, commands)]
        velocities = commands
    # Safety and eventual goal convergence are different requirements.
    if count == 2 or bias > 0:
        assert all(math.dist(p, g) < .15 for p, g in zip(positions, goals))


def make_mode(priority=10):
    mode = AutomaticMode.__new__(AutomaticMode)
    mode._vn = 'Drone1'
    mode._swarm_algorithm = 'orca'
    mode._swarm_passing_bias_mps = .1
    mode._swarm_peer_readers = {'Drone2': object()}
    mode._swarm_peer_states = {'Drone2': dict(position_xy=(5, 0), velocity_xy=(-1, 0), timestamp=10**9, received_monotonic=100.)}
    mode._swarm_peer_poll_interval_s = .2
    mode._swarm_max_timestamp_skew_s = .5
    mode._swarm_yield_priorities = {'Drone1': priority, 'Drone2': 5}
    mode._swarm_spawn_offsets = {'Drone1': (0, 0, 0)}
    mode._swarm_safety_distance_m = 2
    mode._swarm_prediction_horizon_s = 3
    mode._params = SimpleNamespace(forward_speed_mps=2)
    mode._occ_grid_params = SimpleNamespace(horizontal_band_half_height_m=1)
    mode._traj_params = SimpleNamespace(hard_clearance_m=.5)
    mode._swept_command_clearance = Mock(return_value=math.inf)
    mode._log_throttled = Mock()
    state = SimpleNamespace(position_ned_m=(-5, 0, -1), linear_velocity_ned_mps=(1, 0, 0), timestamp=10**9)
    return mode, state


@pytest.mark.parametrize('priority', [0, 10])
@pytest.mark.parametrize('failure', ['missing', 'old', 'skew', 'solver', 'lidar'])
def test_failed_filter_stops_both_priorities(monkeypatch, priority, failure):
    monkeypatch.setattr('flight_modes.automatic_mode.time.monotonic', lambda: 100.)
    mode, state = make_mode(priority)
    if failure == 'missing':
        mode._swarm_peer_states.clear()
    elif failure == 'old':
        mode._swarm_peer_states['Drone2']['received_monotonic'] = 98.
    elif failure == 'skew':
        mode._swarm_peer_states['Drone2']['timestamp'] = 1
    elif failure == 'solver':
        monkeypatch.setattr(orca_2d, 'compute_orca_velocity', lambda **kwargs: None)
    elif failure == 'lidar':
        mode._swept_command_clearance.return_value = .1
    x, y, result = mode._apply_inter_uav_orca(state, 1, 0, 0, .1, [])
    assert (x, y) == (0, 0)
    assert result.startswith('safety_hold_')


def test_orca_world_velocity_converts_to_body(monkeypatch):
    monkeypatch.setattr('flight_modes.automatic_mode.time.monotonic', lambda: 100.)
    monkeypatch.setattr(orca_2d, 'compute_orca_velocity', lambda **kwargs: (0, 1))
    mode, state = make_mode()
    x, y, result = mode._apply_inter_uav_orca(state, 1, 0, math.pi/2, .1, [])
    assert (x, y) == pytest.approx((1, 0), abs=1e-10)
    assert result == 'orca'

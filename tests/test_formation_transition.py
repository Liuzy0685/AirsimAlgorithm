"""Measured transition behavior and geometry for B's five shapes."""

import itertools
import math

import pytest

from formation.controller import FormationController
from formation.transition import FormationTransition
from test_formation_controller import config, snapshot


@pytest.mark.parametrize('shape', ['line', 'column', 'vee', 'diamond', 'ring'])
@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_shape_slots_keep_minimum_spacing(shape, count):
    planner = FormationController(config(count, shape), task_id='mission')
    snap = snapshot(count)
    slots = [tuple(snap.states['Drone1'].position_ned_m),
             *planner.slot_targets(snap.states['Drone1']).values()]
    assert all(math.dist(a[:2], b[:2]) >= 3 - 1e-8
               for a, b in itertools.combinations(slots, 2))


def test_switch_moves_one_target_per_frame_and_requires_measured_dwell():
    snap = snapshot(3, positions={2: (-3, 3, -2), 3: (-3, -3, -2)})
    planner = FormationController(config(3, 'vee'), task_id='mission')
    transition = FormationTransition(safety_distance_m=2, max_speed_mps=1,
                                     tolerance_m=.25, dwell_s=1)
    old = transition.targets(snap, planner.slot_targets(snap.states['Drone1']),
                             shape='vee', now=0)
    planner.switch_shape('line')
    nominal = planner.slot_targets(snap.states['Drone1'])
    first = transition.targets(snap, nominal, shape='line', now=.5)
    moved = [name for name in first if math.dist(old[name], first[name]) > 1e-6]
    assert len(moved) == 1
    assert math.dist(old[moved[0]], first[moved[0]]) <= .5 + 1e-8
    assert transition.phase == 'switching'
    assert first != nominal
    for frame in range(1, 35):
        for name, target in first.items():
            snap.states[name].position_ned_m = list(target)
        first = transition.targets(snap, nominal, shape='line', now=.5 + frame * .5)
        if transition.phase == 'nominal':
            break
    assert transition.phase == 'nominal'
    assert all(math.dist(first[name], nominal[name]) <= .25 for name in first)


def test_switch_holds_when_known_obstacle_or_map_boundary_blocks_path():
    snap = snapshot(2, positions={2: (-3, 0, -2)})
    planner = FormationController(config(2, 'column'), task_id='mission')
    transition = FormationTransition(safety_distance_m=2, max_speed_mps=1,
                                     tolerance_m=.25)
    transition.targets(snap, planner.slot_targets(snap.states['Drone1']),
                       shape='column', now=0)
    planner.switch_shape('line')
    nominal = planner.slot_targets(snap.states['Drone1'])
    blocked = transition.targets(snap, nominal, shape='line', now=.5,
        map_obstacles=[{'min_ned_m': [-3.1, -.5, -3],
                        'max_ned_m': [-2.5, .5, -1]}], obstacle_clearance_m=.1)
    assert blocked['Drone2'] == (-3, 0, -2)
    assert transition.reason == 'transition_path_unavailable'
    transition.reset()
    planner.switch_shape('column')
    transition.targets(snap, planner.slot_targets(snap.states['Drone1']),
                       shape='column', now=0)
    planner.switch_shape('line')
    blocked = transition.targets(snap, nominal, shape='line', now=.5,
                                 bounds_xy=(-3.1, -2.8, -1, 1))
    assert blocked['Drone2'] == (-3, 0, -2)
    assert transition.reason == 'slot_out_of_bounds'


def test_obstacle_recovery_moves_from_trail_target_and_switch_interval_holds():
    snap = snapshot(2, positions={2: (-3, 0, -2)})
    planner = FormationController(config(2, 'column'), task_id='mission')
    transition = FormationTransition(safety_distance_m=2, max_speed_mps=1,
                                     tolerance_m=.25, dwell_s=1, min_switch_interval_s=2)
    initial = planner.slot_targets(snap.states['Drone1'])
    transition.targets(snap, initial, shape='column', now=0)
    trail = {'Drone2': (-4, 0, -2)}
    held = transition.targets(snap, initial, shape='column', now=.2,
                              obstructed=True, trail_targets=trail)
    assert held == trail and transition.phase == 'deforming'
    snap.states['Drone2'].position_ned_m = list(trail['Drone2'])
    recovering = transition.targets(snap, initial, shape='column', now=.4)
    assert math.dist(recovering['Drone2'], trail['Drone2']) <= .2 + 1e-8
    assert transition.phase == 'recovering'
    planner.switch_shape('line')
    switching = transition.targets(snap, planner.slot_targets(snap.states['Drone1']),
                                   shape='line', now=.9)
    assert transition.phase == 'switching'
    assert math.dist(switching['Drone2'], recovering['Drone2']) <= .5 + 1e-8
    planner.switch_shape('column')
    transition.targets(snap, initial, shape='column', now=1.0)
    assert transition.reason == 'switch_interval'


def test_five_vehicle_moving_vee_line_column_keeps_target_separation():
    planner = FormationController(config(5, 'vee'), task_id='mission')
    snap = snapshot(5)
    initial = planner.slot_targets(snap.states['Drone1'])
    for name, point in initial.items():
        snap.states[name].position_ned_m = list(point)
    transition = FormationTransition(safety_distance_m=2, max_speed_mps=1,
                                     tolerance_m=.25, dwell_s=1)
    transition.targets(snap, initial, shape='vee', now=0)
    now = 0.
    for shape in ('line', 'column'):
        planner.switch_shape(shape)
        for frame in range(160):
            now += .5
            snap.states['Drone1'].position_ned_m[0] += .05
            nominal = planner.slot_targets(snap.states['Drone1'])
            targets = transition.targets(snap, nominal, shape=shape, now=now)
            positions = [snap.states['Drone1'].position_ned_m, *targets.values()]
            assert all(math.dist(a[:2], b[:2]) >= 2 - 1e-8
                       for a, b in itertools.combinations(positions, 2))
            for name, target in targets.items():
                snap.states[name].position_ned_m = list(target)
            if transition.phase == 'nominal':
                break
        assert transition.phase == 'nominal', (shape, transition.reason,
                                               transition.assignment, transition.offsets)

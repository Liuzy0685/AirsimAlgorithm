"""Ideal synchronous velocity replays using the production ORCA and CPA gate.

No AirSim connection or flight command. These are algorithm regressions, not
quadrotor dynamics, delayed communication, narrow-corridor or flight acceptance.
"""
import argparse
import itertools
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.models import VehicleState
from control.safety_filter import SafetyFilter
from coordination.collision_predictor import predict_pairwise_conflict
from coordination.operator_common import atomic_json
from coordination.orca_2d import compute_gateway_velocity


def scenarios():
    return {
        'head_on': ([(-6., 0.), (6., 0.)], [(6., 0.), (-6., 0.)], [1., 1.]),
        'crossing': ([(-6., 0.), (0., -6.)], [(6., 0.), (0., 6.)], [1., 1.]),
        'overtaking': ([(-7., 0.), (-3., 0.)], [(7., 0.), (4., 0.)], [1., .35]),
        'stationary_peer': ([(-6., 0.), (0., 0.)], [(6., 0.), (0., 0.)], [1., 0.]),
        'five_way': ([(8*math.cos(i*2*math.pi/5), 8*math.sin(i*2*math.pi/5)) for i in range(5)],
                     [(-8*math.cos(i*2*math.pi/5), -8*math.sin(i*2*math.pi/5)) for i in range(5)], [1.]*5),
    }


def replay(positions, goals, speeds, *, steps=1400, dt=.1):
    positions = list(positions)
    velocities = [(0., 0.) for _ in positions]
    minimum = math.inf
    rejected = 0
    infeasible = 0
    trace = []
    safety = SafetyFilter(max_horizontal_speed_mps=max(speeds))
    for step in range(steps):
        states = [VehicleState(position_ned_m=[*p, -2.], linear_velocity_ned_mps=[*v, 0.])
                  for p, v in zip(positions, velocities)]
        commands = []
        for i, p in enumerate(positions):
            delta = tuple(g-x for g, x in zip(goals[i], p))
            distance = math.hypot(*delta)
            scale = min(.7, speeds[i]/max(distance, 1e-12))
            preferred = tuple(d*scale for d in delta)
            neighbors = [(q, velocities[j]) for j, q in enumerate(positions) if j != i]
            # A stationary peer never promises a reciprocal action.
            command = ((0., 0.) if speeds[i] == 0 else compute_gateway_velocity(
                position=p, velocity=velocities[i], preferred_velocity=preferred,
                neighbors=neighbors, safety_distance_m=2.+2*max(speeds)*dt+.05, horizon_s=3.,
                time_step_s=dt, max_speed_mps=speeds[i], minimum_distance_m=2.01,
                passing_bias_mps=min(.1, speeds[i]), responsibility=1.))
            if command is None:
                infeasible += 1
                command = (0., 0.)
            decision = safety.check_velocity((*command, 0.), own_state=states[i],
                neighbors=[s for j, s in enumerate(states) if j != i])
            if not decision.allowed:
                rejected += 1
                # A rejected proposal is not a verified stop. End the replay
                # instead of integrating an unchecked zero command.
                return dict(passed=False, reason=decision.reason, step=step,
                            minimum_segment_separation_m=minimum if math.isfinite(minimum) else None,
                            vehicle_index=i, rejected=rejected, infeasible=infeasible, trace=trace)
            commands.append(command)
        for i, j in itertools.combinations(range(len(positions)), 2):
            prediction = predict_pairwise_conflict([*positions[i], -2.], [*commands[i], 0.],
                [*positions[j], -2.], [*commands[j], 0.], safety_distance_m=2., horizon_s=dt)
            minimum = min(minimum, prediction.closest_distance_m)
        positions = [tuple(p[k]+v[k]*dt for k in range(2)) for p, v in zip(positions, commands)]
        velocities = commands
        trace.append(dict(t=(step+1)*dt, positions=positions, velocities=commands))
        errors = [math.dist(p, g) for p, g in zip(positions, goals)]
        if max(errors) < .15 and max(math.hypot(*v) for v in velocities) < .15:
            return dict(passed=minimum >= 2., reason='all_goals_reached', step=step+1,
                        minimum_segment_separation_m=minimum, maximum_final_error_m=max(errors),
                        rejected=rejected, infeasible=infeasible, trace=trace)
    return dict(passed=False, reason='time_limit', step=steps,
                minimum_segment_separation_m=minimum, maximum_final_error_m=max(errors),
                rejected=rejected, infeasible=infeasible, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = {name: replay(*case) for name, case in scenarios().items()}
    atomic_json(args.output, dict(evidence_type='ideal_velocity_replay_not_flight',
        assumptions='Synchronous perfect velocity tracking; no LiDAR, dynamics or transport delay', results=results))
    for name, result in results.items():
        print(name, {k:v for k,v in result.items() if k != 'trace'})
    return 0 if all(r['passed'] for r in results.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())

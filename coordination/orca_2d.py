"""Small 2D ORCA velocity solver for horizontal multi-UAV coordination.

Velocities and positions use a shared Cartesian frame (the caller uses NED x/y).
The implementation follows the reciprocal velocity obstacle construction and
solves the resulting half-plane constraints inside a maximum-speed circle.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

Vector2 = Tuple[float, float]


def _dot(a: Vector2, b: Vector2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _det(a: Vector2, b: Vector2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _add(a: Vector2, b: Vector2) -> Vector2:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Vector2, b: Vector2) -> Vector2:
    return a[0] - b[0], a[1] - b[1]


def _scale(a: Vector2, value: float) -> Vector2:
    return a[0] * value, a[1] * value


def _length_sq(a: Vector2) -> float:
    return _dot(a, a)


def _normalize(a: Vector2) -> Vector2:
    length = math.sqrt(_length_sq(a))
    return (a[0] / length, a[1] / length) if length > 1e-12 else (1.0, 0.0)


def _orca_line(
    position: Vector2,
    velocity: Vector2,
    peer_position: Vector2,
    peer_velocity: Vector2,
    radius: float,
    horizon_s: float,
    time_step_s: float,
    responsibility: float = 0.5,
) -> tuple[Vector2, Vector2]:
    """Return an ORCA line as (point, direction), following RVO2 conventions."""
    relative_position = _sub(peer_position, position)
    relative_velocity = _sub(velocity, peer_velocity)
    distance_sq = _length_sq(relative_position)
    radius_sq = radius * radius

    if distance_sq > radius_sq:
        inverse_horizon = 1.0 / horizon_s
        w = _sub(relative_velocity, _scale(relative_position, inverse_horizon))
        w_length_sq = _length_sq(w)
        dot = _dot(w, relative_position)
        if dot < 0.0 and dot * dot > radius_sq * w_length_sq:
            unit_w = _normalize(w)
            direction = (unit_w[1], -unit_w[0])
            u = _scale(unit_w, radius * inverse_horizon - math.sqrt(w_length_sq))
        else:
            leg = math.sqrt(max(0.0, distance_sq - radius_sq))
            if _det(relative_position, w) > 0.0:
                direction = (
                    (relative_position[0] * leg - relative_position[1] * radius) / distance_sq,
                    (relative_position[0] * radius + relative_position[1] * leg) / distance_sq,
                )
            else:
                direction = (
                    -(relative_position[0] * leg + relative_position[1] * radius) / distance_sq,
                    -(-relative_position[0] * radius + relative_position[1] * leg) / distance_sq,
                )
            u = _sub(_scale(direction, _dot(relative_velocity, direction)), relative_velocity)
    else:
        inverse_step = 1.0 / time_step_s
        w = _sub(relative_velocity, _scale(relative_position, inverse_step))
        unit_w = _normalize(w)
        direction = (unit_w[1], -unit_w[0])
        u = _scale(unit_w, radius * inverse_step - math.sqrt(_length_sq(w)))

    return _add(velocity, _scale(u, responsibility)), direction


def compute_orca_velocity(
    *,
    position: Sequence[float],
    velocity: Sequence[float],
    preferred_velocity: Sequence[float],
    neighbors: Iterable[tuple[Sequence[float], Sequence[float]]],
    safety_distance_m: float,
    horizon_s: float,
    time_step_s: float,
    max_speed_mps: float,
    passing_bias_mps: float = 0.0,
    responsibility: float = 0.5,
) -> Optional[Vector2]:
    """Choose the feasible velocity closest to preferred, or ``None``.

    Each neighbor is ``(position_xy, velocity_xy)``. Default responsibility
    is reciprocal (0.5). Use 1.0 when the dispatch gate must remain safe
    against the neighbor's measured velocity without assuming its next action.
    Coordinates and velocities are horizontal 2D.
    Optional clockwise preference bias breaks symmetric encounters. It only
    changes the optimization objective; all ORCA constraints remain mandatory.
    """
    vectors = (position, velocity, preferred_velocity)
    if any(len(v) != 2 for v in vectors):
        raise ValueError("ORCA positions and velocities must be 2D")
    p = tuple(float(x) for x in position)
    v = tuple(float(x) for x in velocity)
    preferred = tuple(float(x) for x in preferred_velocity)
    params = (safety_distance_m, horizon_s, time_step_s, max_speed_mps, passing_bias_mps, responsibility)
    if not all(math.isfinite(x) for x in (*p, *v, *preferred, *params)):
        raise ValueError("ORCA inputs must be finite")
    if safety_distance_m <= 0 or horizon_s <= 0 or time_step_s <= 0 or max_speed_mps <= 0:
        raise ValueError("ORCA distances, times and speed must be positive")
    if passing_bias_mps < 0 or passing_bias_mps > max_speed_mps:
        raise ValueError("passing_bias_mps must be between zero and max_speed_mps")
    if isinstance(responsibility, bool) or not 0.5 <= responsibility <= 1.0:
        raise ValueError("responsibility must be between 0.5 and 1.0")

    lines = []
    predicted_encounter = False
    for peer_position, peer_velocity in neighbors:
        if len(peer_position) != 2 or len(peer_velocity) != 2:
            raise ValueError("ORCA neighbor positions and velocities must be 2D")
        q = tuple(float(x) for x in peer_position)
        qv = tuple(float(x) for x in peer_velocity)
        if not all(math.isfinite(x) for x in (*q, *qv)):
            raise ValueError("ORCA neighbor inputs must be finite")
        lines.append(_orca_line(p, v, q, qv, safety_distance_m, horizon_s, time_step_s, responsibility))
        relative_position = _sub(q, p)
        closing = _sub(preferred, qv)
        closing_sq = _length_sq(closing)
        closest_t = min(horizon_s, max(0.0, _dot(relative_position, closing) / closing_sq)) if closing_sq > 1e-12 else 0.0
        if _length_sq(_sub(relative_position, _scale(closing, closest_t))) <= safety_distance_m ** 2:
            predicted_encounter = True

    preferred_length = math.sqrt(_length_sq(preferred))
    if predicted_encounter and passing_bias_mps > 0 and preferred_length > 1e-9:
        bias = min(passing_bias_mps, preferred_length * 0.2)
        clockwise = (-preferred[1] / preferred_length, preferred[0] / preferred_length)
        preferred = _add(preferred, _scale(clockwise, bias))

    def feasible(candidate: Vector2) -> bool:
        if _length_sq(candidate) > max_speed_mps * max_speed_mps + 1e-8:
            return False
        return all(_det(direction, _sub(point, candidate)) <= 1e-7 for point, direction in lines)

    candidates: list[Vector2] = []
    preferred_speed = math.sqrt(_length_sq(preferred))
    candidates.append(
        _scale(preferred, min(1.0, max_speed_mps / preferred_speed))
        if preferred_speed > 1e-12 else (0.0, 0.0)
    )
    for point, direction in lines:
        # Orthogonal projection onto the ORCA boundary.
        candidates.append(_add(point, _scale(direction, _dot(_sub(preferred, point), direction))))
        # Boundary intersections with the maximum-speed circle.
        # The closest point on this line to the origin is perpendicular to
        # its direction. A line outside the circle has no intersections.
        center = _sub(point, _scale(direction, _dot(point, direction)))
        discriminant = max_speed_mps * max_speed_mps - _length_sq(center)
        if discriminant < -1e-10:
            continue
        discriminant = max(0.0, discriminant)
        offset = math.sqrt(discriminant)
        candidates.extend((_add(center, _scale(direction, offset)), _sub(center, _scale(direction, offset))))

    # Intersections of pairs of ORCA boundaries.
    for index, (point_a, direction_a) in enumerate(lines):
        for point_b, direction_b in lines[index + 1:]:
            denominator = _det(direction_a, direction_b)
            if abs(denominator) <= 1e-10:
                continue
            distance = _det(_sub(point_b, point_a), direction_b) / denominator
            candidates.append(_add(point_a, _scale(direction_a, distance)))

    valid = [candidate for candidate in candidates if feasible(candidate)]
    if not valid:
        return None
    return min(valid, key=lambda candidate: _length_sq(_sub(candidate, preferred)))


def compute_gateway_velocity(*, minimum_distance_m=None, **kwargs) -> Optional[Vector2]:
    """Full-responsibility candidate certified against measured peer velocities.

    A single convex ORCA half-plane can exclude the other side of a velocity
    obstacle. If its full-responsibility solution is infeasible, try a bounded
    polar set in the original (nonconvex) CPA-safe region. This fallback does
    not claim to satisfy reciprocal ORCA half-planes or guarantee convergence.
    Every returned velocity respects speed and finite-horizon measured-peer CPA.
    """
    kwargs = dict(kwargs)
    kwargs['neighbors'] = list(kwargs['neighbors'])
    kwargs['responsibility'] = 1.0
    result = compute_orca_velocity(**kwargs)
    p = tuple(kwargs['position'])
    preferred = tuple(kwargs['preferred_velocity'])
    speed = kwargs['max_speed_mps']
    radius = kwargs['safety_distance_m'] if minimum_distance_m is None else minimum_distance_m
    if not math.isfinite(radius) or not 0 < radius <= kwargs['safety_distance_m']:
        raise ValueError('invalid CPA certification distance')
    horizon = kwargs['horizon_s']

    def safe(candidate):
        if _length_sq(candidate) > speed*speed + 1e-10:
            return False
        for q, qv in kwargs['neighbors']:
            relative = _sub(q, p)
            closing = _sub(candidate, qv)
            v2 = _length_sq(closing)
            t = min(horizon, max(0., _dot(relative, closing)/v2)) if v2 > 1e-12 else 0.
            if _length_sq(_sub(relative, _scale(closing, t))) < radius*radius - 1e-10:
                return False
        return True

    if result is not None and safe(result):
        return result
    length = math.hypot(*preferred)
    objective = preferred
    if length > 1e-9:
        bias = min(kwargs.get('passing_bias_mps', 0.), length*.2)
        objective = (preferred[0]-preferred[1]/length*bias, preferred[1]+preferred[0]/length*bias)
    candidates = [(0.,0.), tuple(kwargs['velocity'])]
    if length > 1e-12:
        candidates.append(_scale(preferred, min(1.,speed/length)))
    for fraction in (.25,.5,.75,1.):
        for index in range(72):
            angle = index*2*math.pi/72
            candidates.append((speed*fraction*math.cos(angle), speed*fraction*math.sin(angle)))
    valid = [v for v in candidates if safe(v)]
    return min(valid, key=lambda v:_length_sq(_sub(v, objective))) if valid else None

"""Rate-limited, serialized formation changes using observed fleet positions."""

from __future__ import annotations

import itertools
import heapq
import math
import time

from coordination.navigation import segment_clear


def _local(point, leader):
    dx = point[0] - leader.position_ned_m[0]
    dy = point[1] - leader.position_ned_m[1]
    c, s = math.cos(leader.yaw_rad), math.sin(leader.yaw_rad)
    return (dx * c + dy * s, -dx * s + dy * c)


def _world(offset, leader):
    c, s = math.cos(leader.yaw_rad), math.sin(leader.yaw_rad)
    x, y, z = leader.position_ned_m
    return (x + offset[0] * c - offset[1] * s,
            y + offset[0] * s + offset[1] * c, z)


def _point_segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = max(0., min(1., ((point[0] - a[0]) * dx +
                             (point[1] - a[1]) * dy) / denom)) if denom else 0.
    return math.dist(point[:2], (a[0] + t * dx, a[1] + t * dy))


def _box_clear(a, b, box, margin):
    lo, hi = box['min_ned_m'], box['max_ned_m']
    if not (lo[2] - margin <= a[2] <= hi[2] + margin):
        return True
    lower, upper = 0., 1.
    for i in (0, 1):
        start, delta = a[i], b[i] - a[i]
        minimum, maximum = lo[i] - margin, hi[i] + margin
        if abs(delta) < 1e-12:
            if minimum <= start <= maximum:
                continue
            return True
        near, far = (minimum - start) / delta, (maximum - start) / delta
        lower, upper = max(lower, min(near, far)), min(upper, max(near, far))
        if lower > upper:
            return True
    return False


class FormationTransition:
    """Move one slot at a time; completion needs measured dwell, not an ACK."""

    def __init__(self, *, safety_distance_m, max_speed_mps, tolerance_m,
                 dwell_s=1., min_switch_interval_s=2.):
        self.separation = safety_distance_m
        self.speed = max_speed_mps
        self.tolerance = tolerance_m
        self.dwell = dwell_s
        self.interval = min_switch_interval_s
        self.reset()

    def reset(self):
        self.shape = None
        self.assignment = {}
        self.offsets = {}
        self.started_at = None
        self.last_at = None
        self.stable_since = None
        self.last_switch = -math.inf
        self.was_obstructed = False
        self.recovering = False
        self.phase = "nominal"
        self.reason = ""

    def _assign(self, snapshot, slots, leader):
        names = tuple(slots)
        destinations = tuple(_local(slots[n], leader) for n in names)
        starts = tuple(self.offsets[n] for n in names)
        best = None
        for order in itertools.permutations(range(len(names))):
            cost = sum(math.dist(starts[i], destinations[order[i]]) for i in range(len(names)))
            # Penalize assignments whose straight paths cross; execution also
            # checks separation one step at a time.
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = starts[i], destinations[order[i]]
                    c, d = starts[j], destinations[order[j]]
                    if (_point_segment_distance(c, a, b) < self.separation and
                            _point_segment_distance(a, c, d) < self.separation):
                        cost += self.separation
            candidate = (cost, order)
            if best is None or candidate < best:
                best = candidate
        self.assignment = {name: names[best[1][i]] for i, name in enumerate(names)}

    def assigned_targets(self, nominal):
        return {name: nominal[self.assignment.get(name, name)] for name in nominal}

    def _detour(self, start, goal, safe):
        """Bounded local search; each returned edge has passed the same gate."""
        h = .5
        margin = 8.
        deadline = time.monotonic() + .02
        low = (min(start[0], goal[0]) - margin, min(start[1], goal[1]) - margin)
        high = (max(start[0], goal[0]) + margin, max(start[1], goal[1]) + margin)
        queue = [(math.dist(start, goal), 0., (0, 0))]
        cost = {(0, 0): 0.}
        parent = {}
        expanded = 0
        while queue and expanded < 1200 and time.monotonic() < deadline:
            _, old_cost, cell = heapq.heappop(queue)
            if old_cost != cost.get(cell):
                continue
            position = (start[0] + cell[0] * h, start[1] + cell[1] * h)
            if math.dist(position, goal) <= h and safe(position, goal):
                path = [goal, position]
                while cell in parent:
                    cell = parent[cell]
                    path.append((start[0] + cell[0] * h, start[1] + cell[1] * h))
                path.reverse()
                return path[1:]
            expanded += 1
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1),
                           (1,1), (1,-1), (-1,1), (-1,-1)):
                other = (cell[0] + dx, cell[1] + dy)
                candidate = (start[0] + other[0] * h, start[1] + other[1] * h)
                if (not low[0] <= candidate[0] <= high[0]
                        or not low[1] <= candidate[1] <= high[1]
                        or not safe(position, candidate)):
                    continue
                new_cost = old_cost + h * math.hypot(dx, dy)
                if new_cost < cost.get(other, math.inf):
                    cost[other] = new_cost
                    parent[other] = cell
                    heapq.heappush(queue, (new_cost + math.dist(candidate, goal),
                                           new_cost, other))
        return []

    def targets(self, snapshot, nominal, *, shape, now, obstacle_points=(),
                obstacle_clearance_m=0., bounds_xy=None, map_obstacles=(),
                obstructed=False, trail_targets=None):
        leader = snapshot.states[next(name for name in snapshot.states if name not in nominal)]
        names = tuple(nominal)
        if self.shape is None:
            self.shape = shape
            self.assignment = {name: name for name in names}
            self.offsets = {name: _local(nominal[name], leader) for name in names}
            self.last_at = now
        if obstructed:
            self.was_obstructed = True
            self.stable_since = None
            if trail_targets is not None:
                self.offsets = {name: _local(trail_targets[name], leader) for name in names}
            self.phase, self.reason = "deforming", "obstacle_constrained"
            self.last_at = now
            return {name: _world(self.offsets[name], leader) for name in names}
        if self.was_obstructed:
            self.was_obstructed = False
            self.recovering = True
            self.started_at = now
            self.stable_since = None
        if shape != self.shape:
            if now - self.last_switch >= self.interval:
                self._assign(snapshot, nominal, leader)
                self.shape = shape
                self.started_at = now
                self.last_switch = now
                self.stable_since = None
                self.recovering = False
            else:
                self.phase, self.reason = "switching", "switch_interval"
                self.last_at = now
                return {name: _world(self.offsets[name], leader) for name in names}
        desired = {name: _local(nominal[self.assignment[name]], leader) for name in names}
        if self.started_at is None and shape == self.shape:
            self.phase, self.reason = "nominal", ""
            self.offsets = desired
            self.last_at = now
            return {name: _world(desired[name], leader) for name in names}
        dt = max(0., min(now - self.last_at, .5)) if self.last_at is not None else 0.
        self.last_at = now
        pending = sorted(names, key=lambda n: math.dist(self.offsets[n], desired[n]))
        failure = ""
        for name in pending:
            old, goal = self.offsets[name], desired[name]
            distance = math.dist(old, goal)
            if distance <= 1e-6:
                continue
            current = _world(old, leader)
            observed = snapshot.states[name].position_ned_m
            if math.dist(current, observed) > max(1., 2 * self.tolerance):
                failure = "waiting_for_slot_tracking"
                continue
            peers = [leader.position_ned_m]
            peers += [snapshot.states[n].position_ned_m for n in names if n != name]
            peers += [_world(self.offsets[n], leader) for n in names if n != name]
            def safe(a, b):
                begin, end = _world(a, leader), _world(b, leader)
                return (bounds_xy is None or
                        (bounds_xy[0] <= end[0] <= bounds_xy[1] and
                         bounds_xy[2] <= end[1] <= bounds_xy[3])) and (
                        segment_clear(begin, end, obstacle_points, obstacle_clearance_m)
                    and all(_box_clear(begin, end, box, obstacle_clearance_m)
                            for box in map_obstacles)
                    and all(_point_segment_distance(peer, begin, end) >= self.separation
                            for peer in peers))
            goal_world = _world(goal, leader)
            if bounds_xy is not None and not (bounds_xy[0] <= goal_world[0] <= bounds_xy[1]
                                               and bounds_xy[2] <= goal_world[1] <= bounds_xy[3]):
                failure = "slot_out_of_bounds"
                continue
            if not safe(goal, goal):
                failure = "target_slot_blocked"
                continue
            step = min(distance, self.speed * dt)
            if step <= 0:
                continue
            candidate = tuple(old[i] + (goal[i] - old[i]) * step / distance for i in range(2))
            if not safe(old, candidate):
                route = self._detour(old, goal, safe)
                if not route:
                    failure = "transition_path_unavailable"
                    continue
                waypoint = route[0]
                leg = math.dist(old, waypoint)
                candidate = tuple(old[i] + (waypoint[i] - old[i]) * min(step, leg) / leg
                                  for i in range(2))
                if not safe(old, candidate):
                    failure = "transition_path_unavailable"
                    continue
            self.offsets[name] = candidate
            break
        targets = {name: _world(self.offsets[name], leader) for name in names}
        assigned = {name: _world(desired[name], leader) for name in names}
        converged = all(math.dist(targets[name], assigned[name]) <= self.tolerance
                        and math.dist(snapshot.states[name].position_ned_m, assigned[name]) <= self.tolerance
                        and math.dist(snapshot.states[name].linear_velocity_ned_mps,
                                      leader.linear_velocity_ned_mps) <= .2 for name in names)
        if converged:
            if self.stable_since is None:
                self.stable_since = now
            if now - self.stable_since >= self.dwell:
                self.started_at = None
                self.recovering = False
                self.offsets = desired
                self.phase, self.reason = "nominal", ""
                return assigned
        else:
            self.stable_since = None
        self.phase = "recovering" if self.recovering else "switching"
        self.reason = (failure or "verifying_dwell") if converged else failure
        return targets

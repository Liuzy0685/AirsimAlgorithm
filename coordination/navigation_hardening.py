"""Opt-in local-navigation policies. No simulator access or safety authority.

These heuristics may improve measured behavior, not guarantee convergence or
performance. Their output must pass the existing raw-LiDAR and peer gateway.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HardeningConfig:
    enabled: bool = False
    memory: bool = True
    recovery: bool = True
    smoothing: bool = True
    # None preserves existing YAML behavior: smoothing used to enable both.
    latency_limit: bool | None = None
    passage: bool = True
    diagnostics: bool = True
    free_confirmations: int = 3
    max_memory_cells: int = 8000
    clearing_rays: int = 48
    stall_window_s: float = 6.
    stall_displacement_m: float = .20
    stall_progress_m: float = .15
    failure_ttl_s: float = 30.
    acceleration_mps2: float = .35
    braking_mps2: float = .35
    passage_wait_s: float = 8.

    def __post_init__(self):
        for name in ('enabled', 'memory', 'recovery', 'smoothing', 'passage', 'diagnostics'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.latency_limit is not None and type(self.latency_limit) is not bool:
            raise ValueError('latency_limit must be boolean or null')
        for name, lower, upper in (('free_confirmations', 2, 10), ('max_memory_cells', 100, 8000),
                                   ('clearing_rays', 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'invalid {name}')

        for name, lower, upper in (('stall_window_s', 2., 30.), ('stall_displacement_m', .05, 1.),
                ('stall_progress_m', .05, 1.), ('failure_ttl_s', 5., 120.),
                ('acceleration_mps2', .05, 2.), ('braking_mps2', .05, 2.), ('passage_wait_s', 2., 30.)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'invalid {name}')

    @property
    def latency_limit_enabled(self):
        return self.smoothing if self.latency_limit is None else self.latency_limit


def stop_category(reason):
    text = str(reason).lower()
    for words, category in (
        (('cleanup_unverified', 'manual_intervention', 'no_verified_cleanup'), 'cleanup_unverified'),
        (('lidar_unavailable', 'lidar_sim_time', 'perception_stale', 'invalid_lidar'), 'perception_unavailable'),
        (('snapshot_unavailable', 'authoritative_snapshot'), 'state_unavailable'),
        (('formation_expired', 'formation_wait', 'waiting_for_current', 'waiting_for_command'), 'command_unavailable'),
        (('leader_wait',), 'formation_wait'),
        (('passage_wait', 'passage_timeout', 'passage_retreat', 'inter_uav_yield', 'orca_fallback_hold', 'inter_vehicle_conflict'), 'traffic_wait'),
        (('lidar_clearance', 'obstacle', 'navigation_blocked', 'smooth_blocked'), 'obstacle_blocked'),
        (('controller_lease', 'vehicle_lock', 'unauthorized'), 'control_authority'),
    ):
        if any(word in text for word in words):
            return category
    return 'other'


class RuntimeDiagnostics:
    """Bounded observations; never renew sensor timestamps or control permits."""
    def __init__(self, capacity=120):
        self.samples = deque(maxlen=capacity)
        self.reason = ''

    def record(self, reason='', **values):
        self.reason = reason or self.reason
        self.samples.append({k: float(v) for k, v in values.items()
            if type(v) in (int, float) and math.isfinite(v) and v >= 0})

    def summary(self):
        keys = sorted({k for row in self.samples for k in row})
        result = {}
        for key in keys:
            values = sorted(row[key] for row in self.samples if key in row)
            result[key] = dict(p50=values[(len(values)-1)//2],
                              p95=values[math.ceil(.95*len(values))-1], maximum=values[-1])
        return dict(reason=self.reason, category=stop_category(self.reason), samples=len(self.samples), timing=result)


class ObservedStaticMemory:
    """Bounded observed occupancy; age alone is not evidence of free space.

    Only measured rays passing close to a cell center may clear it after
    several distinct scans. Peer regions are temporary, not persistent walls.
    This is mapping using provided poses, not SLAM or semantic classification.
    """
    def __init__(self, resolution, config):
        self.h, self.cfg = resolution, config
        self.cells, self.free_votes, self.dynamic = {}, {}, {}
        self.last_scan = -1

    def observe(self, points, now, *, origin=None, scan_id=None, dynamic_regions=(), clearing_points=None):
        if scan_id is not None:
            if type(scan_id) is not int or scan_id <= self.last_scan:
                return
            self.last_scan = scan_id
        points = np.asarray(points, float).reshape(-1, 2)
        if not np.all(np.isfinite(points)):
            raise ValueError('nonfinite observed memory points')
        if origin is not None and not all(math.isfinite(v) for v in origin[:2]):
            raise ValueError('invalid ray origin')
        regions = [(float(x), float(y), float(r)) for x, y, r in dynamic_regions]
        if any(not all(math.isfinite(v) for v in region) or region[2] <= 0 for region in regions):
            raise ValueError('invalid dynamic region')
        def temporary(x, y):
            return any(math.hypot(x-a, y-b) <= r + self.h for a, b, r in regions)
        occupied = set()
        for x, y in np.unique(np.rint(points / self.h).astype(int), axis=0):
            key = (int(x), int(y))
            if temporary(x*self.h, y*self.h):
                self.dynamic[key] = now
            else:
                self.cells[key] = now
                self.free_votes.pop(key, None)
            occupied.add(key)
        # Limit ray work independently of raw cloud size. Do not create rays
        # from interpolated endpoints or treat lack of returns as free space.
        free = set()
        rays = points if clearing_points is None else np.asarray(clearing_points,float).reshape(-1,2)
        if not np.all(np.isfinite(rays)):
            raise ValueError('nonfinite clearing ray')
        if origin is not None and len(rays):
            start = np.asarray(origin[:2], float)
            count = min(self.cfg.clearing_rays, len(rays))
            for index in np.linspace(0, len(rays)-1, count, dtype=int):
                end = rays[index]
                delta = end - start
                length = float(np.linalg.norm(delta))
                if length <= 3*self.h:
                    continue
                # Sub-cell sampling follows a real measured ray. Require a
                # center-near pass, and leave two cells before its hit untouched.
                samples=start+np.arange(self.h,length-2*self.h,self.h/2)[:,None]*delta/length
                keys=np.unique(np.rint(samples/self.h).astype(int),axis=0)
                for ix,iy in keys:
                    key=(int(ix),int(iy))
                    if key not in self.cells or key in occupied:
                        continue
                    cx,cy=ix*self.h,iy*self.h
                    projection=((cx-start[0])*delta[0]+(cy-start[1])*delta[1])/(length*length)
                    error=math.hypot(cx-start[0]-projection*delta[0],cy-start[1]-projection*delta[1])
                    if error <= .2*self.h and not temporary(cx,cy):
                        free.add(key)
        for key in free:
            self.free_votes[key] = self.free_votes.get(key, 0) + 1
            if self.free_votes[key] >= self.cfg.free_confirmations:
                self.cells.pop(key, None)
                self.free_votes.pop(key, None)
        self.dynamic = {k:t for k,t in self.dynamic.items() if 0 <= now-t < .6}
        if len(self.cells) > self.cfg.max_memory_cells:
            keep = sorted(self.cells, key=self.cells.get, reverse=True)[:self.cfg.max_memory_cells]
            self.cells = {k:self.cells[k] for k in keep}
            self.free_votes = {k:v for k,v in self.free_votes.items() if k in self.cells}

    def points(self, now):
        self.dynamic = {k:t for k,t in self.dynamic.items() if 0 <= now-t < .6}
        return np.asarray([(x*self.h, y*self.h) for x,y in self.cells.keys() | self.dynamic.keys()], float).reshape(-1,2)


class ProgressGuard:
    def __init__(self, cfg):
        self.cfg, self.history, self.goal = cfg, deque(maxlen=256), None
        self.last_stall = -math.inf
        self.reason = ''

    def update(self, now, position, goal, *, expected, reason='', commanded_speed=0.):
        category = stop_category(reason)
        if (not expected or category in ('traffic_wait', 'formation_wait', 'command_unavailable',
                                         'perception_unavailable', 'state_unavailable', 'control_authority')):
            self.history.clear()
            self.reason = 'paused_' + category
            return False
        if self.goal is not None and math.dist(goal[:2], self.goal[:2]) > 1.:
            self.history.clear()
        self.goal = tuple(goal)
        # Bound memory without truncating a long configured window at high Hz.
        if self.history and now-self.history[-1][0] < max(.05,self.cfg.stall_window_s/200):
            return False
        self.history.append((now, tuple(position[:2]), math.dist(position[:2], goal[:2])))
        while len(self.history) > 1 and now-self.history[1][0] >= self.cfg.stall_window_s:
            self.history.popleft()
        if not self.history or now-self.history[0][0] < self.cfg.stall_window_s:
            return False
        first = self.history[0]
        displacement = max(math.dist(p, first[1]) for _,p,_ in self.history)
        progress = first[2] - self.history[-1][2]
        stuck = (displacement < self.cfg.stall_displacement_m and progress < self.cfg.stall_progress_m
                 and now-self.last_stall >= self.cfg.stall_window_s)
        if stuck:
            self.last_stall = now
            self.reason = 'command_not_executed' if commanded_speed > .05 else 'no_route_progress'
            self.history.clear()
        return stuck


def latency_speed_cap(clearance, hard_clearance, age, acceleration, max_speed):
    """Conservative clearance heuristic: v*age + v²/(2a) <= free margin.

    It is not an observed-free-space certificate or a full braking model.
    """
    if not math.isfinite(age) or age < 0 or math.isnan(clearance):
        return 0.
    margin = max(0., clearance-hard_clearance-.1)
    return min(max_speed, max(0., -acceleration*age + math.sqrt((acceleration*age)**2 + 2*acceleration*margin)))


class MotionLimiter:
    def __init__(self, acceleration):
        self.acceleration = acceleration
        self.last_at = None
        self.last_command = None

    def accept(self, velocity):
        """Remember the velocity actually accepted after CPA / gateway filters."""
        values=tuple(float(v) for v in velocity)
        if len(values)!=2 or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid accepted motion')
        self.last_command=values

    def limit(self, desired, measured, now):
        # A requested stop / safety override is never delayed by smoothing.
        desired, measured = np.asarray(desired, float), np.asarray(measured, float)
        if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(measured)):
            raise ValueError('nonfinite motion')
        gap=None if self.last_at is None else now-self.last_at
        dt = .05 if gap is None else max(0., min(.2, gap))
        # Rate-limit commands, not the tracking error on every control tick.
        # Reanchoring on measured speed each tick creates a false steady speed
        # ceiling when the vehicle tracks commands with ordinary lag/bias.
        reference=measured if self.last_command is None or gap is None or gap>.3 or gap<0 else np.asarray(self.last_command)
        self.last_at = now
        if np.linalg.norm(desired) < 1e-9:
            self.last_command=(0.,0.)
            return (0., 0.)
        delta = desired - reference
        magnitude = float(np.linalg.norm(delta))
        step = self.acceleration * dt
        if magnitude > step:
            delta *= step / magnitude
        result = reference + delta
        # Smoothing must not exceed the requested speed cap.
        length, cap = float(np.linalg.norm(result)), float(np.linalg.norm(desired))
        if length > cap:
            result *= cap/length
        self.last_command=tuple(float(v) for v in result)
        return self.last_command


class PassageAdvisor:
    """Stable total-order right of way, not a distributed reservation proof.

    Applied only in a locally narrow passage. Final CPA checks remain required.
    Timeout requests a verified retreat; it never forces a grant or clears an
    occupied resource. Stale/missing state is handled by the authoritative link.
    """
    def __init__(self, cfg, own_name, priorities):
        self.cfg, self.name, self.priorities = cfg, own_name, priorities
        self.wait_since = None
        self.state, self.peer = 'CLEAR', None

    def update(self, now, position, waypoint, peers, *, narrow, safety_distance):
        if not narrow:
            self.state = 'EXITING' if self.state not in ('CLEAR', 'EXITING') else 'CLEAR'
            self.wait_since, self.peer = None, None
            return False, False
        delta = np.asarray(waypoint[:2]) - position[:2]
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return False, False
        direction = delta/length
        conflicts = []
        for name, state in peers.items():
            point,velocity=state[:2]
            rel = np.asarray(point[:2]) - position[:2]
            along, across = float(rel@direction), abs(float(rel[0]*direction[1]-rel[1]*direction[0]))
            if 0 < along < max(4., safety_distance*2) and across < safety_distance+.3:
                # A stationary leader waiting for its followers is not an
                # oncoming vehicle. Use a fresh navigation intent if stopped;
                # otherwise leave unknown parked peers to existing CPA/staging.
                approach=float(np.asarray(velocity[:2])@direction)<-.03
                if len(state)>2 and state[2] is not None:
                    intent=np.asarray(state[2][:2])-point[:2]
                    norm=float(np.linalg.norm(intent))
                    if norm>.2:approach=approach or float(intent@direction)/norm<-.25
                elif self.peer==name:
                    approach=True
                if approach:
                    conflicts.append(name)
        if not conflicts:
            self.state, self.wait_since, self.peer = 'EXITING', None, None
            return False, False
        rank = lambda name: (-self.priorities.get(name, 0.), name)
        winner = min([self.name, *conflicts], key=rank)
        if winner == self.name:
            self.state, self.wait_since, self.peer = 'ENTERING', None, None
            if any(math.dist(position[:2], peers[n][0][:2]) < safety_distance*1.5 for n in conflicts):
                self.state = 'OCCUPYING'
            return False, False
        self.peer = winner
        if self.wait_since is None:
            self.wait_since = now
        expired = now-self.wait_since >= self.cfg.passage_wait_s
        self.state = 'BLOCKED_TIMEOUT' if expired else 'YIELDING'
        return True, expired

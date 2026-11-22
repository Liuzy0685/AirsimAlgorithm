"""Contracts for the local operator API; no flight-control calls."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid


class OperatorError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


def require(condition, code, message, status=400):
    if not condition:
        raise OperatorError(code, message, status)


def number(value, label, *, minimum=-10000., maximum=10000.):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, 'invalid_number', f'Invalid {label}')
    return float(value)


def identifier(value, label='request_id'):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value),
            'invalid_identifier', f'Invalid {label}')
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(canonical(value), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class OperatorMap:
    """Explicit partial map. Bounds are an editing extent, never free-space proof."""
    def __init__(self, value):
        self.value = json.loads(canonical(value))
        identifier(value.get('map_id'), 'map_id')
        self.asset_path = value.get('asset_path', '/Game/' + value['map_id'])
        require(isinstance(self.asset_path, str) and re.fullmatch(
            r'/Game/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', self.asset_path),
            'invalid_map', 'Need a project /Game asset path without traversal')
        require(value.get('frame') == 'world_ned' and value.get('units') == 'm',
                'invalid_map', 'Map must use World NED meters')
        b = value.get('bounds_xy_m', [])
        require(isinstance(b, list) and len(b) == 4, 'invalid_map', 'Need XY editing bounds')
        self.bounds = [number(v, 'map bounds') for v in b]
        require(b[0] < b[1] and b[2] < b[3], 'invalid_map', 'Invalid map bounds')
        obstacles = value.get('obstacles', [])
        require(isinstance(obstacles, list) and len(obstacles) <= 2000,
                'invalid_map', 'Invalid obstacle collection')
        for item in obstacles:
            require(item.get('shape') == 'aabb', 'invalid_map', 'Only measured AABBs supported')
            lo, hi = item.get('min_ned_m', []), item.get('max_ned_m', [])
            require(len(lo) == len(hi) == 3, 'invalid_map', 'Invalid obstacle bounds')
            for v in (*lo, *hi):
                number(v, 'obstacle bounds')
            require(all(a <= b for a, b in zip(lo, hi)), 'invalid_map', 'Inverted obstacle bounds')
        self.revision = hashlib.sha256(canonical(value).encode()).hexdigest()

    def public(self):
        return dict(self.value, asset_path=self.asset_path, map_revision=self.revision,
                    bounds_semantics='editing_extent_not_free_space', collision_certified=False)

    def asset_file(self, project):
        content = (Path(project).resolve().parent / 'Content').resolve()
        asset = (content / (self.asset_path[len('/Game/'):] + '.umap')).resolve()
        require(asset.is_relative_to(content), 'invalid_map', 'Asset must stay inside project Content')
        return asset

    def check_point(self, point, clearance):
        x, y, z = point
        lo_x, hi_x, lo_y, hi_y = self.bounds
        require(lo_x <= x <= hi_x and lo_y <= y <= hi_y,
                'goal_out_of_bounds', 'Target is outside the map editing extent', 422)
        for item in self.value.get('obstacles', []):
            lo, hi = item['min_ned_m'], item['max_ned_m']
            if lo[2] - clearance <= z <= hi[2] + clearance:
                dx = max(lo[0] - x, 0., x - hi[0])
                dy = max(lo[1] - y, 0., y - hi[1])
                require(math.hypot(dx, dy) >= clearance, 'goal_near_known_obstacle',
                        f'Target is too close to known obstacle {item.get("id", "")}', 422)


class RuntimeDirectoryLock:
    """Prevent two API supervisors from owning the same durable runtime directory."""
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.handle = (Path(directory) / 'operator.lock').open('a+b')
        self.handle.seek(0)
        self.handle.write(b'0')
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise OperatorError('operator_already_running', 'Runtime directory is already owned', 409)

    def close(self):
        self.handle.close()

"""
Sector measurement data model.

An immutable snapshot of one spatial sector after point-cloud filtering
and distance estimation.  All angles are in radians.

ROUND 3.3: Added FOV observability fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorMeasurement:
    """Single-sector distance measurement.

    Attributes:
        name: Sector name (e.g. ``"front"``, ``"frontLeft"``).
        distance_m: Estimated obstacle distance in metres.
            Set to ``max_range_m`` when ``has_return`` is ``False``.
        point_count: Number of filtered points that fell inside this sector.
        has_return: ``True`` if at least ``min_points`` points were present
            and a distance estimate was computed.
        azimuth_min_rad: Lower azimuth bound (radians).
        azimuth_max_rad: Upper azimuth bound (radians).
        elevation_min_rad: Lower elevation bound (radians).
        elevation_max_rad: Upper elevation bound (radians).
        max_range_m: Sensor or sector maximum range (metres).
        min_points: Minimum points required to consider ``has_return=True``.
        observable_by_fov: ``True`` if this sector is fully covered by the
            LiDAR FOV (both horizontal and vertical).
        fov_coverage_fraction: Fraction [0.0, 1.0] of the sector's solid-angle
            area that is covered by the LiDAR FOV.  1.0 = fully observable,
            0.0 = unobservable, 0.x = partially observable.
    """

    name: str
    distance_m: float
    point_count: int
    has_return: bool
    azimuth_min_rad: float
    azimuth_max_rad: float
    elevation_min_rad: float
    elevation_max_rad: float
    max_range_m: float
    min_points: int
    observable_by_fov: bool = False
    fov_coverage_fraction: float = 0.0

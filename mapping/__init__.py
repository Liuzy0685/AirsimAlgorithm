"""Mapping package — persistent occupancy grid and local distance field.

This package provides the geometric map-memory layer for trajectory-centric
local planning.  It does **not** perform SLAM: pose comes from AirSim ground
truth, so only mapping (not localisation) is required.
"""
from mapping.occupancy_grid import (
    OccupancyGridMap,
    OccupancyGridParams,
    UNKNOWN,
    FREE,
    OCCUPIED,
)
from mapping.distance_field import DistanceField

__all__ = [
    "OccupancyGridMap",
    "OccupancyGridParams",
    "DistanceField",
    "UNKNOWN",
    "FREE",
    "OCCUPIED",
]

"""
Point-cloud to spatial-sector conversion.

Converts a filtered SensorLocalFrame point cloud into discrete spatial
sectors defined in ``perception.yaml``.  Does NOT use world yaw — the
SensorLocalFrame +X axis IS the drone forward direction.

Angle definitions
-----------------
azimuth_rad = atan2(y, x)      0° = front, +90° = right, -90° = left
elevation_rad = atan2(-z, sqrt(x² + y²))   positive = up, negative = down

Sector boundaries (ROUND 3.1)
-----------------------------
Horizontal sectors use half-open intervals **[min, max)** so that
boundary points fall into exactly one sector.  Vertical overlap
(e.g. up ∩ frontUp) is intentional — documented in perception.yaml.

ROUND 3.3 changes
-----------------
- Accepts ``Sequence[SectorDef]`` (validated objects) instead of
  ``List[dict]`` — no second unvalidated config path.
- Float32/float64 boundary snapping for deterministic half-open azimuth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from models.sector_measurement import SectorMeasurement
from models.directional_distances import DirectionalDistances
from perception.perception_config import SectorDef

# ---------------------------------------------------------------------------
# Distance strategies
# ---------------------------------------------------------------------------


def _distance_minimum(_pts: np.ndarray, distance: np.ndarray) -> float:
    return float(np.min(distance)) if distance.size > 0 else float("inf")


def _distance_nearest_k_median(
    _pts: np.ndarray, distance: np.ndarray, k: int = 3
) -> float:
    if distance.size == 0:
        return float("inf")
    top_k = np.partition(distance, min(k - 1, distance.size - 1))[:k]
    return float(np.median(top_k))


def _distance_percentile(
    _pts: np.ndarray, distance: np.ndarray, percentile: float = 10.0
) -> float:
    if distance.size == 0:
        return float("inf")
    return float(np.percentile(distance, percentile))


_DISTANCE_STRATEGIES = {
    "minimum": _distance_minimum,
    "nearest_k_median": _distance_nearest_k_median,
    "percentile": _distance_percentile,
}

# ---------------------------------------------------------------------------
# Angle helpers (ROUND 3.1 — half-open azimuth)
# ---------------------------------------------------------------------------


def _normalize_angle_rad(angle_rad: float) -> float:
    """Wrap a scalar angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _is_full_circle(a_min_rad: float, a_max_rad: float) -> bool:
    """True when the raw azimuth range covers the full circle."""
    return (a_max_rad - a_min_rad) >= (2.0 * math.pi - 1e-9)


# Tiny tolerance — applied ONLY to one side so adjacent sectors
# do not both expand and double-count boundary points.
_EPS = 1e-9  # rad (~ 5.7e-8 deg)

# ROUND 3.3: Snapping tolerance for float32 boundary points.
# When an azimuth angle is within this tolerance of a named sector
# boundary, it is snapped to the exact boundary before half-open
# membership is evaluated.  This prevents float32 quantization from
# placing a boundary point in the wrong sector.
_SNAP_TOLERANCE_RAD = 1e-6  # ~ 5.7e-5 deg


def _collect_boundaries_rad(sector_defs: Sequence[SectorDef]) -> np.ndarray:
    """Collect all unique azimuth boundaries from sector definitions.

    Returns a sorted array of boundary angles in radians, in [-π, π).
    Both min and max of each sector are collected.  +180° is unified to -π.

    Boundaries are NOT rounded — they must be exact so that snapped
    points align with the half-open interval boundaries.
    """
    boundaries: List[float] = []
    for s in sector_defs:
        a_min = _normalize_angle_rad(math.radians(s.azimuth_min_deg))
        a_max = _normalize_angle_rad(math.radians(s.azimuth_max_deg))
        boundaries.append(a_min)
        boundaries.append(a_max)

    # Sort and deduplicate with tolerance (float64-safe: _SNAP_TOLERANCE_RAD ≈ 1e-6)
    sorted_bounds = sorted(boundaries)
    unique: List[float] = []
    for b in sorted_bounds:
        if not unique or abs(b - unique[-1]) > (_SNAP_TOLERANCE_RAD * 0.1):
            unique.append(b)
    return np.array(unique, dtype=np.float64)


def _snap_to_boundaries(
    azimuth_rad: np.ndarray,
    boundaries_rad: np.ndarray,
) -> np.ndarray:
    """Snap azimuth angles to the nearest named boundary if within tolerance.

    Only snaps when the distance to a boundary is < _SNAP_TOLERANCE_RAD.
    Returns a copy with snapped values; the original is not modified.
    """
    if boundaries_rad.size == 0:
        return azimuth_rad.copy()

    result = _normalize_angle_rad(azimuth_rad).astype(np.float64)

    for b in boundaries_rad:
        diff = _normalize_angle_rad(result - b)
        mask = np.abs(diff) < _SNAP_TOLERANCE_RAD
        result[mask] = b

    return result


def _in_azimuth_half_open(
    azimuth: np.ndarray,
    a_min_rad: float,
    a_max_rad: float,
) -> np.ndarray:
    """Half-open **[min, max)** azimuth membership.

    - Full-circle ranges → all points accepted.
    - Normal sectors: min <= angle < max  (upper bound EXCLUSIVE).
    - Wrap-around sectors: angle >= min OR angle < max.
    - +180° and -180° are both normalised to -π; the back sector
      which wraps [157.5°, -157.5°] catches both.
    """
    if _is_full_circle(a_min_rad, a_max_rad):
        return np.ones_like(azimuth, dtype=bool)

    a_min_norm = _normalize_angle_rad(a_min_rad)
    a_max_norm = _normalize_angle_rad(a_max_rad)
    azi = _normalize_angle_rad(azimuth)

    if a_min_norm <= a_max_norm:
        # Normal range: [min, max)
        return (azi >= a_min_norm) & (azi < a_max_norm)
    else:
        # Wrap-around: [min, pi) ∪ [-pi, max)
        return (azi >= a_min_norm) | (azi < a_max_norm)


def _in_elevation(
    elevation: np.ndarray,
    e_min_rad: float,
    e_max_rad: float,
) -> np.ndarray:
    """Elevation membership — closed interval [min, max] for both sides.

    Vertical overlap (up ∩ frontUp etc.) is by design.
    """
    return (elevation >= e_min_rad - _EPS) & (elevation <= e_max_rad + _EPS)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def pointcloud_to_directional_distances(
    filtered_points_sensor: np.ndarray,
    sector_defs: Sequence[SectorDef],
    default_max_range_m: float = 40.0,
    default_min_points: int = 3,
    distance_strategy: str = "nearest_k_median",
    nearest_k: int = 3,
    percentile: float = 10.0,
    frame_valid: bool = True,
    invalid_reason: Optional[str] = None,
    raw_timestamp_ns: int = 0,
    received_monotonic_seconds: float = 0.0,
    fov_compatible: bool = False,
    fov_invalid_sectors: tuple = (),
    fov_observability: Optional[Dict[str, Tuple[bool, float]]] = None,
) -> DirectionalDistances:
    """Convert a filtered SensorLocalFrame point cloud into directional distances.

    ROUND 3.1: If ``frame_valid=True`` but the filtered point cloud is empty,
    returns ``frame_valid=False, invalid_reason="empty_filtered_pointcloud"``.

    ROUND 3.3:
    - ``sector_defs`` is now ``Sequence[SectorDef]`` (already-validated objects).
      No ``float()`` / ``int()`` / ``.get()`` fallback path inside this function.
    - Float32 boundary snapping is applied before sector membership.
    - FOV observability per sector is propagated into each SectorMeasurement.
    """
    if not frame_valid:
        return DirectionalDistances(
            frame_valid=False,
            invalid_reason=invalid_reason,
            raw_timestamp_ns=raw_timestamp_ns,
            received_monotonic_seconds=received_monotonic_seconds,
        )

    # --- Defend against empty input (ROUND 3.1) ---
    if filtered_points_sensor.size == 0 or filtered_points_sensor.shape[0] == 0:
        return DirectionalDistances(
            frame_valid=False,
            invalid_reason="empty_filtered_pointcloud",
            raw_timestamp_ns=raw_timestamp_ns,
            received_monotonic_seconds=received_monotonic_seconds,
        )

    strategy_fn = _DISTANCE_STRATEGIES.get(distance_strategy)
    if strategy_fn is None:
        raise ValueError(f"Unknown distance_strategy: {distance_strategy!r}")

    dist = np.linalg.norm(filtered_points_sensor, axis=1)
    x = filtered_points_sensor[:, 0]
    y = filtered_points_sensor[:, 1]
    z = filtered_points_sensor[:, 2]

    azimuth_raw = np.arctan2(y, x)
    r_xy = np.sqrt(x * x + y * y)
    elevation = np.arctan2(-z, r_xy)

    # ROUND 3.3: Snap angles to sector boundaries before classification
    boundaries = _collect_boundaries_rad(sector_defs)
    azimuth = _snap_to_boundaries(azimuth_raw, boundaries)

    minimum_distance_m = float(np.min(dist)) if dist.size > 0 else float("inf")

    if fov_observability is None:
        fov_observability = {}

    sectors: Dict[str, SectorMeasurement] = {}
    legacy_map: Dict[str, str] = {}
    for sdef in sector_defs:
        name = sdef.name
        legacy_name = sdef.legacy_name
        a_min = math.radians(sdef.azimuth_min_deg)
        a_max = math.radians(sdef.azimuth_max_deg)
        e_min = math.radians(sdef.elevation_min_deg)
        e_max = math.radians(sdef.elevation_max_deg)
        s_max_range = sdef.max_range_m
        s_min_points = sdef.min_points
        s_strategy = sdef.distance_strategy

        in_azi = _in_azimuth_half_open(azimuth, a_min, a_max)
        in_ele = _in_elevation(elevation, e_min, e_max)
        mask = in_azi & in_ele

        sector_dist = dist[mask]
        n_pts = int(np.sum(mask))

        if n_pts >= s_min_points:
            s_fn = _DISTANCE_STRATEGIES.get(s_strategy, strategy_fn)
            if s_strategy == "nearest_k_median":
                est = _distance_nearest_k_median(
                    filtered_points_sensor[mask], sector_dist, nearest_k
                )
            elif s_strategy == "percentile":
                est = _distance_percentile(
                    filtered_points_sensor[mask], sector_dist, percentile
                )
            else:
                est = s_fn(filtered_points_sensor[mask], sector_dist)
            has_return = True
        else:
            est = s_max_range
            has_return = False

        # ROUND 3.3: Look up FOV observability for this sector
        obs_tuple = fov_observability.get(name, (False, 0.0))
        obs_by_fov, fov_frac = obs_tuple[0], obs_tuple[1]

        sectors[name] = SectorMeasurement(
            name=name,
            distance_m=est,
            point_count=n_pts,
            has_return=has_return,
            azimuth_min_rad=a_min,
            azimuth_max_rad=a_max,
            elevation_min_rad=e_min,
            elevation_max_rad=e_max,
            max_range_m=s_max_range,
            min_points=s_min_points,
            observable_by_fov=obs_by_fov,
            fov_coverage_fraction=fov_frac,
        )
        legacy_map[name] = legacy_name

    return DirectionalDistances(
        frame_valid=True,
        invalid_reason=None,
        raw_timestamp_ns=raw_timestamp_ns,
        received_monotonic_seconds=received_monotonic_seconds,
        minimum_distance_m=minimum_distance_m,
        sectors=sectors,
        max_range_m=default_max_range_m,
        legacy_map=legacy_map,
        fov_compatible=fov_compatible,
        fov_invalid_sectors=fov_invalid_sectors,
    )

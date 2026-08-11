"""LiDAR sensor FOV loader and sector-coverage validator. ROUND 3.3.

Validates SensorType, Enabled, DataFrame, zero-width FOV, and
vertical_lower < vertical_upper.  No silent type coercion.

validate_sector_fov_coverage now handles both horizontal azimuth AND
vertical elevation coverage, including ±180° wrap, 360° full circle,
forward-180°, partial coverage, and zero-width intersections.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

from perception.perception_config import PerceptionConfig, SectorDef


@dataclass(frozen=True)
class SensorFov:
    horizontal_start_deg: float
    horizontal_end_deg: float
    vertical_lower_deg: float
    vertical_upper_deg: float
    range_m: float

    @property
    def horizontal_full_circle(self) -> bool:
        span = abs(self.horizontal_end_deg - self.horizontal_start_deg)
        return span >= 359.9

    @property
    def vertical_span_deg(self) -> float:
        return self.vertical_upper_deg - self.vertical_lower_deg


@dataclass
class SectorFovStatus:
    legacy_name: str
    horizontal_coverage_fraction: float   # 0..1
    vertical_coverage_fraction: float     # 0..1
    fully_observable: bool
    partially_observable: bool
    unobservable: bool
    intersection_azimuth_deg: float       # true intersection width in degrees
    intersection_elevation_deg: float
    note: str = ""


# ---------------------------------------------------------------------------
# Helper: wrap an angle to [-180, 180)
# ---------------------------------------------------------------------------

def _wrap_180(deg: float) -> float:
    """Wrap a scalar angle in degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def _azimuth_intersection_deg(
    sec_min: float, sec_max: float,
    fov_min: float, fov_max: float,
    fov_full_circle: bool,
) -> Tuple[float, float, float]:
    """Compute the true azimuth intersection width (in degrees) between a
    sector's azimuth range and the LiDAR FOV azimuth range.

    Returns:
        (overlap_lo, overlap_hi, width_deg) where width_deg is the
        positive-area intersection width in [0, 360].
    """
    if fov_full_circle:
        # Full 360°: entire sector azimuth range is covered.
        span = (sec_max - sec_min) % 360.0
        if span < 0:
            span += 360.0
        if span == 0:
            span = 360.0
        return sec_min, sec_max, span

    # Normalise all boundaries to [-180, 180)
    s_min = _wrap_180(sec_min)
    s_max = _wrap_180(sec_max)
    f_min = _wrap_180(fov_min)
    f_max = _wrap_180(fov_max)

    # Sector span (accounting for wrap)
    if s_min <= s_max:
        s_span = s_max - s_min
        s_wraps = False
    else:
        s_span = (s_max + 360.0) - s_min
        s_wraps = True

    # FOV span
    if f_min <= f_max:
        f_span = f_max - f_min
        f_wraps = False
    else:
        f_span = (f_max + 360.0) - f_min
        f_wraps = True

    # Brute-force: sample at 1° resolution along the sector's azimuth span.
    # This handles all wrap cases correctly.
    _STEP = 0.5  # degrees
    steps = max(1, int(s_span / _STEP))
    covered_steps = 0
    for i in range(steps + 1):
        a = s_min + (s_span * i / steps)
        a = _wrap_180(a)

        # Check if a is inside the FOV azimuth range
        if f_wraps:
            in_fov = (a >= f_min) or (a < f_max)
        else:
            in_fov = (a >= f_min) and (a <= f_max)

        if in_fov:
            covered_steps += 1

    fraction = covered_steps / (steps + 1) if (steps + 1) > 0 else 0.0
    width_deg = s_span * fraction

    return s_min, s_max, width_deg


def _elevation_intersection_deg(
    sec_min: float, sec_max: float,
    fov_lower: float, fov_upper: float,
) -> Tuple[float, float, float]:
    """Compute elevation intersection width in degrees.

    Returns:
        (overlap_lo, overlap_hi, width_deg)
    """
    overlap_lo = max(sec_min, fov_lower)
    overlap_hi = min(sec_max, fov_upper)
    width = max(0.0, overlap_hi - overlap_lo)
    return overlap_lo, overlap_hi, width


# ---------------------------------------------------------------------------
# FOV loader with full metadata validation (ROUND 3.3)
# ---------------------------------------------------------------------------


def load_lidar_fov(
    settings_path: Union[str, Path],
    vehicle_name: str,
    lidar_name: str,
) -> SensorFov:
    """Load LiDAR FOV parameters from an AirSim settings.json.

    ROUND 3.3: Also validates:
    - SensorType == 6
    - Enabled is true
    - DataFrame == "SensorLocalFrame"
    - HorizontalFOVStart != HorizontalFOVEnd (no zero-width FOV)
    - vertical_lower < vertical_upper
    - No silent type coercion (bool→int, string→number, etc.)
    """
    with open(settings_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    vehicles = raw.get("Vehicles", {})
    if not isinstance(vehicles, dict):
        raise ValueError("Vehicles must be a dict")
    vehicle = vehicles.get(vehicle_name)
    if not isinstance(vehicle, dict):
        raise ValueError(f"Vehicle {vehicle_name!r} not found in settings")

    sensors = vehicle.get("Sensors", {})
    if not isinstance(sensors, dict):
        raise ValueError(f"Sensors must be a dict for {vehicle_name}")
    sensor = sensors.get(lidar_name)
    if not isinstance(sensor, dict):
        raise ValueError(f"LiDAR {lidar_name!r} not found for {vehicle_name}")

    # --- Validate SensorType == 6 ---
    st = sensor.get("SensorType")
    if isinstance(st, bool) or not isinstance(st, (int, float)):
        raise ValueError(
            f"SensorType must be a number, got {type(st).__name__} ({st!r})"
        )
    if int(st) != 6:
        raise ValueError(
            f"SensorType must be 6 (LiDAR), got {st}. "
            f"Sector directions are only valid for SensorLocalFrame LiDAR data."
        )

    # --- Validate Enabled is true ---
    enabled = sensor.get("Enabled")
    if enabled is not True:
        raise ValueError(
            f"Enabled must be true, got {type(enabled).__name__} ({enabled!r}). "
            f"Cannot validate FOV for a disabled LiDAR sensor."
        )

    # --- Validate DataFrame == "SensorLocalFrame" ---
    dataframe = sensor.get("DataFrame")
    if not isinstance(dataframe, str):
        raise ValueError(
            f"DataFrame must be a string, got {type(dataframe).__name__} ({dataframe!r})"
        )
    if dataframe != "SensorLocalFrame":
        raise ValueError(
            f"DataFrame must be 'SensorLocalFrame', got {dataframe!r}. "
            f"Sector azimuth/elevation definitions assume SensorLocalFrame coordinates."
        )

    # --- Validate numeric fields with no silent coercion ---
    def _num(key, lo, hi):
        v = sensor.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} must be a number, got {type(v).__name__} ({v!r})")
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{key} must be finite")
        if not (lo <= v <= hi):
            raise ValueError(f"{key}={v} must be in [{lo},{hi}]")
        return float(v)

    h_start = _num("HorizontalFOVStart", -360, 360)
    h_end = _num("HorizontalFOVEnd", -360, 360)
    v_upper = _num("VerticalFOVUpper", -90, 90)
    v_lower = _num("VerticalFOVLower", -90, 90)
    rng = _num("Range", 0.1, 10000)

    # --- Validate vertical ordering ---
    if v_lower >= v_upper:
        raise ValueError(
            f"VerticalFOVLower ({v_lower}) must be < "
            f"VerticalFOVUpper ({v_upper})"
        )

    # --- Validate no zero-width horizontal FOV ---
    h_span = abs(h_end - h_start)
    if h_span < 1e-9:
        raise ValueError(
            f"HorizontalFOVStart ({h_start}) and HorizontalFOVEnd ({h_end}) "
            f"must not form a zero-width FOV"
        )

    return SensorFov(
        horizontal_start_deg=h_start,
        horizontal_end_deg=h_end,
        vertical_lower_deg=v_lower,
        vertical_upper_deg=v_upper,
        range_m=rng,
    )


# ---------------------------------------------------------------------------
# FOV coverage validator — full horizontal + vertical (ROUND 3.3)
# ---------------------------------------------------------------------------


def validate_sector_fov_coverage(
    config: PerceptionConfig,
    fov: SensorFov,
) -> List[SectorFovStatus]:
    """Check each sector against the real LiDAR FOV.

    ROUND 3.3: Checks BOTH horizontal azimuth AND vertical elevation
    coverage.  A sector is ``fully_observable`` only when both dimensions
    are completely covered by the LiDAR FOV.

    Returns a list of status objects with coverage fractions,
    intersection widths, and classification flags.
    """
    fov_full = fov.horizontal_full_circle
    results: List[SectorFovStatus] = []

    for sector in config.sectorization.sectors:
        # --- Horizontal coverage ---
        _, _, h_width = _azimuth_intersection_deg(
            sector.azimuth_min_deg, sector.azimuth_max_deg,
            fov.horizontal_start_deg, fov.horizontal_end_deg,
            fov_full,
        )

        # Sector horizontal span
        h_span = abs(sector.azimuth_max_deg - sector.azimuth_min_deg)
        if sector.azimuth_min_deg > sector.azimuth_max_deg:
            # Wrap-around sector (e.g. back: [157.5, -157.5])
            h_span = (sector.azimuth_max_deg + 360.0) - sector.azimuth_min_deg
        if h_span <= 0:
            h_span = 360.0  # full-circle sector

        h_frac = min(1.0, h_width / h_span) if h_span > 0 else 1.0

        # --- Vertical coverage ---
        _, _, v_width = _elevation_intersection_deg(
            sector.elevation_min_deg, sector.elevation_max_deg,
            fov.vertical_lower_deg, fov.vertical_upper_deg,
        )
        v_span = sector.elevation_max_deg - sector.elevation_min_deg
        v_frac = min(1.0, v_width / v_span) if v_span > 0 else 0.0

        # --- Classification ---
        # Tolerance for floating-point: fraction >= 0.9999 is fully covered
        _TOL = 1e-4

        h_full = h_frac >= (1.0 - _TOL)
        v_full = v_frac >= (1.0 - _TOL)

        fully = h_full and v_full
        partially = (not fully) and (h_frac > _TOL and v_frac > _TOL)
        unobs = not fully and not partially

        # Build note
        parts = []
        if not h_full:
            parts.append(f"horizontal: {h_frac*100:.1f}% covered")
        if not v_full:
            parts.append(f"vertical: {v_frac*100:.1f}% covered")
        if fully:
            parts.append("fully observable")
        elif partially:
            parts.append("partially observable")
        else:
            parts.append("unobservable")

        results.append(SectorFovStatus(
            legacy_name=sector.legacy_name,
            horizontal_coverage_fraction=h_frac,
            vertical_coverage_fraction=v_frac,
            fully_observable=fully,
            partially_observable=partially,
            unobservable=unobs,
            intersection_azimuth_deg=h_width,
            intersection_elevation_deg=v_width,
            note="; ".join(parts),
        ))

    return results


# ---------------------------------------------------------------------------
# Max range check — ROUND 3.3: returns errors, not warnings
# ---------------------------------------------------------------------------


def check_max_range_against_fov(
    config: PerceptionConfig, fov: SensorFov
) -> List[str]:
    """Return error messages if any configured max_range exceeds the
    physical sensor Range.  ROUND 3.3: these are errors, not warnings."""
    errors: List[str] = []
    if config.pointcloud.max_range_m > fov.range_m:
        errors.append(
            f"pointcloud.max_range_m ({config.pointcloud.max_range_m}) "
            f"exceeds LiDAR Range ({fov.range_m})"
        )
    for s in config.sectorization.sectors:
        if s.max_range_m > fov.range_m:
            errors.append(
                f"Sector {s.name}: max_range_m ({s.max_range_m}) "
                f"exceeds LiDAR Range ({fov.range_m})"
            )
    return errors

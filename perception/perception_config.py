"""Perception configuration loader — ROUND 3.2 strict validation.

All validators reject bool-as-int, string-as-number, and other silent
conversions.  No ``bool()``, ``float()``, ``int()``, or ``str()`` casts
are applied to user-supplied config values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml

# ---------------------------------------------------------------------------
# Validated data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfExclusionConfig:
    enabled: bool
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float


@dataclass(frozen=True)
class VoxelDownsampleConfig:
    enabled: bool
    voxel_size_m: float


@dataclass(frozen=True)
class PointCloudFilterConfig:
    min_range_m: float
    max_range_m: float
    self_exclusion: SelfExclusionConfig
    voxel_downsample: VoxelDownsampleConfig


@dataclass(frozen=True)
class SectorDef:
    name: str
    legacy_name: str
    azimuth_min_deg: float
    azimuth_max_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    max_range_m: float
    min_points: int
    distance_strategy: str


@dataclass(frozen=True)
class SectorizationConfig:
    default_max_range_m: float
    default_min_points: int
    default_distance_strategy: str
    nearest_k: int
    percentile: float
    sectors: List[SectorDef]


@dataclass(frozen=True)
class PerceptionConfig:
    pointcloud: PointCloudFilterConfig
    sectorization: SectorizationConfig


# ---------------------------------------------------------------------------
# Strict validators — no silent type coercion
# ---------------------------------------------------------------------------


def _require_dict(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict, got {type(value).__name__} ({value!r})")
    return value


def _require_bool(value, label: str) -> bool:
    if value is True or value is False:
        return value
    raise ValueError(f"{label} must be true or false, got {type(value).__name__} ({value!r})")


def _require_finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got bool ({value!r})")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be int or float, got {type(value).__name__} ({value!r})")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return float(value)


def _require_positive_number(value, label: str) -> float:
    v = _require_finite_number(value, label)
    if v <= 0:
        raise ValueError(f"{label} must be > 0, got {v}")
    return v


def _require_nonnegative_number(value, label: str) -> float:
    v = _require_finite_number(value, label)
    if v < 0:
        raise ValueError(f"{label} must be >= 0, got {v}")
    return v


def _require_positive_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an int, got bool ({value!r})")
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an int, got {type(value).__name__} ({value!r})")
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value}")
    return value


def _require_nonempty_string(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__} ({value!r})")
    if not value.strip():
        raise ValueError(f"{label} must not be empty or whitespace-only")
    return value


def _require_strategy(value, label: str) -> str:
    s = _require_nonempty_string(value, label)
    allowed = {"minimum", "nearest_k_median", "percentile"}
    if s not in allowed:
        raise ValueError(f"{label} must be one of {allowed}, got {s!r}")
    return s


def _require_azimuth_deg(value, label: str) -> float:
    v = _require_finite_number(value, label)
    if not (-180.0 <= v <= 180.0):
        raise ValueError(f"{label} must be -180..180, got {v}")
    return v


def _require_elevation_deg(value, label: str) -> float:
    v = _require_finite_number(value, label)
    if not (-90.0 <= v <= 90.0):
        raise ValueError(f"{label} must be -90..90, got {v}")
    return v


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_perception_config(config_path: Union[str, Path]) -> PerceptionConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Perception config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a dict")

    # --- pointcloud ---
    pc = _require_dict(raw.get("pointcloud", {}), "pointcloud")

    min_range = _require_nonnegative_number(pc.get("min_range_m", 0.2), "pointcloud.min_range_m")
    max_range = _require_positive_number(pc.get("max_range_m", 40.0), "pointcloud.max_range_m")
    if min_range >= max_range:
        raise ValueError(f"min_range_m ({min_range}) must be < max_range_m ({max_range})")

    # --- self_exclusion ---
    se = _require_dict(pc.get("self_exclusion", {}), "pointcloud.self_exclusion")
    se_enabled = _require_bool(se.get("enabled", True), "self_exclusion.enabled")
    se_xmin = _require_finite_number(se.get("x_min_m", -0.8), "self_exclusion.x_min_m")
    se_xmax = _require_finite_number(se.get("x_max_m", 0.8), "self_exclusion.x_max_m")
    se_ymin = _require_finite_number(se.get("y_min_m", -0.8), "self_exclusion.y_min_m")
    se_ymax = _require_finite_number(se.get("y_max_m", 0.8), "self_exclusion.y_max_m")
    se_zmin = _require_finite_number(se.get("z_min_m", -0.5), "self_exclusion.z_min_m")
    se_zmax = _require_finite_number(se.get("z_max_m", 0.5), "self_exclusion.z_max_m")
    if se_xmin >= se_xmax:
        raise ValueError(f"self_exclusion x_min ({se_xmin}) must be < x_max ({se_xmax})")
    if se_ymin >= se_ymax:
        raise ValueError(f"self_exclusion y_min ({se_ymin}) must be < y_max ({se_ymax})")
    if se_zmin >= se_zmax:
        raise ValueError(f"self_exclusion z_min ({se_zmin}) must be < z_max ({se_zmax})")

    # --- voxel_downsample ---
    vd = _require_dict(pc.get("voxel_downsample", {}), "pointcloud.voxel_downsample")
    vd_enabled = _require_bool(vd.get("enabled", True), "voxel_downsample.enabled")
    vd_size = _require_positive_number(vd.get("voxel_size_m", 0.1), "voxel_downsample.voxel_size_m")

    # --- sectorization ---
    sz = _require_dict(raw.get("sectorization", {}), "sectorization")

    def_max_range = _require_positive_number(
        sz.get("default_max_range_m", 40.0), "sectorization.default_max_range_m"
    )
    def_min_points = _require_positive_int(
        sz.get("default_min_points", 3), "sectorization.default_min_points"
    )
    def_strategy = _require_strategy(
        sz.get("default_distance_strategy", "nearest_k_median"),
        "sectorization.default_distance_strategy",
    )
    nearest_k = _require_positive_int(sz.get("nearest_k", 3), "sectorization.nearest_k")
    percentile = _require_finite_number(sz.get("percentile", 10.0), "sectorization.percentile")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"sectorization.percentile must be 0–100, got {percentile}")

    sectors_raw = sz.get("sectors", [])
    if not isinstance(sectors_raw, list):
        raise ValueError("sectorization.sectors must be a list")

    sectors: List[SectorDef] = []
    seen_names: set = set()
    for sdef in sectors_raw:
        _require_dict(sdef, "sector entry")
        name = _require_nonempty_string(sdef.get("name", ""), "sector.name")
        if name in seen_names:
            raise ValueError(f"Duplicate sector name {name!r}")
        seen_names.add(name)

        legacy_name = _require_nonempty_string(sdef.get("legacy_name", ""), "sector.legacy_name")

        a_min = _require_azimuth_deg(sdef.get("azimuth_min_deg", 0), f"{name}.azimuth_min_deg")
        a_max = _require_azimuth_deg(sdef.get("azimuth_max_deg", 0), f"{name}.azimuth_max_deg")
        if a_min == a_max:
            raise ValueError(f"{name}: azimuth_min_deg == azimuth_max_deg ({a_min}) not allowed")
        e_min = _require_elevation_deg(sdef.get("elevation_min_deg", 0), f"{name}.elevation_min_deg")
        e_max = _require_elevation_deg(sdef.get("elevation_max_deg", 0), f"{name}.elevation_max_deg")
        if e_min >= e_max:
            raise ValueError(f"{name}: elevation_min ({e_min}) must be < elevation_max ({e_max})")

        s_max_range = _require_positive_number(
            sdef.get("max_range_m", def_max_range), f"{name}.max_range_m"
        )
        s_min_points = _require_positive_int(
            sdef.get("min_points", def_min_points), f"{name}.min_points"
        )
        s_strategy = _require_strategy(
            sdef.get("distance_strategy", def_strategy), f"{name}.distance_strategy"
        )

        sectors.append(SectorDef(
            name=name, legacy_name=legacy_name,
            azimuth_min_deg=a_min, azimuth_max_deg=a_max,
            elevation_min_deg=e_min, elevation_max_deg=e_max,
            max_range_m=s_max_range, min_points=s_min_points,
            distance_strategy=s_strategy,
        ))

    # --- Cross-sector: legacy_name uniqueness & completeness ---
    if len(sectors) == 0:
        raise ValueError("sectors list must not be empty")

    legacy_seen: set = set()
    for s in sectors:
        if s.legacy_name in legacy_seen:
            raise ValueError(f"Duplicate legacy_name {s.legacy_name!r}")
        legacy_seen.add(s.legacy_name)

    _REQUIRED = {
        "front", "back", "left", "right", "up", "down",
        "frontLeft", "frontRight", "backLeft", "backRight",
        "frontUp", "frontDown", "leftUp", "rightUp", "leftDown", "rightDown",
    }
    missing = _REQUIRED - legacy_seen
    extra = legacy_seen - _REQUIRED
    if missing:
        raise ValueError(f"Missing required legacy_name(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown legacy_name(s): {sorted(extra)}")

    return PerceptionConfig(
        pointcloud=PointCloudFilterConfig(
            min_range_m=min_range, max_range_m=max_range,
            self_exclusion=SelfExclusionConfig(
                enabled=se_enabled, x_min_m=se_xmin, x_max_m=se_xmax,
                y_min_m=se_ymin, y_max_m=se_ymax,
                z_min_m=se_zmin, z_max_m=se_zmax,
            ),
            voxel_downsample=VoxelDownsampleConfig(enabled=vd_enabled, voxel_size_m=vd_size),
        ),
        sectorization=SectorizationConfig(
            default_max_range_m=def_max_range, default_min_points=def_min_points,
            default_distance_strategy=def_strategy,
            nearest_k=nearest_k, percentile=percentile,
            sectors=sectors,
        ),
    )

"""AirborneFixtureConfig — ROUND 4.7. Strict validation, no unknown fields."""
from __future__ import annotations
import math
from dataclasses import dataclass, fields
from typing import Dict, Set


def _rf(value, label: str, lo=None, hi=None) -> float:
    if isinstance(value, bool): raise ValueError(f"{label} must be a number, got bool")
    if not isinstance(value, (int, float)): raise ValueError(f"{label} must be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v): raise ValueError(f"{label} must be finite, got {v}")
    if lo is not None and v < lo: raise ValueError(f"{label} must be >= {lo}, got {v}")
    if hi is not None and v > hi: raise ValueError(f"{label} must be <= {hi}, got {v}")
    return v

def _ri(value, label: str, lo=1) -> int:
    if isinstance(value, bool): raise ValueError(f"{label} must be an int, got bool")
    if not isinstance(value, int): raise ValueError(f"{label} must be an int, got {type(value).__name__}")
    if value < lo: raise ValueError(f"{label} must be >= {lo}, got {value}")
    return value


@dataclass(frozen=True)
class AirborneFixtureConfig:
    target_z_range: tuple
    max_vertical_speed_mps: float
    default_hover_duration_s: float
    takeoff_timeout_s: float
    preflight_lidar_frames: int
    altitude_tolerance_m: float
    hover_stabilization_frames: int
    landing_confirmation_frames: int
    min_preflight_filtered_points: int
    takeoff_delta_z_m: float
    vehicle_type: str
    vc_safety_radius_m: float
    vc_vertical_margin_m: float
    upward_sensor_enabled: bool
    upward_sensor_provider: str
    max_altitude_error_m: float
    max_horizontal_drift_m: float
    max_horizontal_speed_mps: float
    hs_max_vertical_speed_mps: float
    max_tilt_rad: float
    stationary_speed_threshold_mps: float

    _KNOWN_TOP_KEYS = {"target_z_range_m", "max_vertical_speed_mps", "default_hover_duration_s",
                        "takeoff_timeout_s", "preflight_lidar_frames", "altitude_tolerance_m",
                        "hover_stabilization_frames", "landing_confirmation_frames",
                        "minimum_preflight_filtered_points", "takeoff_delta_z_m", "vehicle_type",
                        "vertical_clearance", "upward_sensor", "hover_safety"}
    _KNOWN_VC_KEYS = {"safety_radius_m", "vertical_margin_m"}
    _KNOWN_US_KEYS = {"enabled", "provider", "_note"}
    _KNOWN_HS_KEYS = {"max_altitude_error_m", "max_horizontal_drift_m", "max_horizontal_speed_mps",
                       "max_vertical_speed_mps", "max_tilt_rad", "stationary_speed_threshold_mps"}

    @classmethod
    def from_dict(cls, d: Dict) -> "AirborneFixtureConfig":
        af = d.get("airborne_fixture", {})
        if not isinstance(af, dict): raise ValueError("airborne_fixture must be a dict")

        # Reject unknown keys
        extra = set(af.keys()) - cls._KNOWN_TOP_KEYS
        if extra: raise ValueError(f"Unknown airborne_fixture keys: {sorted(extra)}")

        zr = af.get("target_z_range_m", [-3.0, -0.5])
        if not isinstance(zr, list) or len(zr) != 2: raise ValueError("target_z_range_m must be [lo, hi]")
        z_lo, z_hi = _rf(zr[0], "target_z_range_m[0]"), _rf(zr[1], "target_z_range_m[1]")
        if z_lo >= z_hi: raise ValueError(f"target_z_range_m: {z_lo} >= {z_hi}")

        max_vs = _rf(af.get("max_vertical_speed_mps", 0.5), "max_vertical_speed_mps", 0.01, 0.5)
        hover_dur = _rf(af.get("default_hover_duration_s", 30), "default_hover_duration_s", 1.0, 3600.0)
        to = _rf(af.get("takeoff_timeout_s", 20.0), "takeoff_timeout_s", 1.0, 120.0)
        pf = _ri(af.get("preflight_lidar_frames", 3), "preflight_lidar_frames")
        alt = _rf(af.get("altitude_tolerance_m", 0.2), "altitude_tolerance_m", 0.05, 1.0)
        stab = _ri(af.get("hover_stabilization_frames", 3), "hover_stabilization_frames")
        land_fr = _ri(af.get("landing_confirmation_frames", 3), "landing_confirmation_frames")
        min_pts = _ri(af.get("minimum_preflight_filtered_points", 10), "minimum_preflight_filtered_points")
        tdz = _rf(af.get("takeoff_delta_z_m", -2.0), "takeoff_delta_z_m", -5.0, -0.5)
        if abs(tdz - (-2.0)) > 1e-9:
            raise ValueError(f"takeoff_delta_z_m must be -2.0 (SimpleFlight audited value), got {tdz}")
        vt = af.get("vehicle_type", "SimpleFlight")
        if not isinstance(vt, str) or vt != "SimpleFlight":
            raise ValueError(f"vehicle_type must be 'SimpleFlight', got {vt!r}")

        vc = af.get("vertical_clearance", {})
        if not isinstance(vc, dict): raise ValueError("vertical_clearance must be a dict")
        extra_vc = set(vc.keys()) - cls._KNOWN_VC_KEYS
        if extra_vc: raise ValueError(f"Unknown vertical_clearance keys: {sorted(extra_vc)}")
        vc_radius = _rf(vc.get("safety_radius_m", 1.5), "safety_radius_m", 0.1, 10.0)
        vc_margin = _rf(vc.get("vertical_margin_m", 0.3), "vertical_margin_m", 0.05, 2.0)

        us = af.get("upward_sensor", {})
        if not isinstance(us, dict): raise ValueError("upward_sensor must be a dict")
        extra_us = set(us.keys()) - cls._KNOWN_US_KEYS
        if extra_us: raise ValueError(f"Unknown upward_sensor keys: {sorted(extra_us)}")
        us_enabled = us.get("enabled", False)
        if us_enabled is not True and us_enabled is not False:
            raise ValueError(f"upward_sensor.enabled must be bool, got {type(us_enabled).__name__}")
        us_provider = us.get("provider", "none")
        if not isinstance(us_provider, str): raise ValueError("upward_sensor.provider must be a string")

        hs = af.get("hover_safety", {})
        if not isinstance(hs, dict): raise ValueError("hover_safety must be a dict")
        extra_hs = set(hs.keys()) - cls._KNOWN_HS_KEYS
        if extra_hs: raise ValueError(f"Unknown hover_safety keys: {sorted(extra_hs)}")
        alt_err = _rf(hs.get("max_altitude_error_m", 0.3), "max_altitude_error_m", 0.05, 2.0)
        hdrift = _rf(hs.get("max_horizontal_drift_m", 0.5), "max_horizontal_drift_m", 0.05, 5.0)
        hspeed = _rf(hs.get("max_horizontal_speed_mps", 0.5), "max_horizontal_speed_mps", 0.01, 2.0)
        vspeed = _rf(hs.get("max_vertical_speed_mps", 0.3), "max_vertical_speed_mps", 0.01, 1.0)
        tilt = _rf(hs.get("max_tilt_rad", 0.35), "max_tilt_rad", 0.01, 1.0)
        stt = _rf(hs.get("stationary_speed_threshold_mps", 0.2), "stationary_speed_threshold_mps", 0.01, 1.0)

        return cls(target_z_range=(z_lo, z_hi), max_vertical_speed_mps=max_vs,
                   default_hover_duration_s=hover_dur, takeoff_timeout_s=to,
                   preflight_lidar_frames=pf, altitude_tolerance_m=alt,
                   hover_stabilization_frames=stab, landing_confirmation_frames=land_fr,
                   min_preflight_filtered_points=min_pts, takeoff_delta_z_m=tdz,
                   vehicle_type=vt, vc_safety_radius_m=vc_radius,
                   vc_vertical_margin_m=vc_margin, upward_sensor_enabled=us_enabled,
                   upward_sensor_provider=us_provider, max_altitude_error_m=alt_err,
                   max_horizontal_drift_m=hdrift, max_horizontal_speed_mps=hspeed,
                   hs_max_vertical_speed_mps=vspeed, max_tilt_rad=tilt,
                   stationary_speed_threshold_mps=stt)

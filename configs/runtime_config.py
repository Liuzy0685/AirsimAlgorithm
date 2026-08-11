"""
Runtime configuration loader for LiDAR parameters.

Provides a reusable, testable function that parses and validates
LiDAR settings from ``vehicle.yaml``.  All validation rules are
centralised here so that both the smoke-test script and the future
main avoidance loop use the same safe defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

# ---------------------------------------------------------------------------
# Safe defaults
# ---------------------------------------------------------------------------

DEFAULT_LIDAR_FRAME_TIMEOUT_SECONDS: float = 0.5
DEFAULT_MAX_CONSECUTIVE_INVALID: int = 10

# ---------------------------------------------------------------------------
# Validated config data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AirSimConfig:
    """AirSim connection parameters from vehicle.yaml."""

    vehicle_name: str
    lidar_name: str


@dataclass(frozen=True)
class LidarRuntimeConfig:
    """Immutable, validated LiDAR runtime configuration.

    Attributes:
        airsim: AirSim connection parameters (vehicle_name, lidar_name).
        frame_timeout_seconds:
            Maximum age of the last *new* timestamp before a frame is
            considered stale.  Clamped to [0.05, 10.0] seconds.
        max_consecutive_invalid:
            Number of consecutive invalid frames before escalation.
            Must be a positive int ≤ 10000.
    """

    airsim: AirSimConfig
    frame_timeout_seconds: float
    max_consecutive_invalid: int


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_lidar_runtime_config(
    config_path: Union[str, Path],
) -> LidarRuntimeConfig:
    """Load and validate LiDAR runtime settings from a YAML file.

    Parameters
    ----------
    config_path:
        Path to a ``vehicle.yaml`` file.

    Returns
    -------
    LidarRuntimeConfig
        Frozen, validated configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    yaml.YAMLError
        If the file contains invalid YAML syntax.
    ValueError
        If any configuration value fails validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # --- airsim section (vehicle_name, lidar_name) ---
    airsim_cfg = cfg.get("airsim")
    if not isinstance(airsim_cfg, dict):
        raise ValueError("airsim section must be a dict in vehicle.yaml")

    vehicle_name = airsim_cfg.get("vehicle_name")
    if not isinstance(vehicle_name, str) or not vehicle_name.strip():
        raise ValueError(f"airsim.vehicle_name must be a non-empty string, got {vehicle_name!r}")
    lidar_name = airsim_cfg.get("lidar_name")
    if not isinstance(lidar_name, str) or not lidar_name.strip():
        raise ValueError(f"airsim.lidar_name must be a non-empty string, got {lidar_name!r}")

    # --- lidar section ---
    lidar_cfg = cfg.get("lidar")
    if not isinstance(lidar_cfg, dict):
        lidar_cfg = {}

    frame_timeout_seconds = _validate_frame_timeout(
        lidar_cfg.get("frame_timeout_seconds", DEFAULT_LIDAR_FRAME_TIMEOUT_SECONDS)
    )
    max_consecutive_invalid = _validate_max_consecutive_invalid(
        lidar_cfg.get("max_consecutive_invalid", DEFAULT_MAX_CONSECUTIVE_INVALID)
    )

    return LidarRuntimeConfig(
        airsim=AirSimConfig(vehicle_name=vehicle_name.strip(), lidar_name=lidar_name.strip()),
        frame_timeout_seconds=frame_timeout_seconds,
        max_consecutive_invalid=max_consecutive_invalid,
    )


# ---------------------------------------------------------------------------
# Validators (public so tests can reuse them)
# ---------------------------------------------------------------------------


def _validate_frame_timeout(value) -> float:
    """Validate and return a safe ``frame_timeout_seconds``.

    Rules
    -----
    - Must be ``int`` or ``float`` (NOT ``bool``).
    - Must be finite.
    - Must be in [0.05, 10.0] seconds.
    - Strings (``"0.5"``) are NOT accepted.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"frame_timeout_seconds must be a number, got bool ({value!r})"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"frame_timeout_seconds must be int or float, got {type(value).__name__} ({value!r})"
        )
    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"frame_timeout_seconds must be finite, got {value!r}"
        )
    fval = float(value)
    if not (0.05 <= fval <= 10.0):
        raise ValueError(
            f"frame_timeout_seconds={fval} out of range [0.05, 10.0]"
        )
    return fval


def _validate_max_consecutive_invalid(value) -> int:
    """Validate and return a safe ``max_consecutive_invalid``.

    Rules
    -----
    - Must be ``int`` (NOT ``bool``).
    - Must be > 0 and ≤ 10000.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"max_consecutive_invalid must be an int, got bool ({value!r})"
        )
    if not isinstance(value, int):
        raise ValueError(
            f"max_consecutive_invalid must be an int, got {type(value).__name__} ({value!r})"
        )
    if value <= 0:
        raise ValueError(
            f"max_consecutive_invalid must be > 0, got {value}"
        )
    if value > 10000:
        raise ValueError(
            f"max_consecutive_invalid={value} exceeds maximum 10000"
        )
    return value

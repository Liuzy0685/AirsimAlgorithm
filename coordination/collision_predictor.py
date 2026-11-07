"""Constant-velocity pairwise conflict screening for multi-UAV experiments.

This module only predicts and reports proximity risk. It does not generate or
dispatch flight commands.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CollisionPrediction:
    conflict: bool
    time_to_closest_s: float
    closest_distance_m: float
    current_distance_m: float


def predict_pairwise_conflict(
    position_a: Sequence[float],
    velocity_a: Sequence[float],
    position_b: Sequence[float],
    velocity_b: Sequence[float],
    *,
    safety_distance_m: float,
    horizon_s: float,
) -> CollisionPrediction:
    """Predict closest 3D separation under constant velocity over a horizon."""
    vectors = (position_a, velocity_a, position_b, velocity_b)
    if any(len(vector) != 3 for vector in vectors):
        raise ValueError("positions and velocities must be 3D")
    values = [float(value) for vector in vectors for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("positions and velocities must be finite")
    if not math.isfinite(safety_distance_m) or safety_distance_m <= 0:
        raise ValueError("safety_distance_m must be finite and > 0")
    if not math.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be finite and >= 0")

    relative_position = [float(b) - float(a) for a, b in zip(position_a, position_b)]
    relative_velocity = [float(b) - float(a) for a, b in zip(velocity_a, velocity_b)]
    current_distance = math.sqrt(sum(value * value for value in relative_position))
    speed_sq = sum(value * value for value in relative_velocity)
    if speed_sq <= 1e-12:
        closest_time = 0.0
    else:
        closest_time = -sum(
            p * v for p, v in zip(relative_position, relative_velocity)
        ) / speed_sq
        closest_time = min(horizon_s, max(0.0, closest_time))
    closest_distance = math.sqrt(sum(
        (p + v * closest_time) ** 2
        for p, v in zip(relative_position, relative_velocity)
    ))
    return CollisionPrediction(
        conflict=closest_distance <= safety_distance_m,
        time_to_closest_s=closest_time,
        closest_distance_m=closest_distance,
        current_distance_m=current_distance,
    )


def load_spawn_offsets(
    settings_json: str | Path, vehicle_names: Sequence[str],
) -> Mapping[str, tuple[float, float, float]]:
    """Read each AirSim vehicle's configured NED spawn offset from settings.

    AirSim reports multirotor positions relative to that vehicle's starting
    point. Adding the configured initial offset expresses positions in the
    PlayerStart NED frame used by the multi-vehicle settings.
    """
    with Path(settings_json).open("r", encoding="utf-8") as stream:
        settings = json.load(stream)
    vehicles = settings.get("Vehicles") or {}
    if not isinstance(vehicles, dict):
        raise ValueError("settings.json Vehicles must be an object")
    offsets: dict[str, tuple[float, float, float]] = {}
    for name in vehicle_names:
        item = vehicles.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"vehicle {name!r} is missing from settings.json Vehicles")
        offset = tuple(float(item.get(axis, 0.0)) for axis in ("X", "Y", "Z"))
        if not all(math.isfinite(value) for value in offset):
            raise ValueError(f"vehicle {name!r} has a non-finite spawn offset")
        offsets[name] = offset
    return offsets


def load_scene_marker_offsets(
    client,
    actor_names: Mapping[str, str],
) -> Mapping[str, tuple[float, float, float]]:
    """Read each vehicle's PlayerStart marker by actor name or tag in world NED."""
    offsets: dict[str, tuple[float, float, float]] = {}
    for vehicle_name, actor_name in actor_names.items():
        try:
            pose = client.simGetObjectPose(str(actor_name))
            position = pose.position
            offset = (
                float(position.x_val),
                float(position.y_val),
                float(position.z_val),
            )
        except Exception as exc:
            raise ValueError(
                f"could not read PlayerStart marker/tag {actor_name!r} for "
                f"vehicle {vehicle_name!r}: {exc}"
            ) from exc
        if not all(math.isfinite(value) for value in offset):
            raise ValueError(
                f"PlayerStart marker/tag {actor_name!r} for vehicle "
                f"{vehicle_name!r} was not found by AirSim or has an invalid pose"
            )
        offsets[str(vehicle_name)] = offset
    return offsets


def to_common_ned(
    local_position_ned: Sequence[float],
    spawn_offset_ned: Sequence[float],
) -> tuple[float, float, float]:
    """Translate vehicle-home-relative position into common PlayerStart NED."""
    if len(local_position_ned) != 3 or len(spawn_offset_ned) != 3:
        raise ValueError("positions and offsets must be 3D")
    result = tuple(float(p) + float(o) for p, o in zip(local_position_ned, spawn_offset_ned))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("transformed position must be finite")
    return result

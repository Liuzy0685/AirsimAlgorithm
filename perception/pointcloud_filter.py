"""
Point-cloud filter.

Pure NumPy, testable, no AirSim / UE4 / ROS / Open3D dependency.

Operates on **SensorLocalFrame** point clouds (N×3).
Does NOT modify the caller's input array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Result of a point-cloud filtering pass.

    Attributes:
        filtered_points_sensor: M×3 NumPy array (SensorLocalFrame).  Empty (0,3)
            when ``valid`` is ``False``.
        input_point_count: Number of points before filtering.
        output_point_count: Number of points after filtering.
        removed_nonfinite_count: Points removed for NaN / inf.
        removed_min_range_count: Points removed below ``min_range_m``.
        removed_max_range_count: Points removed above ``max_range_m``.
        removed_self_body_count: Points removed inside the self-exclusion box.
        voxel_reduction_count: Points removed by voxel down-sampling.
        valid: ``True`` if filtering completed successfully.
        invalid_reason: Human-readable reason when ``valid`` is ``False``.
    """

    filtered_points_sensor: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    input_point_count: int = 0
    output_point_count: int = 0
    removed_nonfinite_count: int = 0
    removed_min_range_count: int = 0
    removed_max_range_count: int = 0
    removed_self_body_count: int = 0
    voxel_reduction_count: int = 0
    valid: bool = False
    invalid_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Filter function
# ---------------------------------------------------------------------------


def filter_pointcloud(
    points: np.ndarray,
    min_range_m: float = 0.2,
    max_range_m: float = 40.0,
    self_exclusion: Optional[dict] = None,
    voxel_downsample: bool = False,
    voxel_size_m: float = 0.1,
) -> FilterResult:
    """Filter a SensorLocalFrame point cloud.

    Filter order:
    1. Validate input is N×3
    2. Convert to explicit float32
    3. Remove NaN / inf
    4. Compute 3-D Euclidean distances
    5. Remove points below ``min_range_m``
    6. Remove points above ``max_range_m``
    7. Remove points inside the self-exclusion bounding box
    8. Optional voxel down-sampling (deterministic, keeps nearest per voxel)

    Parameters
    ----------
    points:
        N×3 NumPy array in SensorLocalFrame.
    min_range_m:
        Minimum 3-D distance (metres).
    max_range_m:
        Maximum 3-D distance (metres).
    self_exclusion:
        Dict with keys ``enabled`` (bool) and ``x/y/z_min/max_m`` (float).
        If ``None`` or ``enabled=False``, self-exclusion is skipped.
    voxel_downsample:
        If ``True``, apply voxel grid down-sampling after range filtering.
    voxel_size_m:
        Voxel edge length in metres.  Must be > 0.

    Returns
    -------
    FilterResult
    """
    # --- 1. Validate shape ---
    if not isinstance(points, np.ndarray):
        return FilterResult(
            valid=False,
            invalid_reason=f"expected np.ndarray, got {type(points).__name__}",
        )
    if points.ndim != 2 or points.shape[1] != 3:
        return FilterResult(
            valid=False,
            invalid_reason=f"expected N×3 array, got {points.shape}",
        )

    input_count = points.shape[0]
    if input_count == 0:
        return FilterResult(
            input_point_count=0,
            valid=False,
            invalid_reason="empty",
        )

    # --- 2. Convert to explicit float32 (copy — do not mutate caller) ---
    arr = points.astype(np.float32, copy=True)

    # --- 3. Remove NaN / inf ---
    finite_mask = np.isfinite(arr).all(axis=1)
    removed_nonfinite = input_count - int(np.sum(finite_mask))
    arr = arr[finite_mask]

    if arr.size == 0:
        return FilterResult(
            input_point_count=input_count,
            output_point_count=0,
            removed_nonfinite_count=removed_nonfinite,
            valid=False,
            invalid_reason="all_points_nonfinite",
        )

    # --- 4. Compute 3-D Euclidean distances ---
    dist = np.linalg.norm(arr, axis=1)

    # --- 5. Remove points below min_range_m ---
    min_mask = dist >= min_range_m
    removed_min = int(np.sum(~min_mask))
    arr = arr[min_mask]
    dist = dist[min_mask]

    # --- 6. Remove points above max_range_m ---
    max_mask = dist <= max_range_m
    removed_max = int(np.sum(~max_mask))
    arr = arr[max_mask]
    dist = dist[max_mask]

    # If range filtering removed all remaining points, that's an invalid result.
    if arr.size == 0:
        return FilterResult(
            input_point_count=input_count,
            output_point_count=0,
            removed_nonfinite_count=removed_nonfinite,
            removed_min_range_count=removed_min,
            removed_max_range_count=removed_max,
            valid=False,
            invalid_reason="all_points_outside_range",
        )

    # --- 7. Self-exclusion ---
    removed_self = 0
    if self_exclusion and self_exclusion.get("enabled", False):
        x_min = float(self_exclusion["x_min_m"])
        x_max = float(self_exclusion["x_max_m"])
        y_min = float(self_exclusion["y_min_m"])
        y_max = float(self_exclusion["y_max_m"])
        z_min = float(self_exclusion["z_min_m"])
        z_max = float(self_exclusion["z_max_m"])

        inside = (
            (arr[:, 0] >= x_min)
            & (arr[:, 0] <= x_max)
            & (arr[:, 1] >= y_min)
            & (arr[:, 1] <= y_max)
            & (arr[:, 2] >= z_min)
            & (arr[:, 2] <= z_max)
        )
        removed_self = int(np.sum(inside))
        arr = arr[~inside]
        dist = dist[~inside]

        if arr.size == 0:
            return FilterResult(
                input_point_count=input_count,
                output_point_count=0,
                removed_nonfinite_count=removed_nonfinite,
                removed_min_range_count=removed_min,
                removed_max_range_count=removed_max,
                removed_self_body_count=removed_self,
                valid=False,
                invalid_reason="all_points_self_excluded",
            )

    # --- 8. Voxel down-sample (deterministic, keeps nearest per voxel) ---
    voxel_reduction = 0
    if voxel_downsample and arr.size > 0:
        if voxel_size_m <= 0:
            raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}")

        voxel_indices = np.floor(arr / voxel_size_m).astype(np.int64)

        # Build unique key per voxel: encode (ix, iy, iz) → single int64.
        # We use a simple hash that works for signed indices:
        # shift by 21 bits (2^21 ≈ 2M — enough for ±1M voxels at 0.05 m → ±51 km).
        shift = np.int64(21)
        keys = (
            (voxel_indices[:, 0].astype(np.int64) << (2 * shift))
            + (voxel_indices[:, 1].astype(np.int64) << shift)
            + voxel_indices[:, 2].astype(np.int64)
        )

        # For each unique voxel key, keep the point with the smallest distance.
        _, unique_idx, counts = np.unique(
            keys, return_index=True, return_counts=True
        )

        # Sort unique_idx to find the nearest point per voxel.
        # np.unique already returns the first occurrence; we need the argmin
        # per group.  Use a loop-free approach with argsort.
        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        sorted_dist = dist[order]

        # Boundaries between groups.
        change = np.diff(sorted_keys) != 0
        group_starts = np.concatenate(([0], np.flatnonzero(change) + 1))
        group_ends = np.concatenate((group_starts[1:], [len(sorted_keys)]))

        keep_indices = np.zeros(len(group_starts), dtype=np.intp)
        for i in range(len(group_starts)):
            gs = group_starts[i]
            ge = group_ends[i]
            # Nearest point = min distance within group
            local_idx = gs + np.argmin(sorted_dist[gs:ge])
            keep_indices[i] = order[local_idx]

        keep_indices.sort()
        voxel_reduction = arr.shape[0] - len(keep_indices)
        arr = arr[keep_indices]

    return FilterResult(
        filtered_points_sensor=arr,
        input_point_count=input_count,
        output_point_count=arr.shape[0],
        removed_nonfinite_count=removed_nonfinite,
        removed_min_range_count=removed_min,
        removed_max_range_count=removed_max,
        removed_self_body_count=removed_self,
        voxel_reduction_count=voxel_reduction,
        valid=True,
        invalid_reason=None,
    )

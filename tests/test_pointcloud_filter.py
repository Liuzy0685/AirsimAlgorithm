"""
Unit tests for perception/pointcloud_filter.py.

Pure NumPy — no AirSim / UE4 dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.pointcloud_filter import filter_pointcloud


class TestNormalFilter:
    def test_n_by_3(self):
        pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0)
        assert r.valid
        assert r.output_point_count == 3

    def test_does_not_modify_input(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        original = pts.copy()
        filter_pointcloud(pts)
        assert np.array_equal(pts, original)


class TestNanInf:
    def test_nan_removed(self):
        pts = np.array([[1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [2.0, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0)
        assert r.valid
        assert r.output_point_count == 2
        assert r.removed_nonfinite_count == 1

    def test_inf_removed(self):
        pts = np.array([[1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0)
        assert r.removed_nonfinite_count == 1


class TestRangeFilter:
    def test_below_min_removed(self):
        pts = np.array([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.2, max_range_m=10.0)
        assert r.output_point_count == 1
        assert r.removed_min_range_count == 1

    def test_above_max_removed(self):
        pts = np.array([[1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=40.0)
        assert r.output_point_count == 1
        assert r.removed_max_range_count == 1


class TestSelfExclusion:
    def test_inside_box_removed(self):
        pts = np.array([[0.3, 0.0, 0.0], [2.0, 0.0, 0.0]])  # [0.3,0,0] inside box, within range
        se = {
            "enabled": True,
            "x_min_m": -0.8, "x_max_m": 0.8,
            "y_min_m": -0.8, "y_max_m": 0.8,
            "z_min_m": -0.5, "z_max_m": 0.5,
        }
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0, self_exclusion=se)
        assert r.output_point_count == 1
        assert r.removed_self_body_count == 1

    def test_outside_box_kept(self):
        pts = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [0.3, 0.0, 0.1]])
        se = {
            "enabled": True,
            "x_min_m": -0.5, "x_max_m": 0.5,
            "y_min_m": -0.5, "y_max_m": 0.5,
            "z_min_m": -0.5, "z_max_m": 0.5,
        }
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0, self_exclusion=se)
        assert r.output_point_count == 2


class TestVoxelDownsample:
    def test_voxel_reduces(self):
        np.random.seed(42)
        pts = np.random.randn(1000, 3).astype(np.float32) * 0.05 + np.array([3.0, 0.0, 0.0])
        r = filter_pointcloud(
            pts, min_range_m=0.1, max_range_m=10.0,
            voxel_downsample=True, voxel_size_m=0.1,
        )
        assert r.valid
        assert r.output_point_count < pts.shape[0]
        assert r.voxel_reduction_count > 0

    def test_voxel_deterministic(self):
        np.random.seed(99)
        pts = np.random.randn(500, 3).astype(np.float32) * 0.05 + np.array([3.0, 0.0, 0.0])
        r1 = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0,
                               voxel_downsample=True, voxel_size_m=0.1)
        r2 = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0,
                               voxel_downsample=True, voxel_size_m=0.1)
        assert r1.output_point_count == r2.output_point_count
        # Results should be identical (sorted)
        assert np.array_equal(
            np.sort(r1.filtered_points_sensor, axis=0),
            np.sort(r2.filtered_points_sensor, axis=0),
        )

    def test_voxel_disabled(self):
        pts = np.array([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10.0,
                              voxel_downsample=False, voxel_size_m=0.1)
        assert r.output_point_count == 2
        assert r.voxel_reduction_count == 0


class TestEdgeCases:
    def test_empty_array(self):
        pts = np.empty((0, 3))
        r = filter_pointcloud(pts)
        assert not r.valid
        assert r.invalid_reason == "empty"

    def test_wrong_shape(self):
        pts = np.array([[1.0, 2.0]])
        r = filter_pointcloud(pts)
        assert not r.valid
        assert "N×3" in r.invalid_reason

    def test_not_ndarray(self):
        r = filter_pointcloud([1.0, 2.0, 3.0])
        assert not r.valid


class TestAllEmptyFilterSafety:
    """ROUND 3.1: All filtering paths that produce 0 points must return valid=False."""

    def test_all_min_range_filtered(self):
        pts = np.array([[0.01, 0, 0], [0.02, 0, 0]])
        r = filter_pointcloud(pts, min_range_m=0.2, max_range_m=40)
        assert not r.valid
        assert r.invalid_reason == "all_points_outside_range"

    def test_all_max_range_filtered(self):
        pts = np.array([[50.0, 0, 0], [60.0, 0, 0]])
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=40)
        assert not r.valid
        assert r.invalid_reason == "all_points_outside_range"

    def test_all_self_excluded(self):
        pts = np.array([[0.3, 0, 0], [0.0, 0.4, 0]])
        se = {"enabled": True, "x_min_m": -0.8, "x_max_m": 0.8,
              "y_min_m": -0.8, "y_max_m": 0.8, "z_min_m": -0.5, "z_max_m": 0.5}
        r = filter_pointcloud(pts, min_range_m=0.1, max_range_m=10, self_exclusion=se)
        assert not r.valid
        assert r.invalid_reason == "all_points_self_excluded"

    def test_all_nonfinite(self):
        pts = np.array([[float("nan"), 0, 0], [float("inf"), 0, 0]])
        r = filter_pointcloud(pts)
        assert not r.valid
        assert r.invalid_reason == "all_points_nonfinite"

    def test_invalid_filter_no_legacy_safety(self):
        """An invalid filter result must not be usable for legacy distances."""
        pts = np.array([[0.01, 0, 0]])
        r = filter_pointcloud(pts, min_range_m=0.2)
        assert not r.valid
        # The filtered point cloud is empty
        assert r.filtered_points_sensor.size == 0

    def test_all_filtered(self):
        pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
        r = filter_pointcloud(pts, min_range_m=0.2)
        assert not r.valid
        assert r.invalid_reason == "all_points_outside_range"

    def test_all_nan(self):
        pts = np.array([[float("nan"), 0.0, 0.0]])
        r = filter_pointcloud(pts)
        assert not r.valid

    def test_bad_voxel_size(self):
        pts = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            filter_pointcloud(pts, voxel_downsample=True, voxel_size_m=0.0)

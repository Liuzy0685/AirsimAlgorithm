"""ROUND 3.2: strict config validation — no silent type coercion."""
from __future__ import annotations
import math, sys, tempfile
from pathlib import Path
import pytest
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
from perception.perception_config import load_perception_config

def _y(content):
    t=tempfile.NamedTemporaryFile(mode="w",suffix=".yaml",delete=False,encoding="utf-8")
    t.write(content);t.close();return t.name

_BASE="pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors:\n    - name: a\n      legacy_name: front\n      azimuth_min_deg: -10\n      azimuth_max_deg: 10\n      elevation_min_deg: -10\n      elevation_max_deg: 10\n"

class TestLoadReal:
    def test_loads(self):
        cfg=load_perception_config(str(_PROJECT_ROOT/"configs"/"perception.yaml"))
        assert len(cfg.sectorization.sectors)==16

class TestBoolStrict:
    def test_enabled_string_rejected(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\n  self_exclusion:\n    enabled: \"false\"\n"))
    def test_enabled_int_rejected(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\n  self_exclusion:\n    enabled: 1\n"))

class TestNumericStrict:
    def test_percentile_bool(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  percentile: true\n"))
    def test_percentile_string(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  percentile: \"10.0\"\n"))
    def test_min_range_negative(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: -1\n  max_range_m: 40\n"))
    def test_max_range_negative(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: -1\n"))
    def test_voxel_size_negative(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\n  voxel_downsample:\n    voxel_size_m: -0.1\n"))
    def test_string_number_rejected(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: \"0.2\"\n  max_range_m: 40\n"))
    def test_min_points_string(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  default_min_points: \"3\"\n"))
    def test_min_points_float(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  default_min_points: 3.9\n"))
    def test_sector_max_range_negative(self):
        with pytest.raises(ValueError): load_perception_config(_y(_BASE+"      max_range_m: -1\n"))

class TestSectorNameStrict:
    def test_name_int(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors:\n    - name: 123\n      legacy_name: front\n      azimuth_min_deg: 0\n      azimuth_max_deg: 10\n      elevation_min_deg: 0\n      elevation_max_deg: 10\n"))
    def test_azimuth_equal(self):
        with pytest.raises(ValueError,match="azimuth_min_deg == azimuth_max_deg"): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors:\n    - name: a\n      legacy_name: front\n      azimuth_min_deg: 45\n      azimuth_max_deg: 45\n      elevation_min_deg: 0\n      elevation_max_deg: 10\n"))
    def test_angle_string(self):
        with pytest.raises(ValueError): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors:\n    - name: a\n      legacy_name: front\n      azimuth_min_deg: \"-10\"\n      azimuth_max_deg: 10\n      elevation_min_deg: 0\n      elevation_max_deg: 10\n"))

class TestDuplicate:
    def test_duplicate_internal_name(self):
        p=_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors:\n    - name: a\n      legacy_name: front\n      azimuth_min_deg: -10\n      azimuth_max_deg: 10\n      elevation_min_deg: -10\n      elevation_max_deg: 10\n    - name: a\n      legacy_name: frontLeft\n      azimuth_min_deg: -67.5\n      azimuth_max_deg: -22.5\n      elevation_min_deg: -22.5\n      elevation_max_deg: 22.5\n")
        with pytest.raises(ValueError,match="Duplicate"): load_perception_config(p)

class TestTopLevel:
    def test_not_dict(self):
        with pytest.raises(ValueError,match="dict"): load_perception_config(_y("[]\n"))
    def test_sectors_not_list(self):
        with pytest.raises(ValueError,match="list"): load_perception_config(_y("pointcloud:\n  min_range_m: 0.2\n  max_range_m: 40\nsectorization:\n  sectors: {}\n"))

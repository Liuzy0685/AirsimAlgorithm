"""
Unit tests for configs/runtime_config.py — LiDAR config loading.

No AirSim import or RPC connection required.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.runtime_config import (
    LidarRuntimeConfig,
    _validate_frame_timeout,
    _validate_max_consecutive_invalid,
    load_lidar_runtime_config,
)
from sensors.lidar_reader import LidarReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests — load_lidar_runtime_config
# ---------------------------------------------------------------------------

class TestLoadLidarRuntimeConfig:
    def test_yaml_timeout_loaded(self):
        """frame_timeout_seconds: 1.25 → config value is 1.25."""
        path = _make_yaml("""
airsim:
  vehicle_name: Drone1
  lidar_name: LidarSensor1
lidar:
  frame_timeout_seconds: 1.25
""")
        cfg = load_lidar_runtime_config(path)
        assert cfg.frame_timeout_seconds == 1.25

    def test_lidar_section_missing_uses_default(self):
        """No lidar section → safe default."""
        path = _make_yaml("""
airsim:
  vehicle_name: Drone1
  lidar_name: LidarSensor1
""")
        cfg = load_lidar_runtime_config(path)
        assert cfg.frame_timeout_seconds == 0.5

    def test_timeout_missing_uses_default(self):
        """lidar section present but frame_timeout_seconds missing."""
        path = _make_yaml("""
airsim:
  vehicle_name: Drone1
  lidar_name: LidarSensor1
lidar:
  max_consecutive_invalid: 5
""")
        cfg = load_lidar_runtime_config(path)
        assert cfg.frame_timeout_seconds == 0.5
        assert cfg.max_consecutive_invalid == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_lidar_runtime_config("/nonexistent/config.yaml")

    def test_yaml_parse_error(self):
        path = _make_yaml("{{{bad: [")
        with pytest.raises(Exception):  # yaml.YAMLError
            load_lidar_runtime_config(path)


# ---------------------------------------------------------------------------
# Tests — _validate_frame_timeout
# ---------------------------------------------------------------------------

class TestValidateFrameTimeout:
    def test_valid_int(self):
        assert _validate_frame_timeout(2) == 2.0

    def test_valid_float(self):
        assert _validate_frame_timeout(0.75) == 0.75

    def test_min_boundary(self):
        assert _validate_frame_timeout(0.05) == 0.05

    def test_max_boundary(self):
        assert _validate_frame_timeout(10.0) == 10.0

    # --- Rejected values ---

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(0.0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(-1)

    def test_below_min_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(0.01)

    def test_above_max_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(10.01)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(float("inf"))

    def test_string_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout("0.5")

    def test_bool_true_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(True)

    def test_bool_false_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(False)

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            _validate_frame_timeout(None)


# ---------------------------------------------------------------------------
# Tests — _validate_max_consecutive_invalid
# ---------------------------------------------------------------------------

class TestValidateMaxConsecutiveInvalid:
    def test_valid(self):
        assert _validate_max_consecutive_invalid(10) == 10

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(-5)

    def test_float_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(3.5)

    def test_string_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid("10")

    def test_bool_true_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(True)

    def test_bool_false_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(False)

    def test_above_10000_rejected(self):
        with pytest.raises(ValueError):
            _validate_max_consecutive_invalid(10001)


# ---------------------------------------------------------------------------
# Tests — LidarReader self-validation (ROUND 2.3)
# ---------------------------------------------------------------------------

class TestLidarReaderTimeoutValidation:
    """LidarReader.__init__ must reject bad timeout values itself."""

    def _make_adapter(self):
        adapter = MagicMock()
        adapter.vehicle_name = "Drone1"
        adapter.lidar_name = "LidarSensor1"
        return adapter

    def test_valid_timeout_accepted(self):
        adapter = self._make_adapter()
        reader = LidarReader(adapter, frame_timeout_seconds=1.5)
        assert reader._frame_timeout_seconds == 1.5

    def test_default_accepted(self):
        adapter = self._make_adapter()
        reader = LidarReader(adapter)
        assert reader._frame_timeout_seconds == 0.5

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=0.0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=-1.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=float("inf"))

    def test_string_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds="0.5")  # type: ignore

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=True)  # type: ignore

    def test_below_min_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=0.01)

    def test_above_max_rejected(self):
        with pytest.raises(ValueError):
            LidarReader(self._make_adapter(), frame_timeout_seconds=10.01)

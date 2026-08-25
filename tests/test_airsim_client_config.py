"""
Unit tests for AirSimClientAdapter configuration logic.

Tests config defaults, YAML loading, kwargs override, and validation.
No RPC connection is ever attempted.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.airsim_client import AirSimClientAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml(content: str) -> str:
    """Write a temporary YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests — defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_no_config_no_kwargs(self):
        adapter = AirSimClientAdapter()
        assert adapter._ip == "127.0.0.1"
        assert adapter._port == 41451
        assert adapter._vehicle_name == "Drone1"
        assert adapter._lidar_name == "LidarSensor1"

    def test_default_readonly(self):
        adapter = AirSimClientAdapter()
        assert adapter.readonly is True


# ---------------------------------------------------------------------------
# Tests — YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_yaml_values_loaded(self):
        yaml_path = _make_yaml("""
airsim:
  ip: "10.0.0.1"
  port: 5000
  vehicle_name: "TestDrone"
  lidar_name: "TestLidar"
""")
        adapter = AirSimClientAdapter(config_path=yaml_path)
        assert adapter._ip == "10.0.0.1"
        assert adapter._port == 5000
        assert adapter._vehicle_name == "TestDrone"
        assert adapter._lidar_name == "TestLidar"

    def test_partial_yaml(self):
        """YAML with only some fields — rest stay at defaults."""
        yaml_path = _make_yaml("""
airsim:
  port: 9999
""")
        adapter = AirSimClientAdapter(config_path=yaml_path)
        assert adapter._ip == "127.0.0.1"
        assert adapter._port == 9999
        assert adapter._vehicle_name == "Drone1"

    def test_empty_yaml(self):
        yaml_path = _make_yaml("{}")
        adapter = AirSimClientAdapter(config_path=yaml_path)
        assert adapter._ip == "127.0.0.1"
        assert adapter._port == 41451


# ---------------------------------------------------------------------------
# Tests — kwargs override
# ---------------------------------------------------------------------------

class TestKwargsOverride:
    def test_kwargs_override_yaml(self):
        yaml_path = _make_yaml("""
airsim:
  ip: "10.0.0.1"
  port: 5000
  vehicle_name: "YamlDrone"
  lidar_name: "YamlLidar"
""")
        adapter = AirSimClientAdapter(
            config_path=yaml_path,
            ip="192.168.1.1",
            vehicle_name="KwargDrone",
        )
        assert adapter._ip == "192.168.1.1"       # kwargs wins
        assert adapter._port == 5000               # YAML (no kwargs)
        assert adapter._vehicle_name == "KwargDrone"  # kwargs wins
        assert adapter._lidar_name == "YamlLidar"  # YAML (no kwargs)

    def test_all_kwargs_override(self):
        yaml_path = _make_yaml("""
airsim:
  ip: "10.0.0.1"
  port: 5000
  vehicle_name: "YamlDrone"
  lidar_name: "YamlLidar"
""")
        adapter = AirSimClientAdapter(
            config_path=yaml_path,
            ip="a.b.c.d",
            port=12345,
            vehicle_name="KwargDrone",
            lidar_name="KwargLidar",
        )
        assert adapter._ip == "a.b.c.d"
        assert adapter._port == 12345
        assert adapter._vehicle_name == "KwargDrone"
        assert adapter._lidar_name == "KwargLidar"

    def test_kwargs_only_no_yaml(self):
        adapter = AirSimClientAdapter(
            ip="172.16.0.1",
            port=8080,
            vehicle_name="MyDrone",
            lidar_name="MyLidar",
        )
        assert adapter._ip == "172.16.0.1"
        assert adapter._port == 8080
        assert adapter._vehicle_name == "MyDrone"
        assert adapter._lidar_name == "MyLidar"


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

class TestPortValidation:
    def test_port_zero_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port=0)

    def test_port_65536_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port=65536)

    def test_port_negative_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port=-1)

    # --- Strict type checks (ROUND 2.3) ---

    def test_port_string_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port="41451")

    def test_port_float_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port=41451.0)

    def test_port_bool_rejected(self):
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(port=True)


class TestPortInYaml:
    """Strict port validation from YAML source."""

    def test_yaml_port_string_rejected(self):
        yaml_path = _make_yaml("""
airsim:
  port: "41451"
""")
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(config_path=yaml_path)

    def test_yaml_port_float_rejected(self):
        yaml_path = _make_yaml("""
airsim:
  port: 41451.0
""")
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(config_path=yaml_path)

    def test_yaml_port_bool_rejected(self):
        yaml_path = _make_yaml("""
airsim:
  port: true
""")
        with pytest.raises(ValueError, match="port"):
            AirSimClientAdapter(config_path=yaml_path)


class TestNameValidation:
    def test_empty_vehicle_name_rejected(self):
        with pytest.raises(ValueError, match="vehicle_name"):
            AirSimClientAdapter(vehicle_name="")

    def test_empty_lidar_name_rejected(self):
        with pytest.raises(ValueError, match="lidar_name"):
            AirSimClientAdapter(lidar_name="")

    def test_whitespace_only_vehicle_name_rejected(self):
        with pytest.raises(ValueError, match="vehicle_name"):
            AirSimClientAdapter(vehicle_name="   ")

    def test_whitespace_only_lidar_name_rejected(self):
        with pytest.raises(ValueError, match="lidar_name"):
            AirSimClientAdapter(lidar_name="\t\n")


class TestFileErrors:
    def test_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AirSimClientAdapter(config_path="/nonexistent/path/config.yaml")

    def test_yaml_parse_error(self):
        yaml_path = _make_yaml("{{{invalid: yaml: [")
        with pytest.raises(Exception):  # yaml.YAMLError or similar
            AirSimClientAdapter(config_path=yaml_path)

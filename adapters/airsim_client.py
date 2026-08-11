"""
AirSim RPC client adapter.

Responsibilities
----------------
- Lazily connect to the AirSim RPC server (NEVER at import time).
- Read connection parameters from a YAML config or explicit kwargs.
- Wrap ``airsim.MultirotorClient`` for common operations.
- Provide a read-only mode that refuses control commands.
- Offer a safe ``close()`` that does NOT send flight commands.

Config priority
---------------
1. YAML config file (if ``config_path`` is provided).
2. Explicit kwargs (``ip``, ``port``, ``vehicle_name``, ``lidar_name``)
   override the YAML values when they are **not None**.

Loading the ``airsim`` package
------------------------------
This module does NOT require ``pip install airsim``.  It expects the
``airsim`` package to be importable via one of:

1. ``sys.path`` already includes the AirSim PythonClient directory
   (e.g. when running from inside that directory and ``setup_path``
   has been called).

2. ``PYTHONPATH`` points to the PythonClient directory before launch.

3. The caller sets the environment variable
   ``AIRSIM_PYTHONCLIENT_PATH`` to the absolute path of the
   ``PythonClient`` directory.  This module will prepend it to
   ``sys.path`` at connection time (lazy).

See README.md for detailed setup instructions.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Module-level port validation (ROUND 2.3 — strict)
# ------------------------------------------------------------------

def _validate_port_value(value, source: str = "kwargs") -> int:
    """Strictly validate an RPC port value.

    Rejects ``bool`` (which is a subclass of ``int``), strings, floats,
    and any value outside the 1–65535 range.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"port must be an int, got bool ({value!r}) from {source}"
        )
    if not isinstance(value, int):
        raise ValueError(
            f"port must be an int, got {type(value).__name__} ({value!r}) from {source}"
        )
    if value < 1 or value > 65535:
        raise ValueError(
            f"port must be 1–65535, got {value} from {source}"
        )
    return value


# ------------------------------------------------------------------
# AirSimClientAdapter
# ------------------------------------------------------------------


class AirSimClientAdapter:
    """Lazy-connecting wrapper around ``airsim.MultirotorClient``.

    The AirSim connection is **not** established at construction time.
    Call ``connect()`` explicitly when ready.

    Parameters
    ----------
    config_path:
        Path to a YAML config file (see ``configs/vehicle.yaml``).
    ip:
        Override RPC IP.  Default ``"127.0.0.1"``.
    port:
        Override RPC port.  Default ``41451``.
    vehicle_name:
        Override vehicle name.  Default ``"Drone1"``.
    lidar_name:
        Override LiDAR sensor name.  Default ``"LidarSensor1"``.
    readonly:
        If ``True`` (default), all control methods raise ``RuntimeError``.
        Set to ``False`` only when flight control is explicitly required.
    """

    # Hard defaults (used when neither YAML nor kwargs provide a value).
    _DEFAULT_IP = "127.0.0.1"
    _DEFAULT_PORT = 41451
    _DEFAULT_VEHICLE = "Drone1"
    _DEFAULT_LIDAR = "LidarSensor1"

    def __init__(
        self,
        config_path: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        vehicle_name: Optional[str] = None,
        lidar_name: Optional[str] = None,
        readonly: bool = True,
    ) -> None:
        # Step 1 — start with hard defaults.
        self._ip: str = self._DEFAULT_IP
        self._port: int = self._DEFAULT_PORT
        self._vehicle_name: str = self._DEFAULT_VEHICLE
        self._lidar_name: str = self._DEFAULT_LIDAR
        self._readonly: bool = readonly

        # Step 2 — overlay YAML config if provided.
        if config_path is not None:
            self._load_config(config_path)

        # Step 3 — overlay all non-None explicit kwargs (with strict validation).
        if ip is not None:
            self._ip = ip
        if port is not None:
            self._port = _validate_port_value(port, source="kwargs")
        if vehicle_name is not None:
            self._vehicle_name = vehicle_name
        if lidar_name is not None:
            self._lidar_name = lidar_name

        # Step 4 — validate.
        self._validate_config()

        self._client: Any = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Read YAML config and overlay onto current values."""
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        airsim_cfg = cfg.get("airsim", {})
        if "ip" in airsim_cfg:
            self._ip = airsim_cfg["ip"]
        if "port" in airsim_cfg:
            self._port = _validate_port_value(airsim_cfg["port"], source="YAML")
        if "vehicle_name" in airsim_cfg:
            self._vehicle_name = airsim_cfg["vehicle_name"]
        if "lidar_name" in airsim_cfg:
            self._lidar_name = airsim_cfg["lidar_name"]

        lidar_cfg = cfg.get("lidar", {})
        # (lidar config consumed by LidarReader, not here)

    @staticmethod
    def _validate_kwarg_port(port) -> int:
        """Validate a port value from a kwarg (may be int or None)."""
        if port is None:
            return None  # type: ignore[return-value]
        return _validate_port_value(port, source="kwargs")

    @staticmethod
    def _validate_name(value: str, label: str) -> str:
        """Validate a name is a non-empty non-blank string."""
        if not isinstance(value, str):
            raise ValueError(
                f"{label} must be a string, got {type(value).__name__} ({value!r})"
            )
        if not value.strip():
            raise ValueError(f"{label} must not be empty or whitespace-only")
        return value

    def _validate_config(self) -> None:
        """Sanity-check configuration values."""
        if not isinstance(self._port, int) or isinstance(self._port, bool):
            raise ValueError(f"port must be an int, got {type(self._port).__name__} ({self._port!r})")
        if self._port < 1 or self._port > 65535:
            raise ValueError(f"port must be 1–65535, got {self._port!r}")
        self._vehicle_name = self._validate_name(self._vehicle_name, "vehicle_name")
        self._lidar_name = self._validate_name(self._lidar_name, "lidar_name")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish the RPC connection.

        Resolves the ``airsim`` import lazily so that import errors only
        surface when a connection is actually attempted — not at module
        load time.

        Raises
        ------
        ImportError
            If ``airsim`` cannot be imported (see docstring).
        ConnectionError
            If the RPC handshake fails.
        """
        if self._connected:
            logger.info("Already connected — skipping.")
            return

        self._ensure_airsim_importable()
        import airsim  # type: ignore[import-untyped]

        logger.info(
            "Connecting to AirSim RPC at %s:%d …", self._ip, self._port
        )
        try:
            self._client = airsim.MultirotorClient(
                ip=self._ip, port=self._port
            )
            self._client.confirmConnection()
            self._connected = True
            logger.info("AirSim connection confirmed (client ver %d).", 1)
        except Exception as exc:
            self._client = None
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to AirSim at {self._ip}:{self._port}: {exc}"
            ) from exc

    def close(self) -> None:
        """Safely release the RPC client.

        Does **not** send any flight control commands (no disarm,
        no land, no release of API control).
        """
        if self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception:
                pass
        self._client = None
        self._connected = False
        logger.info("AirSim client closed.")

    # ------------------------------------------------------------------
    # AirSim package resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_airsim_importable() -> None:
        """Prepend the AirSim PythonClient dir to ``sys.path`` if needed."""
        try:
            import airsim  # noqa: F401
            return
        except ImportError:
            pass

        candidate = os.environ.get("AIRSIM_PYTHONCLIENT_PATH", "")
        if candidate:
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                import airsim  # noqa: F401
                return
            except ImportError:
                pass

        raise ImportError(
            "Cannot import 'airsim'.  "
            "Set AIRSIM_PYTHONCLIENT_PATH to the AirSim-main/PythonClient "
            "directory, or add it to PYTHONPATH.  See README.md."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Access the underlying ``airsim.MultirotorClient``."""
        if not self._connected or self._client is None:
            raise RuntimeError("Not connected.  Call connect() first.")
        return self._client

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name

    @property
    def lidar_name(self) -> str:
        return self._lidar_name

    @property
    def readonly(self) -> bool:
        return self._readonly

    # ------------------------------------------------------------------
    # Read-only operations (always permitted)
    # ------------------------------------------------------------------

    def confirm_connection(self) -> bool:
        """Verify the RPC link is alive."""
        try:
            self.client.confirmConnection()
            return True
        except Exception:
            return False

    def list_vehicles(self) -> List[str]:
        """Return the list of vehicle names known to AirSim."""
        try:
            vehicles = self.client.listVehicles()
            return list(vehicles) if vehicles else []
        except Exception as exc:
            raise ConnectionError(
                f"listVehicles() failed: {exc}"
            ) from exc

    def validate_vehicle_present(self) -> bool:
        """Check that ``vehicle_name`` appears in ``listVehicles()``.

        Returns
        -------
        bool
            ``True`` if the configured vehicle name is present.
        """
        vehicles = self.list_vehicles()
        present = self._vehicle_name in vehicles
        if not present:
            logger.warning(
                "Vehicle %r not found in %s", self._vehicle_name, vehicles
            )
        return present

    def get_raw_client(self) -> Any:
        """Expose the underlying AirSim client for direct API calls.

        Callers (e.g. sensor readers) use this for ``getLidarData()``,
        ``getMultirotorState()``, ``simGetCollisionInfo()``.
        """
        return self.client

    # ------------------------------------------------------------------
    # Control guard
    # ------------------------------------------------------------------

    def _assert_writable(self) -> None:
        if self._readonly:
            raise RuntimeError(
                "AirSimClientAdapter is in read-only mode.  "
                "Set readonly=False to enable control operations."
            )

    def enable_readonly(self) -> None:
        self._readonly = True

    def disable_readonly(self) -> None:
        """Allow control operations.  Call deliberately — NOT automatic."""
        logger.warning("Read-only mode DISABLED.  Control commands are now permitted.")
        self._readonly = False

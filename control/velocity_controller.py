"""
Safe velocity control interface.

This module provides two explicitly-named velocity-sending functions:

- ``send_velocity_world_ned()``  → ``moveByVelocityAsync`` (world NED)
- ``send_velocity_body_frd()``   → ``moveByVelocityBodyFrameAsync`` (body FRD)

Coordinate semantics
--------------------
``send_velocity_world_ned``:
    vx = North / world X
    vy = East  / world Y
    vz = Down  / world Z

``send_velocity_body_frd``:
    vx = Forward (body X)
    vy = Right   (body Y)
    vz = Down    (body Z)

Safety rules (enforced in this module)
--------------------------------------
1. NaN / inf in any velocity component → rejected.
2. Horizontal speed (sqrt(vx²+vy²)) clamped to
   ``max_horizontal_speed_mps``.
3. Vertical speed (|vz|) clamped to ``max_vertical_speed_mps``.
4. Yaw rate clamped to ``max_yaw_rate_radps``.
5. ``duration`` must be positive and ≤ a reasonable upper bound.
6. ``vehicle_name`` must be passed explicitly (no default).

Yaw rate unit boundary
----------------------
- **Internal**: all yaw rates are in **rad/s**.
- ``max_yaw_rate_radps`` is in rad/s.
- AirSim ``YawMode(is_rate=True)`` expects **deg/s**.
- Conversion happens ONLY inside ``_build_yaw_mode_from_radps()``
  at the API boundary:
      ``yaw_or_rate = math.degrees(yaw_rate_radps)``

Dependency injection
--------------------
``VelocityController`` accepts an optional ``airsim_module`` parameter.
At runtime this is the real ``airsim`` package; in unit tests a fake
module (e.g. ``MagicMock``) is injected so no ``airsim`` import is
required.

Controlling authority
---------------------
This module does **NOT** automatically:
- call ``enableApiControl``
- call ``armDisarm``
- call ``takeoff``
- call ``hover``
- call ``land``
- call ``disarm``
- call ``releaseApiControl``

These are available as public methods but must be called **explicitly**
by the operator or a higher-level supervisor.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel — raised when input fails validation
# ---------------------------------------------------------------------------

class VelocityCommandRejected(ValueError):
    """Raised when a velocity command fails validation."""


# ---------------------------------------------------------------------------
# Velocity controller
# ---------------------------------------------------------------------------

class VelocityController:
    """Safe wrapper around AirSim velocity move commands.

    Parameters
    ----------
    adapter:
        A connected ``AirSimClientAdapter`` (readonly **must** be ``False``).
    airsim_module:
        The ``airsim`` Python module (or a fake for testing).
        Used for ``DrivetrainType``, ``YawMode``, and async calls.
        If ``None``, resolved lazily at first use.
    config_path:
        Optional path to a ``vehicle.yaml`` config for limit defaults.
    max_horizontal_speed_mps:
        Hard cap on horizontal speed (m/s).  Default 2.0.
    max_vertical_speed_mps:
        Hard cap on vertical speed (m/s).  Default 0.5.
    max_yaw_rate_radps:
        Hard cap on yaw rate (rad/s).  Default 0.5.
    command_duration_seconds:
        Default command duration (s).  Default 0.2.
    """

    def __init__(
        self,
        adapter,
        airsim_module: Any = None,
        config_path: Optional[str] = None,
        max_horizontal_speed_mps: float = 2.0,
        max_vertical_speed_mps: float = 0.5,
        max_yaw_rate_radps: float = 0.5,
        command_duration_seconds: float = 0.2,
    ) -> None:
        self._adapter = adapter
        self._airsim = airsim_module  # None → lazy resolve

        # Load config overrides if provided.
        if config_path is not None:
            import yaml
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            ctrl = cfg.get("control", {})
            max_horizontal_speed_mps = ctrl.get(
                "max_horizontal_speed_mps", max_horizontal_speed_mps
            )
            max_vertical_speed_mps = ctrl.get(
                "max_vertical_speed_mps", max_vertical_speed_mps
            )
            max_yaw_rate_radps = ctrl.get(
                "max_yaw_rate_radps", max_yaw_rate_radps
            )
            command_duration_seconds = ctrl.get(
                "command_duration_seconds", command_duration_seconds
            )

        self.max_horizontal_speed_mps = max_horizontal_speed_mps
        self.max_vertical_speed_mps = max_vertical_speed_mps
        self.max_yaw_rate_radps = max_yaw_rate_radps
        self.command_duration_seconds = command_duration_seconds

    # ------------------------------------------------------------------
    # Lazy airsim resolution
    # ------------------------------------------------------------------

    def _get_airsim(self):
        """Return the airsim module, resolving lazily if needed."""
        if self._airsim is None:
            import airsim  # type: ignore[import-untyped]
            self._airsim = airsim
        return self._airsim

    # ------------------------------------------------------------------
    # Main velocity commands
    # ------------------------------------------------------------------

    def send_velocity_world_ned(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: Optional[float] = None,
        vehicle_name: Optional[str] = None,
        yaw_rate: Optional[float] = None,
    ) -> None:
        """Send world-frame NED velocity via ``moveByVelocityAsync``.

        Parameters
        ----------
        vx: North velocity (m/s).
        vy: East velocity (m/s).
        vz: Down velocity (m/s).  Positive = descend.
        duration: Command duration in seconds.  Default from config.
        vehicle_name: Target vehicle.  **Must** be provided.
        yaw_rate: Optional yaw rate (**rad/s**).  Converted to deg/s
            at the AirSim API boundary.

        Raises
        ------
        VelocityCommandRejected
            If any validation check fails.
        """
        if vehicle_name is None:
            raise VelocityCommandRejected("vehicle_name must be provided")

        vx, vy, vz = self._validate_velocity(vx, vy, vz)
        dur = self._validate_duration(duration)
        yaw_rate = self._validate_yaw_rate(yaw_rate)

        self._adapter._assert_writable()

        a = self._get_airsim()
        client = self._adapter.get_raw_client()
        yaw_mode = self._build_yaw_mode_from_radps(yaw_rate)

        logger.debug(
            "moveByVelocityAsync: vx=%.3f vy=%.3f vz=%.3f dur=%.3f yaw_rate_radps=%s",
            vx, vy, vz, dur, yaw_rate,
        )
        client.moveByVelocityAsync(
            vx, vy, vz, dur,
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=vehicle_name,
        )

    def send_velocity_body_frd(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: Optional[float] = None,
        vehicle_name: Optional[str] = None,
        yaw_rate: Optional[float] = None,
    ):
        """Send body-frame FRD velocity via ``moveByVelocityBodyFrameAsync``.

        Returns the AirSim future for the caller to join() if needed.

        Parameters
        ----------
        vx: Forward velocity (m/s).
        vy: Right velocity (m/s).
        vz: Down velocity (m/s).
        duration: Command duration in seconds.  Default from config.
        vehicle_name: Target vehicle.  **Must** be provided.
        yaw_rate: Optional yaw rate (**rad/s**).  Converted to deg/s
            at the AirSim API boundary.

        Raises
        ------
        VelocityCommandRejected
            If any validation check fails.
        """
        if vehicle_name is None:
            raise VelocityCommandRejected("vehicle_name must be provided")

        vx, vy, vz = self._validate_velocity(vx, vy, vz)
        dur = self._validate_duration(duration)
        yaw_rate = self._validate_yaw_rate(yaw_rate)

        self._adapter._assert_writable()

        a = self._get_airsim()
        client = self._adapter.get_raw_client()
        yaw_mode = self._build_yaw_mode_from_radps(yaw_rate)

        logger.debug(
            "moveByVelocityBodyFrameAsync: vx=%.3f vy=%.3f vz=%.3f dur=%.3f yaw_rate_radps=%s",
            vx, vy, vz, dur, yaw_rate,
        )
        return client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, dur,
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=vehicle_name,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_velocity(
        self, vx: float, vy: float, vz: float
    ) -> Tuple[float, float, float]:
        """Check for NaN/inf and clamp magnitude."""
        for val, label in [(vx, "vx"), (vy, "vy"), (vz, "vz")]:
            if math.isnan(val) or math.isinf(val):
                raise VelocityCommandRejected(
                    f"Velocity component {label}={val} is NaN or inf"
                )

        # Horizontal clamp
        h_speed = math.sqrt(vx * vx + vy * vy)
        if h_speed > self.max_horizontal_speed_mps:
            scale = self.max_horizontal_speed_mps / h_speed
            vx *= scale
            vy *= scale
            logger.debug(
                "Horizontal speed clamped: %.3f → %.3f m/s",
                h_speed, self.max_horizontal_speed_mps,
            )

        # Vertical clamp
        if abs(vz) > self.max_vertical_speed_mps:
            vz = math.copysign(self.max_vertical_speed_mps, vz)
            logger.debug(
                "Vertical speed clamped to ±%.3f m/s", self.max_vertical_speed_mps
            )

        return vx, vy, vz

    def _validate_duration(self, duration: Optional[float]) -> float:
        """Check duration is positive and reasonable."""
        dur = duration if duration is not None else self.command_duration_seconds
        if math.isnan(dur) or math.isinf(dur) or dur <= 0.0:
            raise VelocityCommandRejected(f"Invalid duration: {dur}")
        if dur > 10.0:
            raise VelocityCommandRejected(
                f"Duration {dur}s exceeds max 10s — split into shorter commands"
            )
        return dur

    def _validate_yaw_rate(self, yaw_rate: Optional[float]) -> Optional[float]:
        """Validate and clamp yaw rate in **rad/s**."""
        if yaw_rate is None:
            return None
        if math.isnan(yaw_rate) or math.isinf(yaw_rate):
            raise VelocityCommandRejected(
                f"yaw_rate={yaw_rate} is NaN or inf"
            )
        if abs(yaw_rate) > self.max_yaw_rate_radps:
            yaw_rate = math.copysign(self.max_yaw_rate_radps, yaw_rate)
            logger.debug("Yaw rate clamped to ±%.3f rad/s", self.max_yaw_rate_radps)
        return yaw_rate

    def _build_yaw_mode_from_radps(self, yaw_rate_radps: Optional[float]):
        """Build an AirSim YawMode object, converting rad/s → deg/s.

        **Unit boundary**: AirSim ``YawMode(is_rate=True)`` expects
        ``yaw_or_rate`` in **degrees/s**.  This method performs the
        conversion:
            ``yaw_or_rate = math.degrees(yaw_rate_radps)``
        """
        a = self._get_airsim()
        if yaw_rate_radps is not None:
            yaw_rate_degps = math.degrees(yaw_rate_radps)
            return a.YawMode(is_rate=True, yaw_or_rate=yaw_rate_degps)
        return a.YawMode()  # default: is_rate=True, yaw_or_rate=0.0

    # ------------------------------------------------------------------
    # Authority management (NOT auto-called — caller must invoke explicitly)
    # ------------------------------------------------------------------

    def enable_api_control(self, vehicle_name: Optional[str] = None) -> None:
        """Explicitly take API control.  NOT called automatically."""
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("enableApiControl(vehicle_name=%r)", vname)
        self._adapter.get_raw_client().enableApiControl(True, vname)

    def arm(self, vehicle_name: Optional[str] = None) -> None:
        """Explicitly arm the drone.  NOT called automatically."""
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("armDisarm(True, vehicle_name=%r)", vname)
        self._adapter.get_raw_client().armDisarm(True, vname)

    def takeoff(self, vehicle_name: Optional[str] = None, timeout_sec: float = 20.0) -> None:
        """Initiate takeoff and block until complete (``takeoffAsync().join()``).

        NOT called automatically.
        """
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("takeoff(vehicle_name=%r, timeout_sec=%.1f)", vname, timeout_sec)
        future = self._adapter.get_raw_client().takeoffAsync(
            vehicle_name=vname, timeout_sec=timeout_sec
        )
        future.join()

    def hover(self, vehicle_name: Optional[str] = None) -> None:
        """Hover in place via ``hoverAsync().join()``.

        Uses the dedicated hoverAsync API — NOT a 0.2 s zero-velocity
        command that would expire.
        """
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("hover(vehicle_name=%r)", vname)
        future = self._adapter.get_raw_client().hoverAsync(vehicle_name=vname)
        future.join()

    def land(self, vehicle_name: Optional[str] = None, timeout_sec: float = 20.0) -> None:
        """Initiate landing and block until complete (``landAsync().join()``).

        NOT called automatically.
        """
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("land(vehicle_name=%r, timeout_sec=%.1f)", vname, timeout_sec)
        future = self._adapter.get_raw_client().landAsync(
            vehicle_name=vname, timeout_sec=timeout_sec
        )
        future.join()

    def disarm(self, vehicle_name: Optional[str] = None) -> None:
        """Explicitly disarm the drone.  NOT called automatically."""
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("armDisarm(False, vehicle_name=%r)", vname)
        self._adapter.get_raw_client().armDisarm(False, vname)

    def release_api_control(self, vehicle_name: Optional[str] = None) -> None:
        """Explicitly release API control.  NOT called automatically."""
        self._adapter._assert_writable()
        vname = vehicle_name or self._adapter.vehicle_name
        logger.info("enableApiControl(False, vehicle_name=%r)", vname)
        self._adapter.get_raw_client().enableApiControl(False, vname)

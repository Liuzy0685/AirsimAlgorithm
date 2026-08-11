"""
Vehicle state reader.

Reads the full multirotor state from AirSim via ``getMultirotorState()``
and produces a ``VehicleState`` object.

Coordinate system
-----------------
All position and velocity values are in **NED** (North-East-Down):
    +X = North / forward
    +Y = East  / right
    +Z = Down

Orientation / angles
--------------------
- ``orientation_quaternion_xyzw`` is in AirSim native order: ``[x, y, z, w]``
  (from ``Quaternionr.x_val/.y_val/.z_val/.w_val``).
- ``roll_rad``, ``pitch_rad``, ``yaw_rad`` are derived via the injected
  ``euler_converter`` callable.
- **Runtime**: ``airsim.to_eularian_angles()`` is used, which returns
  ``(pitch, roll, yaw)`` — confirmed in AirSim 1.8.1 ``airsim/utils.py:80``.
- **Unit tests** inject a fake converter (e.g. a lambda); no ``airsim``
  import is required.
- All angles are stored in **radians** internally.

Dependency injection
--------------------
``StateReader`` accepts an optional ``euler_converter`` callable::

    reader = StateReader(adapter)                  # uses real airsim at runtime
    reader = StateReader(adapter, euler_converter=my_fake)  # test mode
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from adapters.airsim_client import AirSimClientAdapter
from models.vehicle_state import VehicleState

logger = logging.getLogger(__name__)


class StateReader:
    """Reads ``getMultirotorState()`` and returns a ``VehicleState``.

    Parameters
    ----------
    adapter:
        A connected ``AirSimClientAdapter``.
    vehicle_name:
        Override vehicle name.  Defaults to the adapter's value.
    euler_converter:
        Callable ``(quaternion) -> (pitch, roll, yaw)`` in radians.
        If ``None`` (default), uses ``airsim.to_eularian_angles`` at
        runtime — which means ``airsim`` must be importable at call time.
        Inject a fake for unit tests that run without ``airsim``.
    """

    def __init__(
        self,
        adapter: AirSimClientAdapter,
        vehicle_name: Optional[str] = None,
        euler_converter: Optional[Callable] = None,
    ) -> None:
        self._adapter = adapter
        self._vehicle_name = vehicle_name or adapter.vehicle_name
        self._euler_converter = euler_converter  # None → resolve at read time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> VehicleState:
        """Acquire one multirotor state snapshot.

        Returns
        -------
        VehicleState
        """
        raw = self._adapter.get_raw_client().getMultirotorState(
            vehicle_name=self._vehicle_name
        )
        received_mono = time.monotonic()
        return self._build_state(raw, received_mono)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _build_state(self, raw, received_mono: float = 0.0) -> VehicleState:
        # Resolve euler converter lazily.
        converter = self._euler_converter
        if converter is None:
            import airsim  # type: ignore[import-untyped]
            converter = airsim.to_eularian_angles

        kin = raw.kinematics_estimated
        pos = kin.position
        vel = kin.linear_velocity
        ang = kin.angular_velocity
        quat = kin.orientation  # Quaternionr

        # ---------------------------------------------------------------
        # CRITICAL: to_eularian_angles() returns (pitch, roll, yaw)
        # Confirmed: airsim/utils.py:80  →  return (pitch, roll, yaw)
        # ---------------------------------------------------------------
        pitch, roll, yaw = converter(quat)

        return VehicleState(
            position_ned_m=[float(pos.x_val), float(pos.y_val), float(pos.z_val)],
            linear_velocity_ned_mps=[
                float(vel.x_val),
                float(vel.y_val),
                float(vel.z_val),
            ],
            angular_velocity_body_radps=[
                float(ang.x_val),
                float(ang.y_val),
                float(ang.z_val),
            ],
            orientation_quaternion_xyzw=[
                float(quat.x_val),
                float(quat.y_val),
                float(quat.z_val),
                float(quat.w_val),
            ],
            roll_rad=float(roll),
            pitch_rad=float(pitch),
            yaw_rad=float(yaw),
            timestamp=int(raw.timestamp),
            ready=bool(raw.ready),
            can_arm=bool(raw.can_arm),
            landed_state=int(raw.landed_state),
            received_monotonic_seconds=received_mono,
        )

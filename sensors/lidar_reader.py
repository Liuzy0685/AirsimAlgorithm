"""
LiDAR point cloud reader.

Reads raw point clouds from AirSim via ``getLidarData()`` and produces
validated ``LidarFrame`` objects.

Invalid frame reasons (aligned with SAFETY_FINDINGS category system):

=============== ====================================================
invalid_reason   Condition
=============== ====================================================
rpc_error        ``getLidarData()`` raised an exception
empty             Point-cloud array has zero elements
malformed         Array length is not divisible by 3
bad_values        Array contains NaN or inf
missing_sensor    LiDAR name not found / sensor missing
stale             Raw timestamp unchanged beyond ``frame_timeout_seconds``
unknown_error     Unexpected failure during frame building (logged)
=============== ====================================================

Timestamp strategy (revised — ROUND 2.2)
----------------------------------------
- ``raw_timestamp_ns`` — AirSim ``time_stamp`` as-is (large uint64,
  Unreal-time epoch).  Used for: detecting frame updates.
- ``received_monotonic_seconds`` — ``time.monotonic()`` recorded **after**
  the RPC call returns (or after catching the RPC exception).  This
  represents the moment the data arrived in the Python process.
- ``last_new_timestamp_monotonic`` — ``time.monotonic()`` of the last
  time ``raw_timestamp_ns`` *changed*.
- The sentinel value for "never read" is ``None`` (NOT ``0``), so
  timestamp=0 data is handled correctly.
- A frame is ``stale`` when ``raw_timestamp_ns`` has NOT changed AND
  ``current_monotonic - last_new_timestamp_monotonic > frame_timeout_seconds``.
  A single repeated timestamp is NOT immediately stale — the read rate
  may exceed the LiDAR update rate.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import numpy as np

from adapters.airsim_client import AirSimClientAdapter
from models.lidar_frame import LidarFrame

logger = logging.getLogger(__name__)


class LidarReader:
    """Reads and validates LiDAR point clouds from AirSim.

    Parameters
    ----------
    adapter:
        A connected ``AirSimClientAdapter``.
    vehicle_name:
        Explicit vehicle name override.
    lidar_name:
        Explicit LiDAR name override.
    frame_timeout_seconds:
        Maximum age of the last *new* timestamp before a frame is
        considered stale.  Default 0.5 s.
    monotonic_clock:
        Inject a monotonic-clock callable (``-> float``) for testing.
        Defaults to ``time.monotonic``.
    """

    # Allowed range for frame_timeout_seconds (ROUND 2.3).
    _MIN_TIMEOUT = 0.05
    _MAX_TIMEOUT = 10.0

    def __init__(
        self,
        adapter: AirSimClientAdapter,
        vehicle_name: Optional[str] = None,
        lidar_name: Optional[str] = None,
        frame_timeout_seconds: float = 0.5,
        monotonic_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        # --- Validate frame_timeout_seconds (ROUND 2.3) ---
        if isinstance(frame_timeout_seconds, bool):
            raise ValueError(
                f"frame_timeout_seconds must be a number, got bool ({frame_timeout_seconds!r})"
            )
        if not isinstance(frame_timeout_seconds, (int, float)):
            raise ValueError(
                f"frame_timeout_seconds must be int or float, "
                f"got {type(frame_timeout_seconds).__name__} ({frame_timeout_seconds!r})"
            )
        if math.isnan(frame_timeout_seconds) or math.isinf(frame_timeout_seconds):
            raise ValueError(
                f"frame_timeout_seconds must be finite, got {frame_timeout_seconds!r}"
            )
        if not (self._MIN_TIMEOUT <= frame_timeout_seconds <= self._MAX_TIMEOUT):
            raise ValueError(
                f"frame_timeout_seconds={frame_timeout_seconds} "
                f"out of range [{self._MIN_TIMEOUT}, {self._MAX_TIMEOUT}]"
            )

        self._adapter = adapter
        self._vehicle_name = vehicle_name or adapter.vehicle_name
        self._lidar_name = lidar_name or adapter.lidar_name
        self._frame_timeout_seconds = float(frame_timeout_seconds)
        self._clock = monotonic_clock if monotonic_clock is not None else time.monotonic

        # Staleness tracking — None = "never read yet" (ROUND 2.2 fix).
        self._last_raw_ts: Optional[int] = None
        self._last_new_ts_monotonic: Optional[float] = None
        self._consecutive_stale: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> LidarFrame:
        """Acquire one LiDAR frame with full validation.

        Returns
        -------
        LidarFrame
            Always returns a ``LidarFrame``.  Check ``frame_valid`` to
            determine whether the data is safe to use for navigation.
            When ``frame_valid`` is ``False``, ``point_cloud_sensor`` is
            an empty (0,3) array — never a populated array that could be
            misused.
        """
        # --- RPC call FIRST, then record time (ROUND 2.2 fix) ---
        try:
            raw = self._adapter.get_raw_client().getLidarData(
                lidar_name=self._lidar_name,
                vehicle_name=self._vehicle_name,
            )
            received_mono = self._clock()
        except Exception as exc:
            received_mono = self._clock()
            logger.error("getLidarData() RPC error: %s", exc)
            return LidarFrame(
                frame_valid=False,
                invalid_reason="rpc_error",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                received_monotonic_seconds=received_mono,
            )

        # Build the frame — catch-all for unexpected errors.
        try:
            return self._build_frame(raw, received_mono)
        except Exception as exc:
            logger.exception("Unexpected error building LidarFrame: %s", exc)
            return LidarFrame(
                frame_valid=False,
                invalid_reason="unknown_error",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                received_monotonic_seconds=received_mono,
            )

    # ------------------------------------------------------------------
    # Frame construction & validation
    # ------------------------------------------------------------------

    def _build_frame(self, raw, received_mono: float) -> LidarFrame:
        """Validate raw AirSim ``LidarData`` and produce a ``LidarFrame``."""

        # --- Read raw timestamp (may raise AttributeError → unknown_error) ---
        raw_ts_ns = int(raw.time_stamp)

        # --- 1. Check for missing sensor ---
        point_cloud_flat = getattr(raw, "point_cloud", None)
        if point_cloud_flat is None or (
            isinstance(point_cloud_flat, float) and point_cloud_flat == 0.0
        ):
            return LidarFrame(
                frame_valid=False,
                invalid_reason="missing_sensor",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                raw_timestamp_ns=raw_ts_ns,
                received_monotonic_seconds=received_mono,
            )

        # --- 2. Convert to NumPy array ---
        try:
            arr = np.asarray(point_cloud_flat, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            logger.error("Cannot convert point_cloud to float32 array: %s", exc)
            return LidarFrame(
                frame_valid=False,
                invalid_reason="unknown_error",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                raw_timestamp_ns=raw_ts_ns,
                received_monotonic_seconds=received_mono,
            )

        # --- 3. Empty array ---
        if arr.size == 0:
            return LidarFrame(
                frame_valid=False,
                invalid_reason="empty",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                raw_timestamp_ns=raw_ts_ns,
                received_monotonic_seconds=received_mono,
            )

        # --- 4. Malformed: length not divisible by 3 ---
        if arr.size % 3 != 0:
            logger.warning(
                "LiDAR point_cloud length %d not divisible by 3", arr.size
            )
            return LidarFrame(
                frame_valid=False,
                invalid_reason="malformed",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                point_count=arr.size // 3,
                raw_timestamp_ns=raw_ts_ns,
                received_monotonic_seconds=received_mono,
            )

        # --- 5. Check for NaN / inf ---
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        if has_nan or has_inf:
            logger.warning(
                "LiDAR point_cloud contains NaN=%s inf=%s", has_nan, has_inf
            )
            return LidarFrame(
                frame_valid=False,
                invalid_reason="bad_values",
                vehicle_name=self._vehicle_name,
                lidar_name=self._lidar_name,
                point_count=arr.size // 3,
                raw_timestamp_ns=raw_ts_ns,
                received_monotonic_seconds=received_mono,
            )

        # --- 6. Staleness check (revised — ROUND 2.2) ---
        # Sentinel: None = "never read yet".
        if self._last_raw_ts is None:
            # First frame — record and accept regardless of timestamp value.
            self._consecutive_stale = 0
            self._last_raw_ts = raw_ts_ns
            self._last_new_ts_monotonic = received_mono
        elif raw_ts_ns == self._last_raw_ts:
            # Timestamp unchanged — check if we've exceeded the timeout.
            age = received_mono - self._last_new_ts_monotonic
            if age > self._frame_timeout_seconds:
                self._consecutive_stale += 1
                logger.warning(
                    "LiDAR timestamp %d unchanged for %.3f s (> %.3f s) — stale",
                    raw_ts_ns, age, self._frame_timeout_seconds,
                )
                return LidarFrame(
                    frame_valid=False,
                    invalid_reason="stale",
                    vehicle_name=self._vehicle_name,
                    lidar_name=self._lidar_name,
                    raw_timestamp_ns=raw_ts_ns,
                    received_monotonic_seconds=received_mono,
                )
            else:
                # Short repeat — read rate > LiDAR rate; still valid.
                self._consecutive_stale += 1
                logger.debug(
                    "LiDAR timestamp repeated (age=%.3f s ≤ timeout) — still valid",
                    age,
                )
        else:
            # Timestamp changed — this is a fresh frame.
            self._consecutive_stale = 0
            self._last_raw_ts = raw_ts_ns
            self._last_new_ts_monotonic = received_mono

        # --- All checks passed — reshape to N×3. ---
        n_points = arr.size // 3
        point_cloud = arr.reshape((n_points, 3))

        # Extract sensor pose (may raise AttributeError → unknown_error).
        sensor_pose = {
            "position": {
                "x": float(raw.pose.position.x_val),
                "y": float(raw.pose.position.y_val),
                "z": float(raw.pose.position.z_val),
            },
            "orientation": {
                "w": float(raw.pose.orientation.w_val),
                "x": float(raw.pose.orientation.x_val),
                "y": float(raw.pose.orientation.y_val),
                "z": float(raw.pose.orientation.z_val),
            },
        }

        return LidarFrame(
            point_cloud_sensor=point_cloud,
            raw_timestamp_ns=raw_ts_ns,
            received_monotonic_seconds=received_mono,
            sensor_pose=sensor_pose,
            frame_valid=True,
            invalid_reason=None,
            point_count=n_points,
            vehicle_name=self._vehicle_name,
            lidar_name=self._lidar_name,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def consecutive_stale_count(self) -> int:
        return self._consecutive_stale

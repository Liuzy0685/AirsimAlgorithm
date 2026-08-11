"""
Collision state reader.

Reads collision info from AirSim via ``simGetCollisionInfo()`` and
produces a ``CollisionState`` object.

New-collision detection is built in::

    reader = CollisionReader(adapter)
    curr = reader.read()
    if curr.is_new_collision_event:
        handle_collision(curr)

The reader tracks ``raw_timestamp`` across calls.  A collision is "new"
when ``has_collided`` is ``True`` **and** ``raw_timestamp`` differs from
the previous collision timestamp (including the first time a non-zero
timestamp appears).  The ``is_new_collision_event`` field is populated
automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from adapters.airsim_client import AirSimClientAdapter
from models.collision_state import CollisionState

logger = logging.getLogger(__name__)


class CollisionReader:
    """Reads ``simGetCollisionInfo()`` and returns a ``CollisionState``.

    Parameters
    ----------
    adapter:
        A connected ``AirSimClientAdapter``.
    vehicle_name:
        Override vehicle name.  Defaults to the adapter's value.
    """

    def __init__(
        self,
        adapter: AirSimClientAdapter,
        vehicle_name: Optional[str] = None,
    ) -> None:
        self._adapter = adapter
        self._vehicle_name = vehicle_name or adapter.vehicle_name
        # Track previous collision timestamp for new-event detection.
        self._prev_collision_ts: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> CollisionState:
        """Acquire current collision info.

        Returns
        -------
        CollisionState
            Includes ``is_new_collision_event`` — ``True`` for a
            genuinely new collision.
        """
        raw = self._adapter.get_raw_client().simGetCollisionInfo(
            vehicle_name=self._vehicle_name
        )
        received_mono = time.monotonic()
        return self._build_state(raw, received_mono)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _build_state(self, raw, received_mono: float = 0.0) -> CollisionState:
        # AirSim time_stamp is an integer — preserve it as int.
        raw_ts = int(raw.time_stamp)
        has_collided = bool(raw.has_collided)

        # New collision event if:
        #   - has_collided is True
        #   - timestamp is non-zero
        #   - timestamp differs from previous collision timestamp
        is_new = (
            has_collided
            and raw_ts != 0
            and raw_ts != self._prev_collision_ts
        )

        if is_new:
            logger.info("New collision detected: ts=%d, object=%r", raw_ts, raw.object_name)
            self._prev_collision_ts = raw_ts

        return CollisionState(
            has_collided=has_collided,
            is_new_collision_event=is_new,
            object_name=str(raw.object_name) if raw.object_name else "",
            object_id=int(raw.object_id),
            impact_point_ned_m=[
                float(raw.impact_point.x_val),
                float(raw.impact_point.y_val),
                float(raw.impact_point.z_val),
            ],
            normal_ned=[
                float(raw.normal.x_val),
                float(raw.normal.y_val),
                float(raw.normal.z_val),
            ],
            position_ned_m=[
                float(raw.position.x_val),
                float(raw.position.y_val),
                float(raw.position.z_val),
            ],
            penetration_depth=float(raw.penetration_depth),
            raw_timestamp=raw_ts,
            received_monotonic_seconds=received_mono,
        )

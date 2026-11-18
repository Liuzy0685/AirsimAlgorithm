"""CommandAuthority — per-vehicle single-controller leases (step 4, §5).

Protocol `docs/多机通信与控制协议.md` §5:

    "每架机仅有一个有效控制者。A 的 CommandAuthority 按 vehicle_name 管理租约、
    控制进程身份和续租；租约失效后网关拒绝旧控制者，接管者必须完成显式交接。"

A lease is granted to a *controller identity* (a process ``producer_id`` /
``stream_id`` pair, or any opaque non-empty string).  All times are monotonic
seconds.  A lease auto-expires after ``lease_ttl_s`` unless renewed, so a dead
control process cannot permanently lock out a takeover.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Lease:
    vehicle_name: str
    controller_id: str
    granted_at: float
    expires_at: float


class CommandAuthority:
    """Manage one live controller lease per vehicle."""

    def __init__(
        self,
        *,
        lease_ttl_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(lease_ttl_s) not in (int, float) or not math.isfinite(lease_ttl_s) or lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        self._lease_ttl_s = float(lease_ttl_s)
        self._clock = clock
        self._leases: dict[str, Lease] = {}

    # ------------------------------------------------------------------
    # Lease operations
    # ------------------------------------------------------------------

    def acquire(self, vehicle_name: str, controller_id: str,
                now: Optional[float] = None, *, ttl_s: Optional[float] = None) -> bool:
        """Grant/renew a lease for ``controller_id``.

        Returns ``False`` (and keeps the existing lease) when a *different*
        controller holds a live lease — the caller must perform an explicit
        handover instead.
        """
        self._validate(vehicle_name, controller_id)
        now = self._now(now)
        current = self._leases.get(vehicle_name)
        if current is not None and current.expires_at > now and current.controller_id != controller_id:
            return False
        if ttl_s is not None and (not math.isfinite(ttl_s) or ttl_s <= 0):
            raise ValueError("ttl_s must be finite and positive")
        self._grant(vehicle_name, controller_id, now, ttl_s)
        return True

    def renew(self, vehicle_name: str, controller_id: str,
              now: Optional[float] = None) -> bool:
        """Extend a lease held by the same controller."""
        self._validate(vehicle_name, controller_id)
        now = self._now(now)
        current = self._leases.get(vehicle_name)
        if current is None or current.controller_id != controller_id:
            return False
        if current.expires_at <= now:
            return False  # lapsed; must re-acquire
        self._grant(vehicle_name, controller_id, now)
        return True

    def release(self, vehicle_name: str, controller_id: str,
                now: Optional[float] = None) -> bool:
        """Release a lease held by ``controller_id``."""
        self._validate(vehicle_name, controller_id)
        now = self._now(now)
        current = self._leases.get(vehicle_name)
        if current is None or current.controller_id != controller_id:
            return False
        del self._leases[vehicle_name]
        return True

    def handover(self, vehicle_name: str, from_controller_id: str,
                 to_controller_id: str, now: Optional[float] = None) -> bool:
        """Explicitly transfer a live lease from one controller to another."""
        self._validate(vehicle_name, from_controller_id)
        self._validate(vehicle_name, to_controller_id)
        if from_controller_id == to_controller_id:
            return False
        now = self._now(now)
        current = self._leases.get(vehicle_name)
        if current is None or current.controller_id != from_controller_id:
            return False
        if current.expires_at <= now:
            del self._leases[vehicle_name]
            return False
        self._grant(vehicle_name, to_controller_id, now)
        return True

    def is_authorized(self, vehicle_name: str, controller_id: str,
                      now: Optional[float] = None) -> bool:
        """``True`` when ``controller_id`` holds a live lease."""
        self._validate(vehicle_name, controller_id)
        now = self._now(now)
        current = self._leases.get(vehicle_name)
        return (
            current is not None
            and current.controller_id == controller_id
            and current.expires_at > now
        )

    def check_and_expire(self, now: Optional[float] = None) -> list[str]:
        """Drop expired leases; return the list of freed vehicle names."""
        now = self._now(now)
        expired = [
            name for name, lease in self._leases.items()
            if lease.expires_at <= now
        ]
        for name in expired:
            del self._leases[name]
        return expired

    # ------------------------------------------------------------------
    # Introspection / helpers
    # ------------------------------------------------------------------

    def lease(self, vehicle_name: str) -> Optional[Lease]:
        return self._leases.get(vehicle_name)

    @property
    def vehicles(self) -> tuple[str, ...]:
        return tuple(sorted(self._leases))

    def _grant(self, vehicle_name: str, controller_id: str, now: float, ttl_s=None) -> None:
        self._leases[vehicle_name] = Lease(
            vehicle_name=vehicle_name,
            controller_id=controller_id,
            granted_at=now,
            expires_at=now + (self._lease_ttl_s if ttl_s is None else ttl_s),
        )

    def _now(self, now: Optional[float]) -> float:
        return self._clock() if now is None else now

    @staticmethod
    def _validate(vehicle_name: str, controller_id: str) -> None:
        if not isinstance(vehicle_name, str) or not vehicle_name.strip():
            raise ValueError("vehicle_name must be non-empty")
        if not isinstance(controller_id, str) or not controller_id.strip():
            raise ValueError("controller_id must be non-empty")

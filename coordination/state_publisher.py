"""Authoritative swarm-state publisher (step 2).

Wraps a :class:`coordination.state_provider.StateProvider` and turns each
collected snapshot into a protocol ``swarm.snapshot`` envelope.

Fail-closed guarantee (ARCHITECTURE / docs): a snapshot whose coordinate
transform has **not** passed ``CoordinateTransformer.validate()`` is never
published as authoritative state.  ``StateProvider.get_snapshot()`` already
refuses to collect in that case; this publisher converts that refusal into
``None`` (nothing is emitted) rather than a partial/legacy frame, so B and C
treat the service as offline instead of consuming unconfirmed coordinates.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from coordination.state_provider import StateProvider, SwarmStateSnapshot
from coordination.wire import build_swarm_snapshot_envelope

WallClock = Callable[[], float]
MonoClock = Callable[[], float]


class SwarmStatePublisher:
    """Collect one authoritative snapshot and serialize it for broadcast."""

    def __init__(
        self,
        provider: StateProvider,
        *,
        producer_id: str,
        stream_id: str,
        wall_clock: WallClock = time.time,
        mono_clock: MonoClock = time.monotonic,
    ) -> None:
        if not isinstance(provider, StateProvider):
            raise ValueError("provider must be a StateProvider")
        if not producer_id or not producer_id.strip():
            raise ValueError("producer_id must be non-empty")
        if not stream_id or not stream_id.strip():
            raise ValueError("stream_id must be non-empty")
        self._provider = provider
        self._producer_id = producer_id
        self._stream_id = stream_id
        self._wall_clock = wall_clock
        self._mono_clock = mono_clock
        self._latest_envelope: Optional[dict] = None
        self._latest_snapshot: Optional[SwarmStateSnapshot] = None
        self._authoritative = False
        self._last_error: Optional[str] = None

    @property
    def configured_names(self) -> tuple[str, ...]:
        return self._provider.vehicle_names

    @property
    def authoritative(self) -> bool:
        """``True`` when the latest published envelope is authoritative."""
        return self._authoritative

    @property
    def latest_envelope(self) -> Optional[dict]:
        return self._latest_envelope

    @property
    def latest_snapshot(self) -> Optional[SwarmStateSnapshot]:
        return self._latest_snapshot

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def collect_and_publish(self) -> Optional[dict]:
        """Collect one snapshot and return the envelope, or ``None``.

        Returns ``None`` (and records ``last_error``) when collection is
        refused — most importantly when the coordinate transform has not been
        validated.  Callers must broadcast only the returned envelope; a
        ``None`` result means "publish nothing this cycle".
        """
        try:
            snapshot = self._provider.get_snapshot()
        except Exception as exc:  # noqa: BLE001 - fail-closed on any refusal
            self._latest_envelope = None
            self._latest_snapshot = None
            self._authoritative = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

        envelope = build_swarm_snapshot_envelope(
            snapshot,
            configured_names=self.configured_names,
            producer_id=self._producer_id,
            stream_id=self._stream_id,
            wall_clock=self._wall_clock,
            mono_clock=self._mono_clock,
        )
        self._latest_envelope = envelope
        self._latest_snapshot = snapshot
        self._authoritative = True
        self._last_error = None
        return envelope

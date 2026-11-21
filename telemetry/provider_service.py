"""Periodic bridge from ``StateProvider`` to ``SwarmPublisher``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Any

from coordination.state_provider import StateProvider, SwarmStateSnapshot
from telemetry.publisher import SwarmPublisher

LOGGER = logging.getLogger("telemetry.provider_service")


class StatePublisherService:
    """Collect real provider snapshots and publish them at a fixed interval."""

    def __init__(
        self,
        provider: StateProvider,
        publisher: SwarmPublisher,
        *,
        interval_s: float = 0.2,
        display_provider: Callable[[SwarmStateSnapshot], Mapping[str, Any] | None]
        | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.provider = provider
        self.publisher = publisher
        self.interval_s = float(interval_s)
        self.display_provider = display_provider
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    async def publish_once(self) -> dict[str, Any]:
        # StateProvider uses synchronous AirSim RPC readers.  Keep those calls
        # off aiohttp's event loop so a slow vehicle cannot block WebSockets.
        snapshot = await asyncio.to_thread(self.provider.get_snapshot)
        display = self.display_provider(snapshot) if self.display_provider else None
        return await self.publisher.publish_snapshot(
            snapshot,
            vehicle_names=self.provider.vehicle_names,
            display=display,
        )

    async def run(self) -> None:
        self.stop_event.clear()
        while not self.stop_event.is_set():
            started = asyncio.get_running_loop().time()
            try:
                await self.publish_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed read becomes an unavailable cycle.  The next cycle
                # can recover; subscribers keep the normal C offline timeout.
                LOGGER.exception("StateProvider snapshot cycle failed")
            delay = max(0.0, self.interval_s - (asyncio.get_running_loop().time() - started))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def start(self) -> asyncio.Task[None]:
        if self.task is None or self.task.done():
            self.stop_event.clear()
            self.task = asyncio.create_task(self.run())
        return self.task

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None


__all__ = ["StatePublisherService"]

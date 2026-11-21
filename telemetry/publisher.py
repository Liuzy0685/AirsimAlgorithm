"""A-side WebSocket publisher for real :class:`SwarmStateSnapshot` data.

The publisher is deliberately small and read-only from the point of view of
the consumers: it accepts connection handshakes and broadcasts
``swarm.snapshot`` frames.  It does not expose AirSim and it does not accept
formation or flight-control commands.  The A process should call
``publish_snapshot`` immediately after ``StateProvider.get_snapshot()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from aiohttp import WSMsgType, web

from common.constants import PROTOCOL_VERSION
from coordination.state_provider import SwarmStateSnapshot
from telemetry.project_adapter import snapshot_to_swarm_frame
from telemetry.protocol import MAX_FRAME_BYTES, ProtocolError, validate_swarm_snapshot

LOGGER = logging.getLogger("telemetry.publisher")


class SwarmPublisher:
    """Publish authoritative A snapshots to local WebSocket subscribers."""

    def __init__(
        self,
        *,
        producer_id: str = "a-state-service",
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/ws/swarm",
        stream_id: str | None = None,
        handshake_timeout_s: float = 5.0,
    ) -> None:
        if not producer_id.strip():
            raise ValueError("producer_id must be a non-empty string")
        if not host or not path.startswith("/"):
            raise ValueError("host and path are required")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")
        if handshake_timeout_s <= 0:
            raise ValueError("handshake_timeout_s must be positive")
        self.producer_id = producer_id
        self.host = host
        self.port = port
        self.path = path
        self.stream_id = stream_id or str(uuid.uuid4())
        self.handshake_timeout_s = float(handshake_timeout_s)
        self.clients: set[web.WebSocketResponse] = set()
        self.latest_frame: dict[str, Any] | None = None

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_FRAME_BYTES)
        app.router.add_get(self.path, self.websocket_handler)
        return app

    async def websocket_handler(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_BYTES, heartbeat=20)
        await ws.prepare(request)
        try:
            if request.query.get("protocol_version") == str(PROTOCOL_VERSION):
                hello = {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": "connection.hello",
                    "role": request.query.get("role"),
                    "producer_id": request.query.get("producer_id"),
                    "stream_id": request.query.get("stream_id"),
                }
            else:
                message = await asyncio.wait_for(ws.receive(), timeout=self.handshake_timeout_s)
                if message.type != WSMsgType.TEXT:
                    raise ProtocolError("first message must be connection.hello")
                if len(message.data.encode("utf-8")) > MAX_FRAME_BYTES:
                    raise ProtocolError("frame exceeds 2 MB")
                hello = json.loads(message.data)
            self._validate_hello(hello)
            self.clients.add(ws)
            # A snapshot may have been published while the subscriber's
            # handshake was still in flight.  Send the latest authoritative
            # frame once so a reconnect does not wait for the next cycle.
            if self.latest_frame is not None:
                await ws.send_json(self.latest_frame)
            async for incoming in ws:
                if incoming.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                # The state publisher is intentionally read-only.  Ignore any
                # later browser/client messages rather than interpreting them
                # as control requests.
        except (asyncio.TimeoutError, json.JSONDecodeError, ProtocolError, TypeError, ValueError) as exc:
            LOGGER.warning("Rejected swarm subscriber: %s", exc)
        finally:
            self.clients.discard(ws)
            if not ws.closed:
                await ws.close()
        return ws

    def _validate_hello(self, hello: Any) -> None:
        if not isinstance(hello, Mapping):
            raise ProtocolError("connection.hello must be an object")
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported protocol_version")
        if hello.get("type") != "connection.hello":
            raise ProtocolError("first message must be connection.hello")
        role = hello.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ProtocolError("connection.hello.role must be a non-empty string")
        producer_id = hello.get("producer_id")
        if not isinstance(producer_id, str) or not producer_id.strip():
            raise ProtocolError("connection.hello.producer_id must be a non-empty string")
        stream_id = hello.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ProtocolError("connection.hello.stream_id must be a non-empty string")

    async def publish_snapshot(
        self,
        snapshot: SwarmStateSnapshot,
        *,
        vehicle_names: Sequence[str],
        display: Mapping[str, Any] | None = None,
        sent_at_unix_ms: int | None = None,
        snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        """Serialize and broadcast one real StateProvider snapshot.

        The returned frame is useful for logging and tests.  It is validated
        against the same v1 contract consumed by C before any client receives
        it.
        """
        frame = snapshot_to_swarm_frame(
            snapshot,
            vehicle_names=vehicle_names,
            producer_id=self.producer_id,
            stream_id=self.stream_id,
            sent_at_unix_ms=sent_at_unix_ms,
            snapshot_id=snapshot_id,
            display=display,
        )
        normalized = validate_swarm_snapshot(frame, expected_producer=self.producer_id)
        self.latest_frame = normalized
        if self.clients:
            clients = tuple(self.clients)
            results = await asyncio.gather(
                *(client.send_json(normalized) for client in clients),
                return_exceptions=True,
            )
            for client, result in zip(clients, results):
                if isinstance(result, Exception):
                    self.clients.discard(client)
        return normalized

    async def close(self) -> None:
        clients = tuple(self.clients)
        self.clients.clear()
        if clients:
            await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="A read-only swarm snapshot publisher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/ws/swarm")
    args = parser.parse_args()
    publisher = SwarmPublisher(host=args.host, port=args.port, path=args.path)
    web.run_app(publisher.create_app(), host=publisher.host, port=publisher.port)


if __name__ == "__main__":
    main()

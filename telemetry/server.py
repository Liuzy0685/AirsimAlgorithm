"""Read-only bridge from A's swarm snapshots to the browser dashboard."""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

from aiohttp import ClientSession, WSMsgType, web
import yaml

from telemetry.protocol import MAX_FRAME_BYTES, ProtocolError, SnapshotValidator, snapshot_to_telemetry

LOGGER = logging.getLogger("telemetry")
ROOT = Path(__file__).resolve().parents[1]
GATEWAY_KEY = web.AppKey("telemetry_gateway", object)


@dataclass(frozen=True)
class TelemetryConfig:
    upstream_url: str
    frontend_host: str
    frontend_port: int
    telemetry_path: str
    reconnect_delays_s: tuple[float, ...]
    frontend_offline_after_s: float
    expected_producer: str


def load_config(path: str | Path = ROOT / "configs" / "telemetry.yaml") -> TelemetryConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError("telemetry config must be a YAML mapping")
    host = values.get("frontend_host")
    port = values.get("frontend_port")
    route = values.get("telemetry_path")
    delays = values.get("reconnect_delays_s")
    timeout = values.get("frontend_offline_after_s")
    if host != "127.0.0.1":
        raise ValueError("telemetry frontend must bind to 127.0.0.1")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("frontend_port must be an integer from 1 to 65535")
    if not isinstance(route, str) or not route.startswith("/"):
        raise ValueError("telemetry_path must be an absolute URL path")
    if not isinstance(delays, list) or not delays or any(type(v) not in (int, float) or v <= 0 for v in delays):
        raise ValueError("reconnect_delays_s must be a non-empty list of positive numbers")
    if type(timeout) not in (int, float) or timeout <= 0:
        raise ValueError("frontend_offline_after_s must be positive")
    upstream = values.get("upstream_url")
    producer = values.get("expected_producer")
    if not isinstance(upstream, str) or not upstream.startswith("ws://127.0.0.1:"):
        raise ValueError("upstream_url must use the local ws://127.0.0.1 endpoint")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("expected_producer must be a non-empty string")
    return TelemetryConfig(upstream, host, port, route, tuple(float(v) for v in delays), float(timeout), producer)


class TelemetryGateway:
    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self.latest: dict[str, Any] | None = None
        self.last_received: float | None = None
        self.coordination_status: dict[str, dict[str, Any]] = {}
        self.upstream_connected = False
        self.stream_id = str(uuid.uuid4())
        self.validator = SnapshotValidator(expected_producer=config.expected_producer)
        self.clients: set[web.WebSocketResponse] = set()
        self.stop_event = asyncio.Event()

    def current_frame(self) -> dict[str, Any]:
        fresh = self.last_received is not None and time.monotonic() - self.last_received <= self.config.frontend_offline_after_s
        if fresh and self.latest is not None:
            return self.latest
        return {
            "type": "telemetry",
            "protocol_version": 1,
            "frame": "world_ned",
            "timestamp": int(time.time() * 1000),
            "source": self.config.expected_producer,
            "stream_id": self.stream_id,
            "sequence_id": None,
            "drones": {},
            "goal": None,
            "status": "OFFLINE",
            "obstacles": [],
            "snapshot": None,
            "coordination_state": "OFFLINE",
            "coordination": {},
            "active_command": None,
            "diagnostics": {"source_sent_at_unix_ms": None, "received_at_unix_ms": int(time.time() * 1000), "transport_latency_ms": None},
        }

    async def websocket_handler(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_BYTES, heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            await ws.send_json(self.current_frame())
            async for _message in ws:
                # Browser connections are subscribers only; ignore all client messages.
                pass
        finally:
            self.clients.discard(ws)
        return ws

    async def upstream_loop(self) -> None:
        attempt = 0
        while not self.stop_event.is_set():
            try:
                parsed = urlsplit(self.config.upstream_url)
                query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                query.update({
                    "protocol_version": "1",
                    "role": "c_telemetry",
                    "producer_id": "c-telemetry-gateway",
                    "stream_id": self.stream_id,
                })
                upstream_url = urlunsplit(parsed._replace(query=urlencode(query)))
                async with ClientSession() as session:
                    async with session.ws_connect(
                        upstream_url, max_msg_size=MAX_FRAME_BYTES, heartbeat=20
                    ) as ws:
                        attempt = 0
                        self.upstream_connected = True
                        async for message in ws:
                            if message.type == WSMsgType.TEXT:
                                try:
                                    if len(message.data.encode("utf-8")) > MAX_FRAME_BYTES:
                                        raise ProtocolError("frame exceeds 2 MB")
                                    raw = json.loads(message.data)
                                    if isinstance(raw, dict) and raw.get("type") == "coordination.status":
                                        self._accept_coordination_status(raw)
                                        continue
                                    normalized = self.validator.accept(raw)
                                    self.latest = snapshot_to_telemetry(normalized).as_dict()
                                    self.last_received = time.monotonic()
                                    self._merge_coordination_status()
                                except (json.JSONDecodeError, ProtocolError, TypeError, ValueError) as exc:
                                    LOGGER.warning("Rejected upstream frame: %s", exc)
                            elif message.type in (WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.info("A state service unavailable: %s", exc)
            finally:
                self.upstream_connected = False
            delay = self.config.reconnect_delays_s[min(attempt, len(self.config.reconnect_delays_s) - 1)]
            attempt += 1
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def broadcast_loop(self) -> None:
        while not self.stop_event.is_set():
            frame = self.current_frame()
            stale = [client for client in self.clients if client.closed]
            for client in stale:
                self.clients.discard(client)
            if self.clients:
                clients = tuple(self.clients)
                results = await asyncio.gather(
                    *(client.send_json(frame) for client in clients), return_exceptions=True
                )
                for client, result in zip(clients, results):
                    if isinstance(result, Exception):
                        self.clients.discard(client)
            await asyncio.sleep(0.1)

    def _accept_coordination_status(self, frame: Any) -> None:
        if not isinstance(frame, dict):
            raise ProtocolError("coordination.status must be an object")
        if frame.get("protocol_version") != 1 or frame.get("type") != "coordination.status":
            raise ProtocolError("unsupported coordination.status envelope")
        if frame.get("producer_id") != self.config.expected_producer:
            raise ProtocolError("unexpected producer_id")
        stream_id = frame.get("stream_id")
        sequence_id = frame.get("sequence_id")
        sent_at = frame.get("sent_at_unix_ms")
        payload = frame.get("payload")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ProtocolError("coordination.status stream_id is invalid")
        if type(sequence_id) is not int or sequence_id < 0 or type(sent_at) is not int or sent_at < 0:
            raise ProtocolError("coordination.status sequence/timestamp is invalid")
        if not isinstance(payload, dict):
            raise ProtocolError("coordination.status payload must be an object")
        name = payload.get("vehicle_name")
        mode = payload.get("state")
        reason = payload.get("reason", "")
        if not isinstance(name, str) or not name.strip() or mode not in {
            "NORMAL", "CONFLICT_DETECTED", "AVOIDING", "CONFLICT_CLEARED",
            "FORMATION_RECOVERING", "EMERGENCY",
        } or not isinstance(reason, str):
            raise ProtocolError("coordination.status fields are invalid")
        previous = self.coordination_status.get(name)
        if previous and previous["stream_id"] == stream_id and sequence_id <= previous["sequence_id"]:
            return
        navigation = payload.get("navigation")
        self.coordination_status[name] = {
            "state": mode,
            "reason": reason,
            "active_command_id": payload.get("active_command_id"),
            "task_id": payload.get("task_id", ""),
            "sim_timestamp_ns": payload.get("sim_timestamp_ns", 0),
            "navigation": navigation if isinstance(navigation, dict) else None,
            "stream_id": stream_id,
            "sequence_id": sequence_id,
            "received_monotonic": time.monotonic(),
        }
        self._merge_coordination_status()

    def _merge_coordination_status(self) -> None:
        if not self.latest:
            return
        now = time.monotonic()
        fresh = {
            name: {key: value for key, value in status.items() if key not in {"received_monotonic", "stream_id", "sequence_id"}}
            for name, status in self.coordination_status.items()
            if 0 <= now - status["received_monotonic"] <= 2.0
        }
        self.latest["coordination"] = fresh
        if fresh:
            states = {item["state"] for item in fresh.values()}
            self.latest["coordination_state"] = next(iter(states)) if len(states) == 1 else "MIXED"
        for name, drone in self.latest.get("drones", {}).items():
            status = fresh.get(name)
            drone["coordination_state"] = status["state"] if status else "UNKNOWN"
            drone["coordination_reason"] = status["reason"] if status else ""
            drone["active_command_id"] = status["active_command_id"] if status else None

    async def start(self, _app: web.Application) -> None:
        self.stop_event.clear()
        self.tasks = [asyncio.create_task(self.upstream_loop()), asyncio.create_task(self.broadcast_loop())]

    async def stop(self, _app: web.Application) -> None:
        self.stop_event.set()
        for task in getattr(self, "tasks", []):
            task.cancel()
        await asyncio.gather(*getattr(self, "tasks", []), return_exceptions=True)


def create_app(config: TelemetryConfig | None = None) -> web.Application:
    gateway = TelemetryGateway(config or load_config())
    app = web.Application(client_max_size=MAX_FRAME_BYTES)
    app[GATEWAY_KEY] = gateway

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(ROOT / "frontend" / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/static/", ROOT / "frontend" / "static", show_index=False)
    app.router.add_get(gateway.config.telemetry_path, gateway.websocket_handler)
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.stop)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="C read-only telemetry gateway")
    parser.add_argument("--config", default=str(ROOT / "configs" / "telemetry.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    web.run_app(create_app(config), host=config.frontend_host, port=config.frontend_port)


if __name__ == "__main__":
    main()

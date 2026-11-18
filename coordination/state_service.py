"""A 状态服务 — publishes ``swarm.snapshot`` and accepts ``formation.command``.

This module is split into two layers:

- :class:`SwarmStateService` — the synchronous, socket-free handler: client
  registration (handshake identity), message authorization by role, snapshot
  broadcast, ``formation.command`` → ``command.ack`` via the ingress, and
  ``coordination.status`` downlink.  It is fully unit-testable with fake
  transports.

- :func:`run_swarm_state_server` — thin asyncio/WebSocket wiring over
  :mod:`coordination.ws`.  It binds ``ws://127.0.0.1:8765/ws/swarm`` (default)
  and drives the handler.  Handshake identity is declared as query parameters:

      /ws/swarm?role=b_formation&producer_id=b-formation-service&stream_id=b-boot-uuid

Disconnect / reconnect handling:
- On disconnect the session is removed, so a dead B can no longer publish
  ``formation.command`` (liveness is enforced by the transport).
- On reconnect the client must handshake again; sequence counters in the
  ingress are **not** reset for the same ``stream_id`` (per protocol §3/§4).
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

from common.models import FormationCommand
from coordination.command_ingress import FormationCommandIngress
from coordination.state_publisher import SwarmStatePublisher
from coordination.wire import (
    ROLES,
    TYPE_COORDINATION_STATUS,
    TYPE_FORMATION_COMMAND,
    UPLINK_AUTHORIZED_ROLES,
    WireError,
    build_command_ack,
    build_coordination_status_payload,
    build_envelope,
    parse_envelope,
    parse_formation_command_payload,
)
from coordination.ws import (
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    FrameDecoder,
    WebSocketError,
    build_101_response,
    encode_close_frame,
    encode_pong_frame,
    encode_ping_frame,
    encode_text_frame,
    parse_http_upgrade,
)

WS_SWARM_PATH = "/ws/swarm"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

SendCallback = Callable[[dict], None]


@dataclass(frozen=True)
class ClientSession:
    client_id: int
    role: str
    producer_id: str
    stream_id: str
    vehicle_name: str = ""


class SwarmStateService:
    """Socket-free state/command service handler."""

    def __init__(
        self,
        publisher: SwarmStatePublisher,
        ingress: FormationCommandIngress,
        *,
        producer_id: str,
        stream_id: str,
        clock: Callable[[], float] = time.monotonic,
        on_accepted_command: Optional[Callable[[FormationCommand, ClientSession], None]] = None,
        snapshot_max_age_s: float = 0.6,
        task_id: Optional[str] = None,
        require_controller: bool = False,
        formation_targets: Optional[set[str]] = None,
    ) -> None:
        if not isinstance(publisher, SwarmStatePublisher):
            raise ValueError("publisher must be a SwarmStatePublisher")
        if not isinstance(ingress, FormationCommandIngress):
            raise ValueError("ingress must be a FormationCommandIngress")
        self._publisher = publisher
        self._ingress = ingress
        self._producer_id = producer_id
        self._stream_id = stream_id
        self._clock = clock
        self._on_accepted_command = on_accepted_command
        self._clients: dict[int, ClientSession] = {}
        self._senders: dict[int, SendCallback] = {}
        self._next_client_id = 0
        self._ack_sequence = publisher.latest_envelope["sequence_id"] if publisher.latest_envelope else 0
        self._last_snapshot_envelope = publisher.latest_envelope
        self._accepted_commands: dict[str, FormationCommand] = {}
        self._command_clients: dict[str, int] = {}
        self._snapshot_max_age_s = snapshot_max_age_s
        self._task_id = task_id
        self._require_controller = require_controller
        self._formation_targets = formation_targets
        self._status_sequences = {}
        self._last_status = {}

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def register_client(
        self, *, role: str, producer_id: str, stream_id: str, send: SendCallback,
        vehicle_name: str = "",
    ) -> int:
        """Register a handshaked client; returns its integer id."""
        if role not in ROLES:
            raise WebSocketError(f"unknown role {role!r}")
        if not producer_id or not producer_id.strip():
            raise WebSocketError("producer_id must be non-empty")
        if not stream_id or not stream_id.strip():
            raise WebSocketError("stream_id must be non-empty")
        if not callable(send):
            raise ValueError("send must be callable")
        if role == "a_controller" and self._require_controller:
            if vehicle_name not in self._publisher.configured_names:
                raise WebSocketError("unknown controller vehicle")
            if any(c.role == role and c.vehicle_name == vehicle_name for c in self._clients.values()):
                raise WebSocketError("controller already registered")
        self._next_client_id += 1
        client_id = self._next_client_id
        self._clients[client_id] = ClientSession(
            client_id=client_id, role=role, producer_id=producer_id, stream_id=stream_id,
            vehicle_name=vehicle_name,
        )
        self._senders[client_id] = send
        return client_id

    def unregister_client(self, client_id: int) -> None:
        """Remove a disconnected client (its command session is dropped)."""
        session = self._clients.pop(client_id, None)
        self._senders.pop(client_id, None)
        if session is not None and session.role == "a_controller" and session.vehicle_name:
            command = self._accepted_commands.pop(session.vehicle_name, None)
            self._command_clients.pop(session.vehicle_name, None)
            self.publish_status(vehicle_name=session.vehicle_name, state="EMERGENCY",
                reason="controller_disconnected", active_command_id=command.lifecycle.command_id if command else None)
        for name, origin in list(self._command_clients.items()):
            if origin == client_id:
                command = self._accepted_commands.pop(name, None)
                self._command_clients.pop(name, None)
                self.publish_status(vehicle_name=name, state="EMERGENCY",
                    reason="formation_source_disconnected",
                    active_command_id=command.lifecycle.command_id if command else None)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # Authoritative time basis
    # ------------------------------------------------------------------

    @property
    def latest_sim_time_ns(self) -> Optional[int]:
        """A's latest AirSim sim time from the newest authoritative snapshot."""
        snap = self._publisher.latest_snapshot
        if snap is None:
            return None
        if (not snap.collection_complete or set(snap.states) != set(self._publisher.configured_names)
                or not 0 <= self._clock() - snap.timestamp <= self._snapshot_max_age_s):
            return None
        stamps = [s.timestamp for s in snap.states.values() if s.timestamp > 0]
        return max(stamps) if stamps else None

    def latest_formation_command(self, vehicle_name: str) -> Optional[FormationCommand]:
        command = self._accepted_commands.get(vehicle_name)
        sim_time = self.latest_sim_time_ns
        if (command is None or sim_time is None
                or self._command_clients.get(vehicle_name) not in self._clients
                or not self._ingress.is_adoptable(command, sim_time_ns=sim_time)):
            return None
        return command

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    def handle_message(self, client_id: int, raw_text: str) -> Optional[dict]:
        """Route one inbound frame; returns an optional envelope to send back."""
        session = self._clients.get(client_id)
        if session is None:
            return None
        try:
            envelope = parse_envelope(raw_text)
        except WireError as exc:
            if session.role == "b_formation":
                return self._reject_ack("", f"malformed_frame: {exc}")
            return None

        # Authorization by message type (§2: never trust a body's self report).
        allowed = UPLINK_AUTHORIZED_ROLES.get(envelope.message_type, set())
        if session.role not in allowed:
            if session.role == "b_formation":
                return self._reject_ack(
                    "", f"unauthorized_message_type:{envelope.message_type}"
                )
            return None

        if envelope.message_type == "command.request":
            request_id = envelope.payload.get("request_id")
            if (envelope.producer_id != session.producer_id or envelope.stream_id != session.stream_id
                    or not isinstance(request_id, str) or not request_id.strip()
                    or not session.vehicle_name):
                return None
            command = self.latest_formation_command(session.vehicle_name)
            self._ack_sequence += 1
            return build_envelope(message_type="command.permit", producer_id=self._producer_id,
                stream_id=self._stream_id, sequence_id=self._ack_sequence,
                payload=dict(request_id=request_id, command=asdict(command) if command else None,
                    remaining_ttl_s=self._ingress.remaining_ttl_s(command) if command else 0.0))

        if envelope.message_type == TYPE_FORMATION_COMMAND:
            return self._handle_formation_command(session, envelope)
        if envelope.message_type == TYPE_COORDINATION_STATUS:
            if (envelope.producer_id != session.producer_id or envelope.stream_id != session.stream_id
                    or envelope.payload.get("vehicle_name") != session.vehicle_name):
                return None
            last = self._status_sequences.get((session.producer_id, session.stream_id), -1)
            if envelope.sequence_id <= last:
                return None
            try:
                payload = build_coordination_status_payload(**envelope.payload)
            except (WireError, TypeError, ValueError):
                return None
            self._status_sequences[(session.producer_id, session.stream_id)] = envelope.sequence_id
            self.publish_status(**payload)
        return None

    def _handle_formation_command(self, session: ClientSession, envelope) -> Optional[dict]:
        try:
            command = parse_formation_command_payload(envelope.payload)
        except WireError as exc:
            return self._reject_ack("", f"invalid_command: {exc}")

        command_id = command.lifecycle.command_id if command.lifecycle is not None else ""
        if envelope.producer_id != session.producer_id or envelope.stream_id != session.stream_id:
            return self._reject_ack(command_id, "envelope_identity_mismatch")
        if command.vehicle_name not in self._publisher.configured_names:
            return self._reject_ack(command_id, "unknown_vehicle")
        if self._formation_targets is not None and command.vehicle_name not in self._formation_targets:
            return self._reject_ack(command_id, "vehicle_not_formation_follower")
        if not command.valid:
            return self._reject_ack(command_id, "invalid_formation_command")
        if self._task_id is not None and (command.lifecycle is None or command.lifecycle.task_id != self._task_id):
            return self._reject_ack(command_id, "task_id_mismatch")
        targets = [c for c in self._clients.values()
                   if c.role == "a_controller" and c.vehicle_name == command.vehicle_name]
        if self._require_controller and not targets:
            return self._reject_ack(command_id, "controller_unavailable")
        sim_time_ns = self.latest_sim_time_ns
        if sim_time_ns is None:
            return self._reject_ack(command_id, "no_authoritative_snapshot")

        acceptance = self._ingress.accept(
            command,
            connection=session,
            envelope_sequence_id=envelope.sequence_id,
            sim_time_ns=sim_time_ns,
        )
        if acceptance.accepted:
            self._accepted_commands[command.vehicle_name] = command
            self._command_clients[command.vehicle_name] = session.client_id
            self._ack_sequence += 1
            forwarded = build_envelope(message_type=TYPE_FORMATION_COMMAND,
                producer_id=self._producer_id, stream_id=self._stream_id,
                sequence_id=self._ack_sequence, payload=asdict(command))
            for target in targets:
                try:
                    self._senders[target.client_id](forwarded)
                except (OSError, ConnectionError, BufferError):
                    self.unregister_client(target.client_id)
                    self._accepted_commands.pop(command.vehicle_name, None)
                    self._command_clients.pop(command.vehicle_name, None)
                    return self._reject_ack(command_id, "controller_unavailable")
            if self._on_accepted_command is not None:
                self._on_accepted_command(command, session)
        return self._ack(acceptance)

    def _reject_ack(self, command_id: str, reason: str) -> dict:
        return self._ack_commands(False, reason, command_id)

    def _ack(self, acceptance) -> dict:
        return self._ack_commands(
            acceptance.accepted, acceptance.reason, acceptance.command_id
        )

    def _ack_commands(self, accepted: bool, reason: str, command_id: str) -> dict:
        self._ack_sequence += 1
        return build_command_ack(
            accepted=accepted,
            reason=reason,
            command_id=command_id,
            producer_id=self._producer_id,
            stream_id=self._stream_id,
            sequence_id=self._ack_sequence,
        )

    # ------------------------------------------------------------------
    # Outbound broadcast
    # ------------------------------------------------------------------

    def broadcast_snapshot(self, envelope: dict) -> None:
        """Broadcast the authoritative snapshot to every connected client."""
        self._ack_sequence += 1
        envelope = dict(envelope, sequence_id=self._ack_sequence)
        self._last_snapshot_envelope = envelope
        self._broadcast(envelope)

    def _broadcast(self, envelope):
        for client_id, send in list(self._senders.items()):
            try:
                send(envelope)
            except (OSError, ConnectionError, BufferError):
                self.unregister_client(client_id)

    def publish_status(self, *, vehicle_name: str, state: str, reason: str = "",
                       active_command_id: Optional[str] = None,
                       snapshot_id: Optional[int] = None,
                       task_id: str = "", sim_timestamp_ns: int = 0) -> dict:
        """Broadcast a ``coordination.status`` downlink to B and C."""
        payload = build_coordination_status_payload(
            vehicle_name=vehicle_name, state=state, reason=reason,
            active_command_id=active_command_id, snapshot_id=snapshot_id,
            task_id=task_id, sim_timestamp_ns=sim_timestamp_ns,
        )
        self._last_status[vehicle_name] = payload
        self._ack_sequence += 1
        envelope = build_envelope(
            message_type=TYPE_COORDINATION_STATUS,
            producer_id=self._producer_id,
            stream_id=self._stream_id,
            sequence_id=self._ack_sequence,
            payload=payload,
        )
        self._broadcast(envelope)
        return envelope


# ---------------------------------------------------------------------------
# Asyncio WebSocket transport
# ---------------------------------------------------------------------------


def parse_handshake_query(path: str) -> tuple[str, dict[str, str]]:
    """Split a request target into its path and query parameters."""
    parsed = urllib.parse.urlsplit(path)
    return parsed.path, {
        k: v for k, v in urllib.parse.parse_qsl(parsed.query)
    }


async def run_swarm_state_server(
    service: SwarmStateService,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    collect: Optional[Callable] = None,
    poll_interval_s: float = 0.2,
    heartbeat_s: float = 1.0,
    disconnect_timeout_s: float = 2.0,
    on_ready: Optional[Callable] = None,
) -> None:
    """Serve the swarm state/command WebSocket endpoint forever."""
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("only loopback binding is supported")
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client_id: Optional[int] = None
        try:
            data = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, asyncio.LimitOverrunError):
            writer.close()
            return

        info = parse_http_upgrade(data)
        if info is None:
            writer.close()
            return
        headers = info["headers"]
        if (headers.get("upgrade", "").lower() != "websocket"
                or "upgrade" not in headers.get("connection", "").lower()
                or headers.get("sec-websocket-version") != "13"):
            writer.close()
            return
        path, query = parse_handshake_query(info["path"])
        if path != WS_SWARM_PATH:
            writer.close()
            return
        if query.get("protocol_version") != "1":
            writer.close()
            return
        key = info["headers"].get("sec-websocket-key")
        if not key:
            writer.close()
            return

        def send_json(obj: dict) -> None:
            if writer.is_closing() or writer.transport.get_write_buffer_size() > 2_000_000:
                writer.transport.abort()
                raise BufferError("slow subscriber disconnected")
            writer.write(encode_text_frame(json.dumps(obj, separators=(",", ":"), allow_nan=False)))

        try:
            client_id = service.register_client(
                role=query.get("role", ""),
                producer_id=query.get("producer_id", ""),
                stream_id=query.get("stream_id", ""),
                send=send_json,
                vehicle_name=query.get("vehicle_name", ""),
            )
        except (WebSocketError, ValueError):
            writer.close()
            return

        decoder = FrameDecoder()
        async def ping_client():
            while True:
                await asyncio.sleep(heartbeat_s)
                writer.write(encode_ping_frame())
                await asyncio.wait_for(writer.drain(), disconnect_timeout_s)
        ping_task = None
        try:
            writer.write(build_101_response(key))
            await asyncio.wait_for(writer.drain(), disconnect_timeout_s)
            if service.latest_sim_time_ns is not None and service._last_snapshot_envelope is not None:
                send_json(service._last_snapshot_envelope)
            ping_task = asyncio.create_task(ping_client())
            while True:
                chunk = await asyncio.wait_for(reader.read(65536), disconnect_timeout_s)
                if not chunk:
                    break
                try:
                    frames = decoder.feed(chunk)
                except WebSocketError:
                    break
                for opcode, payload in frames:
                    if opcode == OP_TEXT:
                        reply = service.handle_message(
                            client_id, payload.decode("utf-8")
                        )
                        if reply is not None:
                            send_json(reply)
                    elif opcode == OP_PING:
                        writer.write(encode_pong_frame(payload))
                        await writer.drain()
                    elif opcode == OP_CLOSE:
                        writer.write(encode_close_frame(1000))
                        await writer.drain()
                        return
                await asyncio.wait_for(writer.drain(), disconnect_timeout_s)
        except (ConnectionError, OSError, BufferError, asyncio.TimeoutError, UnicodeError, WebSocketError):
            pass
        finally:
            if ping_task is not None:
                ping_task.cancel()
                await asyncio.gather(ping_task, return_exceptions=True)
            if client_id is not None:
                service.unregister_client(client_id)
            writer.close()

    async def publish_loop():
        while True:
            started = time.monotonic()
            if collect is not None:
                envelope = await collect()
                if envelope is not None:
                    service.broadcast_snapshot(envelope)
            await asyncio.sleep(max(0.001, poll_interval_s - (time.monotonic() - started)))

    async def heartbeat_loop():
        while True:
            for name in service._publisher.configured_names:
                if service.latest_sim_time_ns is None:
                    service.publish_status(vehicle_name=name, state="EMERGENCY", reason="snapshot_unavailable")
                else:
                    payload = service._last_status.get(name, dict(vehicle_name=name, state="NORMAL"))
                    service.publish_status(**payload)
            await asyncio.sleep(heartbeat_s)

    server = await asyncio.start_server(handle, host, port)
    publisher_task = asyncio.create_task(publish_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    server_task = asyncio.create_task(server.serve_forever())
    try:
        if on_ready:
            on_ready(server)
        async with server:
            done, _ = await asyncio.wait((server_task, publisher_task, heartbeat_task), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
    finally:
        for task in (publisher_task, heartbeat_task, server_task):
            task.cancel()
        await asyncio.gather(publisher_task, heartbeat_task, server_task, return_exceptions=True)

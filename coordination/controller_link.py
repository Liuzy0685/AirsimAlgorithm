"""A controller's subscription cache. Never falls back to private peer RPC reads."""
import asyncio
import copy
from dataclasses import replace
import math
import logging
import threading
import time
import uuid
from collections import deque

from coordination.client import SwarmConnection
from coordination.wire import build_envelope, parse_swarm_snapshot, parse_formation_command_payload

logger = logging.getLogger(__name__)


class ControllerLink:
    def __init__(self, cfg, vehicle_name, names, *, clock=time.monotonic):
        self.cfg, self.vehicle_name, self.names = cfg, vehicle_name, set(names)
        self.clock = clock
        self.producer_id = f"a-controller:{vehicle_name}"
        self.stream_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._snapshot = self._command = None
        self._pending_status = deque(maxlen=64)
        self._connected = False
        self._state_stream = None
        self._last_sequence = -1
        self._last_snapshot_id = -1
        self._received_command_at = 0.0
        self._command_sequence = {}
        self._requests = {}

    def start(self):
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True,
                                        name=f"swarm-sub-{self.vehicle_name}")
        self._thread.start()
        deadline = self.clock() + self.cfg.startup_timeout_s
        while self.clock() < deadline:
            if self.snapshot() is not None:
                return
            self._stop.wait(0.05)
        self.close()
        raise RuntimeError("no fresh complete authoritative A snapshot before takeoff")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def snapshot(self):
        with self._lock:
            snap = self._snapshot
            if (not self._connected or snap is None or not snap.collection_complete or snap.errors
                    or set(snap.states) != self.names
                    or not 0 <= self.clock() - snap.timestamp < self.cfg.snapshot_max_age_s):
                return None
            return copy.deepcopy(snap)

    def command(self):
        with self._lock:
            command = self._command
            snap = self._snapshot
            if not self._connected or snap is None or command is None:
                return None
            if (not snap.collection_complete or snap.errors
                    or not 0 <= self.clock() - snap.timestamp < self.cfg.snapshot_max_age_s
                    or not command.lifecycle.is_current(sim_time_ns=max(s.timestamp for s in snap.states.values()),
                        received_monotonic_seconds=self._received_command_at, now_monotonic_seconds=self.clock())):
                return None
            return copy.deepcopy(command)

    def ingest(self, envelope):
        """Called only for frames received on the connected A channel."""
        if envelope.producer_id != "a-state-service":
            raise ValueError("unexpected authoritative producer")
        with self._lock:
            if envelope.stream_id != self._state_stream:
                self._state_stream = envelope.stream_id
                self._snapshot = self._command = None
                self._last_sequence = -1
                self._last_snapshot_id = -1
                self._command_sequence.clear()
            if envelope.message_type == "swarm.snapshot":
                if envelope.sequence_id <= self._last_sequence:
                    return
                snap = parse_swarm_snapshot(envelope, received_at=self.clock())
                if snap.sequence_id <= self._last_snapshot_id:
                    return
                self._last_snapshot_id = snap.sequence_id
                self._snapshot = snap
                self._last_sequence = envelope.sequence_id
                if not self._snapshot.collection_complete:
                    self._command = None
            elif envelope.message_type == "command.permit":
                sent_at = self._requests.pop(envelope.payload.get("request_id"), None)
                if sent_at is None:
                    return
                raw = envelope.payload.get("command")
                if raw is None:
                    self._command = None
                    return
                remaining = envelope.payload.get("remaining_ttl_s")
                if type(remaining) not in (int, float) or not math.isfinite(remaining):
                    raise ValueError("invalid command permit lifetime")
                # Subtract the whole locally measured round trip. This is
                # conservative and cannot restart A's original reception TTL.
                remaining -= max(0.0, self.clock() - sent_at)
                if remaining <= 0:
                    self._command = None
                    return
                cmd = parse_formation_command_payload(raw)
                life = cmd.require_lifecycle()
                if cmd.vehicle_name != self.vehicle_name or life.task_id != self.cfg.task_id:
                    raise ValueError("formation command routing mismatch")
                key = (life.stream_id, life.task_id)
                if life.sequence_id <= self._command_sequence.get(key, -1):
                    return
                self._command_sequence[key] = life.sequence_id
                cmd.lifecycle = replace(life, ttl_s=min(life.ttl_s, remaining))
                self._command, self._received_command_at = cmd, self.clock()
            elif (envelope.message_type == "coordination.status" and
                    envelope.payload.get("vehicle_name") == self.vehicle_name and
                    envelope.payload.get("reason") in ("formation_source_disconnected", "snapshot_unavailable")):
                self._command = None

    def command_request(self, sequence):
        request_id = str(uuid.uuid4())
        with self._lock:
            now = self.clock()
            self._requests = {key: at for key, at in self._requests.items() if now-at < 2}
            self._requests[request_id] = now
        return build_envelope(message_type="command.request", producer_id=self.producer_id,
            stream_id=self.stream_id, sequence_id=sequence, payload=dict(request_id=request_id))

    def report(self, state, reason="", command=None):
        snap = self.snapshot()
        payload = dict(vehicle_name=self.vehicle_name, state=state, reason=reason,
            active_command_id=command.lifecycle.command_id if command else None,
            snapshot_id=snap.sequence_id if snap else None, task_id=self.cfg.task_id,
            sim_timestamp_ns=max(s.timestamp for s in snap.states.values()) if snap else 0)
        with self._lock:
            if self._pending_status and self._pending_status[-1]["state"] == state:
                self._pending_status[-1] = payload
            else:
                self._pending_status.append(payload)

    async def _run(self):
        delay, sequence = 0.5, 0
        while not self._stop.is_set():
            connection = None
            try:
                connection = await SwarmConnection.connect(self.cfg.url, role="a_controller",
                    producer_id=self.producer_id, stream_id=self.stream_id, vehicle_name=self.vehicle_name)
                with self._lock:
                    self._connected = True
                delay, last_ping, last_request = 0.5, 0.0, 0.0
                while not self._stop.is_set():
                    if self.clock() - last_ping >= self.cfg.heartbeat_s:
                        await connection.send_frame(9, b"A")
                        last_ping = self.clock()
                    if self.clock() - last_request >= 0.1:
                        sequence += 1
                        await connection.send(self.command_request(sequence))
                        last_request = self.clock()
                    with self._lock:
                        pending = list(self._pending_status)
                        self._pending_status.clear()
                    for payload in pending:
                        sequence += 1
                        await connection.send(build_envelope(message_type="coordination.status",
                            producer_id=self.producer_id, stream_id=self.stream_id,
                            sequence_id=sequence, payload=payload))
                    try:
                        envelope = await asyncio.wait_for(connection.receive(), 0.1)
                        self.ingest(envelope)
                    except asyncio.TimeoutError:
                        pass
            except Exception as exc:
                logger.warning("swarm_channel_disconnected vehicle=%s reason=%s", self.vehicle_name, exc)
            finally:
                with self._lock:
                    self._connected = False
                    self._snapshot = self._command = None
                    self._requests.clear()
                if connection:
                    await connection.close()
            # Interruptible backoff with the protocol's 0.5/1/2/5 second cap.
            until = self.clock() + delay
            while not self._stop.is_set() and self.clock() < until:
                await asyncio.sleep(0.1)
            delay = min(5, delay * 2 if delay < 2 else 5)


class SnapshotPeerReader:
    def __init__(self, mode, name):
        self.mode, self.name = mode, name

    def read(self):
        snap = self.mode._authoritative_tick_snapshot
        if snap is None or self.name not in snap.states:
            raise RuntimeError("authoritative peer state unavailable")
        state = copy.deepcopy(snap.states[self.name])
        offset = self.mode._swarm_spawn_offsets[self.name]
        state.position_ned_m = [p-o for p, o in zip(state.position_ned_m, offset)]
        return state

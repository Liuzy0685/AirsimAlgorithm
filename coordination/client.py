"""Small loopback WebSocket client shared by A controllers and B's runner."""
import asyncio
import base64
import json
import os
import urllib.parse
from collections import deque

from coordination.ws import FrameDecoder, WebSocketError, accept_key, _encode_header
from coordination.wire import parse_envelope


class SwarmConnection:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.decoder = FrameDecoder(expect_masked=False)
        self.pending = deque()

    @classmethod
    async def connect(cls, url, *, role, producer_id, stream_id, vehicle_name=""):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "ws" or parsed.hostname not in ("127.0.0.1", "::1"):
            raise ValueError("only loopback ws endpoints are supported")
        reader, writer = await asyncio.wait_for(asyncio.open_connection(parsed.hostname, parsed.port), 5)
        key = base64.b64encode(os.urandom(16)).decode()
        query = urllib.parse.urlencode(dict(protocol_version=1, role=role,
            producer_id=producer_id, stream_id=stream_id, vehicle_name=vehicle_name))
        target = f"{parsed.path}?{query}"
        writer.write((f"GET {target} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
                      "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        try:
            await writer.drain()
            response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            lines = response.decode("ascii").split("\r\n")
            headers = {key.lower(): value.strip() for line in lines[1:] if ":" in line
                       for key, value in [line.split(":", 1)]}
            if (not lines[0].startswith("HTTP/1.1 101 ") or
                    headers.get("sec-websocket-accept", "") != accept_key(key)):
                raise WebSocketError("upgrade rejected")
            return cls(reader, writer)
        except BaseException:
            writer.close()
            raise

    async def send_frame(self, opcode, payload):
        mask = os.urandom(4)
        masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        self.writer.write(_encode_header(opcode, len(payload), mask=True) + mask + masked)
        await self.writer.drain()

    async def send(self, envelope):
        await self.send_frame(1, json.dumps(envelope, allow_nan=False).encode())

    async def receive(self):
        while True:
            if not self.pending:
                chunk = await self.reader.read(65536)
                if not chunk:
                    raise ConnectionError("A state service disconnected")
                self.pending.extend(self.decoder.feed(chunk))
                continue
            opcode, payload = self.pending.popleft()
            if opcode == 1:
                return parse_envelope(payload.decode("utf-8"))
            if opcode == 8:
                raise ConnectionError("A state service closed connection")
            if opcode == 9:
                await self.send_frame(10, payload)

    async def close(self):
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 1)
        except (asyncio.TimeoutError, OSError):
            self.writer.transport.abort()

"""Minimal server-side WebSocket (RFC 6455) codec — stdlib only.

The v1 A/B/C topology is loopback-only and small (a handful of clients), so we
implement just enough of RFC 6455 to run without third-party dependencies:

- HTTP ``Upgrade`` handshake with ``Sec-WebSocket-Accept``,
- server→client **unmasked** text/close/pong frames,
- client→server **masked** text/close/ping frames (single, unfragmented).

Pure frame functions and the handshake are separate from :class:`FrameDecoder`
so they can be unit-tested without a socket.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Optional

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class WebSocketError(Exception):
    """A WebSocket protocol violation."""


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


def accept_key(sec_websocket_key: str) -> str:
    """Compute the ``Sec-WebSocket-Accept`` value for a handshake key."""
    digest = hashlib.sha1((sec_websocket_key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def parse_http_upgrade(data: bytes) -> Optional[dict]:
    """Parse a complete HTTP upgrade request into ``{"path", "headers"}``.

    Returns ``None`` when the request line is not a GET upgrade.  ``headers``
    keys are lower-cased.
    """
    text = data.decode("latin-1")
    lines = text.split("\r\n")
    if not lines:
        return None
    parts = lines[0].split(" ")
    if len(parts) < 3 or parts[0].upper() != "GET":
        return None
    path = parts[1]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return {"path": path, "headers": headers}


def build_101_response(sec_websocket_key: str) -> bytes:
    """Build the ``101 Switching Protocols`` response bytes."""
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(sec_websocket_key)}\r\n"
        "\r\n"
    ).encode("ascii")


# ---------------------------------------------------------------------------
# Frame encoding (server → client, unmasked)
# ---------------------------------------------------------------------------


def _encode_header(opcode: int, length: int, *, fin: bool = True, mask: bool = False) -> bytes:
    first = (0x80 if fin else 0x00) | (opcode & 0x0F)
    second = 0x80 if mask else 0x00
    if length < 126:
        return bytes((first, second | length))
    if length < 65536:
        return bytes((first, second | 126)) + struct.pack(">H", length)
    return bytes((first, second | 127)) + struct.pack(">Q", length)


def encode_text_frame(text: str) -> bytes:
    payload = text.encode("utf-8")
    return _encode_header(OP_TEXT, len(payload)) + payload


def encode_close_frame(code: int = 1000) -> bytes:
    return _encode_header(OP_CLOSE, 2) + struct.pack(">H", code)


def encode_pong_frame(payload: bytes) -> bytes:
    return _encode_header(OP_PONG, len(payload)) + payload


def encode_ping_frame(payload: bytes = b"A") -> bytes:
    return _encode_header(OP_PING, len(payload)) + payload


# ---------------------------------------------------------------------------
# Frame decoding (client → server, masked)
# ---------------------------------------------------------------------------


def _decode_frame(buf: bytes, *, expect_masked: bool = True) -> Optional[tuple[int, bytes, int]]:
    """Decode one masked frame from ``buf``.

    Returns ``(opcode, payload, consumed)`` or ``None`` if the buffer does not
    yet contain a complete frame.  Raises ``WebSocketError`` on malformed
    input (non-masked client frame, reserved bits, oversized frame).
    """
    if len(buf) < 2:
        return None
    b0 = buf[0]
    b1 = buf[1]
    fin = bool(b0 & 0x80)
    rsv = b0 & 0x70
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F

    if rsv:
        raise WebSocketError("reserved bits must be zero")
    if masked != expect_masked:
        raise WebSocketError("client frames must be masked" if expect_masked else "server frames must be unmasked")
    if opcode not in (OP_TEXT, OP_CLOSE, OP_PING, OP_PONG):
        raise WebSocketError("unsupported opcode")
    if opcode >= 8 and (not fin or length > 125):
        raise WebSocketError("invalid control frame")

    offset = 2
    if length == 126:
        if len(buf) < offset + 2:
            return None
        length = struct.unpack(">H", buf[offset:offset + 2])[0]
        offset += 2
    elif length == 127:
        if len(buf) < offset + 8:
            return None
        length = struct.unpack(">Q", buf[offset:offset + 8])[0]
        offset += 8

    if length > 2_000_000:
        raise WebSocketError("frame payload exceeds limit")
    if len(buf) < offset + (4 if masked else 0):
        return None
    mask_key = buf[offset:offset + 4] if masked else b"\0\0\0\0"
    offset += 4 if masked else 0
    if len(buf) < offset + length:
        return None

    payload = bytearray(length)
    for i in range(length):
        payload[i] = buf[offset + i] ^ mask_key[i % 4]

    if not fin:
        raise WebSocketError("fragmented messages are not supported")
    return opcode, bytes(payload), offset + length


class FrameDecoder:
    """Incremental client-frame decoder that yields complete messages."""

    def __init__(self, *, expect_masked: bool = True) -> None:
        self._buf = bytearray()
        self._expect_masked = expect_masked

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        """Append bytes and return completed ``(opcode, payload)`` frames."""
        self._buf.extend(data)
        out: list[tuple[int, bytes]] = []
        while self._buf:
            result = _decode_frame(bytes(self._buf), expect_masked=self._expect_masked)
            if result is None:
                break
            opcode, payload, consumed = result
            del self._buf[:consumed]
            out.append((opcode, payload))
        return out

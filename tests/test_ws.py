"""WebSocket codec tests (RFC 6455 handshake + frames)."""

import pytest

from coordination.ws import (
    OP_TEXT,
    FrameDecoder,
    WebSocketError,
    accept_key,
    build_101_response,
    encode_close_frame,
    encode_pong_frame,
    encode_text_frame,
    parse_http_upgrade,
)


def test_accept_key_rfc_example():
    # Canonical RFC 6455 example vector.
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_build_101_response_contains_accept():
    response = build_101_response("dGhlIHNhbXBsZSBub25jZQ==").decode("ascii")
    assert response.startswith("HTTP/1.1 101 Switching Protocols")
    assert "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in response


def test_parse_http_upgrade():
    raw = (
        b"GET /ws/swarm?role=b_formation HTTP/1.1\r\n"
        b"Host: 127.0.0.1:8765\r\n"
        b"Upgrade: websocket\r\n"
        b"Sec-WebSocket-Key: abc\r\n\r\n"
    )
    info = parse_http_upgrade(raw)
    assert info["path"] == "/ws/swarm?role=b_formation"
    assert info["headers"]["sec-websocket-key"] == "abc"
    assert info["headers"]["upgrade"] == "websocket"


def _masked_frame(payload: bytes, opcode: int = OP_TEXT, mask: bytes = b"\x00\x01\x02\x03") -> bytes:
    length = len(payload)
    b0 = 0x80 | (opcode & 0x0F)
    b1 = 0x80 | length  # masked, length < 126
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes((b0, b1)) + mask + masked


def test_frame_decoder_round_trip_text():
    decoder = FrameDecoder()
    frames = decoder.feed(_masked_frame("hello".encode()))
    assert frames == [(OP_TEXT, b"hello")]


def test_frame_decoder_incremental_feed():
    decoder = FrameDecoder()
    raw = _masked_frame(b'{"x":1}')
    assert decoder.feed(raw[:5]) == []
    assert decoder.feed(raw[5:]) == [(OP_TEXT, b'{"x":1}')]


def test_frame_decoder_rejects_unmasked_client_frame():
    payload = b"hi"
    raw = bytes((0x81, len(payload))) + payload  # MASK bit clear
    with pytest.raises(WebSocketError, match="masked"):
        FrameDecoder().feed(raw)


def test_encode_text_frame_is_unmasked():
    raw = encode_text_frame("hi")
    assert raw[0] == 0x81  # FIN + text
    assert raw[1] & 0x80 == 0  # unmasked (server → client)
    assert raw[1] & 0x7F == 2
    assert raw[2:] == b"hi"


def test_encode_close_and_pong():
    close = encode_close_frame(1000)
    assert close[0] == 0x88
    pong = encode_pong_frame(b"p")
    assert pong[0] == 0x8A

"""Tests for adapters/msgpack_compat.py — msgpack>=1.0 ``encoding`` shim.

AirSim's ``msgpack-rpc-python`` still passes ``encoding=`` to
``msgpack.Packer``/``Unpacker``; msgpack 1.x removed that keyword.  The shim
must make those calls construct cleanly while preserving a UTF-8 str round-trip.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import msgpack

# Importing the module applies the patch (module-level side effect).
from adapters import msgpack_compat  # noqa: F401


def test_packer_accepts_legacy_encoding():
    # msgpack-rpc-python's transport/tcp.py does exactly this.
    packer = msgpack.Packer(encoding="utf-8", default=lambda x: x.to_msgpack())
    assert packer is not None


def test_unpacker_accepts_legacy_encoding():
    unpacker = msgpack.Unpacker(encoding="utf-8")
    assert unpacker is not None


def test_round_trip_preserves_utf8_str():
    payload = {"type": "ping", "vehicle": "无人机"}
    data = msgpack.Packer().pack(payload)
    unpacker = msgpack.Unpacker()
    unpacker.feed(data)
    assert next(unpacker) == payload


def test_patch_is_idempotent():
    # Applying again must not raise or wrap the classes a second time.
    msgpack_compat.patch_msgpack()
    msgpack_compat.patch_msgpack()
    assert msgpack.Packer(encoding="utf-8").pack({"a": 1}) is not None

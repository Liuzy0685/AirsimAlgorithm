"""Compatibility shim: msgpack >= 1.0 removed the ``encoding`` argument.

AirSim's PythonClient depends on ``msgpack-rpc-python`` (0.4.1, 2015), whose
``transport/tcp.py`` still calls ``msgpack.Packer(encoding=...)`` and
``msgpack.Unpacker(encoding=...)``.  msgpack 1.x removed those keywords, so on a
modern interpreter ``airsim.MultirotorClient(...)`` raises::

    TypeError: __init__() got an unexpected keyword argument 'encoding'

This module wraps the two constructors to drop the legacy keyword(s) and
delegate to the real ones.  msgpack 1.x already encodes ``str`` as UTF-8 by
default — exactly what ``encoding='utf-8'`` did in 0.x — so the wire format is
unchanged and AirSim's RPC framing is preserved.  The patch is idempotent.
"""

from __future__ import annotations

import msgpack

_patched = False


def patch_msgpack() -> None:
    """Make ``msgpack.Packer``/``msgpack.Unpacker`` tolerate ``encoding``."""
    global _patched
    if _patched:
        return

    _RealPacker = msgpack.Packer
    _RealUnpacker = msgpack.Unpacker

    class _CompatPacker(_RealPacker):
        def __init__(self, *args, **kwargs):
            kwargs.pop("encoding", None)
            kwargs.pop("unicode_errors", None)
            super().__init__(*args, **kwargs)

    class _CompatUnpacker(_RealUnpacker):
        def __init__(self, *args, **kwargs):
            kwargs.pop("encoding", None)
            kwargs.pop("unicode_errors", None)
            # msgpack 0.x with encoding="utf-8" returned Python strings.
            # Preserve that contract on msgpack 1.x, whose raw default can
            # otherwise turn all AirSim RPC keys and text into bytes.
            kwargs.setdefault("raw", False)
            super().__init__(*args, **kwargs)

    msgpack.Packer = _CompatPacker
    msgpack.Unpacker = _CompatUnpacker
    _patched = True


# Apply once at import time so every consumer of this module is safe.
patch_msgpack()

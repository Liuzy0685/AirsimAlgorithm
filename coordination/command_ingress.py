"""BA command ingress — runtime validation for ``formation.command`` (step 3).

Protocol reference: `docs/多机通信与控制协议.md` §3 `formation.command` and §4.

The data model (:class:`common.models.FormationCommand` / ``CommandLifecycle``)
only carried the fields; this module wires the *runtime* checks that decide
accept / reject with a machine-readable reason:

1. **identity**  — the connection's handshake role must be ``b_formation`` and
   the command lifecycle ``stream_id`` must equal the connection's handshake
   ``stream_id`` (the service never trusts a body's self-reported source).
2. **lifecycle**  — a command without ``CommandLifecycle`` is rejected before
   any cross-process dispatch (``require_lifecycle()``).
3. **duplicate id** — a globally replayed ``command_id`` is rejected.
4. **stale sequence** — per ``(stream_id, vehicle_name, task_id)`` the sequence
   must strictly increase; reconnect does **not** reset the counter (only a
   process restart with a new handshake does, via a new ``stream_id``).
5. **timeout** — both the AirSim simulation window (``issued_at <= sim_time_ns
   < expires_at``) and the receiver-local TTL must be live.  The two clocks are
   checked independently and never subtracted from each other.

Session liveness (disconnect / reconnect) is the *service's* concern: the
service only routes frames from handshaked connections, so a dead B cannot
reach this validator, and a reconnecting B must re-handshake first.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from common.models import FormationCommand
from coordination.wire import ROLE_B_FORMATION

MonoClock = Callable[[], float]


@dataclass(frozen=True)
class CommandAcceptance:
    accepted: bool
    reason: str
    command_id: str


class FormationCommandIngress:
    """Validate incoming ``FormationCommand`` objects and deduplicate them."""

    def __init__(self, *, clock: MonoClock = time.monotonic) -> None:
        self._clock = clock
        self._seen_command_ids: set[str] = set()
        # (stream_id, vehicle_name, task_id) -> highest accepted sequence_id
        self._last_sequence: dict[tuple[str, str, str], int] = {}
        # command_id -> received monotonic seconds (for the dispatch-time TTL)
        self._received_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def accept(
        self,
        command: FormationCommand,
        *,
        connection,
        envelope_sequence_id: int,
        sim_time_ns: int,
        received_monotonic_seconds: Optional[float] = None,
        now_monotonic_seconds: Optional[float] = None,
    ) -> CommandAcceptance:
        """Validate one command and record accepted state.

        Parameters
        ----------
        command:
            An already-parsed ``FormationCommand`` (from
            :func:`coordination.wire.parse_formation_command_payload`).
        connection:
            The authenticated connection; must expose ``.role`` and
            ``.stream_id``.
        envelope_sequence_id:
            The envelope-level ``sequence_id`` (must equal the lifecycle's).
        sim_time_ns:
            A's latest AirSim simulation time in nanoseconds (from the newest
            authoritative snapshot), used to bound the command validity window.
        received_monotonic_seconds / now_monotonic_seconds:
            Receiver-local monotonic times for the TTL check.  Defaults to the
            current instant so the TTL starts when A accepts the message.
        """
        now = now_monotonic_seconds if now_monotonic_seconds is not None else self._clock()
        received = (
            received_monotonic_seconds
            if received_monotonic_seconds is not None
            else now
        )
        command_id = self._command_id(command)

        # 1. Identity — the handshake role must be the formation producer.
        role = getattr(connection, "role", None)
        if role != ROLE_B_FORMATION:
            return CommandAcceptance(False, "unauthorized_role", command_id)
        connection_stream_id = getattr(connection, "stream_id", None)

        # 2. Lifecycle required for cross-process dispatch.
        try:
            lifecycle = command.require_lifecycle()
        except ValueError:
            return CommandAcceptance(False, "missing_lifecycle", command_id)

        command_id = lifecycle.command_id

        # 2b. Lifecycle stream must match the authenticated connection.
        if connection_stream_id != lifecycle.stream_id:
            return CommandAcceptance(
                False, "stream_id_mismatch", command_id,
            )

        # 2c. Envelope sequence must equal lifecycle sequence (§3).
        if type(envelope_sequence_id) is not int or envelope_sequence_id != lifecycle.sequence_id:
            return CommandAcceptance(
                False, "envelope_sequence_mismatch", command_id,
            )

        # 3. Duplicate command_id (global replay detection).
        if command_id in self._seen_command_ids:
            return CommandAcceptance(False, "duplicate_command_id", command_id)

        # 4. Stale sequence within the same stream/vehicle/task.
        key = (lifecycle.stream_id, command.vehicle_name, lifecycle.task_id)
        last = self._last_sequence.get(key)
        if last is not None and lifecycle.sequence_id <= last:
            return CommandAcceptance(
                False, "stale_sequence", command_id,
            )

        # 5. Timeout — simulation window + receiver-local TTL, independently.
        try:
            current = lifecycle.is_current(
                sim_time_ns=sim_time_ns,
                received_monotonic_seconds=received,
                now_monotonic_seconds=now,
            )
        except (TypeError, ValueError) as exc:
            return CommandAcceptance(False, f"invalid_time_basis: {exc}", command_id)
        if not current:
            return CommandAcceptance(False, "expired", command_id)

        # Record accepted state before returning success.
        self._seen_command_ids.add(command_id)
        self._last_sequence[key] = lifecycle.sequence_id
        self._received_at[command_id] = received
        return CommandAcceptance(True, "accepted", command_id)

    def is_adoptable(
        self,
        command: FormationCommand,
        *,
        sim_time_ns: int,
        now_monotonic_seconds: Optional[float] = None,
    ) -> bool:
        """Re-check a previously accepted command's deadlines at dispatch time.

        The receiver-local TTL is re-evaluated here (``now - received`` must
        still be ``< ttl_s``), so a command accepted just before a long
        simulator pause cannot keep executing afterwards.
        """
        now = now_monotonic_seconds if now_monotonic_seconds is not None else self._clock()
        try:
            lifecycle = command.require_lifecycle()
        except ValueError:
            return False
        if lifecycle.command_id not in self._received_at:
            return False
        try:
            return lifecycle.is_current(
                sim_time_ns=sim_time_ns,
                received_monotonic_seconds=self._received_at.get(
                    lifecycle.command_id, now,
                ),
                now_monotonic_seconds=now,
            )
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Introspection (tests / diagnostics)
    # ------------------------------------------------------------------

    @property
    def seen_command_ids(self) -> frozenset:
        return frozenset(self._seen_command_ids)

    def remaining_ttl_s(self, command):
        life = command.require_lifecycle()
        received = self._received_at.get(life.command_id)
        return max(0.0, life.ttl_s - (self._clock() - received)) if received is not None else 0.0

    @property
    def last_sequence(self) -> dict:
        return dict(self._last_sequence)

    @staticmethod
    def _command_id(command: FormationCommand) -> str:
        lifecycle = getattr(command, "lifecycle", None)
        if lifecycle is None:
            return ""
        return str(lifecycle.command_id)

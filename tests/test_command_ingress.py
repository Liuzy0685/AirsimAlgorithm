"""FormationCommandIngress runtime checks (step 3)."""

from types import SimpleNamespace as NS

import pytest

from common.models import CommandLifecycle, FormationCommand
from coordination.command_ingress import FormationCommandIngress


class Clock:
    def __init__(self, now=10.0):
        self.now = now

    def __call__(self):
        return self.now


def _conn(role="b_formation", stream_id="b-stream"):
    return NS(role=role, stream_id=stream_id, producer_id="b-formation-service")


def _command(command_id="c1", sequence_id=1, source="formation", stream_id="b-stream",
             vehicle_name="Drone1", task_id="task-1", ttl_s=0.4,
             issued_at=1_000_000_000, expires_at=2_000_000_000, **kw):
    lc = CommandLifecycle(
        command_id=command_id, sequence_id=sequence_id, issued_at=issued_at,
        expires_at=expires_at, source=source, task_id=task_id,
        stream_id=stream_id, ttl_s=ttl_s,
    )
    return FormationCommand(vehicle_name=vehicle_name, lifecycle=lc, **kw)


def _accept(ingress, command, *, conn=None, envelope_seq=None, sim_time_ns=1_500_000_000, now=10.0):
    if envelope_seq is None:
        envelope_seq = command.lifecycle.sequence_id if command.lifecycle is not None else 0
    return ingress.accept(
        command,
        connection=conn or _conn(),
        envelope_sequence_id=envelope_seq,
        sim_time_ns=sim_time_ns,
        received_monotonic_seconds=now,
        now_monotonic_seconds=now,
    )


def test_accept_valid_command():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command())
    assert result.accepted is True
    assert result.command_id == "c1"
    assert result.reason == "accepted"


def test_reject_unauthorized_role():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command(), conn=_conn(role="c_telemetry"))
    assert result.accepted is False
    assert result.reason == "unauthorized_role"


def test_reject_stream_mismatch():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command(stream_id="other-stream"))
    assert result.reason == "stream_id_mismatch"


def test_reject_missing_lifecycle():
    ingress = FormationCommandIngress()
    command = FormationCommand(vehicle_name="Drone1")
    result = _accept(ingress, command)
    assert result.reason == "missing_lifecycle"
    assert result.command_id == ""


def test_reject_envelope_sequence_mismatch():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command(), envelope_seq=999)
    assert result.reason == "envelope_sequence_mismatch"


def test_reject_duplicate_command_id():
    ingress = FormationCommandIngress()
    assert _accept(ingress, _command(command_id="dup", sequence_id=1)).accepted
    result = _accept(ingress, _command(command_id="dup", sequence_id=2))
    assert result.reason == "duplicate_command_id"


def test_reject_stale_sequence_same_stream_vehicle_task():
    ingress = FormationCommandIngress()
    assert _accept(ingress, _command(sequence_id=5)).accepted
    result = _accept(ingress, _command(command_id="c2", sequence_id=5))
    assert result.reason == "stale_sequence"


def test_accept_increasing_sequence_reconnect_does_not_reset():
    ingress = FormationCommandIngress()
    assert _accept(ingress, _command(command_id="a", sequence_id=3)).accepted
    # Reconnect with the SAME stream_id: sequence must continue, not reset.
    assert _accept(ingress, _command(command_id="b", sequence_id=4)).accepted
    # A fresh stream_id (process restart) starts a new namespace.
    fresh = _command(command_id="c", sequence_id=1, stream_id="new-stream")
    assert _accept(ingress, fresh, conn=_conn(stream_id="new-stream")).accepted


def test_reject_expired_simulation_window():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command(), sim_time_ns=2_000_000_000)
    assert result.reason == "expired"


def test_reject_issued_after_sim_time():
    ingress = FormationCommandIngress()
    result = _accept(ingress, _command(), sim_time_ns=500_000_000)
    assert result.reason == "expired"


def test_reject_receiver_local_ttl_elapsed():
    ingress = FormationCommandIngress()
    # Received at t=10.0 but evaluated at t=10.5 with ttl_s=0.4 → too old.
    result = ingress.accept(
        _command(ttl_s=0.4), connection=_conn(),
        envelope_sequence_id=1, sim_time_ns=1_500_000_000,
        received_monotonic_seconds=10.0, now_monotonic_seconds=10.5,
    )
    assert result.reason == "expired"


def test_is_adoptable_reflects_ttl():
    ingress = FormationCommandIngress()
    command = _command(ttl_s=0.4)
    assert _accept(ingress, command, now=10.0).accepted
    # Still within sim window and within TTL.
    assert ingress.is_adoptable(command, sim_time_ns=1_500_000_000, now_monotonic_seconds=10.2)
    # TTL elapsed → no longer adoptable.
    assert not ingress.is_adoptable(command, sim_time_ns=1_500_000_000, now_monotonic_seconds=10.5)
    # Sim window closed → not adoptable.
    assert not ingress.is_adoptable(command, sim_time_ns=2_000_000_000, now_monotonic_seconds=10.2)

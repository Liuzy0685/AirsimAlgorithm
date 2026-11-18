"""CommandAuthority lease tests (single controller per vehicle)."""

import pytest

from control.command_authority import CommandAuthority


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def test_acquire_then_conflicting_acquire_rejected():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    assert auth.acquire("Drone1", "ctrl-a") is True
    assert auth.is_authorized("Drone1", "ctrl-a")
    # Different live controller cannot grab the lease.
    assert auth.acquire("Drone1", "ctrl-b") is False
    assert auth.is_authorized("Drone1", "ctrl-a")
    assert not auth.is_authorized("Drone1", "ctrl-b")


def test_same_controller_reacquire_renews():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    clock.now = 101.0
    assert auth.acquire("Drone1", "ctrl-a") is True
    assert auth.is_authorized("Drone1", "ctrl-a")


def test_lease_expires_and_can_be_taken_over():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    clock.now = 102.5  # past expiry
    assert not auth.is_authorized("Drone1", "ctrl-a")
    assert auth.acquire("Drone1", "ctrl-b") is True
    assert auth.is_authorized("Drone1", "ctrl-b")


def test_renew_requires_same_holder():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    clock.now = 101.0
    assert auth.renew("Drone1", "ctrl-b") is False
    assert auth.renew("Drone1", "ctrl-a") is True


def test_release_requires_same_holder():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    assert auth.release("Drone1", "ctrl-b") is False
    assert auth.release("Drone1", "ctrl-a") is True
    assert not auth.is_authorized("Drone1", "ctrl-a")


def test_explicit_handover():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    assert auth.handover("Drone1", "ctrl-a", "ctrl-b") is True
    assert auth.is_authorized("Drone1", "ctrl-b")
    assert not auth.is_authorized("Drone1", "ctrl-a")


def test_handover_requires_current_holder():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    assert auth.handover("Drone1", "ctrl-x", "ctrl-b") is False
    assert auth.is_authorized("Drone1", "ctrl-a")


def test_check_and_expire_frees_vehicles():
    clock = Clock()
    auth = CommandAuthority(lease_ttl_s=2.0, clock=clock)
    auth.acquire("Drone1", "ctrl-a")
    auth.acquire("Drone2", "ctrl-a")
    clock.now = 101.0
    auth.renew("Drone2", "ctrl-a")
    clock.now = 102.5  # Drone1 (102.0) expired, Drone2 (103.0) still live
    expired = auth.check_and_expire()
    assert set(expired) == {"Drone1"}
    assert "Drone2" in auth.vehicles


def test_rejects_empty_identifiers():
    auth = CommandAuthority(lease_ttl_s=2.0)
    with pytest.raises(ValueError, match="vehicle_name"):
        auth.acquire("  ", "ctrl-a")
    with pytest.raises(ValueError, match="controller_id"):
        auth.acquire("Drone1", "")

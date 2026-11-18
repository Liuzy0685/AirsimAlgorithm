"""VehicleCommandGateway — the single AirSim flight-control entry (§5).

Every flight-control call (velocity, takeoff, hover, land, emergency stop)
must pass through this gateway.  It enforces, in order:

1. **authority** — the calling ``controller_id`` must hold a live lease from
   :class:`control.command_authority.CommandAuthority` for that vehicle;
2. **safety** — a formation/arbitrated target is gated by
   :class:`control.safety_filter.SafetyFilter` (finite, speed-limited,
   inter-vehicle separation) before it reaches the dispatcher;
3. **dispatch** — the injected dispatcher (normally
   :class:`control.velocity_controller.VelocityController`) is the only object
   that actually talks to AirSim.

The gateway does not import ``airsim`` and does not decide who controls what;
those responsibilities belong to the authority and the safety filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from common.models import FormationCommand
from control.command_authority import CommandAuthority
from control.safety_filter import SafetyFilter
from control.velocity_arbitrator import VelocityArbitrator, VelocityCandidate


@dataclass(frozen=True)
class GatewayResult:
    dispatched: bool
    reason: str


class VehicleCommandGateway:
    """Single, authority-checked, safety-gated flight-control entry point."""

    def __init__(
        self,
        authority: CommandAuthority,
        *,
        dispatcher,
        safety_filter: Optional[SafetyFilter] = None,
        arbitrator: Optional[VelocityArbitrator] = None,
        clock: Callable[[], float] = None,
    ) -> None:
        if not isinstance(authority, CommandAuthority):
            raise ValueError("authority must be a CommandAuthority")
        if not callable(getattr(dispatcher, "send_velocity_world_ned", None)):
            raise ValueError("dispatcher must expose send_velocity_world_ned")
        self._authority = authority
        self._dispatcher = dispatcher
        self._safety = safety_filter or SafetyFilter()
        self._arbitrator = arbitrator or VelocityArbitrator()
        self._clock = clock if clock is not None else __import__("time").monotonic

    # ------------------------------------------------------------------
    # Authority helper
    # ------------------------------------------------------------------

    def _authorized(self, vehicle_name: str, controller_id: str) -> Optional[GatewayResult]:
        if not self._authority.is_authorized(vehicle_name, controller_id):
            return GatewayResult(False, "unauthorized_controller")
        return None

    # ------------------------------------------------------------------
    # Velocity dispatch
    # ------------------------------------------------------------------

    def dispatch_velocity(
        self,
        vehicle_name: str,
        vx: float,
        vy: float,
        vz: float,
        *,
        controller_id: str,
        duration: Optional[float] = None,
    ) -> GatewayResult:
        """Dispatch an explicit World-NED velocity after an authority check."""
        denied = self._authorized(vehicle_name, controller_id)
        if denied is not None:
            return denied
        if not all(math.isfinite(float(c)) for c in (vx, vy, vz)):
            return GatewayResult(False, "non_finite")
        self._dispatcher.send_velocity_world_ned(
            vx, vy, vz, duration=duration, vehicle_name=vehicle_name,
        )
        return GatewayResult(True, "")

    def dispatch_formation(
        self,
        command: FormationCommand,
        *,
        controller_id: str,
        own_state,
        neighbors: Sequence,
        neighbors_complete: bool = True,
        duration: Optional[float] = None,
    ) -> GatewayResult:
        """Gate a formation target through authority + safety, then dispatch."""
        denied = self._authorized(command.vehicle_name, controller_id)
        if denied is not None:
            return denied
        decision = self._safety.check(
            command, own_state=own_state, neighbors=neighbors,
            neighbors_complete=neighbors_complete,
        )
        if not decision.allowed:
            return GatewayResult(False, decision.reason)
        self._dispatcher.send_velocity_world_ned(
            *decision.desired_velocity_ned_mps,
            duration=duration, vehicle_name=command.vehicle_name,
        )
        return GatewayResult(True, "")

    def dispatch_arbitrated(
        self,
        candidates: Sequence[Optional[VelocityCandidate]],
        *,
        vehicle_name: str,
        controller_id: str,
        own_state,
        neighbors: Sequence,
        neighbors_complete: bool = True,
        duration: Optional[float] = None,
    ) -> GatewayResult:
        """Arbitrate, then gate the winner through safety before dispatch."""
        denied = self._authorized(vehicle_name, controller_id)
        if denied is not None:
            return denied
        winner = self._arbitrator.arbitrate(candidates)
        if not winner.valid:
            return GatewayResult(False, winner.reason)
        decision = self._safety.check_velocity(
            winner.desired_velocity_ned_mps, own_state=own_state,
            neighbors=neighbors, neighbors_complete=neighbors_complete,
        )
        if not decision.allowed:
            return GatewayResult(False, decision.reason)
        self._dispatcher.send_velocity_world_ned(
            *decision.desired_velocity_ned_mps,
            duration=duration, vehicle_name=vehicle_name,
        )
        return GatewayResult(True, "")

    # ------------------------------------------------------------------
    # Lifecycle control (also authority-gated)
    # ------------------------------------------------------------------

    def takeoff(self, vehicle_name: str, *, controller_id: str) -> GatewayResult:
        denied = self._authorized(vehicle_name, controller_id)
        if denied is not None:
            return denied
        self._dispatcher.takeoff(vehicle_name=vehicle_name)
        return GatewayResult(True, "")

    def hover(self, vehicle_name: str, *, controller_id: str) -> GatewayResult:
        denied = self._authorized(vehicle_name, controller_id)
        if denied is not None:
            return denied
        self._dispatcher.hover(vehicle_name=vehicle_name)
        return GatewayResult(True, "")

    def land(self, vehicle_name: str, *, controller_id: str) -> GatewayResult:
        denied = self._authorized(vehicle_name, controller_id)
        if denied is not None:
            return denied
        self._dispatcher.land(vehicle_name=vehicle_name)
        return GatewayResult(True, "")

    def emergency_stop(self, vehicle_name: str, *, controller_id: str) -> GatewayResult:
        """Request an immediate verifiable hold (hover) for the vehicle.

        Note: this maps to the dispatcher's ``hover`` hold until a dedicated
        emergency-stop API is available in the target AirSim build.
        """
        return self.hover(vehicle_name, controller_id=controller_id)

    def bind_raw_client(self, client, *, vehicle_name, controller_id):
        """Bind legacy lifecycle/VelocityController calls to this same authority boundary."""
        return GatewayFlightClient(self, client, vehicle_name, controller_id)


class GatewayFlightClient:
    """Restricted legacy adapter: reads pass through; every flight write is gated.

    The process must also own SharedFlightSession's per-vehicle OS lock. Leases
    are local to that owning process; the state service additionally rejects
    duplicate live controller subscriptions. Unknown write APIs are denied.
    """
    _FLIGHT = {"enableApiControl", "armDisarm", "takeoffAsync", "hoverAsync", "landAsync",
               "moveByVelocityAsync", "moveByVelocityBodyFrameAsync", "moveByVelocityZBodyFrameAsync", "moveToZAsync"}

    def __init__(self, gateway, raw, vehicle_name, controller_id):
        self.gateway, self._raw = gateway, raw
        self.vehicle_name, self.controller_id = vehicle_name, controller_id
        self.motion_guard = None
        self.lifecycle_guard = None

    def __getattr__(self, name):
        if name in self._FLIGHT:
            def checked(*args, **kwargs):
                if kwargs.get("vehicle_name") != self.vehicle_name:
                    raise PermissionError("gateway vehicle routing mismatch")
                denied = self.gateway._authorized(self.vehicle_name, self.controller_id)
                if denied:
                    raise PermissionError(denied.reason)
                if self.lifecycle_guard is not None:
                    self.lifecycle_guard(name, args, kwargs)
                if self.motion_guard is not None and (name.startswith("move") or name == "hoverAsync"):
                    args, kwargs = self.motion_guard(name, args, kwargs)
                return getattr(self._raw, name)(*args, **kwargs)
            return checked
        if name.startswith(("get", "simGet", "simList", "simPlot", "simPrint")) or name in {
                "close", "ping", "confirmConnection", "isApiControlEnabled", "listVehicles"}:
            return getattr(self._raw, name)
        raise AttributeError(f"AirSim API {name!r} is not allowed through the control gateway")

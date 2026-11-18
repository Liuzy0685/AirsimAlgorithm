"""B-side coordination status gate for A's authoritative stream."""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.constants import COORDINATION_STATES


@dataclass(frozen=True)
class _Status:
    state: str
    received_at: float
    sequence_id: int


class FormationStatusGate:
    """Fail closed on missing/stale A statuses and retain conflict-clear events.

    A can report ``CONFLICT_CLEARED`` and then ``NORMAL`` between two B
    snapshots. The clear event is consumed only after B emits commands from a
    fresh complete snapshot, so recovery cannot be skipped by message timing.
    """

    def __init__(self, vehicle_names, *, task_id: str, max_age_s: float) -> None:
        self.vehicle_names = frozenset(vehicle_names)
        if not self.vehicle_names or any(not isinstance(n, str) or not n.strip()
                                          for n in self.vehicle_names):
            raise ValueError("vehicle_names must contain nonempty names")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonempty")
        if not isinstance(max_age_s, (int, float)) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.task_id = task_id
        self.max_age_s = max_age_s
        self._statuses: dict[str, _Status] = {}
        self._clear_pending = False

    def reset(self) -> None:
        self._statuses.clear()
        self._clear_pending = False

    def ingest(self, payload: dict, *, sequence_id: int, received_at: float) -> bool:
        if (not isinstance(payload, dict) or type(received_at) not in (int, float)
                or not math.isfinite(received_at)):
            return False
        name, state = payload.get("vehicle_name"), payload.get("state")
        if (not isinstance(name, str) or name not in self.vehicle_names
                or not isinstance(state, str) or state not in COORDINATION_STATES
                or type(sequence_id) is not int or sequence_id < 0):
            return False
        # A's disconnect notification has no task ID but must still inhibit B.
        if payload.get("task_id") != self.task_id and state != "EMERGENCY":
            return False
        previous = self._statuses.get(name)
        if previous is not None and sequence_id <= previous.sequence_id:
            return False
        self._statuses[name] = _Status(state, received_at, sequence_id)
        if state in ("CONFLICT_DETECTED", "AVOIDING", "EMERGENCY"):
            self._clear_pending = False
        elif state == "CONFLICT_CLEARED":
            self._clear_pending = True
        return True

    def state_for_snapshot(self, *, now: float) -> str | None:
        """Return the planner state, or None until every A status is fresh."""
        if (type(now) not in (int, float) or not math.isfinite(now)
                or set(self._statuses) != self.vehicle_names):
            return None
        if any(not 0 <= now - item.received_at < self.max_age_s
               for item in self._statuses.values()):
            self._clear_pending = False
            return None
        states = {item.state for item in self._statuses.values()}
        for blocking in ("EMERGENCY", "AVOIDING", "CONFLICT_DETECTED"):
            if blocking in states:
                return blocking
        if self._clear_pending:
            return "CONFLICT_CLEARED"
        if "FORMATION_RECOVERING" in states:
            return "FORMATION_RECOVERING"
        return "NORMAL"

    def commands_emitted(self) -> None:
        self._clear_pending = False

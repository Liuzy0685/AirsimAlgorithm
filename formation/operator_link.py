"""Nonblocking B control channel for the loopback operator API v1."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from formation.controller import FORMATION_SHAPES, FormationConfig, FormationController
from formation.navigation import FormationNavigation


@dataclass(frozen=True)
class OperatorSettings:
    base_url: str
    mission_id: str
    initial_revision: int
    bearer_token: str

    @classmethod
    def from_swarm(cls, swarm: dict, task_id: str):
        raw = swarm.get("operator_api")
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) != {
            "version", "base_url", "mission_id", "initial_revision", "bearer_token"
        } or type(raw["version"]) is not int or raw["version"] != 1:
            raise ValueError("invalid operator_api configuration")
        parsed = urlsplit(raw["base_url"])
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or not parsed.port or parsed.path.rstrip("/") != "/api/v1"
                or parsed.query or parsed.fragment):
            raise ValueError("operator_api must use local HTTP /api/v1")
        if (raw["mission_id"] != task_id or not isinstance(task_id, str)
                or not task_id.strip() or type(raw["initial_revision"]) is not int
                or raw["initial_revision"] < 1 or not isinstance(raw["bearer_token"], str)
                or not raw["bearer_token"].strip()):
            raise ValueError("operator_api task, revision or credential mismatch")
        return cls(**{k: raw[k] for k in ("base_url", "mission_id", "initial_revision", "bearer_token")})


class OperatorHTTPError(Exception):
    def __init__(self, code: int):
        super().__init__(f"operator HTTP {code}")
        self.code = code


class OperatorTransport:
    def __init__(self, settings: OperatorSettings, *, timeout_s: float = 0.25):
        self.settings, self.timeout_s = settings, timeout_s
        self.base = (settings.base_url.rstrip("/") + "/internal/missions/"
                     + quote(settings.mission_id, safe="") + "/formation/")

    def request(self, action: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request = Request(self.base + action, data=data,
                          headers={"Authorization": "Bearer " + self.settings.bearer_token,
                                   "Content-Type": "application/json"},
                          method="GET" if data is None else "POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError("operator response must be an object")
            return value
        except HTTPError as exc:
            raise OperatorHTTPError(exc.code) from None


def _validate_request(raw: object, *, mission_id: str, revision: int, now_ms: int):
    if not isinstance(raw, dict) or set(raw) != {
        "request_id", "mission_id", "revision", "formation", "expires_at_unix_ms"
    }:
        return "invalid_request"
    if raw["mission_id"] != mission_id or not isinstance(raw["request_id"], str) or not raw["request_id"].strip():
        return "task_or_request_id_mismatch"
    if type(raw["revision"]) is not int or raw["revision"] != revision + 1:
        return "stale_revision"
    if type(raw["expires_at_unix_ms"]) is not int or raw["expires_at_unix_ms"] <= now_ms:
        return "expired"
    value = raw["formation"]
    if (not isinstance(value, dict) or set(value) != {"shape", "spacing_m"}
            or value["shape"] not in FORMATION_SHAPES
            or type(value["spacing_m"]) not in (int, float)
            or not math.isfinite(value["spacing_m"]) or value["spacing_m"] <= 0):
        return "invalid_formation"
    return None


class FormationOperatorLink:
    """HTTP work runs in a task; planner mutations run only on the WS snapshot loop."""

    def __init__(self, settings: OperatorSettings, planner: FormationController,
                 navigator: FormationNavigation, *, safety_distance_m: float,
                 max_speed_mps: float, prediction_horizon_s: float,
                 transport: OperatorTransport | None = None):
        self.settings = settings
        self.planner, self.navigator = planner, navigator
        self.transport = transport or OperatorTransport(settings)
        self.revision = settings.initial_revision
        self._safety_distance = safety_distance_m
        self._max_speed = max_speed_mps
        self._horizon = prediction_horizon_s
        self._request = None
        self._applied_ids: set[str] = set()
        self._result = None
        self._awaiting = None
        self._previous: FormationConfig | None = None
        self._result_outcome: int | None = None
        self._status = None
        self._last_status_stamp = -1
        self._last_heartbeat = 0.0
        self._last_poll = 0.0
        self._last_status_post = 0.0
        self._online_at = 0.0
        self._recovery_after_apply = False
        self._version_conflict = False

    def _result_body(self, request, state, reason):
        return {k: request[k] for k in ("request_id", "revision", "formation")} | {
            "state": state, "reason": reason[:256]
        }

    def _accept_request(self, raw):
        if raw is None:
            self._request = None
        elif isinstance(raw, dict) and self._awaiting is None:
            self._request = raw

    def on_snapshot(self, *, gate_state: str | None, now: float,
                    now_unix_ms: int | None = None) -> bool:
        """Apply at a frame boundary. False means no commands until result ACK."""
        if self._result_outcome is not None:
            outcome = self._result_outcome
            self._result_outcome = None
            self._finish_result(outcome)
        if self._version_conflict:
            return False
        if self._awaiting is not None:
            return False
        request = self._request
        if request is None:
            return True
        now_ms = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
        reason = _validate_request(request, mission_id=self.settings.mission_id,
                                   revision=self.revision, now_ms=now_ms)
        if reason is None and request["request_id"] in self._applied_ids:
            reason = "duplicate_request_id"
        if reason is not None:
            if all(k in request for k in ("request_id", "revision", "formation")):
                self._result = self._result_body(request, "REJECTED", reason)
            self._request = None
            return True
        if gate_state is None or gate_state in ("AVOIDING", "CONFLICT_DETECTED", "EMERGENCY"):
            self._result = self._result_body(request, "DEFERRED", "coordination_not_safe")
            return True
        # A status and operator heartbeat must both be recent before mutation.
        if not 0 <= now - self._online_at < 2:
            self._result = self._result_body(request, "DEFERRED", "operator_channel_unavailable")
            return True
        self._previous = self.planner.config
        try:
            updated = replace(self._previous, **request["formation"])
            self.planner.config = updated
            self.navigator.nominal_spacing_m = updated.spacing_m
            self.navigator.spacing = max(updated.spacing_m, self.navigator.separation + .3,
                self._safety_distance + 2 * self._max_speed * self._horizon + .5)
            self.navigator.reset()
            self.planner._recovering = True
        except (TypeError, ValueError) as exc:
            self.planner.config = self._previous
            self._result = self._result_body(request, "REJECTED", str(exc))
            self._request = None
            return True
        self._awaiting = request
        self._result = self._result_body(request, "APPLIED", "")
        self._status = None
        return False

    def _result_response(self, code: int, result: dict):
        if self._result is not result:
            return
        self._result = None
        if self._awaiting is not None:
            self._result_outcome = code

    def _finish_result(self, code: int):
        if self._awaiting is None:
            return
        result = self._result_body(self._awaiting, "APPLIED", "")
        if code == 200:
            self.revision = result["revision"]
            self._applied_ids.add(result["request_id"])
            self._recovery_after_apply = True
        else:
            self.planner.config = self._previous
            self.navigator.nominal_spacing_m = self._previous.spacing_m
            self.navigator.spacing = max(self._previous.spacing_m, self.navigator.separation + .3,
                self._safety_distance + 2 * self._max_speed * self._horizon + .5)
            self.navigator.reset()
            self.planner._recovering = True
            self._version_conflict = True
        self._awaiting = self._previous = self._request = None

    def planning_state(self, gate_state: str) -> str:
        if (self._recovery_after_apply and gate_state in
                ("NORMAL", "FORMATION_RECOVERING")):
            return "CONFLICT_CLEARED"
        return gate_state

    def commands_emitted(self):
        self._recovery_after_apply = False

    def update_status(self, snapshot, targets, *, mode: str, reason: str):
        if self._awaiting is not None or self._version_conflict or targets is None:
            return
        if not self.planner._valid_snapshot(snapshot, snapshot.timestamp, snapshot.timestamp):
            return
        stamps = [state.timestamp for state in snapshot.states.values()]
        if not stamps or max(stamps) <= self._last_status_stamp:
            return
        if set(targets) != set(self.planner.config.followers):
            return
        errors = {name: math.dist(snapshot.states[name].position_ned_m, target)
                  for name, target in targets.items()}
        self._status = dict(revision=self.revision, nominal_shape=self.planner.config.shape,
            effective_mode=mode, slot_targets_ned_m={n: list(v) for n, v in targets.items()},
            tracking_error_m=errors, sim_timestamp_ns=max(stamps), reason=reason[:256])

    async def run(self, stop):
        """Bounded HTTP calls never run on the WebSocket/planning task."""
        while not stop():
            now = time.monotonic()
            try:
                if not self._version_conflict and now - self._last_heartbeat >= .5:
                    await asyncio.to_thread(self.transport.request, "heartbeat", {
                        "protocol_version": 1, "capabilities": ["formation.configure"]})
                    self._last_heartbeat = now
                    self._online_at = time.monotonic()
                if now - self._last_poll >= .2:
                    response = await asyncio.to_thread(self.transport.request, "request")
                    self._accept_request(response.get("request"))
                    self._last_poll = now
                if self._result is not None:
                    body = self._result
                    try:
                        response = await asyncio.to_thread(self.transport.request, "result", body)
                        if response.get("accepted") is not True:
                            raise ValueError("operator result was not accepted")
                        self._result_response(200, body)
                    except OperatorHTTPError as exc:
                        if exc.code == 409:
                            self._result_response(409, body)
                        else:
                            raise
                if (self._awaiting is None and self._status is not None
                        and now - self._last_status_post >= .5
                        and self._status["sim_timestamp_ns"] > self._last_status_stamp):
                    body = self._status
                    await asyncio.to_thread(self.transport.request, "status", body)
                    self._last_status_stamp = body["sim_timestamp_ns"]
                    self._last_status_post = now
            except (OperatorHTTPError, URLError, OSError, ValueError, TimeoutError):
                pass
            await asyncio.sleep(.05)

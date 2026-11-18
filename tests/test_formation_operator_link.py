"""B runtime configuration contract against A's loopback HTTP facade."""

import asyncio
import threading
import time

import pytest

from coordination.navigation import NavigationConfig
from coordination.operator_common import atomic_json
from coordination.operator_http import OperatorHTTPServer
from formation import FormationController
from formation.navigation import FormationNavigation
from formation.operator_link import (FormationOperatorLink, OperatorSettings,
                                     OperatorTransport, OperatorHTTPError)
from test_formation_controller import config, snapshot
from test_operator_api import create, start, env  # shared A test fixture


def link(settings=None):
    settings = settings or OperatorSettings("http://127.0.0.1:8790/api/v1", "mission", 1, "secret")
    planner = FormationController(config(), task_id=settings.mission_id)
    navigator = FormationNavigation(NavigationConfig(), task_id=settings.mission_id,
        vehicle_names=planner.config.vehicle_names, safety_distance_m=2,
        spacing_m=planner.config.spacing_m, max_speed_mps=1, prediction_horizon_s=3)
    return FormationOperatorLink(settings, planner, navigator,
        safety_distance_m=2, max_speed_mps=1, prediction_horizon_s=3)


def request(*, revision=2, shape="vee", spacing=4, expires=None, mid="mission", rid="switch-1"):
    return dict(request_id=rid, mission_id=mid, revision=revision,
        formation=dict(shape=shape, spacing_m=spacing),
        expires_at_unix_ms=expires or int(time.time() * 1000) + 30_000)


def test_config_absence_and_invalid_identity():
    assert OperatorSettings.from_swarm({}, "mission") is None
    raw = dict(version=1, base_url="http://127.0.0.1:8790/api/v1",
               mission_id="mission", initial_revision=1, bearer_token="secret")
    assert OperatorSettings.from_swarm({"operator_api": raw}, "mission").initial_revision == 1
    with pytest.raises(ValueError):
        OperatorSettings.from_swarm({"operator_api": dict(raw, mission_id="wrong")}, "mission")
    with pytest.raises(ValueError):
        OperatorSettings.from_swarm({"operator_api": dict(raw, base_url="http://example.com/api/v1")}, "mission")


def test_deferred_then_atomic_apply_and_recovery():
    runtime = link()
    runtime._online_at = 10
    runtime._accept_request(request())
    assert runtime.on_snapshot(gate_state="AVOIDING", now=10.1)
    assert runtime._result["state"] == "DEFERRED"
    assert runtime.planner.config.shape == "column"
    assert not runtime.on_snapshot(gate_state="NORMAL", now=10.2)
    assert runtime.planner.config.shape == "vee"
    assert runtime.planner.config.spacing_m == 4
    assert runtime.navigator.nominal_spacing_m == 4
    assert runtime.navigator.spacing >= 4
    body = runtime._result
    runtime._result_response(200, body)
    assert runtime.revision == 1  # HTTP task cannot mutate a planning frame.
    assert runtime.on_snapshot(gate_state="NORMAL", now=10.3)
    assert runtime.revision == 2
    assert runtime.planning_state("NORMAL") == "CONFLICT_CLEARED"
    runtime.commands_emitted()
    assert runtime.planning_state("NORMAL") == "NORMAL"
    runtime._accept_request(request())
    assert runtime.on_snapshot(gate_state="NORMAL", now=10.4)
    assert runtime._result["state"] == "REJECTED"
    assert runtime.planner.config.shape == "vee"


@pytest.mark.parametrize("change, reason", [
    ({"revision": 1}, "stale_revision"),
    ({"revision": 4}, "stale_revision"),
    ({"mid": "other"}, "task_or_request_id_mismatch"),
    ({"expires": 100}, "expired"),
    ({"shape": "circle"}, "invalid_formation"),
])
def test_bad_requests_do_not_change_planner(change, reason):
    runtime = link()
    runtime._online_at = 10
    runtime._accept_request(request(**change))
    assert runtime.on_snapshot(gate_state="NORMAL", now=10.1)
    assert runtime._result["reason"] == reason
    assert runtime.planner.config.shape == "column"


def test_409_rolls_back_and_status_uses_real_targets():
    runtime = link()
    runtime._online_at = 10
    runtime._accept_request(request())
    assert not runtime.on_snapshot(gate_state="NORMAL", now=10.1)
    runtime._result_response(409, runtime._result)
    assert not runtime.on_snapshot(gate_state="NORMAL", now=10.2)
    assert runtime.planner.config.shape == "column"
    assert runtime.navigator.nominal_spacing_m == 3
    assert not runtime.on_snapshot(gate_state="NORMAL", now=10.3)
    snap = snapshot(positions={2: (-2, 0, -2)}, sequence=3)
    targets = runtime.planner.slot_targets(snap.states["Drone1"])
    runtime.update_status(snap, targets, mode="trail", reason="obstacle_constrained")
    assert runtime._status is None
    healthy = link()
    healthy.update_status(snap, targets, mode="trail", reason="obstacle_constrained")
    runtime = healthy
    assert runtime._status["slot_targets_ned_m"] == {"Drone2": [-3, 0, -2]}
    assert runtime._status["tracking_error_m"]["Drone2"] == pytest.approx(1)
    assert runtime._status["sim_timestamp_ns"] == 1_000_000_000
    runtime._last_status_stamp = 1_000_000_000
    runtime._status = None
    runtime.update_status(snap, targets, mode="nominal", reason="")
    assert runtime._status is None


def test_disconnected_operator_defers_without_mutating():
    runtime = link()
    runtime._online_at = 1.0
    runtime._accept_request(request())
    assert runtime.on_snapshot(gate_state="NORMAL", now=10.1)
    assert runtime._result["state"] == "DEFERRED"
    assert runtime._result["reason"] == "operator_channel_unavailable"
    assert runtime.planner.config.shape == "column"


def test_http_timeout_does_not_block_planning_loop():
    class SlowTransport:
        def request(self, action, body=None):
            time.sleep(.35)
            raise OSError("offline")

    async def scenario():
        runtime = link()
        runtime.transport = SlowTransport()
        stop = lambda: False
        task = asyncio.create_task(runtime.run(stop))
        await asyncio.sleep(.03)
        start_at = time.monotonic()
        assert runtime.on_snapshot(gate_state="NORMAL", now=start_at)
        assert time.monotonic() - start_at < .1
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_http_heartbeat_request_result_and_status_with_real_a_server(env):
    mid = create(env)
    start(env, mid)
    mission = env.m.data["missions"][mid]
    mission["state"] = "RUNNING"
    server = OperatorHTTPServer(("127.0.0.1", 0), env.m, "x" * 32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = OperatorSettings(server.manager.url, mid, 1, mission["_b_token"])
        transport = OperatorTransport(settings)
        runtime = link(settings)
        runtime.transport = transport
        reply = transport.request("heartbeat", dict(protocol_version=1,
            capabilities=["formation.configure"]))
        assert reply["online"]
        env.m.configure(mid, dict(request_id="switch-live", expected_revision=1,
            formation=dict(shape="vee", spacing_m=3.)))
        pulled = transport.request("request")["request"]
        assert pulled["request_id"] == "switch-live"
        runtime._online_at = time.monotonic()
        runtime._accept_request(pulled)
        assert runtime.on_snapshot(gate_state="AVOIDING", now=time.monotonic())
        deferred = transport.request("result", runtime._result)
        assert deferred["accepted"]
        assert env.m.get(mid)["applied_formation"]["shape"] == "line"
        assert not runtime.on_snapshot(gate_state="NORMAL", now=time.monotonic())
        body = runtime._result
        applied = transport.request("result", body)
        assert applied["accepted"] and not applied["physical_formation_settled"]
        runtime._result_response(200, body)
        assert runtime.on_snapshot(gate_state="NORMAL", now=time.monotonic())
        assert runtime.revision == 2
        stamp = 1_000_000_000
        atomic_json(env.m.directory / mid / "runtime.json", dict(
            timestamp_unix_ms=int(time.time() * 1000), collection_complete=True,
            vehicles={name: dict(timestamp=stamp) for name in env.p.names}))
        snap = snapshot(positions={2: (1, 0, -2)})
        targets = runtime.planner.slot_targets(snap.states["Drone1"])
        runtime.update_status(snap, targets, mode="recovering", reason="shape_change")
        assert transport.request("status", runtime._status)["accepted"]
        actual = env.m.get(mid)["formation_runtime"]
        assert actual["revision"] == 2
        assert actual["effective_mode"] == "recovering"
        assert actual["tracking_error_m"]["Drone2"] == pytest.approx(5)
        with pytest.raises(Exception):
            transport.request("status", runtime._status)
        assert transport.request("request")["request"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

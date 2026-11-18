"""End-to-end A/B contract and final AirSim write boundary, without a simulator."""
import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
import time

import numpy as np
import pytest

from common.models import CommandLifecycle, FormationCommand, VehicleState
from control.swarm_flight_bridge import SwarmFlightBridge
from control.velocity_controller import VelocityCommandRejected
from coordination.client import SwarmConnection
from coordination.command_ingress import FormationCommandIngress
from coordination.controller_link import ControllerLink
from coordination.coordinate_transform import CoordinateTransformer, CoordinateTransformValidation, VehicleTransformCheck
from coordination.runtime_config import ServiceConfig
from coordination.state_provider import StateProvider
from coordination.state_publisher import SwarmStatePublisher
from coordination.state_service import SwarmStateService, run_swarm_state_server
from coordination.wire import build_envelope, parse_envelope, parse_swarm_snapshot
from formation.controller import FormationConfig, FormationController


class Clock:
    def __init__(self): self.now = time.monotonic()
    def __call__(self): return self.now


class Reader:
    def __init__(self, x): self.x, self.timestamp = x, 1_500_000_000
    def read(self):
        self.timestamp += 1
        return VehicleState(position_ned_m=[self.x, 0, -1], ready=True,
                            timestamp=self.timestamp)


def fixture(count=2):
    clock = Clock()
    names = [f"Drone{i+1}" for i in range(count)]
    tf = CoordinateTransformer({name: (0, 0, 0) for name in names})
    # Pure test fixture only; production always calls validate against AirSim.
    tf._validated = True
    tf._last_validation = CoordinateTransformValidation(validated=True,
        vehicles={name: VehicleTransformCheck(name, ok=True) for name in names})
    provider = StateProvider({name: Reader(i*10) for i, name in enumerate(names)}, tf, clock=clock)
    publisher = SwarmStatePublisher(provider, producer_id="a-state-service", stream_id="a-boot", mono_clock=clock)
    publisher.collect_and_publish()
    ingress = FormationCommandIngress(clock=clock)
    service = SwarmStateService(publisher, ingress, producer_id="a-state-service", stream_id="a-boot",
        clock=clock, task_id="swarm-mission", require_controller=True)
    return clock, names, publisher, ingress, service


def connect_controller(service, publisher, names, name="Drone2"):
    link = ControllerLink(ServiceConfig(), name, names)
    link._connected = True
    link.ingest(parse_envelope(json.dumps(publisher.latest_envelope)))
    holder = []
    def receive(env):
        if env["type"] == "formation.command":
            permit = service.handle_message(holder[0], json.dumps(link.command_request(1)))
            link.ingest(parse_envelope(json.dumps(permit)))
        else:
            link.ingest(parse_envelope(json.dumps(env)))
    cid = service.register_client(role="a_controller", producer_id=link.producer_id,
        stream_id=link.stream_id, vehicle_name=name, send=receive)
    holder.append(cid)
    return link, cid


def command(command_id="c1", seq=1, vehicle="Drone2", **kw):
    life = CommandLifecycle(command_id=command_id, sequence_id=seq, issued_at=1_000_000_000,
        expires_at=2_000_000_000, source="formation", task_id="swarm-mission", stream_id="b-boot", ttl_s=0.4)
    return FormationCommand(vehicle_name=vehicle, desired_velocity_ned_mps=(0.1, 0, 0), lifecycle=life, **kw)


def send_command(service, cmd, cid=None):
    if cid is None:
        cid = service.register_client(role="b_formation", producer_id="b", stream_id="b-boot", send=lambda _: None)
    env = build_envelope(message_type="formation.command", producer_id="b", stream_id="b-boot",
        sequence_id=cmd.lifecycle.sequence_id, payload=asdict(cmd))
    return service.handle_message(cid, json.dumps(env)), cid


@pytest.mark.parametrize("count", [2, 5])
def test_b_plans_from_authoritative_wire_and_a_relays_to_each_controller(count):
    clock, names, publisher, ingress, service = fixture(count)
    links = {name: connect_controller(service, publisher, names, name)[0] for name in names[1:]}
    cfg = FormationConfig(names[0], tuple(names[1:]), "column", 3, .5, .5, .4, .5, .5, 1, .25)
    planner = FormationController(cfg, task_id="swarm-mission", stream_id="b-boot")
    env = parse_envelope(json.dumps(publisher.latest_envelope))
    snapshot = parse_swarm_snapshot(env, received_at=clock())
    commands = planner.plan(snapshot, received_at_monotonic_s=clock(), now_monotonic_s=clock(), snapshot_stream_id=env.stream_id)
    for name, cmd in commands.items():
        ack, _ = send_command(service, cmd)
        assert ack["payload"]["accepted"]
        assert links[name].command().lifecycle.command_id == cmd.lifecycle.command_id
    assert len(commands) == count - 1


def test_disconnect_revokes_cached_command_and_reconnect_preserves_replay_history():
    clock, names, publisher, ingress, service = fixture()
    link, _ = connect_controller(service, publisher, names)
    cmd = command()
    ack, cid = send_command(service, cmd)
    assert ack["payload"]["accepted"] and link.command()
    service.unregister_client(cid)
    assert link.command() is None
    assert service.latest_formation_command("Drone2") is None
    ack, _ = send_command(service, cmd)
    assert ack["payload"]["reason"] == "duplicate_command_id"


@pytest.mark.parametrize("failure,expected", [("unknown", "unknown_vehicle"),
    ("task", "task_id_mismatch"), ("no_controller", "controller_unavailable"),
    ("stale_snapshot", "no_authoritative_snapshot")])
def test_ingress_runtime_rejections(failure, expected):
    clock, names, publisher, ingress, service = fixture()
    if failure != "no_controller": connect_controller(service, publisher, names)
    cmd = command(vehicle="Missing" if failure == "unknown" else "Drone2")
    if failure == "task": cmd.lifecycle = CommandLifecycle(**dict(asdict(cmd.lifecycle), task_id="other-task"))
    if failure == "stale_snapshot": clock.now += 1
    ack, _ = send_command(service, cmd)
    assert ack["payload"]["reason"] == expected


def test_expired_unaccepted_command_is_never_adoptable():
    _, _, _, ingress, _ = fixture()
    assert not ingress.is_adoptable(command(), sim_time_ns=1_500_000_000)


def test_duplicate_controller_cannot_claim_vehicle():
    _, names, publisher, _, service = fixture()
    connect_controller(service, publisher, names)
    with pytest.raises(Exception, match="already registered"):
        connect_controller(service, publisher, names)


class Raw:
    def __init__(self): self.calls = []
    def moveByVelocityAsync(self, *args, **kwargs): self.calls.append((args, kwargs)); return "future"


def bridge_fixture():
    _, names, publisher, _, service = fixture()
    link, _ = connect_controller(service, publisher, names)
    send_command(service, command())
    raw = Raw()
    session = SimpleNamespace(_owns_lock=True, vehicle_name="Drone2", client=raw,
        adapter=SimpleNamespace(_client=raw), state=SimpleNamespace(phase=None))
    params = SimpleNamespace(forward_speed_mps=1., max_vertical_speed_mps=.5, command_duration_s=.2, target_z_ned=-1.)
    config = dict(drones=[dict(vehicle_name=n) for n in names], coordination=dict(safety_distance_m=2., prediction_horizon_s=3.))
    bridge = SwarmFlightBridge(session, config, ServiceConfig(), params)
    bridge.link = link
    bridge.authority.acquire("Drone2", bridge.controller_id)
    bridge.begin_cruise(SimpleNamespace(_traj_params=SimpleNamespace(hard_clearance_m=.5),
        _swarm_spawn_offsets={n: (0, 0, 0) for n in names}))
    bridge.points = np.array([[20., 20., 0.]])
    bridge.points_at = time.monotonic()
    return bridge, raw, service


def test_ack_to_gateway_to_actual_velocity_api():
    bridge, raw, _ = bridge_fixture()
    own = bridge.link.snapshot().states["Drone2"]
    vx, vy, vz, source = bridge.select(0, 0, 0, "trajectory", own)
    assert source == "formation_velocity"
    assert bridge.proxy.moveByVelocityAsync(vx, vy, vz, .2, vehicle_name="Drone2") == "future"
    assert len(raw.calls) == 1


@pytest.mark.parametrize("failure", ["disconnected", "expired", "lidar", "lease", "wrong_vehicle", "peer_conflict"])
def test_gateway_faults_never_reach_raw_airsim(failure):
    bridge, raw, _ = bridge_fixture()
    bridge.adopted_command = bridge.link.command()
    if failure == "disconnected": bridge.link._connected = False
    if failure == "expired": bridge.link._received_command_at -= 1
    if failure == "lidar": bridge.points = np.array([[.1, 0., 0.]])
    if failure == "lease": bridge.authority.release("Drone2", bridge.controller_id)
    if failure == "peer_conflict": bridge.link._snapshot.states["Drone1"].position_ned_m = [10.1, 0, -1]
    with pytest.raises((VelocityCommandRejected, PermissionError)):
        bridge.proxy.moveByVelocityAsync(.1, 0., 0., .2,
            vehicle_name="Drone1" if failure == "wrong_vehicle" else "Drone2")
    assert raw.calls == []


def test_unknown_rpc_write_cannot_bypass_gateway():
    bridge, raw, _ = bridge_fixture()
    with pytest.raises(AttributeError): bridge.proxy.moveToPositionAsync
    with pytest.raises(AttributeError): bridge.proxy.call


def test_relay_permit_deducts_round_trip_without_resetting_a_reception_ttl():
    clock, names, publisher, _, service = fixture()
    link = ControllerLink(ServiceConfig(), "Drone2", names, clock=clock)
    link._connected = True
    link.ingest(parse_envelope(json.dumps(publisher.latest_envelope)))
    aid = service.register_client(role="a_controller", producer_id=link.producer_id,
        stream_id=link.stream_id, vehicle_name="Drone2", send=lambda _: None)
    send_command(service, command())
    request = link.command_request(1)
    clock.now += .1
    permit = service.handle_message(aid, json.dumps(request))
    clock.now += .1
    link.ingest(parse_envelope(json.dumps(permit)))
    assert link.command() is not None
    assert link.command().lifecycle.ttl_s <= .101
    clock.now += .11
    assert link.command() is None


def test_duplicate_snapshot_id_does_not_refresh_cached_state_age():
    clock, names, publisher, _, _ = fixture()
    link = ControllerLink(ServiceConfig(), "Drone2", names, clock=clock)
    link._connected = True
    link.ingest(parse_envelope(json.dumps(publisher.latest_envelope)))
    clock.now += .7
    env = dict(publisher.latest_envelope, sequence_id=100)
    link.ingest(parse_envelope(json.dumps(env)))
    assert link.snapshot() is None


def test_snapshot_status_and_ack_share_a_single_outbound_sequence():
    _, names, publisher, _, service = fixture()
    messages = []
    service.register_client(role="c_telemetry", producer_id="c", stream_id="c1", send=messages.append)
    service.publish_status(vehicle_name="Drone1", state="NORMAL")
    service.broadcast_snapshot(publisher.latest_envelope)
    sequences = [m["sequence_id"] for m in messages]
    assert sequences == sorted(set(sequences))
    assert messages[-1]["payload"]["snapshot_id"] == publisher.latest_snapshot.sequence_id


def test_controller_disconnect_revokes_command_before_new_controller_connects():
    _, names, publisher, _, service = fixture()
    _, aid = connect_controller(service, publisher, names)
    send_command(service, command())
    service.unregister_client(aid)
    assert service.latest_formation_command("Drone2") is None


def test_graceful_stop_keeps_state_service_alive_until_vehicle_cleanup_finishes(monkeypatch):
    from scripts import launch_swarm
    events = []
    class Process:
        def __init__(self, name): self.name, self.stopped, self.pid = name, False, 1
        def poll(self): return 0 if self.stopped else None
        def send_signal(self, signal): events.append((self.name, "signal"))
        def wait(self, timeout): self.stopped = True; events.append((self.name, "cleaned"))
    processes = [(name, Process(name)) for name in ("A-state-service", "Drone1", "Drone2", "B-formation-service")]
    launch_swarm._request_graceful_stop(processes)
    assert events.index(("A-state-service", "signal")) > events.index(("Drone2", "cleaned"))


def test_unverified_emergency_does_not_turn_into_blind_hover_or_land():
    bridge, raw, _ = bridge_fixture()
    bridge.link._connected = False
    with pytest.raises(VelocityCommandRejected): bridge.tick()
    bridge.prepare_cleanup()
    assert bridge.session.state.phase.name == "MANUAL_INTERVENTION_REQUIRED"
    assert raw.calls == []


def test_real_websocket_snapshot_command_ack_and_disconnect():
    async def scenario():
        _, names, publisher, _, service = fixture()
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(run_swarm_state_server(service, port=0,
            on_ready=lambda server: ready.set_result(server.sockets[0].getsockname()[1])))
        clients = []
        try:
            port = await asyncio.wait_for(ready, 2)
            url = f"ws://127.0.0.1:{port}/ws/swarm"
            a = await SwarmConnection.connect(url, role="a_controller", producer_id="a-control", stream_id="ac", vehicle_name="Drone2")
            b = await SwarmConnection.connect(url, role="b_formation", producer_id="b", stream_id="b-boot")
            c = await SwarmConnection.connect(url, role="c_telemetry", producer_id="c", stream_id="c-boot")
            clients.extend([a,b,c])
            for client in clients:
                assert (await asyncio.wait_for(client.receive(), 2)).message_type == "swarm.snapshot"
            cmd = command()
            await b.send(build_envelope(message_type="formation.command", producer_id="b", stream_id="b-boot",
                sequence_id=1, payload=asdict(cmd)))
            async def until(client, message_type):
                while True:
                    env = await asyncio.wait_for(client.receive(), 2)
                    if env.message_type == message_type: return env
            assert (await until(b, "command.ack")).payload["accepted"]
            assert (await until(a, "formation.command")).payload["lifecycle"]["command_id"] == "c1"
            await b.close()
            assert (await until(a, "coordination.status")).payload["reason"] == "formation_source_disconnected"
        finally:
            for client in clients:
                await client.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


def test_background_controller_receives_permit_and_clears_command_on_b_disconnect():
    async def scenario():
        clock, names, publisher, _, service = fixture()
        ready = asyncio.get_running_loop().create_future()
        async def collect():
            clock.now = time.monotonic()
            return publisher.collect_and_publish()
        task = asyncio.create_task(run_swarm_state_server(service, port=0, collect=collect,
            poll_interval_s=.05, on_ready=lambda s: ready.set_result(s.sockets[0].getsockname()[1])))
        link, b = None, None
        try:
            port = await asyncio.wait_for(ready, 2)
            link = ControllerLink(ServiceConfig(port=port, startup_timeout_s=2), "Drone2", names)
            await asyncio.to_thread(link.start)
            b = await SwarmConnection.connect(f"ws://127.0.0.1:{port}/ws/swarm", role="b_formation", producer_id="b", stream_id="b-boot")
            cmd = command()
            await b.send(build_envelope(message_type="formation.command", producer_id="b", stream_id="b-boot",
                sequence_id=1, payload=asdict(cmd)))
            deadline = time.monotonic() + 1
            while link.command() is None and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            assert link.command().lifecycle.command_id == "c1"
            await b.close()
            deadline = time.monotonic() + 1
            while link.command() is not None and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            assert link.command() is None
        finally:
            if b: await b.close()
            if link: await asyncio.to_thread(link.close)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())

"""Network runner for B's existing pure FormationController (no AirSim access)."""
import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coordination.client import SwarmConnection
from coordination.runtime_config import load_config
from coordination.wire import build_envelope, parse_swarm_snapshot
from formation.controller import FormationController, load_formation_config
from formation.runtime import FormationStatusGate


async def run(args):
    swarm, cfg = load_config(args.swarm_config)
    formation = load_formation_config(args.formation_config)
    if set(formation.vehicle_names) != {m["vehicle_name"] for m in swarm["drones"]}:
        raise ValueError("formation/swarm vehicle names must match")
    if formation.leader != cfg.leader:
        raise ValueError("formation leader must match state_service.leader")
    planner = FormationController(formation, task_id=cfg.task_id, stream_id=str(uuid.uuid4()))
    producer = "b-formation-service"
    delay = 0.5
    gate = FormationStatusGate(formation.vehicle_names, task_id=cfg.task_id,
                               max_age_s=cfg.disconnect_timeout_s)
    while True:
        connection = None
        try:
            connection = await SwarmConnection.connect(cfg.url, role="b_formation",
                producer_id=producer, stream_id=planner.stream_id)
            gate.reset()
            delay, last_ping = 0.5, 0.0
            while True:
                if time.monotonic() - last_ping >= cfg.heartbeat_s:
                    await connection.send_frame(9, b"B")
                    last_ping = time.monotonic()
                try:
                    env = await asyncio.wait_for(connection.receive(), cfg.heartbeat_s)
                except asyncio.TimeoutError:
                    continue
                if env.producer_id != "a-state-service":
                    raise ValueError("unexpected A producer")
                if env.message_type == "command.ack":
                    print(f"ACK {env.payload}", flush=True)
                elif env.message_type == "coordination.status":
                    gate.ingest(env.payload, sequence_id=env.sequence_id,
                                received_at=time.monotonic())
                elif env.message_type == "swarm.snapshot":
                    received = time.monotonic()
                    snap = parse_swarm_snapshot(env, received_at=received)
                    state = gate.state_for_snapshot(now=received)
                    if state is None:
                        continue
                    commands = planner.plan(snap, received_at_monotonic_s=received,
                        snapshot_stream_id=env.stream_id, coordination_state=state)
                    for command in commands.values():
                        await connection.send(build_envelope(message_type="formation.command",
                            producer_id=producer, stream_id=planner.stream_id,
                            sequence_id=command.lifecycle.sequence_id, payload=asdict(command)))
                    if commands:
                        gate.commands_emitted()
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"B disconnected: {exc}; reconnect in {delay}s", flush=True)
        finally:
            if connection:
                await connection.close()
        await asyncio.sleep(delay)
        delay = min(5, delay * 2 if delay < 2 else 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm-config", default=str(ROOT / "configs/swarm.yaml"))
    parser.add_argument("--formation-config", default=str(ROOT / "configs/formation.yaml"))
    try:
        asyncio.run(run(parser.parse_args()))
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    import signal
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    raise SystemExit(main())

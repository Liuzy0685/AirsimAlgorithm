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
from formation.navigation import FormationNavigation
from formation.operator_link import FormationOperatorLink, OperatorSettings
from coordination.navigation import NavigationConfig


async def run(args):
    swarm, cfg = load_config(args.swarm_config)
    formation = load_formation_config(args.formation_config)
    if set(formation.vehicle_names) != {m["vehicle_name"] for m in swarm["drones"]}:
        raise ValueError("formation/swarm vehicle names must match")
    if formation.leader != cfg.leader:
        raise ValueError("formation leader must match state_service.leader")
    planner = FormationController(formation, task_id=cfg.task_id, stream_id=str(uuid.uuid4()))
    producer = "b-formation-service"
    gate = FormationStatusGate(formation.vehicle_names, task_id=cfg.task_id,
                               max_age_s=cfg.disconnect_timeout_s)
    navigator=FormationNavigation(NavigationConfig(**swarm.get('navigation',{})),task_id=cfg.task_id,
        vehicle_names=formation.vehicle_names,safety_distance_m=swarm['coordination']['safety_distance_m'],spacing_m=formation.spacing_m,
        max_speed_mps=formation.max_recovery_speed_mps,prediction_horizon_s=swarm['coordination']['prediction_horizon_s'])
    operator_settings = OperatorSettings.from_swarm(swarm, cfg.task_id)
    operator = (FormationOperatorLink(operator_settings, planner, navigator,
        safety_distance_m=swarm['coordination']['safety_distance_m'],
        max_speed_mps=formation.max_recovery_speed_mps,
        prediction_horizon_s=swarm['coordination']['prediction_horizon_s'])
        if operator_settings is not None else None)
    stop_file=Path(args.stop_file) if getattr(args,'stop_file',None) else None
    def stopping():return stop_file is not None and stop_file.exists()
    operator_task = asyncio.create_task(operator.run(stopping)) if operator else None
    try:
        await _run_swarm_loop(cfg, formation, planner, producer, gate, navigator,
                              operator, stopping)
    finally:
        if operator_task:
            operator_task.cancel()
            await asyncio.gather(operator_task, return_exceptions=True)


async def _run_swarm_loop(cfg, formation, planner, producer, gate, navigator,
                          operator, stopping):
    delay = 0.5
    last_navigation_mode = None
    last_snapshot = None
    while not stopping():
        connection = None
        try:
            connection = await SwarmConnection.connect(cfg.url, role="b_formation",
                producer_id=producer, stream_id=planner.stream_id)
            gate.reset()
            navigator.reset()
            delay, last_ping = 0.5, 0.0
            while not stopping():
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
                    navigator.ingest(env.payload,received_at=time.monotonic())
                elif env.message_type == "swarm.snapshot":
                    received = time.monotonic()
                    snap = parse_swarm_snapshot(env, received_at=received)
                    state = gate.state_for_snapshot(now=received)
                    if not snap.collection_complete:
                        continue
                    if not planner._valid_snapshot(snap, received, received):
                        continue
                    if (last_snapshot is not None and last_snapshot[0] == env.stream_id
                            and snap.sequence_id <= last_snapshot[1]):
                        continue
                    last_snapshot = (env.stream_id, snap.sequence_id)
                    if operator and not operator.on_snapshot(gate_state=state, now=received):
                        continue
                    if state is None:
                        continue
                    targets=navigator.targets(snap,planner.slot_targets(snap.states[planner.config.leader]),
                        leader=planner.config.leader,now=received)
                    if targets is None:
                        if operator:
                            nominal = planner.slot_targets(snap.states[planner.config.leader])
                            operator.update_status(snap, nominal, mode='holding', reason='navigation_feedback_unavailable')
                        continue
                    if navigator.mode!=last_navigation_mode:
                        print(f'formation_navigation mode={navigator.mode}',flush=True)
                        last_navigation_mode=navigator.mode
                    commands = planner.plan(snap, received_at_monotonic_s=received,
                        snapshot_stream_id=env.stream_id,
                        coordination_state=operator.planning_state(state) if operator else state,
                        target_overrides=targets)
                    if operator:
                        mode = ('holding' if not commands else 'trail' if navigator.mode == 'trail'
                                else 'recovering' if any(c.mode == 'velocity' for c in commands.values())
                                else 'nominal')
                        operator.update_status(snap, targets, mode=mode,
                                               reason=('obstacle_constrained' if mode == 'trail' else
                                                       state.lower() if mode != 'nominal' else ''))
                    for command in commands.values():
                        if navigator.mode=='trail':command.formation_id=f'{planner.config.shape}:trail'
                        await connection.send(build_envelope(message_type="formation.command",
                            producer_id=producer, stream_id=planner.stream_id,
                            sequence_id=command.lifecycle.sequence_id, payload=asdict(command)))
                    if commands:
                        gate.commands_emitted()
                        if operator:
                            operator.commands_emitted()
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"B disconnected: {exc}; reconnect in {delay}s", flush=True)
        finally:
            if connection:
                await connection.close()
        if stopping():break
        await asyncio.sleep(delay)
        delay = min(5, delay * 2 if delay < 2 else 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm-config", default=str(ROOT / "configs/swarm.yaml"))
    parser.add_argument("--formation-config", default=str(ROOT / "configs/formation.yaml"))
    parser.add_argument('--stop-file',help='Cooperative stop request; does not dispatch any flight command.')
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

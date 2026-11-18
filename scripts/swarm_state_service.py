"""Run A's read-only authoritative state publisher and B command ingress."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.airsim_client import AirSimClientAdapter
from coordination.command_ingress import FormationCommandIngress
from coordination.coordinate_transform import CoordinateTransformer
from coordination.runtime_config import load_config
from coordination.state_provider import StateProvider
from coordination.state_publisher import SwarmStatePublisher
from coordination.state_service import SwarmStateService, run_swarm_state_server


def make_publisher(config, settings_json, service_config):
    adapter = AirSimClientAdapter(readonly=True)
    try:
        adapter.connect()
        client = adapter.get_raw_client()
        names = [m["vehicle_name"] for m in config["drones"]]
        markers = {m["vehicle_name"]: m["start_actor"] for m in config["drones"] if m.get("start_actor")}
        if markers and len(markers) != len(names):
            raise ValueError("all vehicles require start_actor when using markers")
        tf = (CoordinateTransformer.from_scene_markers(client, markers) if markers else
              CoordinateTransformer.from_settings_json(settings_json, names))
        report = tf.validate(client, markers=markers or None)
        for name, check in report.vehicles.items():
            print(f"{name}: validated={check.ok} position_error_m={check.position_error_m} "
                  f"velocity_error_mps={check.velocity_error_mps} yaw_rad={check.yaw_rad}", flush=True)
        if not report.validated:
            raise RuntimeError(f"coordinate validation failed: {report.errors}")
        provider = StateProvider.from_adapter(adapter, names, tf,
            max_age_s=service_config.snapshot_max_age_s,
            allow_legacy_readiness=service_config.allow_legacy_readiness,
            max_skew_s=config["coordination"]["max_state_timestamp_skew_s"])
        publisher = SwarmStatePublisher(provider, producer_id="a-state-service", stream_id=str(uuid.uuid4()))
        publisher.collect_and_publish()
        snap = publisher.latest_snapshot
        if snap is None or not snap.collection_complete:
            raise RuntimeError(f"initial snapshot incomplete: {publisher.last_error or (snap.errors if snap else '')}")
        return adapter, publisher
    except BaseException:
        adapter.close()
        raise


async def run(args):
    config, cfg = load_config(args.swarm_config)
    # AirSim/msgpack RPC clients stay on one dedicated thread, including construction.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="A-state-rpc") as executor:
        loop = asyncio.get_running_loop()
        adapter, publisher = await loop.run_in_executor(executor, make_publisher, config, args.settings_json, cfg)
        service = SwarmStateService(publisher, FormationCommandIngress(),
            producer_id=publisher._producer_id, stream_id=publisher._stream_id,
            snapshot_max_age_s=cfg.snapshot_max_age_s, task_id=cfg.task_id, require_controller=True,
            formation_targets=set(publisher.configured_names) - {cfg.leader})
        async def collect():
            return await loop.run_in_executor(executor, publisher.collect_and_publish)
        def ready(server):
            print(f"A state service ready: {cfg.url}", flush=True)
            if args.ready_file:
                Path(args.ready_file).write_text(publisher._stream_id, encoding="utf-8")
        try:
            await run_swarm_state_server(service, cfg.host, cfg.port, collect=collect,
                poll_interval_s=config["coordination"]["peer_poll_interval_s"],
                heartbeat_s=cfg.heartbeat_s, disconnect_timeout_s=cfg.disconnect_timeout_s, on_ready=ready)
        finally:
            await loop.run_in_executor(executor, adapter.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--swarm-config", default=str(ROOT / "configs/swarm.yaml"))
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"A state service failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import signal
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    raise SystemExit(main())

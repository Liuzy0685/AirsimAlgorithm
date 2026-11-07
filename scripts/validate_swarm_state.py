"""Read-only live validation and sequential RPC timing for StateProvider.

Example:
  python scripts/validate_swarm_state.py --settings-json path/to/settings.json
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.airsim_client import AirSimClientAdapter
from coordination.coordinate_transform import CoordinateTransformer
from coordination.state_provider import StateProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--swarm-config", default="configs/swarm.yaml")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    with open(args.swarm_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    members = config["drones"]
    names = [member["vehicle_name"] for member in members]
    if len(set(names)) != len(names) or not 1 <= len(names) <= 5:
        parser.error("swarm config must define 1..5 distinct vehicles")
    adapter = AirSimClientAdapter(readonly=True)
    try:
        adapter.connect()
        client = adapter.get_raw_client()
        markers = {member["vehicle_name"]: member["start_actor"]
                   for member in members if member.get("start_actor")}
        if markers and len(markers) != len(names):
            parser.error("every vehicle needs start_actor when markers are used")
        if markers:
            tf = CoordinateTransformer.from_scene_markers(client, markers)
        else:
            tf = CoordinateTransformer.from_settings_json(args.settings_json, names)
        report = tf.validate(client, markers=markers or None)
        for name, check in report.vehicles.items():
            print(f"{name}: valid={check.ok} position_error_m={check.position_error_m:.4f} "
                  f"velocity_error_mps={check.velocity_error_mps:.4f} {check.message}")
        if not report.validated:
            print("Transform validation failed; no state snapshots will be used.")
            return 2
        provider = StateProvider.from_adapter(
            adapter, names, tf,
            max_age_s=max(0.5, config.get("coordination", {}).get("peer_poll_interval_s", 0.2) * 3),
            max_skew_s=config.get("coordination", {}).get("max_state_timestamp_skew_s", 0.5),
        )
        for _ in range(args.samples):
            snap = provider.get_snapshot()
            reads = " ".join(f"{name}={duration*1000:.1f}ms"
                             for name, duration in snap.read_duration_s.items())
            print(f"snapshot={snap.sequence_id} complete={snap.collection_complete} "
                  f"total={snap.collection_duration*1000:.1f}ms "
                  f"read_skew={snap.max_time_skew*1000:.1f}ms "
                  f"sim_skew={snap.max_sim_timestamp_skew*1000:.1f}ms "
                  f"{reads} errors={snap.errors}")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())

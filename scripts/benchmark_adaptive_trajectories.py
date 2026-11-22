"""Offline planner timing/safety fixtures; never connects to AirSim or commands flight.

Compare current fixed sampling, fixed curves with segment checks/time budget,
and adaptive curves. Multi-process load is synthetic, not flight/RPC validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from flight_modes.automatic_mode import _build_trajectory_params
from mapping.distance_field import DistanceField
from planners.local_trajectory_planner import LocalTrajectoryPlanner


def fixtures():
    def cylinder(x, y, radius, count):
        return [(x + radius * math.cos(a), y + radius * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, count, endpoint=False)]
    return {
        "open_goal_turn": [],
        "offset_pillar": cylinder(3., .5, .5, 96),
        "narrow_passage": [(float(x), y) for x in np.linspace(0, 6, 120) for y in (-1.35, 1.35)],
        "dead_end": [(float(x), y) for x in np.linspace(0, 4, 80) for y in (-1.2, 1.2)]
                    + [(3., float(y)) for y in np.linspace(-1.2, 1.2, 60)],
        "dense_8000_points": sum((cylinder(x, y, .5, 2000)
                                   for x, y in ((3., -1.6), (3., 1.6), (6., -2.), (6., 2.))), []),
    }


def stats(samples):
    return {"p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)), "max_ms": max(samples)}


def run_vehicle(task):
    vehicle, iterations = task
    cfg = yaml.safe_load((ROOT / "configs/trajectory_planner.yaml").read_text(encoding="utf-8"))
    live = _build_trajectory_params(cfg)
    rows = []
    for name, obstacles in fixtures().items():
        for mode in ("fixed_sampled", "fixed_segment_checked", "adaptive"):
            p = replace(live, adaptive_candidates_enabled=mode != "fixed_sampled",
                        adaptive_proposals=live.adaptive_proposals if mode == "adaptive" else 0,
                        adaptive_refinements=live.adaptive_refinements if mode == "adaptive" else 0)
            planner = LocalTrajectoryPlanner(p)
            df = DistanceField(obstacles)
            wall, compute = [], []
            selected = timed_out = no_plan = unsafe = refined = 0
            max_candidates = 0
            for i in range(iterations + 3):
                # Repeated live goals and poses exercise memory and revalidation;
                # they are fixtures, not an integrated vehicle motion model.
                position = (.01 * (i % 5), 0., -2.)
                goal = (8., 2.4 if name == "open_goal_turn" else 0.)
                started = time.perf_counter()
                result = planner.plan(position, 0., goal, None, df)
                elapsed = (time.perf_counter() - started) * 1000
                if i < 3:
                    continue
                wall.append(elapsed)
                compute.append(result.compute_ms)
                max_candidates = max(max_candidates, result.generated)
                timed_out += int(result.budget_exhausted)
                if result.selected:
                    selected += 1
                    refined += int(result.selected.generation_source != "fixed")
                    unsafe += int(df.trajectory_clearances(result.selected.points)[0] < p.hard_clearance_m - 1e-9)
                else:
                    no_plan += 1
            rows.append(dict(vehicle=vehicle, fixture=name, mode=mode, frames=iterations,
                             obstacle_points=len(obstacles), wall=stats(wall), compute=stats(compute),
                             budget_ms=p.max_compute_ms, budget_exhausted=timed_out,
                             wall_budget_overruns=sum(t > p.max_compute_ms for t in wall),
                             selected=selected, no_plan=no_plan, unsafe_selected=unsafe,
                             selected_adaptive=refined, max_generated=max_candidates))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--workers", type=int, choices=(1, 2, 5), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.iterations <= 1000:
        parser.error("iterations must be between 5 and 1000")
    tasks = [(i + 1, args.iterations) for i in range(args.workers)]
    if args.workers == 1:
        rows = run_vehicle(tasks[0])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for batch in pool.map(run_vehicle, tasks) for row in batch]
    sources = ("planners/local_trajectory_planner.py", "mapping/distance_field.py",
               "configs/trajectory_planner.yaml", "scripts/benchmark_adaptive_trajectories.py")
    payload = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        scope="offline synthetic planner fixtures; no AirSim, RPC, camera load or flight validation",
        baseline_note="fixed baseline uses the current revision's distance-query acceleration; no timing claim against an untouched old checkout",
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        cpu=platform.processor(), workers=args.workers, iterations=args.iterations,
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256_lf={name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                          for name in sources},
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for mode in ("fixed_sampled", "fixed_segment_checked", "adaptive"):
        subset = [r for r in rows if r["mode"] == mode]
        print(json.dumps(dict(mode=mode, frames=sum(r["frames"] for r in subset),
            max_p95_ms=max(r["wall"]["p95_ms"] for r in subset),
            max_ms=max(r["wall"]["max_ms"] for r in subset),
            no_plan=sum(r["no_plan"] for r in subset), unsafe_selected=sum(r["unsafe_selected"] for r in subset),
            budget_exhausted=sum(r["budget_exhausted"] for r in subset))))


if __name__ == "__main__":
    main()

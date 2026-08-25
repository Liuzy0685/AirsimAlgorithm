#!/usr/bin/env python
"""Render a trajectory flight trace CSV to PNG (Phase C0, sec 22).

Usage::

    py scripts/plot_flight_trace.py runs/trace_20260815_120000.csv [-o trace.png]

The CSV is written by ``FlightTraceWriter`` (one row per control frame).  The
plot shows the drone path (world-NED XY), the goal marker, and a color-by
command-source / family scatter.  Matplotlib is optional — if unavailable the
script exits cleanly with a message.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a trajectory flight trace CSV to PNG.")
    p.add_argument("csv_path", help="Path to the FlightTraceWriter CSV.")
    p.add_argument("-o", "--output", default=None, help="Output PNG path (default: <csv>.png).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable — skipping PNG ({exc}).")
        return 0

    xs, ys, sources, families = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (KeyError, ValueError):
                continue
            sources.append(row.get("command_source", ""))
            families.append(row.get("family", ""))

    if not xs:
        print("No rows found.")
        return 1

    source_colors = {
        "trajectory": "tab:blue",
        "recovery": "tab:red",
        "guided_apf": "tab:orange",
        "apf": "tab:green",
        "reactive": "tab:purple",
        "trajectory_no_feasible": "tab:gray",
    }
    colors = [source_colors.get(s, "tab:gray") for s in sources]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, c=colors, s=12, alpha=0.85)
    ax.plot(xs, ys, "-", color="0.85", linewidth=0.8, alpha=0.6)
    # Start / end markers.
    ax.scatter([xs[0]], [ys[0]], marker="o", s=80, facecolor="none",
               edgecolors="k", linewidths=1.5, label="start")
    ax.scatter([xs[-1]], [ys[-1]], marker="*", s=140, c="k", label="end")
    ax.set_xlabel("World X (North, m)")
    ax.set_ylabel("World Y (East, m)")
    ax.set_title(f"Trajectory flight trace — {csv_path.name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(args.output) if args.output else csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out} ({len(xs)} points).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

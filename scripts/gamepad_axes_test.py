#!/usr/bin/env python
"""Diagnose whether a gamepad actually transmits axis/button data.

Samples every joystick for ~10 seconds and reports the min/max of every axis
and the set of buttons pressed during that window.  Run it, then **move the
sticks / press buttons during the 10-second window**, and read the report.

Usage::

    py scripts/gamepad_axes_test.py
    py scripts/gamepad_axes_test.py --seconds 10
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    import pygame

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f"joystick count = {count}")

    if count == 0:
        print("No joystick detected by pygame.")
        return 1

    joys = []
    for i in range(count):
        j = pygame.joystick.Joystick(i)
        j.init()
        joys.append(j)
        print(f"  [{i}] {j.get_name()}  axes={j.get_numaxes()} "
              f"buttons={j.get_numbuttons()} hats={j.get_numhats()}")

    seconds = max(1.0, args.seconds)
    print(f"\nSampling for {seconds:.0f}s — MOVE THE STICKS / PRESS BUTTONS NOW…")

    axis_lo = [[9e9] * j.get_numaxes() for j in joys]
    axis_hi = [[-9e9] * j.get_numaxes() for j in joys]
    buttons_seen = [set() for _ in joys]

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pygame.event.pump()
        for i, j in enumerate(joys):
            for a in range(j.get_numaxes()):
                v = j.get_axis(a)
                if v < axis_lo[i][a]:
                    axis_lo[i][a] = v
                if v > axis_hi[i][a]:
                    axis_hi[i][a] = v
            for b in range(j.get_numbuttons()):
                if j.get_button(b):
                    buttons_seen[i].add(b)
        time.sleep(0.02)

    print("\n=== RESULTS ===\n")
    for i, j in enumerate(joys):
        print(f"[{i}] {j.get_name()}")
        for a in range(j.get_numaxes()):
            lo, hi = axis_lo[i][a], axis_hi[i][a]
            changed = "← CHANGED" if hi - lo > 0.05 else "(no change)"
            print(f"    axis {a:2d}: min={lo:+.2f}  max={hi:+.2f}  {changed}")
        btns = sorted(buttons_seen[i])
        print(f"    buttons pressed: {btns if btns else '(none)'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

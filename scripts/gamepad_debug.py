#!/usr/bin/env python
"""Gamepad debug utility — inspect raw joystick state without AirSim.

Prints a live table of every Xbox axis/button/hat value so you can verify
mapping, deadzone, inversion, and profile selection interactively.

Usage::

    py scripts/gamepad_debug.py
    py scripts/gamepad_debug.py --controller-index 0
    py scripts/gamepad_debug.py --config configs/manual_gamepad.yaml

No AirSim connection is made and no flight commands are issued — this is a
pure input-inspection tool.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.gamepad_config import load_manual_gamepad_config
from flight_modes.gamepad_reader import GamepadReader
from flight_modes.manual_gamepad_mode import ManualGamepadController


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect gamepad input (no AirSim).")
    p.add_argument("--controller-index", type=int, default=0)
    p.add_argument("--config", default=None, help="Optional manual_gamepad.yaml path.")
    p.add_argument("--hz", type=float, default=20.0, help="Refresh rate.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_manual_gamepad_config(args.config)
    reader = GamepadReader(args.controller_index if args.controller_index != 0 else config.controller_index)
    controller = ManualGamepadController(config)

    reader.start()
    period = 1.0 / max(1.0, args.hz)

    print("\n  GAMEPAD DEBUG — press Ctrl+C to exit\n")
    try:
        while True:
            now = time.monotonic()
            state = reader.poll(now)
            cmd = controller.update(
                state, now, airborne=True, takeoff_called=True, landed=False
            )

            if not state.connected:
                print("\r  [waiting for controller …]        ", end="", flush=True)
            else:
                line = (
                    f"\r  LX={state.left_x:+.2f} LY={state.left_y:+.2f} "
                    f"RX={state.right_x:+.2f} RY={state.right_y:+.2f} "
                    f"LT={state.left_trigger:.2f} RT={state.right_trigger:.2f} "
                    f"DPAD=({state.dpad_x:+d},{state.dpad_y:+d}) "
                    f"prof={cmd.speed_profile} "
                    f"vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} vz={cmd.vz:+.2f} "
                    f"yaw={cmd.yaw_rate_radps * 57.2958:+.1f}dps "
                    f"A={int(state.button_a)} Y={int(state.button_y)} "
                    f"LB={int(state.lb)} RB={int(state.rb)} "
                    f"START={int(state.start)} BACK={int(state.back)}"
                )
                print(line, end="", flush=True)

            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  done.\n")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

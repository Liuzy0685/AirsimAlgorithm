"""
Manual flight mode — keyboard-controlled drone operation.

Uses Windows ``msvcrt`` for non-blocking keyboard input.

Control modes:
- ``velocity`` (default): ``send_velocity_body_frd()`` — safe speed control.
  AirSim flight controller computes Roll/Pitch automatically.
- ``attitude``: ``moveByRollPitchYawrateZAsync()`` — direct angle control.
  Strictly limited: max_roll_rad=radians(5), max_pitch_rad=radians(5).

Flow:
    Session starts at INITIALIZED (connected, not airborne).
    T key → session.takeoff_and_climb() — once only.
    G key or Esc → safe land and exit.
    Movement keys (WASD/RF/QE) are IGNORED before takeoff.

Key bindings:
    T     : takeoff (once)
    G     : safe land & exit
    Esc   : safe land & exit
    W / S : forward / backward (vx ±)
    A / D : left / right (vy ∓ / ± in velocity; roll ∓ / ± in attitude)
    R / F : climb / descend
    Q / E : yaw left / right
    Space : hover

Attitude mode directions (corrected):
    W → +pitch (nose down)
    S → -pitch (nose up)
    A → -roll (roll left)
    D → +roll (roll right)
    Q → +yaw_rate (yaw left)
    E → -yaw_rate (yaw right)
    R / F → change target Z by ±0.5 m (NOT blocked by zero roll/pitch/yaw check)

Coordinate conventions (body FRD):
    vx > 0 : forward     vx < 0 : backward
    vy < 0 : left        vy > 0 : right
    vz < 0 : climb       vz > 0 : descend
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger("manual_mode")


class ManualControlType(Enum):
    VELOCITY = auto()
    ATTITUDE = auto()


@dataclass
class ManualModeParams:
    linear_speed_mps: float = 0.5
    vertical_speed_mps: float = 0.3
    yaw_rate_radps: float = 0.5
    command_duration_s: float = 0.2
    loop_interval_s: float = 0.05   # control-loop period (must be ≪ command_duration_s)
    max_roll_rad: float = math.radians(5)
    max_pitch_rad: float = math.radians(5)


class ManualMode:
    """Keyboard-controlled manual flight.

    Connects via session but does NOT auto-takeoff.
    User must press T to take off (once).  G/Esc to land.
    Movement keys ignored before takeoff.
    """

    def __init__(
        self,
        session: Any,
        control_type: ManualControlType = ManualControlType.VELOCITY,
        params: Optional[ManualModeParams] = None,
    ) -> None:
        self._session = session
        self._control_type = control_type
        self._params = params or ManualModeParams()
        self._client = session.client
        self._vn = session.vehicle_name
        self._running = False
        self._hover_sent = False
        self._hover_timer = 0.0

        from control.velocity_controller import VelocityController
        self._vc = VelocityController(
            session.adapter,
            max_horizontal_speed_mps=self._params.linear_speed_mps,
            max_vertical_speed_mps=self._params.vertical_speed_mps,
            max_yaw_rate_radps=self._params.yaw_rate_radps,
            command_duration_seconds=self._params.command_duration_s,
        )

    # ── public API ──

    def run(self) -> None:
        """Start the manual control loop. Blocks until Esc/G or Ctrl+C."""
        self._running = True
        logger.info("Manual mode running — control_type=%s", self._control_type.name)
        self._print_controls()

        try:
            while self._running:
                keys = self._read_keys()

                if "esc" in keys or "g" in keys:
                    logger.info("Land/exit key pressed — safe landing.")
                    break

                if "t" in keys:
                    self._handle_takeoff()
                    time.sleep(self._params.loop_interval_s)
                    continue

                # If not airborne, idle — no control calls, just wait for input
                if not self._session.is_airborne:
                    time.sleep(self._params.loop_interval_s)
                    continue

                if not keys or " " in keys:
                    self._send_hover()
                    time.sleep(self._params.loop_interval_s)
                    continue

                if self._control_type == ManualControlType.VELOCITY:
                    self._handle_velocity_keys(keys)
                else:
                    self._handle_attitude_keys(keys)

                time.sleep(self._params.loop_interval_s)

        except KeyboardInterrupt:
            logger.info("Ctrl+C received.")
        finally:
            self._running = False
            logger.info("Manual mode ended.")

    def stop(self) -> None:
        self._running = False

    # ── takeoff ──

    def _handle_takeoff(self) -> None:
        """T key: call session.takeoff_and_climb() — once only."""
        if self._session.takeoff_called:
            logger.info("Takeoff already completed — ignoring T.")
            return
        try:
            logger.info("Takeoff initiated via T key.")
            self._session.takeoff_and_climb()
            logger.info("Takeoff complete — airborne.")
        except Exception as e:
            logger.error("Takeoff failed: %s", e)

    # ── key reading ──

    @staticmethod
    def _read_keys() -> set:
        try:
            import msvcrt
        except ImportError:
            return set()
        pressed = set()
        while msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'\x1b':
                pressed.add("esc")
            elif ch == b'\x00' or ch == b'\xe0':
                try:
                    ch2 = msvcrt.getch()
                except Exception:
                    continue
                mapping = {b'H': "up", b'P': "down", b'K': "left", b'M': "right"}
                pressed.add(mapping.get(ch2, ""))
            else:
                try: key = ch.decode("utf-8").lower()
                except UnicodeDecodeError: key = ch.decode("latin-1").lower()
                pressed.add(key)
        return pressed

    # ── key dispatch (public for testability) ──

    def _handle_keys(self, keys: set) -> None:
        """Dispatch key presses to the appropriate handler.
        Public so tests can call it directly without the run loop."""
        if not self._session.is_airborne:
            return  # idle — no control calls before takeoff
        if not keys or " " in keys:
            self._send_hover()
            return
        if self._control_type == ManualControlType.VELOCITY:
            self._handle_velocity_keys(keys)
        else:
            self._handle_attitude_keys(keys)

    # ── velocity keys ──

    def _handle_velocity_keys(self, keys: set) -> None:
        vx = vy = vz = 0.0
        yaw_rate = 0.0
        spd = self._params.linear_speed_mps
        vspd = self._params.vertical_speed_mps
        yr = self._params.yaw_rate_radps

        if "w" in keys: vx += spd
        if "s" in keys: vx -= spd
        if "a" in keys: vy -= spd
        if "d" in keys: vy += spd
        if "r" in keys: vz -= vspd
        if "f" in keys: vz += vspd
        if "q" in keys: yaw_rate += yr
        if "e" in keys: yaw_rate -= yr

        if vx == 0 and vy == 0 and vz == 0 and yaw_rate == 0:
            self._send_hover()
        else:
            self._vc.send_velocity_body_frd(
                vx=vx, vy=vy, vz=vz,
                duration=self._params.command_duration_s,
                vehicle_name=self._vn,
                yaw_rate=yaw_rate if yaw_rate != 0 else None,
            )
            self._hover_sent = False

    # ── attitude keys ──

    def _handle_attitude_keys(self, keys: set) -> None:
        """Attitude mode via moveByRollPitchYawrateZAsync.

        Directions (corrected):
            W → +pitch  (nose down)
            S → -pitch  (nose up)
            A → -roll   (roll left)
            D → +roll   (roll right)
            Q → +yaw_rate
            E → -yaw_rate
            R → target_Z -= 0.5 (climb)
            F → target_Z += 0.5 (descend)

        R/F modify target_z regardless of roll/pitch/yaw values.
        Zero roll/pitch/yaw BUT R/F pressed → still sends command.
        """
        roll = pitch = 0.0
        yaw_rate = 0.0
        target_z = self._session.target_z_ned
        z_changed = False

        max_r = self._params.max_roll_rad
        max_p = self._params.max_pitch_rad
        yr = self._params.yaw_rate_radps

        if "a" in keys: roll = -max_r
        if "d" in keys: roll = max_r
        if "w" in keys: pitch = max_p   # +pitch = nose down
        if "s" in keys: pitch = -max_p  # -pitch = nose up
        if "q" in keys: yaw_rate += yr
        if "e" in keys: yaw_rate -= yr
        if "r" in keys: target_z -= 0.5; z_changed = True
        if "f" in keys: target_z += 0.5; z_changed = True
        # Write accumulated altitude back to session for next cycle
        if z_changed:
            self._session.target_z_ned = target_z

        if roll == 0 and pitch == 0 and yaw_rate == 0 and not z_changed:
            self._send_hover()
            return

        dur = self._params.command_duration_s

        self._client.moveByRollPitchYawrateZAsync(
            roll=roll,
            pitch=pitch,
            yaw_rate=yaw_rate,
            z=target_z,
            duration=dur,
            vehicle_name=self._vn,
        )
        self._hover_sent = False

    # ── hover ──

    def _send_hover(self) -> None:
        """Send hover via hoverAsync().join().  Deduplicates — won't re-send
        if already hovering and no movement command was issued in between."""
        if self._hover_sent:
            return
        try:
            self._client.hoverAsync(vehicle_name=self._vn).join()
            self._hover_sent = True
        except Exception as e:
            logger.warning("hoverAsync failed: %s", e)

    # ── display ──

    @staticmethod
    def _print_controls() -> None:
        print("\n" + "=" * 48)
        print("  MANUAL FLIGHT CONTROLS")
        print("=" * 48)
        print("  T       : Takeoff (once)")
        print("  G / Esc : Safe land & exit")
        print("  W / S   : Forward / Backward")
        print("  A / D   : Left / Right")
        print("  R / F   : Climb / Descend")
        print("  Q / E   : Yaw left / right")
        print("  Space   : Hover (stop)")
        print("=" * 48)
        print("  Movement keys IGNORED before takeoff.")
        print("=" * 48 + "\n")

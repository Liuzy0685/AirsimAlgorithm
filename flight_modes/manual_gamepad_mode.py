"""Gamepad manual flight mode — full replacement for keyboard manual control.

Architecture (strictly decoupled)
---------------------------------
    GamepadReader (pygame/SDL)
          │  GamepadState (normalised, scale-free)
          ▼
    ManualGamepadController (pure mapping — no session, no AirSim)
          │  ManualFlightCommand (intent)
          ▼
    ManualGamepadMode (dispatcher → VelocityController → AirSim)

``ManualGamepadController`` is a pure function of ``(GamepadState, time, flags)``
→ ``ManualFlightCommand`` and contains **no** I/O, so the entire mapping is
unit-testable without pygame, AirSim, or a flight session.

Mode-2 layout (default)
-----------------------
    Left stick  Y  : throttle (up = climb, down = descend)
    Left stick  X  : yaw (right = yaw right, left = yaw left)
    Right stick Y  : pitch  (up = forward, down = backward)
    Right stick X  : roll   (right = strafe right, left = strafe left)
    LT / RT        : fine yaw (LT = yaw left, RT = yaw right; overrides LS X)
    D-pad          : trim (up/down = climb/descend, left/right = yaw)

Buttons
-------
    A      : takeoff (once)
    Y      : safe land & exit
    START  : arm  (long-press)
    BACK   : disarm (long-press, implemented as safe land + disarm)
    LB     : SLOW profile (+ dead-man if enabled)
    RB     : FAST profile
    LB+RB  : NORMAL profile

Safety
------
    - Disconnect → safe hover (P0).
    - Dead-man button (optional) released → safe hover.
    - Collision guard → safe hover when within ``emergency_distance_m``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional, Tuple

from flight_modes.gamepad_config import (
    ManualGamepadConfig,
    normalize_gamepad_axis,
)
from flight_modes.gamepad_reader import GamepadReader
from flight_modes.gamepad_state import (
    GamepadState,
    ManualFlightCommand,
    SpeedProfile,
)

logger = logging.getLogger("manual_gamepad_mode")

# String button names → GamepadState attribute names.
_BUTTON_ATTR = {
    "A": "button_a",
    "B": "button_b",
    "X": "button_x",
    "Y": "button_y",
    "LB": "lb",
    "RB": "rb",
    "START": "start",
    "BACK": "back",
    "L3": "l3",
    "R3": "r3",
}


class ManualGamepadController:
    """Pure gamepad → flight-command mapping (no I/O, no session).

    Parameters
    ----------
    config:
        :class:`ManualGamepadConfig`.  Defaults are used when ``None``.
    """

    def __init__(self, config: Optional[ManualGamepadConfig] = None) -> None:
        self._cfg = config or ManualGamepadConfig()

        # Edge detection state (previous button levels).
        self._prev_a = False
        self._prev_y = False
        self._prev_start = False
        self._prev_back = False

        # Long-press tracking.
        self._start_hold_since: Optional[float] = None
        self._back_hold_since: Optional[float] = None
        self._start_fired = False
        self._back_fired = False

    # ── main entry ──

    def update(
        self,
        state: GamepadState,
        now: float,
        *,
        airborne: bool,
        takeoff_called: bool,
        landed: bool,
    ) -> ManualFlightCommand:
        """Map a polled gamepad state to a flight command.

        ``airborne`` is the session phase flag; ``takeoff_called`` prevents a
        second takeoff; ``landed`` is the read-back ``landed_state == Landed``.
        """
        # Button edges.
        a = state.button_a
        y = state.button_y
        start = state.start
        back = state.back

        a_edge = a and not self._prev_a
        y_edge = y and not self._prev_y

        # Long-press edge tracking.
        start_long = self._long_press(
            start, now, self._cfg.arm_hold_s, "_start_hold_since", "_start_fired"
        )
        back_long = self._long_press(
            back, now, self._cfg.disarm_hold_s, "_back_hold_since", "_back_fired"
        )

        self._prev_a = a
        self._prev_y = y
        self._prev_start = start
        self._prev_back = back

        # P0 — disconnect → safe hover.
        if not state.connected:
            return ManualFlightCommand(hover=True, input_ok=False, reason="disconnected")

        # Dead-man safety.
        if self._cfg.require_deadman_button and not self._button(state, self._cfg.deadman_button):
            return ManualFlightCommand(hover=True, input_ok=False, reason="deadman_not_held")

        profile = self._select_profile(state)

        if airborne:
            if y_edge:
                return ManualFlightCommand(land=True, input_ok=True, reason="land_requested")
            if back_long:
                return ManualFlightCommand(disarm=True, input_ok=True, reason="disarm_requested")

            cmd = ManualFlightCommand(speed_profile=profile.name)
            self._fill_motion(cmd, state, profile)
            return cmd

        # ── on the ground ──
        if a_edge and not takeoff_called:
            return ManualFlightCommand(takeoff=True, input_ok=True, reason="takeoff_requested")
        if start_long:
            return ManualFlightCommand(arm=True, input_ok=True, reason="arm_requested")
        return ManualFlightCommand(hover=True, input_ok=True, reason="ground_idle")

    # ── long-press helper ──

    def _long_press(self, held: bool, now: float, hold_s: float,
                    hold_attr: str, fired_attr: str) -> bool:
        if not held:
            setattr(self, hold_attr, None)
            setattr(self, fired_attr, False)
            return False

        hold_since = getattr(self, hold_attr)
        if hold_since is None:
            hold_since = now
            setattr(self, hold_attr, hold_since)
        if getattr(self, fired_attr):
            return False
        if now - hold_since >= hold_s:
            setattr(self, fired_attr, True)
            return True
        return False

    # ── profile ──

    def _select_profile(self, state: GamepadState) -> SpeedProfile:
        lb = self._button(state, self._cfg.slow_button)
        rb = self._button(state, self._cfg.fast_button)
        if lb and rb:
            return SpeedProfile.NORMAL
        if rb:
            return SpeedProfile.FAST
        if lb:
            return SpeedProfile.SLOW
        return SpeedProfile.NORMAL

    def _profile_speeds(self, profile: SpeedProfile) -> Tuple[float, float, float]:
        if profile == SpeedProfile.FAST:
            return (
                self._cfg.fast_horizontal_speed_mps,
                self._cfg.fast_vertical_speed_mps,
                self._cfg.fast_yaw_rate_dps,
            )
        if profile == SpeedProfile.SLOW:
            return (
                self._cfg.slow_horizontal_speed_mps,
                self._cfg.slow_vertical_speed_mps,
                self._cfg.slow_yaw_rate_dps,
            )
        return (
            self._cfg.normal_horizontal_speed_mps,
            self._cfg.normal_vertical_speed_mps,
            self._cfg.normal_yaw_rate_dps,
        )

    # ── motion mapping ──

    def _fill_motion(self, cmd: ManualFlightCommand, state: GamepadState,
                     profile: SpeedProfile) -> None:
        cfg = self._cfg
        h, v, yaw_dps = self._profile_speeds(profile)

        lx = normalize_gamepad_axis(state.left_x, cfg.deadzone, cfg.yaw_expo, cfg.invert_left_x)
        ly = normalize_gamepad_axis(state.left_y, cfg.deadzone, cfg.expo, cfg.invert_left_y)
        rx = normalize_gamepad_axis(state.right_x, cfg.deadzone, cfg.expo, cfg.invert_right_x)
        ry = normalize_gamepad_axis(state.right_y, cfg.deadzone, cfg.expo, cfg.invert_right_y)

        cmd.vx = ry * h            # RS up → forward
        cmd.vy = rx * h            # RS right → strafe right
        cmd.vz = -ly * v           # LS up → climb (negative NED Z)

        # Yaw: trigger priority (fine control), else left-stick X.
        trigger_delta = state.right_trigger - state.left_trigger
        if abs(trigger_delta) > cfg.trigger_deadzone:
            cmd.yaw_rate_radps = math.radians(trigger_delta * cfg.trigger_yaw_rate_dps)
        else:
            cmd.yaw_rate_radps = math.radians(lx * yaw_dps)

        # D-pad trim.
        cmd.vz += state.dpad_y * cfg.trim_vertical_speed_mps
        cmd.yaw_rate_radps += math.radians(state.dpad_x * cfg.trim_yaw_rate_dps)

    @staticmethod
    def _button(state: GamepadState, name: str) -> bool:
        return bool(getattr(state, _BUTTON_ATTR[name], False))


class ManualGamepadMode:
    """Session-bound gamepad flight loop (the AirSim dispatcher).

    Owns the ``GamepadReader``, ``ManualGamepadController``, and a
    ``VelocityController`` configured with the FAST-profile speed caps so the
    yaw-rate clamp never bites below the requested fast yaw rate.
    """

    def __init__(
        self,
        session: Any,
        config: Optional[ManualGamepadConfig] = None,
        reader: Optional[GamepadReader] = None,
    ) -> None:
        self._session = session
        self._config = config or ManualGamepadConfig()
        self._controller = ManualGamepadController(self._config)
        self._reader = reader or GamepadReader(self._config.controller_index)

        self._client = session.client
        self._vn = session.vehicle_name
        self._adapter = session.adapter
        self._running = False
        self._hover_sent = False

        self._last_collision_time: float = -float("inf")
        self._collision_cache: Optional[Tuple[float, bool, str]] = None

        from control.velocity_controller import VelocityController
        self._vc = VelocityController(
            session.adapter,
            max_horizontal_speed_mps=self._config.max_horizontal_speed_mps,
            max_vertical_speed_mps=self._config.max_vertical_speed_mps,
            max_yaw_rate_radps=self._config.max_yaw_rate_radps,
            command_duration_seconds=self._config.command_duration_s,
        )

    # ── public API ──

    def run(self) -> None:
        """Run the gamepad control loop.  Blocks until land/disarm or Ctrl+C."""
        self._running = True
        self._reader.start()

        poll_period = 1.0 / max(1.0, self._config.poll_hz)
        command_period = 1.0 / max(1.0, self._config.command_hz)
        hud_period = 1.0 / max(0.1, self._config.hud_hz)
        last_command = -float("inf")
        last_hud = -float("inf")

        self._print_controls()

        try:
            while self._running:
                now = time.monotonic()
                state = self._reader.poll(now)
                landed = self._read_landed()
                airborne = bool(self._session.is_airborne)
                takeoff_called = bool(self._session.takeoff_called)

                cmd = self._controller.update(
                    state, now,
                    airborne=airborne,
                    takeoff_called=takeoff_called,
                    landed=landed,
                )

                # Discrete actions first (takeoff/arm/land/disarm).
                if self._dispatch_discrete(cmd):
                    if not self._running:
                        break
                    continue

                # Collision guard may override continuous motion → hover.
                cmd = self._apply_collision_guard(cmd)

                if now - last_command >= command_period:
                    self._dispatch_continuous(cmd)
                    last_command = now

                if now - last_hud >= hud_period:
                    self._render_hud(cmd)
                    last_hud = now

                time.sleep(poll_period)

        except KeyboardInterrupt:
            logger.info("Ctrl+C received.")
        finally:
            self._reader.close()
            self._running = False
            logger.info("Manual gamepad mode ended.")

    def stop(self) -> None:
        self._running = False

    # ── discrete dispatch ──

    def _dispatch_discrete(self, cmd: ManualFlightCommand) -> bool:
        """Dispatch takeoff/arm/land/disarm.  Returns True if handled."""
        if cmd.takeoff:
            if self._session.takeoff_called:
                logger.info("Takeoff already completed — ignoring A.")
                return True
            try:
                logger.info("Takeoff requested via A button.")
                self._session.takeoff_and_climb()
                self._hover_sent = False
            except Exception as e:
                logger.error("Takeoff failed: %s", e)
            return True

        if cmd.arm:
            try:
                self._vc.enable_api_control(self._vn)
                self._vc.arm(self._vn)
                logger.info("Arm requested via START long-press.")
            except Exception as e:
                logger.error("Arm failed: %s", e)
            return True

        if cmd.land:
            logger.info("Land requested via Y button — safe landing.")
            try:
                self._session.land_and_disarm()
            except Exception as e:
                logger.error("Land failed: %s", e)
            self._running = False
            return True

        if cmd.disarm:
            logger.info("Disarm requested via BACK long-press — safe land + disarm.")
            try:
                self._session.land_and_disarm()
            except Exception as e:
                logger.error("Disarm failed: %s", e)
            self._running = False
            return True

        return False

    # ── continuous dispatch ──

    def _dispatch_continuous(self, cmd: ManualFlightCommand) -> None:
        if not self._session.is_airborne:
            return
        if (not cmd.input_ok) or cmd.hover or cmd.is_zero_motion:
            self._send_hover()
            return
        try:
            self._vc.send_velocity_body_frd(
                vx=cmd.vx,
                vy=cmd.vy,
                vz=cmd.vz,
                duration=self._config.command_duration_s,
                vehicle_name=self._vn,
                yaw_rate=cmd.yaw_rate_radps if cmd.yaw_rate_radps != 0.0 else None,
            )
            self._hover_sent = False
        except Exception as e:
            logger.warning("Velocity command failed: %s", e)

    def _send_hover(self) -> None:
        if self._hover_sent:
            return
        try:
            self._vc.hover(self._vn)
            self._hover_sent = True
        except Exception as e:
            logger.warning("Hover failed: %s", e)

    # ── collision safety ──

    def _read_collision(self) -> Tuple[float, bool, str]:
        """Read collision info + nearest LiDAR point, cached at ~5 Hz."""
        now = time.monotonic()
        if self._collision_cache is not None and now - self._last_collision_time < 0.2:
            return self._collision_cache
        self._last_collision_time = now

        min_dist = float("inf")
        collided = False
        obj = ""
        try:
            col = self._client.simGetCollisionInfo(vehicle_name=self._vn)
            collided = bool(col.has_collided)
            obj = str(col.object_name) if col.object_name else ""
        except Exception as e:
            logger.debug("collision_info_read_failed: %s", e)

        try:
            lidar_name = self._adapter.lidar_name
            lidar = self._client.getLidarData(lidar_name=lidar_name, vehicle_name=self._vn)
            pts = lidar.point_cloud
            if pts is not None and len(pts) >= 3:
                import numpy as np
                arr = np.asarray(pts, dtype="float32").reshape(-1, 3)
                if arr.shape[0] > 0:
                    sq = arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2
                    min_dist = float(np.sqrt(sq.min()))
        except Exception as e:
            logger.debug("lidar_read_failed: %s", e)

        self._collision_cache = (min_dist, collided, obj)
        return self._collision_cache

    def _collision_brake_required(self, min_dist: float, collided: bool, obj: str) -> bool:
        if not self._config.collision_guard:
            return False
        if collided:
            return True
        if math.isfinite(min_dist) and min_dist < self._config.emergency_distance_m:
            return True
        return False

    def _apply_collision_guard(self, cmd: ManualFlightCommand) -> ManualFlightCommand:
        if not self._config.collision_guard:
            return cmd
        min_dist, collided, obj = self._read_collision()
        if self._collision_brake_required(min_dist, collided, obj):
            if cmd.input_ok and not cmd.hover and not cmd.is_zero_motion:
                logger.warning(
                    "collision_guard_active  min_dist=%.2f  collided=%s  obj=%r",
                    min_dist, collided, obj,
                )
            return ManualFlightCommand(hover=True, input_ok=False, reason="collision_guard")
        return cmd

    # ── state read-back ──

    def _read_landed(self) -> bool:
        try:
            st = self._client.getMultirotorState(vehicle_name=self._vn)
            return int(st.landed_state) == 0
        except Exception:
            return False

    # ── HUD ──

    def _render_hud(self, cmd: ManualFlightCommand) -> None:
        try:
            status = "OK" if cmd.input_ok else f"SAFE:{cmd.reason}"
            msg = (
                f"[GAMEPAD] profile={cmd.speed_profile} "
                f"vx={cmd.vx:.2f} vy={cmd.vy:.2f} vz={cmd.vz:.2f} "
                f"yaw={math.degrees(cmd.yaw_rate_radps):.1f}dps "
                f"{status}"
            )
            self._client.simPrintLogMessage(msg, message_param="", severity=0)
        except Exception as e:
            logger.debug("HUD render failed: %s", e)

    @staticmethod
    def _print_controls() -> None:
        print("\n" + "=" * 52)
        print("  GAMEPAD MANUAL FLIGHT (Mode-2 / Xbox)")
        print("=" * 52)
        print("  LS up/down   : climb / descend")
        print("  LS left/right: yaw left / right")
        print("  RS up/down   : forward / backward")
        print("  RS left/right: strafe left / right")
        print("  LT / RT      : fine yaw (left / right)")
        print("  D-pad        : trim (vertical + yaw)")
        print("  A            : takeoff (once)")
        print("  Y            : safe land & exit")
        print("  START (hold) : arm")
        print("  BACK (hold)  : disarm (safe land)")
        print("  LB           : SLOW   |  RB : FAST  |  LB+RB : NORMAL")
        print("=" * 52 + "\n")

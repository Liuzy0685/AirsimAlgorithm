"""Gamepad manual flight — pygame/SDL joystick reader.

Reads an Xbox-style (XInput) controller through the ``pygame`` joystick
subsystem and emits :class:`~flight_modes.gamepad_state.GamepadState`
snapshots.  Xbox axis/button/hat indices are the single source of truth:

Axes
----
    0 = left stick X     (LSX)    -1 left  … +1 right
    1 = left stick Y     (LSY)    -1 up    … +1 down
    2 = right stick X    (RSX)    -1 left  … +1 right
    3 = right stick Y    (RSY)    -1 up    … +1 down
    4 = left trigger     (LT)     0 released … +1 pressed
    5 = right trigger    (RT)     0 released … +1 pressed

Buttons
-------
    0 = A   1 = B   2 = X   3 = Y   4 = LB   5 = RB
    6 = Back   7 = Start   8 = L3   9 = R3

Hat (D-pad)
-----------
    hat 0 → (x, y) each in ``{-1, 0, +1}``.

The module imports ``pygame`` **lazily** and calls ``pygame.init()`` without
creating any display window.  On Windows, Xbox-style controllers are served
by SDL's HIDAPI/XInput backends, which refresh axis/button state directly
through ``pygame.event.pump()`` — no window is required.  Two earlier
approaches were verified to FAIL on an Xbox 360 controller:

  * ``SDL_VIDEODRIVER=dummy`` — the dummy video driver exposes no input
    devices, so every axis reads 0 and every button reads False.
  * A ``pygame.HIDDEN`` 1×1 window — hidden windows do not receive Windows
    input messages, so joystick state still doesn't refresh.

If pygame is not installed, or no controller is attached, ``poll()`` returns
a disconnected state — it never raises.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from flight_modes.gamepad_state import GamepadState

logger = logging.getLogger("gamepad_reader")

# Xbox (XInput) mapping — single source of truth.
_AXIS_LSX = 0
_AXIS_LSY = 1
_AXIS_RSX = 2
_AXIS_RSY = 3
_AXIS_LT = 4
_AXIS_RT = 5

_BTN_A = 0
_BTN_B = 1
_BTN_X = 2
_BTN_Y = 3
_BTN_LB = 4
_BTN_RB = 5
_BTN_BACK = 6
_BTN_START = 7
_BTN_L3 = 8
_BTN_R3 = 9

_HAT_DPAD = 0

# Well-known Xbox controller name fragments for auto-detection confidence.
_KNOWN_NAME_FRAGMENTS = (
    "xbox",
    "xinput",
    "microsoft",
    "gamepad",
    "controller",
)


class GamepadReader:
    """Polls an Xbox controller via pygame and returns GamepadState snapshots.

    Parameters
    ----------
    controller_index:
        Index into ``pygame.joystick`` device enumeration.  Default 0.
    """

    def __init__(self, controller_index: int = 0) -> None:
        self._controller_index = int(controller_index)
        self._pygame: Optional[object] = None
        self._joy: Optional[object] = None
        self._name: str = ""
        self._attached: bool = False
        self._last_known: bool = False
        self._warned_connected: bool = False

    # ── lifecycle ──

    def start(self) -> None:
        """Initialise pygame and attempt to attach to the controller.

        Never raises — if pygame is unavailable or no controller is present,
        the reader stays in a disconnected state and ``poll()`` reports it.
        """
        self._ensure_pygame()
        if self._pygame is None:
            logger.warning("gamepad_backend_unavailable  backend=pygame")
            return
        self._pump_events()
        self._try_attach()

    def close(self) -> None:
        """Release the joystick, display, and quit pygame cleanly."""
        if self._joy is not None:
            try:
                self._joy.quit()
            except Exception:
                pass
            self._joy = None
        self._attached = False
        pygame = self._pygame
        if pygame is not None:
            try:
                pygame.joystick.quit()
            except Exception:
                pass
            try:
                pygame.display.quit()
            except Exception:
                pass
        self._pygame = None
        self._name = ""

    # ── pygame resolution (lazy, headless-safe) ──

    def _ensure_pygame(self) -> None:
        if self._pygame is not None:
            return
        try:
            import pygame  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "gamepad_pygame_missing  pygame is not installed — "
                "gamepad manual mode will report 'disconnected'."
            )
            self._pygame = None
            return

        # IMPORTANT: do NOT set SDL_VIDEODRIVER=dummy, and do NOT create a
        # window (hidden or otherwise).  Both break gamepad reading on Windows:
        #
        #   * SDL_VIDEODRIVER=dummy — the dummy video driver exposes no input
        #     devices, so every axis reads 0 and every button False.
        #   * A pygame.HIDDEN window — hidden windows do not receive Windows
        #     input messages, so joystick state still doesn't refresh.
        #
        # Xbox-style controllers are served by SDL's HIDAPI/XInput backends,
        # which refresh axis/button state directly via pygame.event.pump() —
        # no window is required.  Verified empirically with an Xbox 360
        # controller: no-window reads full axis range + all buttons; HIDDEN
        # window reads nothing.
        try:
            pygame.init()
            pygame.joystick.init()
            self._pygame = pygame
        except Exception as exc:
            logger.warning("gamepad_pygame_init_failed: %s", exc)
            self._pygame = None

    # ── attach / event pump ──

    def _try_attach(self) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        try:
            count = pygame.joystick.get_count()
        except Exception:
            count = 0

        if count == 0:
            self._attached = False
            self._joy = None
            return

        if self._controller_index >= count:
            logger.warning(
                "gamepad_index_out_of_range  index=%d  available=%d",
                self._controller_index, count,
            )
            self._attached = False
            self._joy = None
            return

        try:
            joy = pygame.joystick.Joystick(self._controller_index)
            joy.init()
            self._joy = joy
            self._attached = True
            self._name = str(joy.get_name())
        except Exception as exc:
            logger.warning("gamepad_attach_failed: %s", exc)
            self._attached = False
            self._joy = None
            self._name = ""

    def _pump_events(self) -> None:
        """Process the SDL event queue (hotplug + button/axis state sync).

        ``pygame.event.pump()`` is called explicitly first because it is the
        call that actually refreshes joystick axis/button state on Windows —
        ``event.get()`` pumps too, but only when events are present, and an
        idle controller produces no events.
        """
        pygame = self._pygame
        if pygame is None:
            return
        try:
            pygame.event.pump()
            for event in pygame.event.get():
                etype = getattr(event, "type", None)
                if etype == getattr(pygame, "JOYDEVICEADDED", None):
                    self._try_attach()
                elif etype == getattr(pygame, "JOYDEVICEREMOVED", None):
                    if self._attached:
                        logger.info("GAMEPAD_DISCONNECTED action=hover")
                    self._attached = False
                    self._joy = None
                    self._name = ""
        except Exception:
            pass

    # ── polling ──

    def poll(self, now: Optional[float] = None) -> GamepadState:
        """Return the current gamepad state (normalised, scale-free).

        A disconnected controller returns ``GamepadState(connected=False)``.
        """
        if now is None:
            now = time.monotonic()

        if self._pygame is None:
            return GamepadState(connected=False, timestamp=now)

        self._pump_events()
        if not self._attached or self._joy is None:
            # Re-attempt attach on every poll (handles late/again-plugged-in).
            self._try_attach()
            if not self._attached or self._joy is None:
                if self._warned_connected:
                    self._warned_connected = False
                return GamepadState(connected=False, timestamp=now)

        if not self._warned_connected:
            self._warned_connected = True
            logger.info("gamepad_connected name=%r index=%d", self._name, self._controller_index)

        return self._read_state(now)

    def _read_state(self, now: float) -> GamepadState:
        pygame = self._pygame
        joy = self._joy

        def axis(idx: int) -> float:
            try:
                return float(joy.get_axis(idx))
            except Exception:
                return 0.0

        def button(idx: int) -> bool:
            try:
                return bool(joy.get_button(idx))
            except Exception:
                return False

        # Xbox triggers report on axes 4/5 with raw range [-1, +1].
        left_trigger = self._trigger(axis(_AXIS_LT))
        right_trigger = self._trigger(axis(_AXIS_RT))

        dpad_x, dpad_y = self._hat()

        return GamepadState(
            left_x=axis(_AXIS_LSX),
            left_y=axis(_AXIS_LSY),
            right_x=axis(_AXIS_RSX),
            right_y=axis(_AXIS_RSY),
            left_trigger=left_trigger,
            right_trigger=right_trigger,
            button_a=button(_BTN_A),
            button_b=button(_BTN_B),
            button_x=button(_BTN_X),
            button_y=button(_BTN_Y),
            lb=button(_BTN_LB),
            rb=button(_BTN_RB),
            start=button(_BTN_START),
            back=button(_BTN_BACK),
            l3=button(_BTN_L3),
            r3=button(_BTN_R3),
            dpad_x=dpad_x,
            dpad_y=dpad_y,
            connected=True,
            timestamp=now,
            name=self._name,
            known_mapping=self._is_known(self._name),
        )

    @staticmethod
    def _trigger(raw: float) -> float:
        """Map an Xbox trigger axis raw ``[-1, +1]`` → ``[0, +1]``.

        Xbox triggers report -1 at rest and +1 fully pressed (SDL's
        ``SDL_CONTROLLER_AXIS_TRIGGERLEFT`` is ``[0, 32767]``, but pygame's
        generic joystick interface exposes the raw ``[-1, +1]`` axis, so we
        remap with ``(raw + 1) / 2``).
        """
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def _hat(self):
        joy = self._joy
        try:
            n = joy.get_numhats()
            if n <= _HAT_DPAD:
                return 0, 0
            x, y = joy.get_hat(_HAT_DPAD)
            return int(x), int(y)
        except Exception:
            return 0, 0

    @staticmethod
    def _is_known(name: str) -> bool:
        """Heuristic: is this a recognised Xbox-style controller name?"""
        lowered = (name or "").lower()
        return any(frag in lowered for frag in _KNOWN_NAME_FRAGMENTS)

"""Gamepad manual flight — pure state/command data model.

This module is intentionally **decoupled** from any backend (pygame/SDL),
any AirSim connection, and any flight session.  It holds only plain
dataclasses so the mapping layer (``manual_gamepad_mode.ManualGamepadController``)
and the reader (``gamepad_reader.GamepadReader``) can be unit-tested in
isolation with no external dependencies.

Coordinate conventions
----------------------
The Xbox axis values are normalised to the range ``[-1, +1]`` **before** they
reach this module.  Triggers are normalised to ``[0, +1]`` (0 = released,
1 = fully pressed).

Yaw sign convention (authoritative for gamepad mode)
----------------------------------------------------
``yaw_rate_radps > 0`` = **yaw right** (clockwise viewed from above).  This
is the AirSim body-frame convention and matches the user-specified tests
(LS right → positive, LS left → negative).  It intentionally **differs** from
the legacy ``manual_mode.py`` keyboard mapping (Q → +yaw_rate), which is
scoped to keyboard mode only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SpeedProfile(Enum):
    """The three LB/RB speed profiles.

    - ``NORMAL`` — both LB and RB released, or both held.
    - ``SLOW``   — LB held (fine control).
    - ``FAST``   — RB held (high-speed transit).
    """

    NORMAL = auto()
    SLOW = auto()
    FAST = auto()


@dataclass
class GamepadState:
    """A single polled snapshot of the gamepad, normalised and scale-free.

    Axes are ``[-1, +1]`` (left_x/left_y/right_x/right_y), triggers are
    ``[0, +1]``, buttons are booleans, and the D-pad is a discrete ``-1/0/+1``
    integer pair.  ``connected`` is ``False`` when no controller is attached.
    """

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0  # 0 = released, 1 = fully pressed
    right_trigger: float = 0.0
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False
    lb: bool = False
    rb: bool = False
    start: bool = False
    back: bool = False
    l3: bool = False
    r3: bool = False
    dpad_x: int = 0   # -1 left, 0 centred, +1 right
    dpad_y: int = 0   # -1 up,   0 centred, +1 down
    connected: bool = False
    timestamp: float = 0.0
    name: str = ""
    known_mapping: bool = False


@dataclass
class ManualFlightCommand:
    """A single resolved flight command produced by the controller mapping.

    This is the *intent* — a body-frame velocity + yaw-rate command plus
    discrete actions — not yet an AirSim API call.  The dispatcher
    (``ManualGamepadMode``) translates it into a ``VelocityController`` call.

    Velocity convention (body FRD):
        ``vx`` forward / ``vy`` right / ``vz`` down (negative = climb).
    ``yaw_rate_radps`` follows the authoritative gamepad convention above.
    """

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate_radps: float = 0.0
    takeoff: bool = False
    land: bool = False
    arm: bool = False
    disarm: bool = False
    hover: bool = False
    speed_profile: str = SpeedProfile.NORMAL.name
    input_ok: bool = True
    reason: str = ""

    @property
    def is_zero_motion(self) -> bool:
        """True when the command requests no translational or yaw motion."""
        return (
            self.vx == 0.0
            and self.vy == 0.0
            and self.vz == 0.0
            and self.yaw_rate_radps == 0.0
        )

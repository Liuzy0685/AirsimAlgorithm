"""Gamepad manual flight — configuration model and YAML loader.

Defines :class:`ManualGamepadConfig`, a frozen dataclass holding every tunable
for the gamepad manual mode, plus ``load_manual_gamepad_config()`` which reads
a YAML file with **strict** validation: unknown keys are rejected and every
value is range-checked, so a typo in the config fails loudly at startup rather
than silently falling back to a default mid-flight.

The loader is pure Python (``yaml`` only) and independent of pygame and AirSim,
so it can be unit-tested without either dependency installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Optional

import yaml

# Valid Xbox button names accepted by config button selectors.
_VALID_BUTTONS = frozenset(
    {"A", "B", "X", "Y", "LB", "RB", "START", "BACK", "L3", "R3"}
)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive range ``[lo, hi]``."""
    return lo if value < lo else hi if value > hi else value


def apply_deadzone(x: float, deadzone: float) -> float:
    """Map a raw ``[-1, +1]`` axis through a symmetric deadzone.

    Values inside ``±deadzone`` collapse to 0; the surviving signal is
    re-scaled so that the output still reaches ±1 at full deflection.
    """
    if deadzone <= 0.0:
        return x
    if abs(x) <= deadzone:
        return 0.0
    return math.copysign((abs(x) - deadzone) / (1.0 - deadzone), x)


def apply_expo(x: float, expo: float) -> float:
    """Apply cubic-response expo curve to a ``[-1, +1]`` input.

    ``expo`` in ``[0, 1]`` blends between linear (0) and fully cubic (1).
    Preserves sign and endpoints (±1 maps to ±1).
    """
    if expo <= 0.0:
        return x
    e = clamp(expo, 0.0, 1.0)
    return (1.0 - e) * x + e * (x ** 3)


def normalize_gamepad_axis(
    raw: float,
    deadzone: float,
    expo: float,
    invert: bool,
) -> float:
    """Full axis pipeline: deadzone → expo → invert → clamp to ``[-1, +1]``."""
    x = apply_deadzone(float(raw), float(deadzone))
    x = apply_expo(x, float(expo))
    if invert:
        x = -x
    return clamp(x, -1.0, 1.0)


@dataclass(frozen=True)
class ManualGamepadConfig:
    """Immutable gamepad manual-mode configuration.

    All fields have safe defaults matching the user spec.  Build a custom
    instance with ``replace()`` or load one from YAML.  Speed units are m/s;
    yaw rates are degrees/s (matching the user-facing config); the controller
    converts them to rad/s at the API boundary.
    """

    # ── backend / timing ──
    backend: str = "pygame"
    controller_index: int = 0
    poll_hz: float = 50.0       # gamepad sampling rate
    command_hz: float = 20.0    # AirSim velocity command rate
    hud_hz: float = 3.0         # on-screen/log HUD refresh rate

    # ── axis shaping ──
    deadzone: float = 0.10
    expo: float = 0.30
    yaw_expo: float = 0.25
    trigger_deadzone: float = 0.05

    # ── speed profiles (m/s and deg/s) ──
    normal_horizontal_speed_mps: float = 1.0
    normal_vertical_speed_mps: float = 0.5
    normal_yaw_rate_dps: float = 45.0
    slow_horizontal_speed_mps: float = 0.4
    slow_vertical_speed_mps: float = 0.25
    slow_yaw_rate_dps: float = 20.0
    fast_horizontal_speed_mps: float = 2.0
    fast_vertical_speed_mps: float = 1.0
    fast_yaw_rate_dps: float = 90.0
    trigger_yaw_rate_dps: float = 35.0

    # ── axis inversion ──
    invert_left_x: bool = False
    invert_left_y: bool = True    # LS up = negative NED Z = climb
    invert_right_x: bool = False
    invert_right_y: bool = True   # RS up = forward

    # ── safety ──
    collision_guard: bool = True
    emergency_distance_m: float = 0.8
    input_timeout_s: float = 0.30   # max gap before "disconnected/timeout"
    require_deadman_button: bool = False
    deadman_button: str = "LB"

    # ── button mapping ──
    slow_button: str = "LB"
    fast_button: str = "RB"

    # ── D-pad trim ──
    trim_vertical_speed_mps: float = 0.25
    trim_yaw_rate_dps: float = 20.0

    # ── long-press durations ──
    arm_hold_s: float = 1.5      # hold START to arm
    disarm_hold_s: float = 1.5   # hold BACK to disarm
    land_button: str = "Y"

    # ── AirSim command duration ──
    command_duration_s: float = 0.2

    # ── derived limits (used to configure the VelocityController) ──
    @property
    def max_horizontal_speed_mps(self) -> float:
        return self.fast_horizontal_speed_mps

    @property
    def max_vertical_speed_mps(self) -> float:
        return self.fast_vertical_speed_mps

    @property
    def max_yaw_rate_radps(self) -> float:
        """Hard yaw-rate cap in rad/s — the larger of FAST and trigger rates."""
        return math.radians(max(self.fast_yaw_rate_dps, self.trigger_yaw_rate_dps))

    def with_collision_guard(self, enabled: bool) -> "ManualGamepadConfig":
        """Return a copy with the collision guard toggled (for tests/CLI)."""
        from dataclasses import replace
        return replace(self, collision_guard=bool(enabled))


# ---------------------------------------------------------------------------
# YAML loader (strict)
# ---------------------------------------------------------------------------

def _as_float(raw: Any, key: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{key}: expected a number, got bool ({raw!r})")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: expected a number, got {raw!r}") from None


def _as_finite_positive(raw: Any, key: str, allow_zero: bool = False) -> float:
    val = _as_float(raw, key)
    if not math.isfinite(val):
        raise ValueError(f"{key}: must be finite, got {val!r}")
    if allow_zero:
        if val < 0:
            raise ValueError(f"{key}: must be >= 0, got {val}")
    elif val <= 0:
        raise ValueError(f"{key}: must be > 0, got {val}")
    return val


def _as_bool(raw: Any, key: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{key}: expected a bool, got {raw!r}")
    return raw


def _as_int(raw: Any, key: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{key}: expected an int, got bool ({raw!r})")
    if not isinstance(raw, int):
        raise ValueError(f"{key}: expected an int, got {raw!r}")
    return raw


def _as_button(raw: Any, key: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{key}: expected a button name string, got {raw!r}")
    name = raw.strip().upper()
    if name not in _VALID_BUTTONS:
        raise ValueError(
            f"{key}: unknown button {raw!r} (expected one of {sorted(_VALID_BUTTONS)})"
        )
    return name


def _as_backend(raw: Any, key: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{key}: expected a string, got {raw!r}")
    backend = raw.strip().lower()
    if backend != "pygame":
        raise ValueError(f"{key}: only 'pygame' is supported, got {raw!r}")
    return backend


# Per-field validators keyed by dataclass field name.
_VALIDATORS = {
    "backend": _as_backend,
    "controller_index": lambda v, k: _as_int(v, k),
    "poll_hz": lambda v, k: _as_finite_positive(v, k),
    "command_hz": lambda v, k: _as_finite_positive(v, k),
    "hud_hz": lambda v, k: _as_finite_positive(v, k),
    "deadzone": lambda v, k: _as_float(v, k),
    "expo": lambda v, k: _as_float(v, k),
    "yaw_expo": lambda v, k: _as_float(v, k),
    "trigger_deadzone": lambda v, k: _as_float(v, k),
    "normal_horizontal_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "normal_vertical_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "normal_yaw_rate_dps": lambda v, k: _as_finite_positive(v, k),
    "slow_horizontal_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "slow_vertical_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "slow_yaw_rate_dps": lambda v, k: _as_finite_positive(v, k),
    "fast_horizontal_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "fast_vertical_speed_mps": lambda v, k: _as_finite_positive(v, k),
    "fast_yaw_rate_dps": lambda v, k: _as_finite_positive(v, k),
    "trigger_yaw_rate_dps": lambda v, k: _as_finite_positive(v, k),
    "invert_left_x": _as_bool,
    "invert_left_y": _as_bool,
    "invert_right_x": _as_bool,
    "invert_right_y": _as_bool,
    "collision_guard": _as_bool,
    "emergency_distance_m": lambda v, k: _as_finite_positive(v, k),
    "input_timeout_s": lambda v, k: _as_finite_positive(v, k),
    "require_deadman_button": _as_bool,
    "deadman_button": _as_button,
    "slow_button": _as_button,
    "fast_button": _as_button,
    "trim_vertical_speed_mps": lambda v, k: _as_finite_positive(v, k, allow_zero=True),
    "trim_yaw_rate_dps": lambda v, k: _as_finite_positive(v, k, allow_zero=True),
    "arm_hold_s": lambda v, k: _as_finite_positive(v, k),
    "disarm_hold_s": lambda v, k: _as_finite_positive(v, k),
    "land_button": _as_button,
    "command_duration_s": lambda v, k: _as_finite_positive(v, k),
}


def load_manual_gamepad_config(path: Optional[str]) -> ManualGamepadConfig:
    """Load a gamepad config from YAML, applying strict validation.

    ``path`` may be ``None`` (return defaults) or a path to a YAML file.
    Unknown keys raise ``ValueError``; invalid values raise ``ValueError``
    with a key-prefixed message.
    """
    if path is None:
        return ManualGamepadConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"gamepad config must be a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(ManualGamepadConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown gamepad config key(s): {sorted(unknown)} "
            f"(expected: {sorted(known)})"
        )

    kwargs: dict = {}
    for key, value in raw.items():
        validator = _VALIDATORS[key]
        kwargs[key] = validator(value, key)

    return ManualGamepadConfig(**kwargs)

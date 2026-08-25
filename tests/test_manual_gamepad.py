"""Tests for gamepad manual flight mode (state, config, controller, mode).

Covers the pure mapping layer (``ManualGamepadController``), the config
loader/validators, the axis-shaping helpers, and the session dispatcher
(``ManualGamepadMode``).  No pygame or AirSim import is required — the reader
degrades gracefully and the mode is driven with a fake reader / mock session.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flight_modes.gamepad_config import (
    ManualGamepadConfig,
    apply_deadzone,
    apply_expo,
    clamp,
    load_manual_gamepad_config,
    normalize_gamepad_axis,
)
from flight_modes.gamepad_reader import GamepadReader
from flight_modes.gamepad_state import (
    GamepadState,
    ManualFlightCommand,
    SpeedProfile,
)
from flight_modes.manual_gamepad_mode import ManualGamepadController, ManualGamepadMode


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _state(**kwargs) -> GamepadState:
    s = GamepadState(connected=True)
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _controller(config=None) -> ManualGamepadController:
    return ManualGamepadController(config or ManualGamepadConfig())


def _update(controller, state, now=1.0, airborne=True, takeoff_called=True, landed=False):
    return controller.update(
        state, now, airborne=airborne, takeoff_called=takeoff_called, landed=landed
    )


def _make_mock_session():
    session = MagicMock()
    session.client = MagicMock()
    session.adapter = MagicMock()
    session.adapter.lidar_name = "LidarSensor1"
    session.vehicle_name = "Drone1"
    session.target_z_ned = -1.0
    session.takeoff_called = False
    session.is_airborne = False
    session.state.phase.name = "INITIALIZED"
    return session


_VC_PATH = "control.velocity_controller.VelocityController"


# ────────────────────────────────────────────────────────────────────────────
# Axis shaping helpers
# ────────────────────────────────────────────────────────────────────────────

class TestAxisShaping:
    def test_clamp(self):
        assert clamp(1.5, -1.0, 1.0) == 1.0
        assert clamp(-2.0, -1.0, 1.0) == -1.0
        assert clamp(0.5, -1.0, 1.0) == 0.5

    def test_deadzone_zero_inside(self):
        assert apply_deadzone(0.05, 0.10) == 0.0

    def test_deadzone_rescales(self):
        assert apply_deadzone(0.55, 0.10) == pytest.approx(0.5)

    def test_deadzone_negative_sign(self):
        assert apply_deadzone(-0.55, 0.10) == pytest.approx(-0.5)

    def test_expo_linear_when_zero(self):
        assert apply_expo(0.5, 0.0) == pytest.approx(0.5)

    def test_expo_preserves_endpoints(self):
        assert apply_expo(1.0, 0.5) == pytest.approx(1.0)
        assert apply_expo(-1.0, 0.5) == pytest.approx(-1.0)

    def test_normalize_invert(self):
        assert normalize_gamepad_axis(1.0, 0.0, 0.0, True) == pytest.approx(-1.0)


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

class TestManualGamepadConfig:
    def test_defaults_sane(self):
        c = ManualGamepadConfig()
        assert c.normal_horizontal_speed_mps > 0
        assert c.max_horizontal_speed_mps == c.fast_horizontal_speed_mps
        assert c.max_vertical_speed_mps == c.fast_vertical_speed_mps
        assert c.max_yaw_rate_radps == pytest.approx(math.radians(90.0))

    def test_fast_exceeds_normal_exceeds_slow(self):
        c = ManualGamepadConfig()
        assert c.fast_horizontal_speed_mps > c.normal_horizontal_speed_mps > c.slow_horizontal_speed_mps

    def test_with_collision_guard_returns_new_instance(self):
        c = ManualGamepadConfig()
        c2 = c.with_collision_guard(False)
        assert c.collision_guard is True
        assert c2.collision_guard is False

    def test_load_none_returns_defaults(self):
        assert load_manual_gamepad_config(None) == ManualGamepadConfig()

    def test_load_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "gp.yaml"
        p.write_text("deadzone: 0.15\nfast_horizontal_speed_mps: 3.0\n", encoding="utf-8")
        c = load_manual_gamepad_config(str(p))
        assert c.deadzone == pytest.approx(0.15)
        assert c.fast_horizontal_speed_mps == pytest.approx(3.0)
        assert c.normal_horizontal_speed_mps == pytest.approx(1.0)  # untouched default

    def test_unknown_key_rejected(self, tmp_path):
        p = tmp_path / "gp.yaml"
        p.write_text("nonsense_key: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown"):
            load_manual_gamepad_config(str(p))

    def test_bad_button_rejected(self, tmp_path):
        p = tmp_path / "gp.yaml"
        p.write_text("slow_button: NOT_A_BUTTON\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown button"):
            load_manual_gamepad_config(str(p))

    def test_negative_speed_rejected(self, tmp_path):
        p = tmp_path / "gp.yaml"
        p.write_text("fast_horizontal_speed_mps: -1.0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_manual_gamepad_config(str(p))


# ────────────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────────────

class TestDataModel:
    def test_command_is_zero_motion(self):
        assert ManualFlightCommand().is_zero_motion is True

    def test_command_not_zero_with_yaw(self):
        assert ManualFlightCommand(yaw_rate_radps=0.1).is_zero_motion is False

    def test_speed_profile_enum(self):
        assert {p.name for p in SpeedProfile} == {"NORMAL", "SLOW", "FAST"}


# ────────────────────────────────────────────────────────────────────────────
# Controller mapping — motion axes (Mode-2)
# ────────────────────────────────────────────────────────────────────────────

class TestControllerMotion:
    def test_rs_up_forward(self):
        cmd = _update(_controller(), _state(right_y=-1.0))
        assert cmd.vx > 0

    def test_rs_down_backward(self):
        cmd = _update(_controller(), _state(right_y=1.0))
        assert cmd.vx < 0

    def test_rs_right_strafe_right(self):
        cmd = _update(_controller(), _state(right_x=1.0))
        assert cmd.vy > 0

    def test_rs_left_strafe_left(self):
        cmd = _update(_controller(), _state(right_x=-1.0))
        assert cmd.vy < 0

    def test_ls_up_climb(self):
        cmd = _update(_controller(), _state(left_y=-1.0))
        assert cmd.vz < 0

    def test_ls_down_descend(self):
        cmd = _update(_controller(), _state(left_y=1.0))
        assert cmd.vz > 0

    def test_ls_right_yaw_right(self):
        cmd = _update(_controller(), _state(left_x=1.0))
        assert cmd.yaw_rate_radps > 0

    def test_ls_left_yaw_left(self):
        cmd = _update(_controller(), _state(left_x=-1.0))
        assert cmd.yaw_rate_radps < 0

    def test_rt_yaw_right_positive(self):
        cmd = _update(_controller(), _state(right_trigger=1.0))
        assert cmd.yaw_rate_radps > 0

    def test_lt_yaw_left_negative(self):
        cmd = _update(_controller(), _state(left_trigger=1.0))
        assert cmd.yaw_rate_radps < 0

    def test_trigger_overrides_ls_x(self):
        # LS pushed right (+1) but LT held → net yaw must be negative (left).
        cmd = _update(_controller(), _state(left_x=1.0, left_trigger=1.0))
        assert cmd.yaw_rate_radps < 0

    def test_deadzone_kills_small_ls_x(self):
        cmd = _update(_controller(), _state(left_x=0.02))
        assert cmd.yaw_rate_radps == 0.0

    def test_centered_sticks_zero_motion(self):
        cmd = _update(_controller(), _state())
        assert cmd.is_zero_motion is True


# ────────────────────────────────────────────────────────────────────────────
# Controller mapping — speed profiles
# ────────────────────────────────────────────────────────────────────────────

class TestSpeedProfiles:
    def test_rb_fast(self):
        cmd = _update(_controller(), _state(right_y=-1.0, rb=True))
        assert cmd.speed_profile == "FAST"
        assert cmd.vx == pytest.approx(ManualGamepadConfig().fast_horizontal_speed_mps)

    def test_lb_slow(self):
        cmd = _update(_controller(), _state(right_y=-1.0, lb=True))
        assert cmd.speed_profile == "SLOW"
        assert cmd.vx == pytest.approx(ManualGamepadConfig().slow_horizontal_speed_mps)

    def test_lb_rb_normal(self):
        cmd = _update(_controller(), _state(right_y=-1.0, lb=True, rb=True))
        assert cmd.speed_profile == "NORMAL"
        assert cmd.vx == pytest.approx(ManualGamepadConfig().normal_horizontal_speed_mps)

    def test_fast_exceeds_normal_velocity(self):
        c = ManualGamepadConfig()
        fast = _update(_controller(c), _state(right_y=-1.0, rb=True))
        normal = _update(_controller(c), _state(right_y=-1.0))
        assert fast.vx > normal.vx


# ────────────────────────────────────────────────────────────────────────────
# Controller mapping — discrete actions & safety
# ────────────────────────────────────────────────────────────────────────────

class TestDiscreteActions:
    def test_a_takeoff(self):
        cmd = _update(_controller(), _state(button_a=True), airborne=False, takeoff_called=False)
        assert cmd.takeoff is True

    def test_a_ignored_if_already_takeoff(self):
        cmd = _update(_controller(), _state(button_a=True), airborne=False, takeoff_called=True)
        assert cmd.takeoff is False

    def test_y_land_when_airborne(self):
        cmd = _update(_controller(), _state(button_y=True), airborne=True)
        assert cmd.land is True

    def test_start_long_press_arm(self):
        c = ManualGamepadConfig()
        controller = _controller(c)
        _update(controller, _state(start=True), now=0.0, airborne=False, takeoff_called=False)
        cmd = _update(controller, _state(start=True), now=0.0 + c.arm_hold_s + 0.1,
                      airborne=False, takeoff_called=False)
        assert cmd.arm is True

    def test_back_long_press_disarm(self):
        c = ManualGamepadConfig()
        controller = _controller(c)
        _update(controller, _state(back=True), now=0.0, airborne=True)
        cmd = _update(controller, _state(back=True), now=0.0 + c.disarm_hold_s + 0.1,
                      airborne=True)
        assert cmd.disarm is True

    def test_disconnect_hover(self):
        cmd = _update(_controller(), _state(connected=False), airborne=True)
        assert cmd.hover is True and cmd.input_ok is False and cmd.reason == "disconnected"

    def test_deadman_released_hover(self):
        c = ManualGamepadConfig(require_deadman_button=True, deadman_button="LB")
        cmd = _update(_controller(c), _state(lb=False), airborne=True)
        assert cmd.hover is True and cmd.reason == "deadman_not_held"

    def test_motion_ignored_on_ground(self):
        cmd = _update(_controller(), _state(right_y=-1.0), airborne=False, takeoff_called=True)
        assert cmd.hover is True


class TestDpadTrim:
    def test_dpad_up_trim_climb(self):
        base = _update(_controller(), _state())
        trimmed = _update(_controller(), _state(dpad_y=-1))
        assert trimmed.vz < base.vz

    def test_dpad_right_trim_yaw_right(self):
        cmd = _update(_controller(), _state(dpad_x=1))
        assert cmd.yaw_rate_radps > 0


# ────────────────────────────────────────────────────────────────────────────
# GamepadReader (no pygame required)
# ────────────────────────────────────────────────────────────────────────────

class TestGamepadReader:
    def test_trigger_remap(self):
        assert GamepadReader._trigger(-1.0) == pytest.approx(0.0)
        assert GamepadReader._trigger(1.0) == pytest.approx(1.0)
        assert GamepadReader._trigger(0.0) == pytest.approx(0.5)

    def test_is_known_xbox(self):
        assert GamepadReader._is_known("Xbox 360 Controller") is True
        assert GamepadReader._is_known("Microsoft XInput") is True

    def test_is_known_unknown(self):
        assert GamepadReader._is_known("Acme Flight Stick") is False

    def test_poll_disconnected_when_pygame_missing(self):
        reader = GamepadReader(0)
        reader._pygame = None  # simulate pygame unavailable
        state = reader.poll(now=1.0)
        assert state.connected is False

    def test_start_without_pygame_does_not_raise(self, monkeypatch):
        reader = GamepadReader(0)
        # Force ImportError on pygame import.
        monkeypatch.setattr(
            "builtins.__import__",
            lambda name, *a, **k: (_ for _ in ()).throw(ImportError("no pygame"))
            if name == "pygame" else __import__(name, *a, **k),
        )
        reader.start()  # must not raise
        assert reader.poll(now=1.0).connected is False


# ────────────────────────────────────────────────────────────────────────────
# ManualGamepadMode dispatcher
# ────────────────────────────────────────────────────────────────────────────

class TestManualGamepadMode:
    def _mode(self, session=None, config=None, reader=None):
        session = session or _make_mock_session()
        with patch(_VC_PATH) as mock_vc_cls:
            mode = ManualGamepadMode(session, config=config or ManualGamepadConfig(), reader=reader)
            return mode, mock_vc_cls.return_value

    def test_takeoff_dispatches_session(self):
        session = _make_mock_session()
        mode, _ = self._mode(session)
        assert mode._dispatch_discrete(ManualFlightCommand(takeoff=True)) is True
        session.takeoff_and_climb.assert_called_once()

    def test_land_dispatches_and_stops(self):
        session = _make_mock_session()
        mode, _ = self._mode(session)
        assert mode._dispatch_discrete(ManualFlightCommand(land=True)) is True
        session.land_and_disarm.assert_called_once()
        assert mode._running is False

    def test_continuous_velocity(self):
        session = _make_mock_session()
        session.is_airborne = True
        mode, vc = self._mode(session)
        mode._dispatch_continuous(ManualFlightCommand(vx=1.0, vy=0.5, vz=-0.3, yaw_rate_radps=0.1))
        kw = vc.send_velocity_body_frd.call_args.kwargs
        assert kw["vx"] == pytest.approx(1.0)
        assert kw["vy"] == pytest.approx(0.5)
        assert kw["vz"] == pytest.approx(-0.3)
        assert kw["yaw_rate"] == pytest.approx(0.1)

    def test_continuous_zero_motion_hover(self):
        session = _make_mock_session()
        session.is_airborne = True
        mode, vc = self._mode(session)
        mode._dispatch_continuous(ManualFlightCommand())
        vc.hover.assert_called_once()

    def test_continuous_noop_when_not_airborne(self):
        session = _make_mock_session()
        session.is_airborne = False
        mode, vc = self._mode(session)
        mode._dispatch_continuous(ManualFlightCommand(vx=1.0))
        vc.send_velocity_body_frd.assert_not_called()

    def test_collision_guard_overrides_motion(self):
        session = _make_mock_session()
        session.is_airborne = True
        mode, _ = self._mode(session)
        mode._read_collision = MagicMock(return_value=(0.5, False, "Wall"))
        cmd = mode._apply_collision_guard(ManualFlightCommand(vx=1.0))
        assert cmd.hover is True and cmd.reason == "collision_guard"

    def test_collision_guard_disabled_passthrough(self):
        session = _make_mock_session()
        session.is_airborne = True
        cfg = ManualGamepadConfig(collision_guard=False)
        mode, _ = self._mode(session, config=cfg)
        cmd = mode._apply_collision_guard(ManualFlightCommand(vx=1.0))
        assert cmd.vx == pytest.approx(1.0)

    def test_collision_brake_on_collided(self):
        session = _make_mock_session()
        mode, _ = self._mode(session)
        assert mode._collision_brake_required(10.0, True, "Wall") is True

    def test_collision_brake_on_close(self):
        session = _make_mock_session()
        mode, _ = self._mode(session)
        assert mode._collision_brake_required(0.5, False, "") is True

    def test_collision_brake_none_when_far(self):
        session = _make_mock_session()
        mode, _ = self._mode(session)
        assert mode._collision_brake_required(5.0, False, "") is False

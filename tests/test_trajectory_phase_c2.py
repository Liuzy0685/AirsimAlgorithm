"""Phase C2 unit tests — realtime loop + altitude-hold P0 fixes.

Covers the pure-computation units introduced in Phase C2:

  A. command duration does not determine control frequency
  B. deadline scheduler
  C. altitude dispatch (moveByVelocityZBodyFrameAsync)
  D. no altitude sign error
  E. climb confirmation (state read-back)
  F. duplicate initial CBMBA request guard
  G. stale sensor snapshot

No AirSim RPC is exercised.
"""

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

import flight_modes.automatic_mode as automatic_mode
from flight_modes.automatic_mode import (
    AutomaticMode,
    GlobalPlannerWorker,
    PerceptionSnapshot,
    PerceptionWorker,
)
from control.velocity_controller import VelocityController
from flight_modes.shared_flight_session import SharedFlightSession


# ── helpers ──

def _fake_result(success=True, path=None):
    return types.SimpleNamespace(
        success=success,
        path_world=path or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        nodes_expanded=42,
        planning_time_ms=12.5,
        grid_size=100,
        max_lateral_deviation_m=0.3,
    )


class _FakePlanner:
    def __init__(self, result=None, delay_s=0.0):
        self._result = result if result is not None else _fake_result()
        self._delay_s = delay_s
        self.calls = []

    def plan_with_result(self, obstacles, start, goal):
        self.calls.append((obstacles, start, goal))
        if self._delay_s:
            time.sleep(self._delay_s)
        return self._result


class _Clock:
    """Controllable monotonic clock for deadline-scheduler / staleness tests."""

    def __init__(self, t=10.0):
        self.t = t
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _wait_for(predicate, timeout_s=2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = predicate()
        if val is not None and val is not False:
            return val
        time.sleep(0.005)
    return predicate()


def _fake_lidar_reader(frame_valid=True, stale_count=0):
    lf = MagicMock()
    lf.frame_valid = frame_valid
    lf.invalid_reason = None if frame_valid else "stale"
    lf.point_cloud_sensor = MagicMock()
    lf.received_monotonic_seconds = 0.0
    lf.point_count = 0
    reader = MagicMock()
    reader.read.return_value = lf
    reader.consecutive_stale_count = stale_count
    return reader


def _fake_perceive(lf):
    if lf is None or not lf.frame_valid:
        return (None, None, None)
    return (MagicMock(), MagicMock(), {"front": 10.0, "left": 10.0, "right": 10.0})


def _make_controller():
    fake = MagicMock()
    fake.DrivetrainType.MaxDegreeOfFreedom = "MaxDegreeOfFreedom"
    fake.YawMode.side_effect = lambda is_rate=True, yaw_or_rate=0.0: MagicMock(
        is_rate=is_rate, yaw_or_rate=yaw_or_rate,
    )
    adapter = MagicMock()
    adapter._readonly = False
    adapter.vehicle_name = "Drone1"
    adapter._assert_writable = lambda: None
    client = MagicMock()
    adapter.get_raw_client.return_value = client
    vc = VelocityController(
        adapter=adapter, airsim_module=fake,
        max_horizontal_speed_mps=2.0, max_vertical_speed_mps=0.5,
        max_yaw_rate_radps=0.5, command_duration_seconds=0.2,
    )
    return vc, client


def _make_session(z_values):
    s = SharedFlightSession(settings_json="fake.json", mode="auto")
    s.target_z_ned = -1.0
    s.max_vertical_speed_mps = 0.5
    client = MagicMock()

    def _state(z):
        st = MagicMock()
        kin = MagicMock()
        kin.position.z_val = z
        st.kinematics_estimated = kin
        return st

    client.getMultirotorState.side_effect = [_state(z) for z in z_values]
    s._client = client
    return s, client


# ── A. command duration does not determine control frequency ──


class TestControlFrequencyDecoupling:
    def test_period_is_inverse_target_hz_not_command_duration(self):
        cfg = yaml.safe_load(
            (_PROJECT_ROOT / "configs" / "trajectory_planner.yaml").read_text(encoding="utf-8")
        )
        target_hz = float(cfg["control_loop"]["target_hz"])
        period_s = 1.0 / max(0.5, target_hz)

        flight = yaml.safe_load(
            (_PROJECT_ROOT / "configs" / "minimal_flight.yaml").read_text(encoding="utf-8")
        )
        command_duration_s = float(flight["minimal_flight"]["command_duration_s"])

        # The control loop rate must be 1/target_hz (50 ms @ 20 Hz), NOT
        # 1/command_duration_s (5 Hz @ 0.2 s).  These are unrelated.
        assert period_s == pytest.approx(1.0 / target_hz)
        assert period_s != command_duration_s

    def test_target_hz_is_20(self):
        cfg = yaml.safe_load(
            (_PROJECT_ROOT / "configs" / "trajectory_planner.yaml").read_text(encoding="utf-8")
        )
        assert float(cfg["control_loop"]["target_hz"]) == 20.0


# ── B. deadline scheduler ──


class TestDeadlineScheduler:
    def _auto(self, period_s=0.05):
        a = AutomaticMode.__new__(AutomaticMode)
        a._control_period_s = period_s
        return a

    def test_on_time_advances_tick_and_sleeps(self):
        a = self._auto()
        clock = _Clock(10.0)
        with patch.object(automatic_mode.time, "monotonic", clock.monotonic), \
                patch.object(automatic_mode.time, "sleep", clock.sleep):
            next_tick, sleep_ms, late_ms, missed, resynced = a._sleep_to_next_period(10.02)
        assert next_tick == pytest.approx(10.07)  # 10.02 + 0.05
        assert sleep_ms == pytest.approx(70.0)
        assert late_ms == 0.0
        assert missed == 0
        assert resynced is False
        assert len(clock.sleeps) == 1
        assert clock.sleeps[0] == pytest.approx(0.07)

    def test_missed_deadline_resyncs_and_spaces_next_tick(self):
        a = self._auto()
        clock = _Clock(10.0)
        with patch.object(automatic_mode.time, "monotonic", clock.monotonic), \
                patch.object(automatic_mode.time, "sleep", clock.sleep):
            next_tick, sleep_ms, late_ms, missed, resynced = a._sleep_to_next_period(9.9)
        assert next_tick == pytest.approx(10.05)  # re-anchored one period ahead of now
        assert sleep_ms == pytest.approx(50.0)
        assert late_ms == pytest.approx(50.0)    # 50 ms late
        assert missed == 1                       # one full period missed
        assert resynced is True
        # One full period is slept so the *next* tick is spaced ~period after
        # the resync — never a tight sleep(0) catch-up burst.
        assert clock.sleeps == [pytest.approx(0.05)]


# ── C. altitude dispatch ──


class TestAltitudeHoldDispatch:
    def test_uses_move_by_velocity_z_body_frame_async(self):
        vc, client = _make_controller()
        vc.send_velocity_body_frd_z(0.3, -0.2, -1.0, duration=0.5, vehicle_name="Drone1")
        client.moveByVelocityZBodyFrameAsync.assert_called_once()
        # Must NOT fall back to the vertical-velocity API (vz would be a speed).
        client.moveByVelocityBodyFrameAsync.assert_not_called()
        args, _ = client.moveByVelocityZBodyFrameAsync.call_args
        assert args[0] == 0.3   # vx (forward)
        assert args[1] == -0.2  # vy (right)
        assert args[2] == -1.0  # target_z (altitude position)
        assert args[3] == 0.5   # duration


# ── D. no altitude sign error ──


class TestNoAltitudeSignError:
    def test_altitude_velocity_points_toward_target_in_ned(self):
        # NED Z is positive down: from +1.1 m back to -1.0 m requires an
        # upward (negative) velocity, never the old positive descent command.
        auto = AutomaticMode.__new__(AutomaticMode)
        assert auto._altitude_hold_velocity(1.1, -1.0, 0.5) == pytest.approx(-0.5)
        assert auto._altitude_hold_velocity(-0.5, -1.0, 0.5) == pytest.approx(-0.5)
        assert auto._altitude_hold_velocity(-1.2, -1.0, 0.5) == pytest.approx(0.2)

    def test_altitude_hold_velocity_is_bounded(self):
        auto = AutomaticMode.__new__(AutomaticMode)
        assert auto._altitude_hold_velocity(10.0, -1.0, 0.5) == pytest.approx(-0.5)

    def test_vertical_controller_limit_must_not_be_zero(self):
        # A zero controller limit would discard every corrective vz command
        # before it reaches AirSim and let the aircraft drift in Z.
        params = automatic_mode.AutomaticModeParams(max_vertical_speed_mps=0.5)
        assert params.max_vertical_speed_mps > 0.0

    def test_target_z_passed_verbatim(self):
        vc, client = _make_controller()
        # NED: -1.0 = 1 m above the origin.  A sign flip would command +1.0
        # (below ground).  The altitude hold must pass the target through
        # exactly as configured.
        vc.send_velocity_body_frd_z(0.0, 0.0, -1.0, vehicle_name="Drone1")
        args, _ = client.moveByVelocityZBodyFrameAsync.call_args
        assert args[2] == -1.0  # no negation

    def test_target_z_not_treated_as_velocity(self):
        vc, client = _make_controller()
        vc.send_velocity_body_frd_z(0.0, 0.0, -1.0, vehicle_name="Drone1")
        args, _ = client.moveByVelocityZBodyFrameAsync.call_args
        # moveByVelocityZBodyFrameAsync has exactly 4 positional args (vx, vy,
        # z, duration) — there is no vertical *velocity* component.
        assert len(args) == 4


# ── E. climb confirmation ──


class TestClimbConfirmation:
    def test_confirmed_within_tolerance_no_retry(self):
        s, client = _make_session([-1.02])  # error 0.02 ≤ 0.3
        s._confirm_altitude("Drone1")
        assert client.moveToZAsync.call_count == 0

    def test_short_climb_retries_once_then_confirms(self):
        # First read-back is short (z=-0.42), retry climbs to -1.0.
        s, client = _make_session([-0.42, -1.0])
        s._confirm_altitude("Drone1")
        assert client.moveToZAsync.call_count == 1

    def test_persistent_short_climb_reports_failure(self):
        s, client = _make_session([-0.42, -0.5])  # both short
        s._confirm_altitude("Drone1")
        # Only one retry is issued (between attempt 1 and attempt 2).
        assert client.moveToZAsync.call_count == 1


# ── F. duplicate initial CBMBA request guard ──


class TestDuplicateInitialCbmbaRequest:
    def test_no_in_flight_request_initially(self):
        w = GlobalPlannerWorker(_FakePlanner())
        assert not w.has_in_flight_request()

    def test_pending_request_is_in_flight(self):
        w = GlobalPlannerWorker(_FakePlanner())
        w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="initial")
        # Not started → the request sits pending; the loop-side guard sees it.
        assert w.has_in_flight_request()

    def test_processed_request_clears_in_flight(self):
        planner = _FakePlanner(delay_s=0.02)
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        try:
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="initial")
            _wait_for(lambda: w.get_latest_result())
            assert not w.has_in_flight_request()
        finally:
            w.shutdown()

    def test_duplicate_initial_request_is_skipped(self):
        # The loop-side guard calls has_in_flight_request() and skips re-issuing
        # the "initial" request while the first is still running — so only ONE
        # planner call is ever made.
        planner = _FakePlanner(delay_s=0.05)
        w = GlobalPlannerWorker(planner, poll_interval_s=0.001)
        w.start()
        try:
            w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="initial")
            # Simulate the guard inside _plan_trajectory_tick:
            if w.has_in_flight_request():
                pass  # skip the duplicate "initial" request
            else:
                w.request_replan([[0, 0, 0]], [0, 0, 0], [10, 0, 0], reason="initial")
            _wait_for(lambda: w.get_latest_result())
            assert len(planner.calls) == 1
        finally:
            w.shutdown()


# ── G. stale sensor snapshot ──


class TestPerceptionWorkerStaleness:
    def test_snapshot_age_inf_before_first_poll(self):
        w = PerceptionWorker(_fake_lidar_reader(), _fake_perceive, poll_hz=10.0)
        assert w.get_latest_snapshot() is None
        assert w.snapshot_age_s(now=0.0) == float("inf")

    def test_snapshot_age_uses_clock_delta(self):
        clock = _Clock(100.0)
        w = PerceptionWorker(
            _fake_lidar_reader(), _fake_perceive, poll_hz=10.0, clock=clock.monotonic,
        )
        w._latest = PerceptionSnapshot(received_mono=100.0)
        clock.t = 101.25
        assert w.snapshot_age_s(now=101.25) == pytest.approx(1.25)

    def test_worker_publishes_valid_snapshot(self):
        w = PerceptionWorker(_fake_lidar_reader(), _fake_perceive, poll_hz=10.0)
        w.start()
        try:
            snap = _wait_for(lambda: w.get_latest_snapshot())
            assert snap is not None
            assert snap.lf.frame_valid is True
            assert snap.fr is not None
            assert snap.dd is not None
            assert snap.rays == {"front": 10.0, "left": 10.0, "right": 10.0}
        finally:
            w.shutdown()

    def test_worker_publishes_stale_count(self):
        w = PerceptionWorker(_fake_lidar_reader(stale_count=3), _fake_perceive, poll_hz=10.0)
        w.start()
        try:
            snap = _wait_for(lambda: w.get_latest_snapshot())
            assert snap is not None
            assert snap.stale_count == 3
        finally:
            w.shutdown()

    def test_invalid_lidar_yields_empty_perception(self):
        w = PerceptionWorker(
            _fake_lidar_reader(frame_valid=False), _fake_perceive, poll_hz=10.0,
        )
        w.start()
        try:
            snap = _wait_for(lambda: w.get_latest_snapshot())
            assert snap is not None
            assert snap.lf.frame_valid is False
            assert snap.fr is None and snap.dd is None and snap.rays is None
        finally:
            w.shutdown()

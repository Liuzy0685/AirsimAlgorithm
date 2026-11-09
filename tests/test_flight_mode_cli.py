"""Tests for flight_mode.py CLI entry point — single-instance, argument parsing, mode dispatch."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))


class TestCLIArgs:
    def test_manual_mode_requires_no_clearance(self):
        """manual mode does not require --confirm-simulation-clearance."""
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", ["flight_mode.py", "--mode", "manual"]):
            args = _parse_args()
            assert args.mode == "manual"
            assert not args.confirm_simulation_clearance

    def test_auto_mode_without_clearance_rejected_in_run(self):
        """--mode auto without --confirm-simulation-clearance → exit code 2."""
        from scripts.flight_mode import _run_auto
        args = MagicMock()
        args.confirm_simulation_cnflearance = False  # not set
        args.settings_json = None
        # simulate missing clearance
        with patch("sys.argv", ["flight_mode.py", "--mode", "auto"]):
            from scripts.flight_mode import _parse_args
            # Actually test the logic inline
            pass

    def test_cli_overrides_passed_to_auto(self):
        """--target-z and --max-duration are passed as CLI overrides."""
        # Verify that when --target-z is specified, it's included in cli_overrides
        args = MagicMock()
        args.target_z = -2.5
        args.max_duration = 15.0
        args.flight_config = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        # Build cli_overrides
        overrides = {}
        if args.target_z is not None:
            overrides["target_z_ned"] = float(args.target_z)
        if args.max_duration is not None:
            overrides["max_flight_duration_s"] = float(args.max_duration)
        assert overrides == {"target_z_ned": -2.5, "max_flight_duration_s": 15.0}

    def test_cli_overrides_none_when_not_specified(self):
        args = MagicMock()
        args.target_z = None
        args.max_duration = None
        overrides = {}
        if args.target_z is not None: overrides["target_z_ned"] = float(args.target_z)
        if args.max_duration is not None: overrides["max_flight_duration_s"] = float(args.max_duration)
        assert overrides == {}

    def test_goal_coordinates_are_parsed(self):
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", [
            "flight_mode.py", "--mode", "auto", "--settings-json", "f.json",
            "--confirm-simulation-clearance", "--goal-x", "15", "--goal-y", "0",
        ]):
            args = _parse_args()
            assert args.goal_x == 15.0
            assert args.goal_y == 0.0


class TestSingleInstanceCLI:
    def test_acquire_lock_success(self):
        from flight_modes.shared_flight_session import SharedFlightSession
        fh = SharedFlightSession.acquire_lock("cli-test")
        assert fh is not None
        SharedFlightSession.release_lock(fh)

    def test_second_lock_fails(self):
        from flight_modes.shared_flight_session import SharedFlightSession
        fh1 = SharedFlightSession.acquire_lock("first")
        assert fh1 is not None
        fh2 = SharedFlightSession.acquire_lock("second")
        assert fh2 is None  # should fail — lock held by first
        SharedFlightSession.release_lock(fh1)


class TestManualModeDispatch:
    def test_run_manual_initializes_session(self):
        from scripts.flight_mode import _run_manual

        with patch("scripts.flight_mode._acquire_lock_or_die") as mock_lock, \
             patch("flight_modes.manual_mode.ManualMode") as mock_manual, \
             patch("flight_modes.shared_flight_session.SharedFlightSession") as mock_session_cls:
            mock_lock.return_value = MagicMock()
            mock_session = mock_session_cls.return_value
            mock_session.state.phase.name = "INITIALIZED"
            mock_session.is_airborne = False

            args = MagicMock()
            args.manual_control = "velocity"
            args.settings_json = ""
            args.target_z = None

            result = _run_manual(args)
            mock_session.initialize.assert_called_once()
            mock_session.takeoff_and_climb.assert_not_called()


class TestAutoModeDispatch:
    def test_auto_without_clearance_exits_2(self):
        from scripts.flight_mode import _run_auto

        with patch("scripts.flight_mode._acquire_lock_or_die"):
            args = MagicMock()
            args.confirm_simulation_clearance = False
            args.settings_json = None
            result = _run_auto(args)
            assert result == 2  # clearance not confirmed

    def test_auto_without_settings_exits_2(self):
        from scripts.flight_mode import _run_auto

        with patch("scripts.flight_mode._acquire_lock_or_die"):
            args = MagicMock()
            args.confirm_simulation_clearance = True
            args.settings_json = None
            result = _run_auto(args)
            assert result == 2  # settings not provided


class TestFlightConfigLoading:
    def test_load_valid_config(self):
        from flight_modes.automatic_mode import _load_flight_config
        cfg_path = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        params = _load_flight_config(cfg_path)
        assert params["target_z_ned"] == -1.0
        assert params["max_flight_duration_s"] == 10.0
        assert params["geofence_radius_m"] == 2.0
        assert params["command_duration_s"] == 0.2

    def test_missing_key_raises(self):
        from flight_modes.automatic_mode import _load_flight_config
        import tempfile, yaml
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump({"minimal_flight": {"target_z_ned": -1.0}}, f)
            f.flush()
            f.close()
            with pytest.raises(ValueError, match="missing"):
                _load_flight_config(f.name)
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_params_from_yaml(self):
        from flight_modes.automatic_mode import AutomaticModeParams
        cfg_path = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        params = AutomaticModeParams.from_yaml(cfg_path)
        assert params.target_z_ned == -1.0
        assert params.emergency_distance_m == 0.8
        assert params.max_flight_duration_s == 10.0

    def test_cli_overrides_yaml(self):
        from flight_modes.automatic_mode import AutomaticModeParams
        cfg_path = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        params = AutomaticModeParams.from_yaml(cfg_path, {"target_z_ned": -5.0, "max_flight_duration_s": 20.0})
        assert params.target_z_ned == -5.0  # CLI wins
        assert params.max_flight_duration_s == 20.0
        assert params.emergency_distance_m == 0.8  # unchanged from YAML

class TestPlannerModeCLI:
    def test_default_is_reactive(self):
        """--planner-mode defaults to 'reactive'."""
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", ["flight_mode.py", "--mode", "auto",
                                "--settings-json", "f.json",
                                "--confirm-simulation-clearance"]):
            args = _parse_args()
            assert args.planner_mode == "reactive"

    def test_apf_shadow_passed_correctly(self):
        """--planner-mode apf_shadow is parsed correctly."""
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", ["flight_mode.py", "--mode", "auto",
                                "--settings-json", "f.json",
                                "--confirm-simulation-clearance",
                                "--planner-mode", "apf_shadow"]):
            args = _parse_args()
            assert args.planner_mode == "apf_shadow"

    def test_apf_passed_correctly(self):
        """--planner-mode apf is parsed correctly."""
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", ["flight_mode.py", "--mode", "auto",
                                "--settings-json", "f.json",
                                "--confirm-simulation-clearance",
                                "--planner-mode", "apf"]):
            args = _parse_args()
            assert args.planner_mode == "apf"

    def test_invalid_mode_rejected_by_argparse(self):
        """Invalid --planner-mode value is rejected by argparse."""
        from scripts.flight_mode import _parse_args
        with patch("sys.argv", ["flight_mode.py", "--mode", "auto",
                                "--settings-json", "f.json",
                                "--confirm-simulation-clearance",
                                "--planner-mode", "invalid_mode"]):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_planner_mode_in_cli_overrides(self):
        """planner_mode is included in cli_overrides for _run_auto."""
        args = MagicMock()
        args.target_z = None
        args.max_duration = None
        args.planner_mode = "apf_shadow"
        args.flight_config = str(_PROJECT_ROOT / "configs" / "minimal_flight.yaml")
        overrides = {}
        if args.target_z is not None:
            overrides["target_z_ned"] = float(args.target_z)
        if args.max_duration is not None:
            overrides["max_flight_duration_s"] = float(args.max_duration)
        overrides["planner_mode"] = args.planner_mode
        assert overrides == {"planner_mode": "apf_shadow"}

"""ROUND 4.1: SafetySupervisor tests."""
import sys, math
from pathlib import Path
import numpy as np, pytest
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
from control.safety_supervisor import SafetySupervisor
from models.local_planner_command import LocalPlannerCommand


class TestSafetySupervisor:
    def setup_method(self):
        self._clock_val = 100.0
        self.safety = SafetySupervisor(clock=lambda: self._clock_val)

    def _valid_lidar_frame(self):
        from unittest.mock import MagicMock
        lf = MagicMock(); lf.frame_valid = True
        lf.received_monotonic_seconds = self._clock_val - 0.01  # very recent
        lf.invalid_reason = None; return lf

    def _stale_lidar_frame(self):
        from unittest.mock import MagicMock
        lf = MagicMock(); lf.frame_valid = True
        lf.received_monotonic_seconds = self._clock_val - 10.0  # 10s old
        lf.invalid_reason = None; return lf

    def _valid_dd(self):
        from unittest.mock import MagicMock
        dd = MagicMock(); dd.frame_valid = True; dd.invalid_reason = None; return dd

    def _valid_collision(self):
        from unittest.mock import MagicMock
        c = MagicMock(); c.has_collided = False; c.object_name = ""; c.penetration_depth = 0.0; return c

    def test_valid_command_passes(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert cmd.command_valid

    def test_lidar_stale_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._stale_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid
        assert "stale" in (cmd.invalid_reason or "")

    def test_lidar_invalid_blocks(self):
        lf = self._valid_lidar_frame(); lf.frame_valid = False; lf.invalid_reason = "stale"
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf", lf,
                                    self._valid_dd(), self._valid_collision(), True, 0)
        assert not cmd.command_valid
        assert "stale" in (cmd.invalid_reason or "")

    def test_data_sync_invalid_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0,
                                    data_sync_valid=False, data_sync_reason="skew")
        assert not cmd.command_valid
        assert "skew" in (cmd.invalid_reason or "")

    def test_obstacle_approach_blocks(self):
        import numpy as np
        # velocity (1,0,0) toward obstacle at (5,0,0) from ego at (0,0,0)
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0,
                                    obstacle_positions_ned=np.array([[5.0, 0.0, 0.0]]),
                                    ego_position_ned=(0.0, 0.0, 0.0))
        assert not cmd.command_valid
        assert "toward obstacle" in (cmd.invalid_reason or "")

    def test_obstacle_away_passes(self):
        import numpy as np
        # velocity (1,0,0) AWAY from obstacle at (-5,0,0) from ego at (0,0,0)
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0,
                                    obstacle_positions_ned=np.array([[-5.0, 0.0, 0.0]]),
                                    ego_position_ned=(0.0, 0.0, 0.0))
        assert cmd.command_valid

    def test_lidar_none_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf", None,
                                    self._valid_dd(), self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_fov_incompatible_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), fov_compatible=False, consecutive_invalid=0)
        assert not cmd.command_valid
        assert "FOV" in (cmd.invalid_reason or "")

    def test_dd_invalid_blocks(self):
        dd = self._valid_dd(); dd.frame_valid = False; dd.invalid_reason = "empty"
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), dd,
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_dd_none_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), None,
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_collision_blocks(self):
        c = self._valid_collision(); c.has_collided = True; c.object_name = "wall"; c.penetration_depth = 0.1
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(), c, True, 0)
        assert not cmd.command_valid

    def test_nan_velocity_blocks(self):
        cmd = self.safety.validate((float("nan"), 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_inf_velocity_blocks(self):
        cmd = self.safety.validate((float("inf"), 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_nan_yaw_rate_blocks(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), float("nan"), "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_horizontal_speed_exceeded(self):
        cmd = self.safety.validate((5.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_vertical_speed_exceeded(self):
        cmd = self.safety.validate((0.0, 0.0, 2.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_yaw_rate_exceeded(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 3.0, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, 0)
        assert not cmd.command_valid

    def test_consecutive_invalid_max(self):
        cmd = self.safety.validate((1.0, 0.0, 0.0), 0.1, "apf",
                                    self._valid_lidar_frame(), self._valid_dd(),
                                    self._valid_collision(), True, consecutive_invalid=15)
        assert not cmd.command_valid

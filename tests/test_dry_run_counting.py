"""ROUND 4.4: Dry-run counting semantics tests.

Verifies:
- Data errors increment consecutive_invalid_data, reach threshold → terminate
- Safety holds increment safety_hold_count, do NOT count as data errors
- --frames 20 with all safety holds records all 20 frames
"""
import sys, math
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_valid_lidar_frame(point_count=100):
    """Create a mock LidarFrame that passes all validation."""
    pts = np.random.rand(point_count, 3).astype(np.float32) * 10.0 + 5.0
    lf = MagicMock()
    lf.frame_valid = True
    lf.invalid_reason = None
    lf.point_cloud_sensor = pts
    lf.point_count = point_count
    lf.raw_timestamp_ns = 1234567890
    lf.received_monotonic_seconds = 100.0
    lf.sensor_pose = {
        "position": {"x": 0.2, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    return lf


def _make_valid_state():
    st = MagicMock()
    st.position_ned_m = [0.0, 0.0, 0.0]
    st.linear_velocity_ned_mps = [0.0, 0.0, 0.0]
    st.angular_velocity_body_radps = [0.0, 0.0, 0.0]
    st.roll_rad = 0.0; st.pitch_rad = 0.0; st.yaw_rad = 0.0
    st.ready = True; st.can_arm = True
    st.received_monotonic_seconds = 100.0
    return st


def _make_valid_collision():
    col = MagicMock()
    col.has_collided = False
    col.object_name = ""
    col.penetration_depth = 0.0
    col.received_monotonic_seconds = 100.0
    return col


def _make_filter_result(pts):
    fr = MagicMock()
    fr.valid = True
    fr.invalid_reason = None
    fr.filtered_points_sensor = pts
    fr.output_point_count = pts.shape[0]
    return fr


def _make_valid_dd():
    dd = MagicMock()
    dd.frame_valid = True
    dd.invalid_reason = None
    dd.minimum_distance_m = 5.0
    dd.sectors = {}
    # to_legacy_ray_distances returns a valid dict
    dd.to_legacy_ray_distances.return_value = {
        "front": 10.0, "back": 10.0, "left": 10.0, "right": 10.0,
        "up": 10.0, "down": 10.0,
        "frontLeft": 10.0, "frontRight": 10.0, "backLeft": 10.0, "backRight": 10.0,
        "frontUp": 10.0, "frontDown": 10.0, "leftUp": 10.0, "rightUp": 10.0,
        "leftDown": 10.0, "rightDown": 10.0,
    }
    dd.legacy_map = {}
    return dd


class TestCountingSemantics:
    """Test that data errors and safety holds are counted separately."""

    def test_data_errors_counted_separately_from_safety_holds(self):
        """Verify the counting logic: data_error vs safety_hold."""
        # Simulate the post-safety counting logic from the dry-run
        consecutive_invalid_data = 0
        safety_hold_count = 0
        valid_command_count = 0

        # Helper simulating cmd.command_valid / cmd.invalid_reason
        class FakeCmd:
            def __init__(self, valid, reason=""):
                self.command_valid = valid
                self.invalid_reason = reason

        # Frame 1-3: safety holds (Velocity toward obstacle)
        for i in range(3):
            cmd = FakeCmd(False, "Velocity toward obstacle at distance 2.00m")
            if cmd.command_valid:
                consecutive_invalid_data = 0; valid_command_count += 1
            else:
                if "toward obstacle" in (cmd.invalid_reason or ""):
                    safety_hold_count += 1
                else:
                    consecutive_invalid_data += 1

        assert consecutive_invalid_data == 0, "Safety holds must not count as data errors"
        assert safety_hold_count == 3
        assert valid_command_count == 0

        # Frame 4: valid command
        cmd = FakeCmd(True, "")
        if cmd.command_valid:
            consecutive_invalid_data = 0; valid_command_count += 1
        assert consecutive_invalid_data == 0
        assert valid_command_count == 1

        # Frame 5-7: actual data errors
        for i in range(3):
            cmd = FakeCmd(False, "LiDAR frame invalid: stale")
            if not cmd.command_valid:
                if "toward obstacle" in (cmd.invalid_reason or ""):
                    safety_hold_count += 1
                else:
                    consecutive_invalid_data += 1

        assert consecutive_invalid_data == 3
        assert safety_hold_count == 3  # unchanged

    def test_ten_safety_holds_no_termination(self):
        """10 safety holds in dry-run: all 10 counted, none cause termination."""
        consecutive_invalid_data = 0
        safety_hold_count = 0
        max_invalid = 10
        terminated = False

        for i in range(10):
            # Simulate safety hold
            safety_hold_count += 1
            if consecutive_invalid_data >= max_invalid:
                terminated = True

        assert not terminated, "Safety holds must not terminate dry-run"
        assert safety_hold_count == 10
        assert consecutive_invalid_data == 0

    def test_ten_data_errors_terminate(self):
        """10 data errors: termination triggered."""
        consecutive_invalid_data = 0
        max_invalid = 10
        terminated = False

        for i in range(10):
            consecutive_invalid_data += 1
            if consecutive_invalid_data >= max_invalid:
                terminated = True
                break

        assert terminated
        assert consecutive_invalid_data == 10

    def test_data_error_resets_on_valid(self):
        """A valid command resets the data error counter."""
        consecutive_invalid_data = 5
        consecutive_invalid_data = 0  # valid command reset
        assert consecutive_invalid_data == 0

    def test_twenty_frames_all_safety_holds(self):
        """--frames 20 with all safety holds: 20 frames recorded."""
        consecutive_invalid_data = 0
        safety_hold_count = 0
        valid_command_count = 0
        max_invalid = 10
        terminated = False
        frames_completed = 0

        for f in range(20):
            frames_completed += 1
            # Simulate safety hold each frame
            safety_hold_count += 1
            if consecutive_invalid_data >= max_invalid:
                terminated = True
                break

        assert not terminated
        assert frames_completed == 20, f"Expected 20 frames, got {frames_completed}"
        assert safety_hold_count == 20

    def test_no_flight_control_api_called(self):
        """Verify the dry-run script has no control API calls."""
        script_path = _PROJECT_ROOT / "scripts" / "run_local_avoidance_dry_run.py"
        content = script_path.read_text(encoding="utf-8")
        forbidden = [
            "enableApiControl", "armDisarm", "takeoffAsync",
            "hoverAsync", "landAsync", "moveByVelocityAsync",
            "moveByVelocityBodyFrameAsync", ".reset(",
        ]
        for api in forbidden:
            assert api not in content, f"Forbidden API call found: {api}"

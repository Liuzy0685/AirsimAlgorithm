from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from scripts import camera_service


class FakeAirSim:
    class ImageType:
        Scene = 0

    class Vector3r:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z

    class Pose:
        def __init__(self, position, orientation):
            self.position, self.orientation = position, orientation

    @staticmethod
    def to_eularian_angles(_orientation):
        return 0.0, 0.0, 0.25

    @staticmethod
    def to_quaternion(*angles):
        return angles


class FakeClient:
    def __init__(self):
        self.calls = []

    def simGetVehiclePose(self, vehicle_name):
        self.calls.append(("pose", vehicle_name))
        return SimpleNamespace(
            position=SimpleNamespace(x_val=1.0, y_val=2.0, z_val=-2.0),
            orientation=object(),
        )

    def simSetCameraPose(self, camera_name, pose, external=False):
        self.calls.append(("set_camera", camera_name, external, pose))

    def simGetImage(self, camera_name, image_type, vehicle_name="", external=False):
        self.calls.append(("image", camera_name, vehicle_name, external))
        return b"frame"


def test_third_person_uses_vehicle_specific_configured_camera(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(camera_service, "_import_airsim", lambda: FakeAirSim)
    hub = camera_service.CameraHub(client, ["Drone2"], camera_service.CameraConfig())

    assert hub._capture("Drone2", "third") == b"frame"
    assert ("set_camera", "Drone2Third", True) == client.calls[1][:3]
    assert ("image", "Drone2Third", "", True) == client.calls[2]


def test_first_person_uses_vehicle_camera_zero(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(camera_service, "_import_airsim", lambda: FakeAirSim)
    hub = camera_service.CameraHub(client, ["Drone1"], camera_service.CameraConfig())

    assert hub._capture("Drone1", "first") == b"frame"
    assert client.calls == [("image", "0", "Drone1", False)]


def test_missing_external_camera_is_rejected_before_capture():
    class SettingsClient:
        def __init__(self):
            self.client = SimpleNamespace(closed=False)
            self.client.close = lambda: setattr(self.client, "closed", True)

        def getSettingsString(self):
            return '{"ExternalCameras":{"Drone1Third":{}}}'

    client = SettingsClient()
    with pytest.raises(RuntimeError, match="Drone2Third"):
        camera_service.validate_external_camera_settings(client, ["Drone1", "Drone2"])
    assert client.client.closed


@pytest.mark.parametrize("settings_name,expected_count", [("settings_2.json", 2), ("settings_5.json", 5)])
def test_acceptance_external_cameras_have_finite_initial_poses(settings_name, expected_count):
    settings_path = Path(__file__).resolve().parents[1] / "configs" / "acceptance" / settings_name
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    cameras = settings["ExternalCameras"]

    assert len(cameras) == expected_count
    for camera in cameras.values():
        assert all(camera[axis] == value for axis, value in {
            "X": 0.0, "Y": 0.0, "Z": -5.0,
            "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0,
        }.items())
        assert camera["CaptureSettings"] == [
            {"ImageType": 0, "Width": 320, "Height": 240, "FOV_Degrees": 90}
        ]

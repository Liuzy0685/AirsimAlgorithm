from types import SimpleNamespace
import json
import math
from pathlib import Path
import asyncio

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


def test_third_person_applies_vehicle_spawn_offset_to_external_camera(monkeypatch):
    class SettingsClient(FakeClient):
        def getSettingsString(self):
            return json.dumps({
                "Vehicles": {"Drone2": {"X": 0.5, "Y": -3.0, "Z": 0.0, "Yaw": 0.0}}
            })

    client = SettingsClient()
    monkeypatch.setattr(camera_service, "_import_airsim", lambda: FakeAirSim)
    hub = camera_service.CameraHub(client, ["Drone2"], camera_service.CameraConfig())

    hub._capture("Drone2", "third")

    camera_pose = client.calls[1][3]
    assert camera_pose.position.x_val == pytest.approx(0.5 + 1.0 - 8.0 * math.cos(0.25))
    assert camera_pose.position.y_val == pytest.approx(-3.0 + 2.0 - 8.0 * math.sin(0.25))
    assert camera_pose.position.z_val == pytest.approx(-2.0 - 4.0)


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


def test_camera_capture_is_started_only_for_selected_view(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(camera_service, "_import_airsim", lambda: FakeAirSim)
    hub = camera_service.CameraHub(client, ["Drone1", "Drone2"], camera_service.CameraConfig())

    async def exercise():
        await hub.start(None)
        assert hub.status()["Drone1"]["first"]["status"] == "idle"
        await hub.set_active("Drone2", "third")
        assert hub.status()["Drone2"]["third"]["active"] is True
        assert hub.status()["Drone1"]["first"]["active"] is False
        await hub.set_active("Drone1", "first")
        assert hub.status()["Drone1"]["first"]["active"] is True
        assert hub.status()["Drone2"]["third"]["active"] is False
        await hub.set_active(mode="all")
        assert sum(item["active"] for views in hub.status().values() for item in views.values()) == 4
        await hub.stop(None)

    asyncio.run(exercise())


def test_camera_capture_supports_an_arbitrary_selected_set(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(camera_service, "_import_airsim", lambda: FakeAirSim)
    hub = camera_service.CameraHub(client, ["Drone1", "Drone2"], camera_service.CameraConfig())

    async def exercise():
        await hub.set_active(
            mode="custom",
            cameras=[
                {"vehicle": "Drone1", "view": "first"},
                {"vehicle": "Drone2", "view": "third"},
            ],
        )
        assert hub.status()["Drone1"]["first"]["active"] is True
        assert hub.status()["Drone2"]["third"]["active"] is True
        assert hub.status()["Drone1"]["third"]["active"] is False
        assert hub.status()["Drone2"]["first"]["active"] is False
        await hub.stop(None)

    asyncio.run(exercise())


def test_camera_custom_selection_requires_at_least_one_camera():
    hub = camera_service.CameraHub(FakeClient(), ["Drone1"], camera_service.CameraConfig())

    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(hub.set_active(mode="custom", cameras=[]))


def test_camera_fps_can_be_changed_at_runtime():
    client = FakeClient()
    hub = camera_service.CameraHub(client, ["Drone1"], camera_service.CameraConfig())

    async def exercise():
        await hub.set_active("Drone1", "first", fps=10)
        assert hub.config.interval_s == pytest.approx(0.1)
        assert camera_service._fps_from_interval(hub.config.interval_s) == 10
        await hub.set_active("Drone1", "first", fps=1)
        assert hub.config.interval_s == pytest.approx(1.0)
        assert camera_service._fps_from_interval(hub.config.interval_s) == 1
        await hub.stop(None)

    asyncio.run(exercise())


def test_camera_fps_rejects_unsupported_values():
    hub = camera_service.CameraHub(FakeClient(), ["Drone1"], camera_service.CameraConfig())

    with pytest.raises(ValueError, match="one of"):
        asyncio.run(hub.set_active("Drone1", "first", fps=7))


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

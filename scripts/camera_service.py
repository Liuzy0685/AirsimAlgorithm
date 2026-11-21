"""Serve low-rate read-only AirSim camera streams for the C dashboard.

The task API deliberately has no video endpoint.  This service is an
independent observer: it reads attached vehicle cameras for first-person
views and repositions one AirSim external camera for cached third-person
views.  It never calls takeoff, land, move, arm, disarm, or API-control APIs.

Run, for example::

    python scripts/camera_service.py --vehicles Drone1 Drone2

The dashboard expects ``/stream/{vehicle}/first`` and
``/stream/{vehicle}/third``. First-person streams use the vehicle camera named
``0``. Third-person streams use a separately configured external camera named
``{vehicle}Third`` (for example, ``Drone1Third``). Frames are intentionally
low-rate; configure the camera resolution in AirSim settings rather than
resizing in the browser.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("camera_service")


def validate_external_camera_settings(client: Any, vehicles: list[str]) -> None:
    """Refuse capture before RPC calls if an external camera is not configured."""
    runtime_settings = json.loads(client.getSettingsString())
    external_cameras = runtime_settings.get("ExternalCameras", {})
    missing = [f"{vehicle}Third" for vehicle in vehicles
               if f"{vehicle}Third" not in external_cameras]
    if missing:
        raw_client = getattr(client, "client", None)
        if raw_client is not None:
            raw_client.close()
        raise RuntimeError(
            "AirSim settings are missing ExternalCameras entries: "
            + ", ".join(missing)
            + ". Restart AirSim with the matching shared settings before starting camera capture."
        )


@dataclass(frozen=True)
class CameraConfig:
    host: str = "127.0.0.1"
    port: int = 8780
    interval_s: float = 1.0
    camera_name: str = "0"
    third_distance_m: float = 8.0
    third_height_m: float = 4.0


def _import_airsim() -> Any:
    candidate = os.environ.get("AIRSIM_PYTHONCLIENT_PATH", "")
    if candidate and candidate not in sys.path:
        sys.path.insert(0, candidate)
    try:
        import airsim  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import airsim; set AIRSIM_PYTHONCLIENT_PATH to the AirSim PythonClient directory"
        ) from exc
    return airsim


class CameraHub:
    """Capture and cache two observer views per configured vehicle."""

    def __init__(self, client: Any, vehicles: list[str], config: CameraConfig) -> None:
        if not vehicles:
            raise ValueError("at least one vehicle is required")
        if config.interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.client = client
        self.vehicles = tuple(dict.fromkeys(vehicles))
        self.config = config
        self.lock = threading.Lock()
        self.latest: dict[tuple[str, str], dict[str, Any]] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.stop_event = asyncio.Event()

    def _capture(self, vehicle: str, view: str) -> bytes:
        airsim = _import_airsim()
        with self.lock:
            if view == "first":
                image = self.client.simGetImage(
                    self.config.camera_name, airsim.ImageType.Scene, vehicle_name=vehicle
                )
            elif view == "third":
                pose = self.client.simGetVehiclePose(vehicle_name=vehicle)
                pitch, _roll, yaw = airsim.to_eularian_angles(pose.orientation)
                del pitch
                distance = self.config.third_distance_m
                height = self.config.third_height_m
                camera_x = pose.position.x_val - distance * math.cos(yaw)
                camera_y = pose.position.y_val - distance * math.sin(yaw)
                camera_z = pose.position.z_val - height
                # AirSim uses NED coordinates: an object above the camera has
                # a smaller Z value, while positive pitch points the camera
                # down toward the target.  The previous negative pitch made
                # the third-person camera look over the drone.
                look_pitch = math.atan2(height, distance)
                orientation = airsim.to_quaternion(look_pitch, 0.0, yaw)
                external_pose = airsim.Pose(
                    airsim.Vector3r(camera_x, camera_y, camera_z), orientation
                )
                external_camera_name = f"{vehicle}Third"
                self.client.simSetCameraPose(
                    external_camera_name, external_pose, external=True
                )
                image = self.client.simGetImage(
                    external_camera_name,
                    airsim.ImageType.Scene,
                    external=True,
                )
            else:
                raise ValueError(f"unsupported camera view: {view}")
        if not image:
            raise RuntimeError(f"AirSim returned no image for {vehicle}/{view}")
        return bytes(image)

    async def capture_loop(self, vehicle: str, view: str) -> None:
        while not self.stop_event.is_set():
            started = time.time()
            try:
                image = await asyncio.to_thread(self._capture, vehicle, view)
                self.latest[(vehicle, view)] = {
                    "image": image,
                    "captured_at_unix_ms": int(time.time() * 1000),
                    "status": "online",
                    "error": "",
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Keep one broken camera from stopping others.
                LOGGER.warning("capture failed for %s/%s: %s", vehicle, view, exc)
                self.latest[(vehicle, view)] = {
                    "image": None,
                    "captured_at_unix_ms": int(time.time() * 1000),
                    "status": "offline",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            delay = max(0.0, self.config.interval_s - (time.time() - started))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def start(self, _app: web.Application) -> None:
        self.stop_event.clear()
        self.tasks = [
            asyncio.create_task(self.capture_loop(vehicle, view))
            for vehicle in self.vehicles
            for view in ("first", "third")
        ]

    async def stop(self, _app: web.Application) -> None:
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    def status(self) -> dict[str, Any]:
        result = {}
        for vehicle in self.vehicles:
            result[vehicle] = {}
            for view in ("first", "third"):
                item = self.latest.get((vehicle, view), {})
                result[vehicle][view] = {
                    "status": item.get("status", "starting"),
                    "captured_at_unix_ms": item.get("captured_at_unix_ms"),
                    "error": item.get("error", ""),
                }
        return result


def create_app(hub: CameraHub) -> web.Application:
    app = web.Application()

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"service": "camera", "vehicles": list(hub.vehicles), "views": ["first", "third"]})

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(
            hub.status(), headers={"Access-Control-Allow-Origin": "http://127.0.0.1:8000"}
        )

    async def stream(request: web.Request) -> web.StreamResponse:
        vehicle = request.match_info["vehicle"]
        view = request.match_info["view"]
        if vehicle not in hub.vehicles or view not in {"first", "third"}:
            raise web.HTTPNotFound(text="unknown camera")
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
        await response.prepare(request)
        try:
            while True:
                item = hub.latest.get((vehicle, view))
                image = item.get("image") if item else None
                if image:
                    await response.write(
                        b"--frame\r\nContent-Type: image/png\r\nContent-Length: "
                        + str(len(image)).encode("ascii")
                        + b"\r\nX-Captured-At-Unix-Ms: "
                        + str(item.get("captured_at_unix_ms", 0)).encode("ascii")
                        + b"\r\n\r\n"
                        + image
                        + b"\r\n"
                    )
                await asyncio.sleep(hub.config.interval_s)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass
        return response

    async def frame(request: web.Request) -> web.Response:
        """Return the latest single PNG frame for still-image previews."""
        vehicle = request.match_info["vehicle"]
        view = request.match_info["view"]
        if vehicle not in hub.vehicles or view not in {"first", "third"}:
            raise web.HTTPNotFound(text="unknown camera")
        item = hub.latest.get((vehicle, view))
        image = item.get("image") if item else None
        if not image:
            raise web.HTTPServiceUnavailable(text="camera frame is not ready")
        return web.Response(
            body=image,
            content_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "http://127.0.0.1:8000",
                "X-Captured-At-Unix-Ms": str(item.get("captured_at_unix_ms", 0)),
            },
        )

    app.router.add_get("/health", health)
    app.router.add_get("/api/cameras", status)
    app.router.add_get("/stream/{vehicle}/{view}", stream)
    app.router.add_get("/frame/{vehicle}/{view}", frame)
    app.on_startup.append(hub.start)
    app.on_cleanup.append(hub.stop)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicles", nargs="+", help="AirSim vehicle names; default is client.listVehicles()")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--interval", type=float, default=1.0, help="capture interval in seconds")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--third-distance", type=float, default=8.0)
    parser.add_argument("--third-height", type=float, default=4.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    airsim = _import_airsim()
    client = airsim.MultirotorClient(timeout_value=3)
    client.confirmConnection()
    vehicles = args.vehicles or list(client.listVehicles())
    validate_external_camera_settings(client, vehicles)
    hub = CameraHub(
        client,
        vehicles,
        CameraConfig(
            host=args.host,
            port=args.port,
            interval_s=args.interval,
            camera_name=args.camera_name,
            third_distance_m=args.third_distance,
            third_height_m=args.third_height,
        ),
    )
    web.run_app(create_app(hub), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Serve low-rate read-only AirSim camera streams for the C dashboard.

The task API deliberately has no video endpoint.  This service is an
independent observer: it reads attached vehicle cameras for first-person
views and repositions one AirSim external camera for cached third-person
views.  It never calls takeoff, land, move, arm, disarm, or API-control APIs.

Run, for example::

    python scripts/camera_service.py --vehicles Drone1 Drone2

The dashboard uses ``/frame/{vehicle}/{view}`` for the selected view and
``/stream/{vehicle}/third``. First-person streams use the vehicle camera named
``0``. Third-person streams use a separately configured external camera named
``{vehicle}Third`` (for example, ``Drone1Third``). Capture is started on
demand by ``POST /api/cameras/active``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
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
SUPPORTED_FPS = (1, 5, 10, 20)
DEFAULT_FPS = 5


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
    # Five FPS is a reasonable observer load for the 320x240 PNG frames.
    interval_s: float = 0.2
    camera_name: str = "0"
    third_distance_m: float = 8.0
    third_height_m: float = 4.0


def _fps_from_interval(interval_s: float) -> int:
    """Return the nearest supported display FPS for status reporting."""
    return min(SUPPORTED_FPS, key=lambda fps: abs((1.0 / interval_s) - fps))


def _interval_for_fps(fps: Any) -> float:
    try:
        value = int(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fps must be one of {SUPPORTED_FPS}") from exc
    if value not in SUPPORTED_FPS:
        raise ValueError(f"fps must be one of {SUPPORTED_FPS}")
    return 1.0 / value


def _load_vehicle_origins(client: Any, vehicles: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Read each vehicle's common NED spawn offset and initial yaw.

    ``simGetVehiclePose`` is relative to the vehicle's local NED origin,
    while external cameras use the shared world NED frame.  Unit-test fakes
    may not expose ``getSettingsString``; in that case the zero origin keeps
    the local-frame behavior used by the fake client.
    """
    getter = getattr(client, "getSettingsString", None)
    if not callable(getter):
        return {vehicle: (0.0, 0.0, 0.0, 0.0) for vehicle in vehicles}
    try:
        settings = json.loads(getter())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AirSim returned invalid settings JSON") from exc
    configured = settings.get("Vehicles", {})
    origins: dict[str, tuple[float, float, float, float]] = {}
    for vehicle in vehicles:
        item = configured.get(vehicle, {})
        try:
            origins[vehicle] = (
                float(item.get("X", 0.0)),
                float(item.get("Y", 0.0)),
                float(item.get("Z", 0.0)),
                math.radians(float(item.get("Yaw", 0.0))),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"AirSim settings contain invalid spawn pose for {vehicle}") from exc
    return origins


def _common_vehicle_pose(airsim: Any, pose: Any, origin: tuple[float, float, float, float]) -> tuple[Any, float]:
    """Convert a local vehicle pose to the shared World NED frame."""
    origin_x, origin_y, origin_z, origin_yaw = origin
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    local_x = float(pose.position.x_val)
    local_y = float(pose.position.y_val)
    common_position = airsim.Vector3r(
        origin_x + cos_yaw * local_x - sin_yaw * local_y,
        origin_y + sin_yaw * local_x + cos_yaw * local_y,
        origin_z + float(pose.position.z_val),
    )
    _pitch, _roll, local_yaw = airsim.to_eularian_angles(pose.orientation)
    return common_position, origin_yaw + local_yaw


def _import_airsim() -> Any:
    candidate = os.environ.get("AIRSIM_PYTHONCLIENT_PATH", "")
    if candidate and candidate not in sys.path:
        sys.path.insert(0, candidate)
    try:
        # AirSim's bundled msgpack-rpc client still passes the removed
        # ``encoding`` argument on modern msgpack releases.  The normal
        # AirSim adapter applies this patch before importing AirSim; this
        # standalone read-only service must do the same.
        from adapters import msgpack_compat  # noqa: F401
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
        self.vehicle_origins = _load_vehicle_origins(client, list(self.vehicles))
        self.lock = threading.Lock()
        self.latest: dict[tuple[str, str], dict[str, Any]] = {}
        self.tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
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
                common_position, yaw = _common_vehicle_pose(
                    airsim, pose, self.vehicle_origins[vehicle]
                )
                distance = self.config.third_distance_m
                height = self.config.third_height_m
                camera_x = common_position.x_val - distance * math.cos(yaw)
                camera_y = common_position.y_val - distance * math.sin(yaw)
                camera_z = common_position.z_val - height
                # AirSim uses NED coordinates: an object above the camera has
                # a smaller Z value.  AirSim's positive pitch rotates the
                # optical axis toward smaller NED Z, so the camera must use a
                # negative pitch to look down at the vehicle.
                look_pitch = -math.atan2(height, distance)
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
        self.tasks.clear()

    async def set_active(
        self,
        vehicle: str = "",
        view: str = "",
        mode: str = "selected",
        fps: Any = None,
        cameras: Any = None,
    ) -> None:
        if fps is not None:
            self.config = replace(self.config, interval_s=_interval_for_fps(fps))
        if mode == "all":
            selected = {(name, item_view) for name in self.vehicles for item_view in ("first", "third")}
        elif mode == "custom":
            if not isinstance(cameras, list) or not cameras:
                raise ValueError("custom mode requires at least one camera")
            selected = set()
            for item in cameras:
                if not isinstance(item, dict):
                    raise ValueError("each camera must contain vehicle and view")
                item_vehicle = str(item.get("vehicle", ""))
                item_view = str(item.get("view", ""))
                if item_vehicle not in self.vehicles or item_view not in {"first", "third"}:
                    raise ValueError("unknown camera selection")
                selected.add((item_vehicle, item_view))
        elif mode == "selected" and vehicle in self.vehicles and view in {"first", "third"}:
            selected = {(vehicle, view)}
        else:
            raise ValueError("unknown camera selection")
        for key, task in list(self.tasks.items()):
            if key in selected:
                continue
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.tasks.pop(key, None)
        for vehicle_name, item_view in selected:
            task = self.tasks.get((vehicle_name, item_view))
            if task is None or task.done():
                self.tasks[(vehicle_name, item_view)] = asyncio.create_task(self.capture_loop(vehicle_name, item_view))

    async def stop(self, _app: web.Application) -> None:
        self.stop_event.set()
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()

    def status(self) -> dict[str, Any]:
        result = {}
        for vehicle in self.vehicles:
            result[vehicle] = {}
            for view in ("first", "third"):
                item = self.latest.get((vehicle, view), {})
                active = (vehicle, view) in self.tasks and not self.tasks[(vehicle, view)].done()
                result[vehicle][view] = {
                    "status": item.get("status", "starting") if active else "idle",
                    "active": active,
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
            hub.status(), headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:8000",
                "Access-Control-Expose-Headers": "X-Camera-Fps",
                "X-Camera-Fps": str(_fps_from_interval(hub.config.interval_s)),
            }
        )

    async def active_options(_request: web.Request) -> web.Response:
        return web.Response(status=204, headers={
            "Access-Control-Allow-Origin": "http://127.0.0.1:8000",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    async def active(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            mode = str(body.get("mode", "selected"))
            vehicle = str(body.get("vehicle", ""))
            view = str(body.get("view", ""))
            fps = body.get("fps")
            cameras = body.get("cameras")
            await hub.set_active(vehicle, view, mode=mode, fps=fps, cameras=cameras)
        except (ValueError, AttributeError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except Exception as exc:
            raise web.HTTPBadRequest(text=f"invalid camera selection: {exc}") from exc
        return web.json_response(
            {"active": {"mode": mode, "vehicle": vehicle, "view": view,
                         "cameras": cameras if mode == "custom" else None,
                         "fps": _fps_from_interval(hub.config.interval_s)}},
            headers={"Access-Control-Allow-Origin": "http://127.0.0.1:8000"},
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
    app.router.add_options("/api/cameras/active", active_options)
    app.router.add_post("/api/cameras/active", active)
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
    parser.add_argument("--interval", type=float, default=1.0 / DEFAULT_FPS, help="initial capture interval in seconds (default: 0.2, about 5 FPS)")
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

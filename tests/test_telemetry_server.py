import asyncio

from aiohttp.test_utils import TestClient, TestServer

from telemetry.server import TelemetryGateway, create_app, load_config


def test_config_loads_and_gateway_starts_offline():
    config = load_config()
    gateway = TelemetryGateway(config)
    frame = gateway.current_frame()
    assert config.frontend_host == "127.0.0.1"
    assert frame["status"] == "OFFLINE"
    assert frame["drones"] == {}


def test_http_dashboard_and_browser_websocket_offline_state():
    async def scenario():
        client = TestClient(TestServer(create_app()))
        await client.start_server()
        try:
            page = await client.get("/")
            assert page.status == 200
            assert "大创无人机任务控制台" in await page.text()
            socket = await client.ws_connect("/ws/telemetry")
            initial = await socket.receive_json()
            assert initial["type"] == "telemetry"
            assert initial["status"] == "OFFLINE"
            await socket.close()
        finally:
            await client.close()

    asyncio.run(scenario())

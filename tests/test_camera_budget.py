import asyncio
import threading
import time

import pytest

from scripts.camera_service import CameraConfig, CameraHub


def hub(**kwargs):
    return CameraHub(object(),['Drone1','Drone2','Drone3','Drone4','Drone5'],CameraConfig(**kwargs))


def test_ten_selected_views_share_a_global_budget_and_report_throttling():
    async def run():
        camera=hub(max_total_fps=10.)
        await camera.set_active(mode='all',fps=20)
        try:
            status=camera.status()
            assert status['Drone1']['first']['requested_fps']==20
            assert status['Drone1']['first']['effective_fps_limit']<=1.
            assert status['Drone1']['first']['throttled']
        finally:await camera.stop(None)
    asyncio.run(run())


def test_slow_camera_rpc_reduces_the_capture_duty_cycle():
    camera=hub(max_total_fps=10.,capture_utilization=.5)
    camera.capture_ema_s=.3
    assert camera.global_capture_period()==pytest.approx(.6)


def test_rpc_work_is_serialized_without_a_thread_backlog(monkeypatch):
    async def run():
        camera=hub(max_total_fps=20.,capture_utilization=1.)
        active=0;maximum=0;starts=[]
        def capture(*args):
            nonlocal active,maximum
            active+=1;maximum=max(maximum,active);starts.append(time.monotonic())
            time.sleep(.01);active-=1
            return b'frame'
        monkeypatch.setattr(camera,'_capture',capture)
        await asyncio.gather(*(camera.budgeted_capture('Drone1','first') for _ in range(3)))
        assert maximum==1
        assert all(b-a>=.035 for a,b in zip(starts,starts[1:]))
    asyncio.run(run())


def test_cancelled_view_keeps_gate_until_its_inflight_read_finishes(monkeypatch):
    async def run():
        camera=hub(max_total_fps=0.,capture_utilization=1.)
        started=threading.Event();release=threading.Event();calls=[]
        def capture(*args):
            calls.append(args)
            if len(calls)==1:
                started.set();release.wait(timeout=2.)
            return b'frame'
        monkeypatch.setattr(camera,'_capture',capture)
        first=asyncio.create_task(camera.budgeted_capture('Drone1','first'))
        try:
            for _ in range(100):
                if started.is_set():break
                await asyncio.sleep(.005)
            assert started.is_set()
            first.cancel()
            second=asyncio.create_task(camera.budgeted_capture('Drone2','first'))
            await asyncio.sleep(.02)
            assert len(calls)==1
            release.set()
            with pytest.raises(asyncio.CancelledError):await first
            assert await second==b'frame'
        finally:release.set()
    asyncio.run(run())


@pytest.mark.parametrize('values',[dict(max_total_fps=-1),dict(max_total_fps=float('nan')),
                                   dict(capture_utilization=0),dict(capture_utilization=2)])
def test_invalid_budgets_are_rejected(values):
    with pytest.raises(ValueError):hub(**values)

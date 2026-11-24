"""Validate the public read-only CLI in an owned, landed instance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from scripts.run_coordination_suite import hidden,port_open
spec=importlib.util.spec_from_file_location('owned_ab',ROOT/'runs/live_navigation_ab_20261123_01/run_live_ab.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
OUT=Path(__file__).resolve().parent/'ground_runtime_cli';OUT.mkdir(exist_ok=False)
assert not any(port_open(p) for p in (41451,8765,8780))
command=[str(h.UE),str(h.PROJECT),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game',
    '-settings='+str(ROOT/'configs/acceptance/settings_5.json'),'-windowed','-ResX=960','-ResY=540','-NoSound','-unattended','-abslog='+str(OUT/'ue.log')]
with (OUT/'stdout.log').open('w',encoding='utf-8') as log:ue=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,**hidden())
client=None;ground=False;report=dict(mode='read_only_no_flight_no_api_acquisition',ue_pid=ue.pid)
try:
    deadline=time.monotonic()+90
    while not port_open(41451):
        if ue.poll() is not None or time.monotonic()>deadline:raise RuntimeError('startup failed')
        time.sleep(.2)
    AirSimClientAdapter._ensure_airsim_importable();import airsim
    client=airsim.MultirotorClient(timeout_value=20)
    names=[f'Drone{i}' for i in range(1,6)]
    for _ in range(5):
        check=h.ground_check(client,names)
        if check['verified_ground_released']:break
        time.sleep(1.)
    assert check['verified_ground_released'];report['initial_ground']=check
    cmd=[sys.executable,'-X','utf8',str(ROOT/'scripts/check_runtime_goal_geometry.py'),
        '--blocks-project',str(h.PROJECT),'--engine-log',str(OUT/'ue.log'),
        '--settings-json','configs/acceptance/settings_5.json','--swarm-config','configs/acceptance/swarm_5_navigation.yaml',
        '--formation-config','configs/acceptance/formation_5.yaml','--map-manifest','configs/operator/flying_example.json',
        '--output',str(OUT/'runtime_goal_geometry.json')]
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=90)
    (OUT/'cli_stdout.txt').write_text(run.stdout,encoding='utf-8');(OUT/'cli_stderr.txt').write_text(run.stderr,encoding='utf-8')
    result=json.loads((OUT/'runtime_goal_geometry.json').read_text(encoding='utf-8'))
    report.update(cli_returncode=run.returncode,runtime=result)
    assert run.returncode==2 and result['active_map_verified'] and result['render_aabb_matches_manifest'] and result['inside_render_mesh'] is True
finally:
    if client:
        check=h.ground_check(client,[f'Drone{i}' for i in range(1,6)]);ground=check['verified_ground_released'];report['final_ground']=check
        if ground:client.simRunConsoleCommand('quit')
        client.client.close()
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if ground:ue.wait(timeout=25)
    print(json.dumps({k:v for k,v in report.items() if k not in ('initial_ground','final_ground','runtime')}))

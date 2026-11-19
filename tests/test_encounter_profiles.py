import json
import math
from types import SimpleNamespace
import pytest
from coordination.runtime_config import load_config
from scripts.build_encounter_configs import build


def test_encounter_profiles_have_separate_settings_and_all_safe_terminal_spacing(tmp_path):
    profiles=build(tmp_path/'profiles')
    for name,profile in profiles.items():
        config,cfg=load_config(profile['swarm_config'])
        settings=json.loads(open(profile['settings_json'],encoding='utf-8').read())
        assert config['navigation']['independent_goals'] and not config['navigation']['adaptive_slots']
        assert cfg.enabled and cfg.task_id=='encounter-'+name
        assert len(settings['Vehicles'])==len(config['drones'])
        for i in range(len(profile['goals'])):
            for j in range(i):
                assert math.dist(profile['goals'][i],profile['goals'][j])>=2.3
                assert math.dist(profile['starts'][i],profile['starts'][j])>=2.3
        for member,goal in zip(config['drones'],profile['goals']):
            spawn=settings['Vehicles'][member['vehicle_name']]
            assert member['goal_x']+spawn['X']==pytest.approx(goal[0])
            assert member['goal_y']+spawn['Y']==pytest.approx(goal[1])


def test_owned_suite_refuses_existing_rpc_before_spawning(tmp_path,monkeypatch):
    from scripts import run_coordination_suite as suite
    monkeypatch.setattr(suite,'port_open',lambda port:port==41451)
    def forbidden(*a,**k):raise AssertionError('Must not launch over an existing simulator')
    monkeypatch.setattr(suite.subprocess,'Popen',forbidden)
    with pytest.raises(RuntimeError,match='Existing service'):
        suite.run(SimpleNamespace(output=tmp_path/'suite'))

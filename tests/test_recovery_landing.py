"""Faults must never turn an accepted recovery request into a false cleanup."""
import json
from types import SimpleNamespace as NS
import pytest

from control.recovery_landing import RecoveryLanding


def vec(x=0.,y=0.,z=0.):return NS(x_val=x,y_val=y,z_val=z)


@pytest.fixture
def recovery(monkeypatch):
    clock=NS(now=1.)
    settings={'Vehicles':{'Drone1':dict(X=0.,Y=0.,Z=0.),'Drone2':dict(X=0.,Y=3.,Z=0.)}}
    profile=NS(names=tuple(settings['Vehicles']),settings=settings,settings_path='unused',safety_distance=2.,
               service=NS(host='127.0.0.1',port=12345,snapshot_max_age_s=.6))
    class Client:
        def __init__(self):
            self.calls=[];self.landed={n:1 for n in profile.names};self.api={n:True for n in profile.names}
            self.stamp=100;self.freeze=False;self.collision='Ground_5';self.new_wall=False;self.release_ok=True
        def listVehicles(self):return list(profile.names)
        def getSettingsString(self):return json.dumps(settings)
        def simIsPause(self):return False
        def getMultirotorState(self,vehicle_name):
            if not self.freeze:self.stamp+=1
            return NS(timestamp=self.stamp,landed_state=self.landed[vehicle_name],
                kinematics_estimated=NS(position=vec(0.,0.,1.1 if self.landed[vehicle_name]==0 else -2.),linear_velocity=vec()))
        def simGetCollisionInfo(self,vehicle_name):
            wall=self.new_wall and ('land',vehicle_name) in self.calls
            return NS(has_collided=True,time_stamp=2 if wall else 1,
                      object_name='Wall' if wall else self.collision,normal=vec(0.,0.,-1.),penetration_depth=0.)
        def isApiControlEnabled(self,vehicle_name):return self.api[vehicle_name]
        def enableApiControl(self,enabled,vehicle_name):
            self.calls.append(('api',vehicle_name,enabled))
            if enabled or self.release_ok:self.api[vehicle_name]=enabled
        def landAsync(self,vehicle_name,timeout_sec):
            self.calls.append(('land',vehicle_name));self.landed[vehicle_name]=0
        def armDisarm(self,value,vehicle_name):self.calls.append(('arm',vehicle_name,value));return True
        def cancelLastTask(self,vehicle_name):self.calls.append(('cancel',vehicle_name))
    c=Client();locks=[];released=[]
    def acquire(mode,name):
        lock=NS(closed=False,name=name);locks.append(lock);return lock
    def release(lock):lock.closed=True;released.append(lock.name)
    monkeypatch.setattr('coordination.operator_map_runtime.verify_map',lambda p,c:None)
    monkeypatch.setattr('coordination.coordinate_transform.CoordinateTransformer.from_settings_json',
                        lambda *a:NS(validate=lambda c:NS(validated=True)))
    monkeypatch.setattr('control.recovery_landing.socket.create_connection',
                        lambda *a,**kw:(_ for _ in ()).throw(ConnectionRefusedError()))
    def make(**kw):
        return RecoveryLanding(profile,c,clock=lambda:clock.now,sleep=lambda t:setattr(clock,'now',clock.now+t),
            lock_acquire=kw.pop('lock_acquire',acquire),lock_release=release,timeout_s=kw.pop('timeout_s',2.),**kw)
    return NS(c=c,p=profile,make=make,locks=locks,released=released)


def test_landing_then_disarm_then_release_with_no_takeoff_or_arming(recovery):
    e=recovery;report=e.make().run()
    assert report['cleanup_verified']
    assert set(e.released)==set(e.p.names)
    for n in e.p.names:
        assert e.c.calls.index(('land',n))<e.c.calls.index(('arm',n,False))<e.c.calls.index(('api',n,False))
        assert report['vehicle_results'][n]['landing_confirmed']
    assert not any(c[0]=='arm' and c[2] is True for c in e.c.calls)


def test_all_vehicle_locks_required_before_any_write(recovery):
    e=recovery
    def acquire(mode,name):return None if name=='Drone2' else NS(closed=False,name=name)
    result=e.make(lock_acquire=acquire).run()
    assert not result['cleanup_verified'] and 'controller_busy' in result['reason']
    assert not e.c.calls
    assert e.released==['Drone1']


def test_new_collision_cancels_landing_and_preserves_api_control(recovery):
    e=recovery;e.c.new_wall=True
    result=e.make().run()
    assert not result['cleanup_verified'] and 'recovery_collision' in result['reason']
    assert ('cancel','Drone1') in e.c.calls
    assert not any(c[0]=='arm' for c in e.c.calls)
    assert all(e.c.api.values())


def test_new_collision_timestamp_is_checked_even_after_one_frame_pulse(recovery):
    e=recovery;e.c.new_wall=True;read=e.c.simGetCollisionInfo
    def collision(**kw):
        c=read(**kw)
        if c.object_name=='Wall':c.has_collided=False
        return c
    e.c.simGetCollisionInfo=collision
    assert not e.make().run()['cleanup_verified']
    assert ('cancel','Drone1') in e.c.calls
    assert not any(c[0]=='arm' for c in e.c.calls)


def test_old_wall_contact_prevents_any_control_write(recovery):
    e=recovery;e.c.collision='Wall'
    assert not e.make().run()['cleanup_verified']
    assert not e.c.calls


def test_frozen_state_cannot_confirm_landing(recovery):
    e=recovery;e.c.freeze=True
    result=e.make().run()
    assert not result['cleanup_verified'] and 'stale' in result['reason']
    assert not any(c[0]=='arm' for c in e.c.calls)


def test_failed_api_release_is_not_cleanup_success(recovery):
    e=recovery;e.c.release_ok=False
    result=e.make().run()
    assert not result['cleanup_verified']
    assert result['vehicle_results']['Drone1']['landing_confirmed']
    assert not result['vehicle_results']['Drone1']['api_control_released']


def test_intersecting_descent_columns_are_rejected(recovery):
    e=recovery;e.p.settings['Vehicles']['Drone2']['Y']=1.
    assert 'recovery_separation' in e.make().run()['reason']
    assert not e.c.calls


def test_known_ground_at_rest_can_be_cleaned_without_land_command(recovery):
    e=recovery;e.c.landed={n:0 for n in e.p.names}
    assert e.make().run()['cleanup_verified']
    assert not any(c[0]=='land' for c in e.c.calls)


def test_lost_os_lock_or_revoked_lease_rejects_next_write(recovery):
    e=recovery;worker=e.make()
    original=e.c.landAsync
    def land(**kw):
        original(**kw)
        worker.authority.release('Drone1',worker.controller_id)
    e.c.landAsync=land
    report=worker.run()
    assert not report['cleanup_verified']
    assert not any(c[0]=='arm' for c in e.c.calls)


def test_landing_rechecked_at_the_disarm_boundary(recovery):
    e=recovery;worker=e.make();get_state=e.c.getMultirotorState
    def state(**kw):
        s=get_state(**kw)
        if kw['vehicle_name'] in worker.confirmed:
            s.landed_state=1;s.kinematics_estimated.position=vec(0,0,-2.)
        return s
    e.c.getMultirotorState=state
    assert not worker.run()['cleanup_verified']
    assert not any(c[0]=='arm' for c in e.c.calls)


@pytest.mark.parametrize('pulse',[True,False])
def test_new_ground_contact_waits_for_height_velocity_and_dwell_when_sdk_flag_lags(recovery,pulse):
    e=recovery;original_state=e.c.getMultirotorState;original_collision=e.c.simGetCollisionInfo
    def state(**kw):
        s=original_state(**kw);s.landed_state=1;return s
    def collision(**kw):
        c=original_collision(**kw)
        if e.c.landed[kw['vehicle_name']]==0:c.time_stamp=2;c.has_collided=pulse
        return c
    e.c.getMultirotorState=state;e.c.simGetCollisionInfo=collision
    refs={n:dict(landed_state=0,position_ned_m=[0.,e.p.settings['Vehicles'][n]['Y'],1.1]) for n in e.p.names}
    result=e.make(timeout_s=5.,ground_reference=refs).run()
    assert result['cleanup_verified'],result
    assert all(r['landing_confirmation']=='verified_ground_contact' for r in result['vehicle_results'].values())


@pytest.mark.parametrize('bad',['normal','depth','height','missing_reference'])
def test_lagging_landed_flag_never_hides_bad_new_ground_event(recovery,bad):
    e=recovery;get_state=e.c.getMultirotorState;get_col=e.c.simGetCollisionInfo
    def state(**kw):s=get_state(**kw);s.landed_state=1;return s
    def collision(**kw):
        c=get_col(**kw)
        if e.c.landed[kw['vehicle_name']]==0:
            c.time_stamp=2
            if bad=='normal':c.normal=vec(1.,0.,0.)
            if bad=='depth':c.penetration_depth=.021
        return c
    e.c.getMultirotorState=state;e.c.simGetCollisionInfo=collision
    refs={n:dict(landed_state=0,position_ned_m=[0.,e.p.settings['Vehicles'][n]['Y'],1.1 if bad!='height' else 4.]) for n in e.p.names}
    if bad=='missing_reference':refs={}
    result=e.make(timeout_s=5.,ground_reference=refs).run()
    assert not result['cleanup_verified']
    assert not any(c[0]=='arm' for c in e.c.calls)

import pytest
from flight_modes.shared_flight_session import SessionPhase
from test_shared_flight_session import _make_session


@pytest.mark.parametrize('failure',['unconfirmed_landing','landing_rpc','disarm','release'])
def test_unverified_cleanup_is_explicit_manual_intervention_and_never_success(failure):
    session,client=_make_session()
    session.state.phase=SessionPhase.AIRBORNE
    if failure=='unconfirmed_landing':session._poll_landed_detailed.return_value=False
    if failure=='landing_rpc':
        client.landAsync.side_effect=ConnectionError('link lost')
        session._poll_landed_detailed.return_value=False
    if failure=='disarm':client.armDisarm.side_effect=ConnectionError('disarm unverified')
    if failure=='release':client.enableApiControl.side_effect=ConnectionError('release unverified')
    assert not session.land_and_disarm()
    assert session.state.phase==SessionPhase.MANUAL_INTERVENTION_REQUIRED
    if failure in ('unconfirmed_landing','landing_rpc'):
        client.armDisarm.assert_not_called()
        client.enableApiControl.assert_not_called()
    calls=client.landAsync.call_count
    assert not session.land_and_disarm()
    assert client.landAsync.call_count==calls


def test_new_session_cannot_adopt_api_control_left_by_unverified_cleanup():
    from flight_modes.shared_flight_session import SessionError
    session,client=_make_session()
    client.isApiControlEnabled.return_value=True
    with pytest.raises(SessionError,match='vehicle_api_already_enabled'):
        session.takeoff_and_climb()
    assert not session.takeoff_called
    client.enableApiControl.assert_not_called()
    client.armDisarm.assert_not_called()
    session.cleanup()
    client.enableApiControl.assert_not_called()

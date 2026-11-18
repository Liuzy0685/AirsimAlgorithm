from dataclasses import replace
import pytest
from control.ground_contact import is_stationary_ground_contact
from models.collision_state import CollisionState
from models.vehicle_state import VehicleState


def fixture():
    return (CollisionState(has_collided=True,object_name="(null)",normal_ned=[0.,0.,-1.],raw_timestamp=123),
            VehicleState(landed_state=0,position_ned_m=[0,0,1.1]))


def test_only_stationary_landed_horizontal_contact_is_a_baseline():
    col,state=fixture()
    assert is_stationary_ground_contact(col,state)
    assert not is_stationary_ground_contact(col,replace(state,landed_state=1))
    assert not is_stationary_ground_contact(col,replace(state,linear_velocity_ned_mps=[.1,0,0]))


@pytest.mark.parametrize("changes", [dict(raw_timestamp=0),dict(normal_ned=[1.,0.,0.]),
    dict(penetration_depth=.1),dict(object_name="Wall"),dict(normal_ned=[0.,0.,float('nan')])])
def test_suspicious_contacts_are_not_hidden(changes):
    col,state=fixture()
    assert not is_stationary_ground_contact(replace(col,**changes),state)

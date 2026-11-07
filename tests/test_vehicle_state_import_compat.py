"""Phase 1 — VehicleState import compatibility.

``common.models`` must re-export the *same* ``VehicleState`` class as
``models.vehicle_state`` (single source of truth, no copy).  The existing
import path must keep working unchanged.
"""

from common.models import VehicleState as CommonVehicleState
from models.vehicle_state import VehicleState as ModelsVehicleState


def test_common_reexports_same_class_object():
    assert CommonVehicleState is ModelsVehicleState


def test_existing_import_path_still_works():
    state = ModelsVehicleState(
        position_ned_m=[1.0, 2.0, 3.0],
        linear_velocity_ned_mps=[0.0, 0.0, 0.0],
        yaw_rad=0.5,
    )
    assert state.position_ned_m == [1.0, 2.0, 3.0]
    assert state.yaw_rad == 0.5


def test_common_reexport_constructs_same_shape():
    state = CommonVehicleState(position_ned_m=[4.0, 5.0, -1.0])
    assert isinstance(state, ModelsVehicleState)
    assert state.position_ned_m == [4.0, 5.0, -1.0]


def test_vehicle_state_repr_is_stable():
    state = ModelsVehicleState(
        position_ned_m=[1.0, 2.0, 3.0],
        linear_velocity_ned_mps=[0.1, 0.2, 0.3],
        yaw_rad=0.0,
    )
    text = repr(state)
    assert "1.00" in text
    assert "pos=" in text

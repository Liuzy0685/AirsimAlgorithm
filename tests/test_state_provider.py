"""StateProvider tests; runnable with unittest or pytest, no AirSim server."""

import math
import unittest
from types import SimpleNamespace as NS

from coordination.coordinate_transform import (
    CoordinateTransformer, CoordinateTransformValidation, VehicleTransformCheck,
    yaw_rotation_matrix,
)
from coordination.state_provider import StateProvider
from models.vehicle_state import VehicleState


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class Reader:
    def __init__(self, state=None, error=None, clock=None, delay=0):
        self.state, self.error, self.clock, self.delay = state, error, clock, delay

    def read(self):
        if self.clock:
            self.clock.now += self.delay
        if self.error:
            raise self.error
        return self.state


def state(timestamp=1, position=(1, 2, -3), velocity=(1, 0, 0), ready=True):
    return VehicleState(position_ned_m=list(position), linear_velocity_ned_mps=list(velocity),
                        timestamp=timestamp, ready=ready)


def provider(count=2, *, clock=None, readers=None, rotations=None, max_age_s=0.5, max_skew_s=0.5):
    clock = clock or Clock()
    offsets = {f"Drone{i}": (10*i, 0, 0) for i in range(1, count+1)}
    tf = CoordinateTransformer(offsets, rotations=rotations)
    tf._validated = True  # Fake validation report; integration test uses validate().
    tf._last_validation = CoordinateTransformValidation(
        validated=True, vehicles={name: VehicleTransformCheck(name, ok=True) for name in offsets}
    )
    readers = readers or {name: Reader(state(timestamp=i)) for i, name in enumerate(offsets, 1)}
    return StateProvider(readers, tf, clock=clock, max_age_s=max_age_s, max_skew_s=max_skew_s), clock


class StateProviderTests(unittest.TestCase):
    def test_one_to_five(self):
        for count in range(1, 6):
            p, _ = provider(count)
            snap = p.get_snapshot()
            self.assertTrue(snap.collection_complete)
            self.assertEqual(len(snap.states), count)
            self.assertEqual(len(p.get_neighbor_states("Drone1")), count-1)
            self.assertEqual(snap.states["Drone1"].position_ned_m, [11.0, 2.0, -3.0])

    def test_failed_and_invalid_reader(self):
        readers = {"Drone1": Reader(state()), "Drone2": Reader(error=OSError("rpc down"))}
        p, _ = provider(readers=readers)
        snap = p.get_snapshot()
        self.assertFalse(snap.collection_complete)
        self.assertEqual(set(snap.states), {"Drone1"})
        self.assertIn("rpc down", snap.errors["Drone2"])
        readers["Drone2"].error = None
        readers["Drone2"].state = state(timestamp=2, ready=False)
        self.assertIn("not ready", p.get_snapshot().errors["Drone2"])

    def test_nonfinite_and_repeated_timestamp(self):
        readers = {"Drone1": Reader(state()), "Drone2": Reader(state(timestamp=2, velocity=(math.nan, 0, 0)))}
        p, _ = provider(readers=readers)
        self.assertIn("finite", p.get_snapshot().errors["Drone2"])
        readers["Drone2"].state = state(timestamp=2)
        self.assertTrue(p.get_snapshot().collection_complete is False)  # Drone1 repeated
        self.assertIn("did not advance", p.get_snapshot().errors["Drone1"])

    def test_rotation_and_attitude(self):
        p, _ = provider(1, rotations={"Drone1": yaw_rotation_matrix(math.pi/2)})
        result = p.get_snapshot().states["Drone1"]
        self.assertAlmostEqual(result.position_ned_m[0], 8.0)
        self.assertAlmostEqual(result.position_ned_m[1], 1.0)
        self.assertAlmostEqual(result.linear_velocity_ned_mps[1], 1.0)
        self.assertAlmostEqual(result.yaw_rad, math.pi/2)

    def test_existing_state_reader_adapter(self):
        def vec(values):
            return NS(x_val=values[0], y_val=values[1], z_val=values[2])
        class Client:
            def getMultirotorState(self, vehicle_name):
                kin = NS(position=vec((1, 2, -3)), linear_velocity=vec((1, 0, 0)),
                         angular_velocity=vec((0, 0, 0)), orientation=NS(x_val=0, y_val=0, z_val=0, w_val=1))
                return NS(kinematics_estimated=kin, timestamp=1, ready=True, can_arm=True, landed_state=0)
        class Adapter:
            vehicle_name = "Drone1"
            def get_raw_client(self):
                return Client()
        tf = CoordinateTransformer({"Drone1": (10, 0, 0)})
        tf._validated = True
        tf._last_validation = CoordinateTransformValidation(
            validated=True, vehicles={"Drone1": VehicleTransformCheck("Drone1", ok=True)}
        )
        p = StateProvider.from_adapter(Adapter(), ["Drone1"], tf, euler_converter=lambda q: (0, 0, 0))
        self.assertEqual(p.get_snapshot().states["Drone1"].position_ned_m, [11.0, 2.0, -3.0])


if __name__ == "__main__":
    unittest.main()

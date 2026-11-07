"""Validated transform and legacy translation comparison."""

import math
import unittest
from types import SimpleNamespace as NS

from coordination.collision_predictor import to_common_ned
from coordination.coordinate_transform import CoordinateTransformer, yaw_rotation_matrix
from coordination.state_provider import StateProvider
from test_state_provider import Reader, state


def vec(v):
    return NS(x_val=v[0], y_val=v[1], z_val=v[2])


class PoseClient:
    def __init__(self, offsets, rotation=None):
        self.offsets, self.rotation = offsets, rotation or {}

    def getMultirotorState(self, name):
        return NS(kinematics_estimated=NS(position=vec((1, 0, 0)), linear_velocity=vec((1, 0, 0)),
                                          orientation=NS(x_val=0, y_val=0, z_val=0, w_val=1)))

    def simGetObjectPose(self, name):
        rotation = self.rotation.get(name)
        relative = (rotation[0][0], rotation[1][0], rotation[2][0]) if rotation else (1, 0, 0)
        offset = self.offsets[name]
        return NS(position=vec(tuple(a+b for a, b in zip(relative, offset))),
                  orientation=NS(x_val=0, y_val=0, z_val=0, w_val=1))


class IntegrationTests(unittest.TestCase):
    def test_translation_matches_legacy_after_validation(self):
        offsets = {"Drone1": (10, 0, 0), "Drone2": (20, 5, 0)}
        tf = CoordinateTransformer(offsets)
        self.assertTrue(tf.validate(PoseClient(offsets)).validated)
        p = StateProvider({name: Reader(state(timestamp=i, position=(1, 0, 0)))
                           for i, name in enumerate(offsets, 1)}, tf)
        snap = p.get_snapshot()
        for name, world in snap.states.items():
            self.assertEqual(tuple(world.position_ned_m), to_common_ned((1, 0, 0), offsets[name]))
            self.assertEqual(tuple(world.linear_velocity_ned_mps), (1, 0, 0))
            self.assertEqual(tf.from_common_ned(name, world.position_ned_m), (1, 0, 0))

    def test_rotation_after_validation(self):
        offsets = {"Drone1": (10, 0, 0)}
        rotation = {"Drone1": yaw_rotation_matrix(math.pi/2)}
        tf = CoordinateTransformer(offsets, rotations=rotation)
        self.assertTrue(tf.validate(PoseClient(offsets, rotation)).validated)
        p = StateProvider({"Drone1": Reader(state(position=(1, 0, 0)))}, tf)
        world = p.get_snapshot().states["Drone1"]
        self.assertAlmostEqual(world.position_ned_m[0], 10)
        self.assertAlmostEqual(world.position_ned_m[1], 1)
        self.assertAlmostEqual(world.linear_velocity_ned_mps[1], 1)

    def test_failed_validation_blocks_collection(self):
        tf = CoordinateTransformer({"Drone1": (99, 0, 0)})
        self.assertFalse(tf.validate(PoseClient({"Drone1": (0, 0, 0)})).validated)
        p = StateProvider({"Drone1": Reader(state())}, tf)
        with self.assertRaises(RuntimeError):
            p.get_snapshot()


if __name__ == "__main__":
    unittest.main()

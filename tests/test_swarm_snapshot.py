"""Snapshot timing, failure and freshness behavior."""

import unittest
from test_state_provider import Reader, provider, state


class SnapshotTests(unittest.TestCase):
    def test_individual_timestamps_duration_and_skew(self):
        p, clock = provider(5)
        for i, reader in enumerate(p._readers.values(), 1):
            reader.clock, reader.delay = clock, 0.01*i
        snap = p.get_snapshot()
        self.assertTrue(snap.collection_complete)
        self.assertAlmostEqual(snap.collection_duration, 0.15)
        self.assertAlmostEqual(snap.max_time_skew, 0.14)
        self.assertAlmostEqual(snap.read_duration_s["Drone5"], 0.05)
        self.assertEqual(len(set(snap.acquired_at.values())), 5)

    def test_skew_rejects_snapshot(self):
        p, clock = provider(max_skew_s=0.05)
        p._readers["Drone2"].clock = clock
        p._readers["Drone2"].delay = 0.1
        snap = p.get_snapshot()
        self.assertFalse(snap.collection_complete)
        self.assertFalse(snap.states)
        self.assertIn("skew", snap.errors["Drone1"])

    def test_sim_timestamp_skew_rejects_snapshot(self):
        p, _ = provider(max_skew_s=0.05)
        p._readers["Drone2"].state = state(timestamp=100_000_002)
        snap = p.get_snapshot()
        self.assertFalse(snap.collection_complete)
        self.assertFalse(snap.states)
        self.assertGreater(snap.max_sim_timestamp_skew, 0.05)

    def test_expired_during_collection_and_afterward(self):
        p, clock = provider(max_age_s=0.05)
        p._readers["Drone2"].clock = clock
        p._readers["Drone2"].delay = 0.1
        snap = p.get_snapshot()
        self.assertIn("expired", snap.errors["Drone1"])
        self.assertIsNone(p.get_vehicle_state("Drone1"))
        self.assertIsNotNone(p.get_vehicle_state("Drone2"))
        clock.now += 0.1
        self.assertFalse(p.get_all_vehicle_states())

    def test_failed_vehicle_not_reused_from_previous_snapshot(self):
        p, _ = provider()
        self.assertTrue(p.get_snapshot().collection_complete)
        p._readers["Drone1"].error = OSError("lost")
        p._readers["Drone2"].state = state(timestamp=3)
        snap = p.get_snapshot()
        self.assertNotIn("Drone1", snap.states)
        self.assertIsNone(p.get_vehicle_state("Drone1"))
        self.assertEqual(snap.sequence_id, 2)


if __name__ == "__main__":
    unittest.main()

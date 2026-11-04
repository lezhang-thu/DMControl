"""Chronology and episode-boundary checks for SPR replay sequences."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import ReplayBuffer


class SPRReplayTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)

    def make_buffer(self, capacity=8, jumps=2, batch_size=16):
        return ReplayBuffer(
            obs_shape=(3, 4, 4), action_shape=(1,), capacity=capacity,
            batch_size=2, device="cpu", image_size=4,
            auxiliary_task_batch_size=batch_size, jumps=jumps)

    def add(self, buffer, value, terminated=False, truncated=False):
        obs = np.full((3, 4, 4), value, dtype=np.uint8)
        buffer.add(obs, np.array([value], dtype=np.float32), value + 0.5,
                   obs + 1, terminated, truncated)

    def test_insufficient_sequence_still_returns_sac_batch(self):
        buffer = self.make_buffer()
        self.add(buffer, 0)
        result = buffer.sample_spr()
        self.assertIsNone(result[-1])
        self.assertEqual(result[0].shape, (2, 3, 4, 4))
        self.assertEqual(buffer.current_auxiliary_batch_size, 0)

    def test_minimum_sequence_allows_terminal_after_final_observation(self):
        buffer = self.make_buffer()
        self.add(buffer, 0)
        self.add(buffer, 1)
        self.add(buffer, 2, terminated=True)
        samples = buffer.sample_spr()[-1]
        self.assertIsNotNone(samples)
        states = samples['observation'][:, :, 0, 0, 0, 0].numpy()
        np.testing.assert_array_equal(states[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(samples['action'][:, :, 0], states)
        np.testing.assert_array_equal(samples['reward'][:, :, 0], states + 0.5)
        self.assertEqual(buffer.current_auxiliary_batch_size, 16)

    def test_both_termination_and_truncation_block_cross_episode_sequences(self):
        for ending in ('terminated', 'truncated'):
            with self.subTest(ending=ending):
                buffer = self.make_buffer()
                self.add(buffer, 0)
                self.add(buffer, 1, **{ending: True})
                self.add(buffer, 2)
                self.assertIsNone(buffer.sample_spr()[-1])
                self.assertEqual(buffer.current_auxiliary_batch_size, 0)

    def test_ring_wrap_preserves_time_and_never_crosses_overwrite_boundary(self):
        buffer = self.make_buffer(batch_size=128)
        for value in range(12):
            self.add(buffer, value)
        samples = buffer.sample_spr()[-1]
        states = samples['observation'][:, :, 0, 0, 0, 0].numpy()
        np.testing.assert_array_equal(np.diff(states, axis=0), np.ones((2, 128)))
        self.assertGreaterEqual(states.min(), 4)
        self.assertLessEqual(states.max(), 11)
        # The physical storage boundary at 7 -> 0 is a valid time sequence.
        self.assertTrue(np.any(np.all(states == np.array([7, 8, 9])[:, None], axis=0)))

    def test_zero_jumps_samples_even_terminal_observations(self):
        buffer = self.make_buffer(jumps=0)
        self.add(buffer, 3, terminated=True)
        samples = buffer.sample_spr()[-1]
        self.assertEqual(samples['observation'].shape, (1, 16, 3, 1, 4, 4))

    def test_empty_replay_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'empty replay'):
            self.make_buffer().sample_spr()


if __name__ == '__main__':
    unittest.main()

"""Environment migration checks without requiring an OpenGL renderer."""

from pathlib import Path
import sys
import unittest

import numpy as np
from dm_control import suite
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from env import make


class EnvironmentTests(unittest.TestCase):
    def test_action_repeat_matches_native_control_suite(self):
        tasks = [('ball_in_cup', 'catch'), ('cartpole', 'swingup'),
                 ('cheetah', 'run'), ('finger', 'spin'),
                 ('reacher', 'easy'), ('walker', 'walk')]
        for domain, task in tasks:
            with self.subTest(domain=domain, task=task):
                env = make(domain, task, seed=7, from_pixels=False, frame_skip=4)
                native = suite.load(domain, task, task_kwargs={'random': 7})
                try:
                    obs, info = env.reset()
                    native.reset()
                    self.assertTrue(env.observation_space.contains(obs))
                    action = np.linspace(-0.5, 0.5, env.action_space.shape[0], dtype=np.float32)
                    spec = native.action_spec()
                    native_action = spec.minimum + (action.astype(spec.dtype) + 1) * 0.5 * (
                        spec.maximum - spec.minimum)
                    expected_reward = 0.0
                    for _ in range(4):
                        step = native.step(native_action)
                        expected_reward += float(step.reward or 0)
                    obs, reward, terminated, truncated, info = env.step(action)
                    expected_obs = np.concatenate([
                        np.asarray(value).reshape(-1) for value in step.observation.values()
                    ]).astype(np.float32)
                    np.testing.assert_allclose(obs, expected_obs, rtol=0, atol=1e-6)
                    self.assertAlmostEqual(reward, expected_reward)
                    self.assertFalse(terminated or truncated)
                    self.assertEqual(info['discount'], step.discount)
                finally:
                    env.close()
                    native.close()

    def test_time_limit_truncates_and_stops_action_repeat(self):
        env = make('cartpole', 'swingup', seed=7, from_pixels=False, frame_skip=1500)
        try:
            env.reset()
            obs, reward, terminated, truncated, info = env.step(np.zeros(1))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['discount'], 1.0)
            self.assertTrue(env.observation_space.contains(obs))
        finally:
            env.close()

    def test_gymnasium_api_and_seeded_resets(self):
        env = make('cartpole', 'swingup', seed=7, from_pixels=False)
        try:
            check_env(env, skip_render_check=True)
            first, _ = env.reset(seed=11)
            first_action = env.action_space.sample()
            first_step = env.step(first_action)
            second, _ = env.reset(seed=11)
            second_action = env.action_space.sample()
            second_step = env.step(second_action)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first_action, second_action)
            np.testing.assert_array_equal(first_step[0], second_step[0])
            self.assertEqual(first_step[1:], second_step[1:])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

"""Environment migration checks without requiring an OpenGL renderer."""

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from env import configure_headless_rendering, make
from dm_control import mujoco, suite


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        # Stub only the renderer. The Control Suite still runs the real dynamics.
        self.frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        renderer = mock.patch.object(mujoco.Physics, "render", return_value=self.frame)
        self.render = renderer.start()
        self.addCleanup(renderer.stop)

    def test_headless_rendering_selects_egl_without_overriding_a_choice(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("DISPLAY", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            configure_headless_rendering()
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")

        with mock.patch.dict(os.environ, {"MUJOCO_GL": "osmesa"}, clear=False):
            os.environ.pop("DISPLAY", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            configure_headless_rendering()
            self.assertEqual(os.environ["MUJOCO_GL"], "osmesa")

        with mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False):
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            configure_headless_rendering()
            self.assertNotIn("MUJOCO_GL", os.environ)

    def test_action_repeat_matches_native_control_suite(self):
        tasks = [('ball_in_cup', 'catch'), ('cartpole', 'swingup'),
                 ('cheetah', 'run'), ('finger', 'spin'),
                 ('reacher', 'easy'), ('walker', 'walk')]
        for domain, task in tasks:
            with self.subTest(domain=domain, task=task):
                env = make(domain, task, seed=7, height=5, width=7, frame_skip=4)
                native = suite.load(domain, task, task_kwargs={'random': 7})
                try:
                    obs, info = env.reset()
                    native.reset()
                    self.assertTrue(env.observation_space.contains(obs))
                    self.render.assert_called_with(height=5, width=7, camera_id=0)
                    np.testing.assert_array_equal(obs, self.frame.transpose(2, 0, 1))
                    action = np.linspace(-0.5, 0.5, env.action_space.shape[0], dtype=np.float32)
                    spec = native.action_spec()
                    native_action = spec.minimum + (action.astype(spec.dtype) + 1) * 0.5 * (
                        spec.maximum - spec.minimum)
                    expected_reward = 0.0
                    for _ in range(4):
                        step = native.step(native_action)
                        expected_reward += float(step.reward or 0)
                    obs, reward, terminated, truncated, info = env.step(action)
                    np.testing.assert_array_equal(obs, self.frame.transpose(2, 0, 1))
                    np.testing.assert_array_equal(
                        env._env.physics.get_state(), native.physics.get_state())
                    self.assertAlmostEqual(reward, expected_reward)
                    self.assertFalse(terminated or truncated)
                    self.assertEqual(info['discount'], step.discount)
                finally:
                    env.close()
                    native.close()

    def test_time_limit_truncates_and_stops_action_repeat(self):
        env = make('cartpole', 'swingup', seed=7, height=5, width=7, frame_skip=1500)
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
        env = make('cartpole', 'swingup', seed=7, height=5, width=7)
        try:
            check_env(env, skip_render_check=True)
            first, _ = env.reset(seed=11)
            first_state = env._env.physics.get_state().copy()
            first_action = env.action_space.sample()
            first_step = env.step(first_action)
            first_step_state = env._env.physics.get_state().copy()
            second, _ = env.reset(seed=11)
            np.testing.assert_array_equal(first_state, env._env.physics.get_state())
            second_action = env.action_space.sample()
            second_step = env.step(second_action)
            np.testing.assert_array_equal(first_step_state, env._env.physics.get_state())
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first_action, second_action)
            np.testing.assert_array_equal(first_step[0], second_step[0])
            self.assertEqual(first_step[1:], second_step[1:])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

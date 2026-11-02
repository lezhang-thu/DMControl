"""Regression checks for the current numerical stack and local scalar logging.

Run with: python -m unittest discover -s tests -v
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from logger import Logger
from spr_sac import SPR, SPRSacAgent
from utils import FrameStack, ReplayBuffer, random_crop, random_crop_2


def make_agent():
    with contextlib.redirect_stdout(io.StringIO()):
        return SPRSacAgent(
            obs_shape=(9, 84, 84), action_shape=(2,), device="cpu",
            augmentation=("crop", "intensity"),
            transition_model_type="deterministic",
            transition_model_layer_width=16, encoder_feature_dim=8,
            latent_dim=16, hidden_dim=16, num_layers=2, num_filters=4,
            jumps=2, log_interval=1, actor_update_freq=1,
            critic_target_update_freq=1,
        )


def make_buffer(obs_shape=(9, 100, 100)):
    return ReplayBuffer(
        obs_shape=obs_shape, action_shape=(2,), capacity=32,
        batch_size=2, device="cpu", image_size=84,
        auxiliary_task_batch_size=2, jumps=2,
    )


class NumericalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)


class LoggingTests(NumericalTestCase):
    def test_scalar_console_and_json_logging_without_backends(self):
        # Import and use the logger with optional logging packages unavailable.
        spec = importlib.util.spec_from_file_location("local_test_logger", SRC / "logger.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"wandb": None, "tensorboard": None,
                                           "torchvision": None,
                                           "torch.utils.tensorboard": None}):
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as directory:
                logger = module.Logger(directory)
                logger.log("train/episode_reward", 4.0, 10)
                logger.log("train/episode_reward", 8.0, 10)
                logger.log("train_critic/loss", torch.tensor(2.0, requires_grad=True), 10)
                logger.log("eval/episode_reward", 5.0, 10)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    logger.dump(10)
                    logger.dump(11)
                self.assertIn("train", output.getvalue())
                self.assertIn("eval", output.getvalue())
                train_lines = (Path(directory) / "train_info.log").read_text().splitlines()
                self.assertEqual(len(train_lines), 1)
                self.assertEqual(json.loads(train_lines[0]), {
                    "episode_reward": 6.0, "critic_loss": 2.0, "step": 10,
                })
                evaluation = json.loads((Path(directory) / "eval_info.log").read_text())
                self.assertEqual(evaluation, {"episode_reward": 5.0, "step": 10})
                self.assertEqual(len(list(Path(directory).iterdir())), 2)


class CropTests(unittest.TestCase):
    def test_rectangular_crop_preserves_channels_and_last_valid_origin(self):
        images = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        with mock.patch("numpy.random.randint", side_effect=[np.array([0, 2]), np.array([0, 4])]):
            crops = random_crop(images, 3)
        self.assertEqual(crops.shape, (2, 3, 3, 3))
        np.testing.assert_array_equal(crops[0], images[0, :, :3, :3])
        np.testing.assert_array_equal(crops[1], images[1, :, 2:, 4:])
        self.assertTrue(crops.flags.c_contiguous)
        self.assertTrue(crops.flags.writeable)

    def test_equal_size_and_temporal_crop(self):
        images = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        np.testing.assert_array_equal(random_crop(images, 4), images)
        temporal_images = np.stack([images, images + 1000])
        np.testing.assert_array_equal(random_crop_2(temporal_images, 4), temporal_images)

    def test_bottom_and_right_edges_are_reachable(self):
        def largest_draw(low, high=None, size=None):
            return np.full(size, (low if high is None else high) - 1)

        image = np.arange(3 * 5 * 7).reshape(1, 3, 5, 7)
        with mock.patch("numpy.random.randint", side_effect=largest_draw):
            crop = random_crop(image, 3)
        np.testing.assert_array_equal(crop, image[:, :, -3:, -3:])

    def test_invalid_size_is_rejected(self):
        for size in (0, -1, 5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                random_crop(np.zeros((2, 3, 4, 6)), size)


class ReplayTests(NumericalTestCase):
    def test_tensor_checkpoint_roundtrip_preserves_episode_boundaries(self):
        buffer = make_buffer(obs_shape=(3, 4, 4))
        # A time limit still bootstraps, whereas a true termination does not.
        endings = [(False, False), (False, True), (True, False)]
        for i, (terminated, truncated) in enumerate(endings):
            observation = np.full((3, 4, 4), i, dtype=np.uint8)
            buffer.add(observation, np.array([i, -i], dtype=np.float32), i + 0.5,
                       observation + 1, terminated, truncated)
        np.testing.assert_array_equal(buffer.not_dones[:3, 0], [1, 1, 0])
        np.testing.assert_array_equal(buffer.real_dones[:3, 0], [0, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            buffer.save(directory)
            checkpoint = next(Path(directory).glob("*.pt"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(len(payload), 6)
            self.assertTrue(all(isinstance(item, torch.Tensor) for item in payload))
            restored = make_buffer(obs_shape=(3, 4, 4))
            restored.load(directory)
        self.assertEqual(restored.idx, 3)
        for name in ("obses", "next_obses", "actions", "rewards", "not_dones", "real_dones"):
            np.testing.assert_array_equal(getattr(restored, name)[:3], getattr(buffer, name)[:3])


class AugmentationTests(NumericalTestCase):
    def make_spr(self, augmentations, probability):
        critic = SimpleNamespace(encoder=nn.Identity())
        with contextlib.redirect_stdout(io.StringIO()):
            return SPR(
                (2,), critic, critic, augmentations,
                transition_model_type="deterministic", transition_model_layer_width=8,
                encoder_feature_dim=4, latent_dim=8, jumps=2, aug_prob=probability,
            )

    def test_all_augmentations_with_full_partial_and_zero_probability(self):
        images = torch.randint(0, 256, (2, 2, 9, 100, 100), dtype=torch.uint8)
        for augmentation in ("crop", "affine", "rrc", "blur", "shift", "intensity", "none"):
            pipeline = (augmentation,) if augmentation in {"crop", "rrc", "shift"} else ("crop", augmentation)
            for probability in (0.0, 0.5, 1.0):
                with self.subTest(augmentation=augmentation, probability=probability):
                    spr = self.make_spr(pipeline, probability)
                    transformed = spr.transform(images, augment=True)
                    evaluation = spr.transform(images, augment=False)
                    self.assertEqual(transformed.shape, (2, 2, 9, 84, 84))
                    self.assertEqual(evaluation.shape, transformed.shape)
                    self.assertTrue(torch.isfinite(transformed).all())
                    self.assertEqual(transformed.dtype, torch.float32)
                    if probability == 0.0:
                        torch.testing.assert_close(transformed, evaluation)

    def test_uint8_and_float_inputs_keep_the_same_pixel_scale(self):
        spr = self.make_spr(("crop", "intensity"), 1.0)
        images = torch.full((2, 9, 100, 100), 128, dtype=torch.uint8)
        torch.manual_seed(19)
        uint8_output = spr.transform(images, augment=True)
        torch.manual_seed(19)
        float_output = spr.transform(images.float(), augment=True)
        torch.testing.assert_close(uint8_output, float_output)
        self.assertGreater(float(uint8_output.mean()), 100.0)
        self.assertLess(float(uint8_output.mean()), 150.0)


class AgentTests(NumericalTestCase):
    def test_cpu_update_changes_actor_critic_temperature_and_spr(self):
        agent = make_agent()
        buffer = make_buffer()
        for _ in range(12):
            observation = np.random.randint(0, 256, (9, 100, 100), dtype=np.uint8)
            next_observation = np.random.randint(0, 256, observation.shape, dtype=np.uint8)
            buffer.add(observation, np.random.uniform(-1, 1, 2).astype(np.float32),
                       1.0, next_observation, False)
        buffer.add_agent(agent)
        tracked = {
            "actor": agent.actor.trunk[0].weight,
            "critic": agent.critic.Q1.trunk[0].weight,
            "temperature": agent.log_alpha,
            "spr": agent.SPR.transition_model.fc.weight,
        }
        before = {name: tensor.detach().clone() for name, tensor in tracked.items()}
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(directory)
            agent.update(buffer, logger, step=1)
            with contextlib.redirect_stdout(io.StringIO()):
                logger.dump(1)
            metrics = json.loads((Path(directory) / "train_info.log").read_text())
        for name, tensor in tracked.items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(tensor).all())
                self.assertFalse(torch.equal(tensor.detach(), before[name]))
        for key in ("critic_loss", "actor_loss", "alpha_loss", "spr_loss"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))
        action = agent.sample_action(observation)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_model_checkpoints_roundtrip_with_weights_only_loading(self):
        agent = make_agent()
        observation = np.full((9, 84, 84), 128, dtype=np.uint8)
        expected_action = agent.select_action(observation)
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory, 7)
            agent.save_spr(directory, 7)
            restored = make_agent()
            restored.load(directory, 7)
            restored.SPR.load_state_dict(torch.load(
                Path(directory) / "spr_7.pt", map_location="cpu", weights_only=True,
            ))
        np.testing.assert_array_equal(restored.select_action(observation), expected_action)
        for original, loaded in ((agent.actor, restored.actor), (agent.critic, restored.critic),
                                 (agent.SPR, restored.SPR)):
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)


class PixelEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(3, 4, 4), dtype=np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {"seed": seed}

    def step(self, action):
        self.count += 1
        observation = np.full(self.observation_space.shape, self.count, dtype=np.uint8)
        return observation, 2.0, bool(action[0] < 0), self.count >= 2, {"count": self.count}


class FrameStackTests(unittest.TestCase):
    def test_gymnasium_reset_step_flags_and_frames(self):
        env = FrameStack(PixelEnv(), 3)
        observation, info = env.reset(seed=42)
        self.assertEqual(info, {"seed": 42})
        self.assertTrue(env.observation_space.contains(observation))
        first, reward, terminated, truncated, _ = env.step(np.array([0.0]))
        np.testing.assert_array_equal(first[:, 0, 0], [0] * 6 + [1] * 3)
        self.assertEqual((reward, terminated, truncated), (2.0, False, False))
        second, _, terminated, truncated, info = env.step(np.array([0.0]))
        np.testing.assert_array_equal(second[:, 0, 0], [0] * 3 + [1] * 3 + [2] * 3)
        self.assertEqual((terminated, truncated, info), (False, True, {"count": 2}))
        reset, _ = env.reset()
        self.assertFalse(np.any(reset))
        _, _, terminated, truncated, _ = env.step(np.array([-1.0]))
        self.assertEqual((terminated, truncated), (True, False))


if __name__ == "__main__":
    unittest.main()

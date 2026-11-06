# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


@torch.no_grad()
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.lerp_(param, tau)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer:
    """Buffer to store environment transitions."""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 batch_size,
                 device,
                 image_size=84,
                 auxiliary_task_batch_size=64,
                 jumps=5):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.jumps = jumps
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty_like(self.obses)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32) # for auxiliary task

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, terminated, truncated=False):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        # Bootstrap across time limits, but never sample SPR across a reset.
        np.copyto(self.not_dones[self.idx], not terminated)
        np.copyto(self.real_dones[self.idx], terminated or truncated)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_spr(self):   # sample batch for auxiliary task
        idxs = np.random.randint(0,
                                 self.capacity - self.jumps -
                                 1 if self.full else self.idx - self.jumps - 1,
                                 size=self.auxiliary_task_batch_size*2)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.jumps + 1).reshape(1, -1) # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)   # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs] # (B, jumps+1)
        idxs = idxs[:self.auxiliary_task_batch_size]

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()   # (B, jumps+1, 3*3=9, 100, 100)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)

        spr_samples = {
            'observation': obses.transpose(0, 1).unsqueeze(3),
            'action': actions.transpose(0, 1),
        }
        return (*self.sample_aug(), spr_samples)

    def sample_aug(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = random_crop(self.obses[idxs], self.image_size)
        next_obses = random_crop(self.next_obses[idxs], self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames.clear()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """Crop each BCHW image independently, including the last valid origin."""
    n, _, height, width = imgs.shape
    if not 0 < output_size <= min(height, width):
        raise ValueError("Crop size must fit inside the image")
    rows = np.random.randint(height - output_size + 1, size=n)
    cols = np.random.randint(width - output_size + 1, size=n)
    windows = sliding_window_view(imgs, (output_size, output_size), axis=(-2, -1))
    return windows[np.arange(n), :, rows, cols].copy()


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

"""Gymnasium interface for the DeepMind Control Suite."""

import os

import gymnasium as gym
import numpy as np


def configure_headless_rendering():
    """Use EGL when no window system is available.

    dm_control prefers GLFW. With libglfw installed and no DISPLAY, the Python
    glfw package warns and glfw.init() returns failure instead of raising, so
    dm_control never falls through to EGL. MuJoCo then fails in mjr_makeContext.
    An explicit MUJOCO_GL value is left unchanged.
    """
    if os.environ.get("MUJOCO_GL"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    os.environ["MUJOCO_GL"] = "egl"


configure_headless_rendering()

from dm_control import suite  # noqa: E402


class ControlSuiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, domain_name, task_name, seed=None, height=100,
                 width=100, frame_skip=1):
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self._domain_name = domain_name
        self._task_name = task_name
        self._height = height
        self._width = width
        self._frame_skip = frame_skip
        self.render_mode = "rgb_array"
        self._env = self._load(seed)

        spec = self._env.action_spec()
        self._action_min = spec.minimum
        self._action_max = spec.maximum
        self._action_dtype = spec.dtype
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=spec.shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(3, height, width), dtype=np.uint8)

    def _load(self, seed):
        return suite.load(
            self._domain_name, self._task_name,
            task_kwargs={"random": seed})

    def _observation(self):
        return np.ascontiguousarray(self.render().transpose(2, 0, 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.close()
            self._env = self._load(seed)
            self.action_space.seed(seed)
        self._env.reset()
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=self._action_dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, "
                             f"got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        action = self._action_min + (action + 1.0) * 0.5 * (
            self._action_max - self._action_min)
        reward = 0.0
        for _ in range(self._frame_skip):
            timestep = self._env.step(action)
            reward += float(timestep.reward or 0.0)
            if timestep.last():
                break
        # Control Suite time limits have discount=1, genuine terminals have 0.
        terminated = bool(timestep.last() and timestep.discount == 0)
        truncated = bool(timestep.last() and not terminated)
        return (self._observation(), reward, terminated, truncated,
                {"discount": float(timestep.discount)})

    def render(self):
        """Render the image used for pixel observations."""
        return self._env.physics.render(
            height=self._height, width=self._width, camera_id=0)

    def close(self):
        self._env.close()


def make(domain_name, task_name, seed=None,
         height=100, width=100, frame_skip=1):
    return ControlSuiteEnv(domain_name, task_name, seed, height, width, frame_skip)

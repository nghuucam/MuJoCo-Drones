"""Wind Gymnasium wrapper that applies wind forces during simulation.

Wraps a BaseAviary and injects wind disturbance forces at each physics step.
"""

import numpy as np
import gymnasium as gym
from typing import Optional

from multi_drone_mujoco.wrappers.wind import WindField, WindConfig, WindModel


class WindWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies wind disturbance to drone environments.

    Adds wind forces to each drone body via xfrc_applied after the base
    environment's physics step. Compatible with any BaseAviary subclass.

    Example
    -------
    >>> from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    >>> from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel
    >>> from multi_drone_mujoco.wrappers.wind_wrapper import WindWrapper
    >>> config = WindConfig(
    ...     model=WindModel.COMBINED,
    ...     constant_wind=np.array([1.0, 0.0, 0.0]),  # 1 m/s headwind
    ...     turbulence_intensity=1.5,
    ...     gust_intensity=0.01,
    ... )
    >>> env = WindWrapper(HoverAviary(), config)
    >>> obs, _ = env.reset()
    """

    def __init__(self, env: gym.Env, wind_config: Optional[WindConfig] = None):
        super().__init__(env)
        self.wind_config = wind_config or WindConfig(model=WindModel.DRYDEN)
        self.wind_field = WindField(self.wind_config)
        self._sync_base_wind_field()

    def _sync_base_wind_field(self):
        base_env = self.env
        while hasattr(base_env, 'env') and not hasattr(base_env, '_wind_field'):
            base_env = base_env.env
        if hasattr(base_env, '_wind_field'):
            base_env._wind_field = self.wind_field

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        seed = kwargs.get("seed", None)
        self.wind_field.reset(seed=seed)
        self._sync_base_wind_field()
        info["wind_config"] = {
            "model": self.wind_config.model.value,
            "constant_wind": self.wind_config.constant_wind.tolist(),
            "turbulence_intensity": self.wind_config.turbulence_intensity,
        }
        return obs, info

    def step(self, action):
        """Step environment with sub-step wind disturbance."""
        self._sync_base_wind_field()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


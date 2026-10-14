import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config_mjx as cfg
from drone_mjx_env import DroneMJXEnv, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import random


class MJXGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = cfg.NUM_ENVS, level: int = 6):
        super().__init__()
        self.num_envs = num_envs
        self.level = level
        self.env = DroneMJXEnv(num_envs=num_envs, level=level)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.act_dim,), dtype=np.float32
        )

        self.rng = random.PRNGKey(42) if HAS_JAX else None
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None and HAS_JAX:
            self.rng = random.PRNGKey(seed)
        self.rng, r_key = random.split(self.rng)
        self.state, obs_jax = self.env.reset(r_key, level=self.level)
        obs = np.asarray(obs_jax)
        info = {}
        return obs, info

    def step(self, actions):
        actions_jax = jnp.asarray(actions)
        self.state, obs_jax, rewards_jax, dones_jax, info_jax = self.env.step(self.state, actions_jax)

        obs = np.asarray(obs_jax)
        rewards = np.asarray(rewards_jax)
        dones = np.asarray(dones_jax)
        truncated = np.zeros_like(dones)

        info = {k: np.asarray(v) for k, v in info_jax.items()}
        return obs, rewards, dones, truncated, info

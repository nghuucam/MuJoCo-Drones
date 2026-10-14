import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config_mjx as cfg
from drone_mjx_env import DroneMJXEnv, HAS_JAX

try:
    from stable_baselines3.common.vec_env import VecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    VecEnv = gym.Env

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import random


class MJXGymEnv(VecEnv):
    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = cfg.NUM_ENVS, level: int = 6):
        self.num_envs = num_envs
        self.level = level
        self.env = DroneMJXEnv(num_envs=num_envs, level=level)

        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32
        )
        act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.act_dim,), dtype=np.float32
        )
        super().__init__(num_envs, obs_space, act_space)
        self.render_mode = None

        self.rng = random.PRNGKey(42) if HAS_JAX else None
        self.state = None
        self._actions = None

    def set_level(self, level: int):
        self.level = int(level)
        self.env.set_level(int(level))

    def get_level(self) -> int:
        return self.level

    def reset(self):
        if HAS_JAX:
            self.rng, r_key = random.split(self.rng)
            self.state, obs_jax = self.env.reset(r_key, level=self.level)
            return np.asarray(obs_jax)
        return np.zeros((self.num_envs, self.env.obs_dim), dtype=np.float32)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        actions_jax = jnp.asarray(self._actions)
        self.state, obs_jax, rewards_jax, dones_jax, info_jax = self.env.step(self.state, actions_jax, level=self.level)

        obs = np.asarray(obs_jax)
        rewards = np.asarray(rewards_jax).reshape(-1)
        dones = np.asarray(dones_jax).reshape(-1)

        wins = np.asarray(info_jax["win"]).reshape(-1)
        cols = np.asarray(info_jax["collision"]).reshape(-1)
        overs = np.asarray(info_jax["over_map"]).reshape(-1)
        dists = np.asarray(info_jax["dist"]).reshape(-1)
        timeouts = np.asarray(info_jax["timeout"]).reshape(-1)
        term_obs = np.asarray(info_jax["terminal_observation"])

        infos = []
        for i in range(self.num_envs):
            is_win = bool(wins[i])
            is_col = bool(cols[i])
            is_succ = bool(is_win and not is_col)
            info_dict = {
                "is_success": is_succ,
                "win": is_win,
                "collision": is_col,
                "over_map": bool(overs[i]),
                "timeout": bool(timeouts[i]),
                "dist": float(dists[i]),
                "level": self.level
            }
            if dones[i]:
                info_dict["terminal_observation"] = term_obs[i]
                info_dict["TimeLimit.truncated"] = bool(timeouts[i])
            infos.append(info_dict)

        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        val = getattr(self, attr_name)
        n = self.num_envs if indices is None else len(indices)
        return [val] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        val = method(*method_args, **method_kwargs)
        n = self.num_envs if indices is None else len(indices)
        return [val] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        if seed is not None and HAS_JAX:
            self.rng = random.PRNGKey(seed)
        return [seed]

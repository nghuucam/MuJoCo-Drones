"""PettingZoo-compatible multi-agent wrapper for aviary environments.

Provides ParallelEnv interface for multi-agent reinforcement learning.
"""

import functools
from typing import Optional

import gymnasium as gym
import numpy as np

try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils import parallel_to_aec, wrappers
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    # Fallback base class
    class ParallelEnv:
        pass

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MultiAgentAviary(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for multi-drone environments.

    Each drone is an independent agent with its own observation and action space.

    Usage:
        env = MultiAgentAviary(num_drones=3)
        observations, infos = env.reset()
        # observations = {"drone0": obs0, "drone1": obs1, "drone2": obs2}
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "multi_agent_aviary_v0"}

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 3,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        target_heights=None,
        initial_xyzs=None,
        neighbourhood_radius: float = 0.5,
        render_mode: Optional[str] = None,
    ):
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo is required: pip install pettingzoo")

        self.num_drones = num_drones
        self.possible_agents = [f"drone{i}" for i in range(num_drones)]
        self.agents = self.possible_agents[:]

        # Target heights
        if target_heights is None:
            self.target_heights = np.linspace(0.7, 1.2, num_drones)
        else:
            self.target_heights = np.array(target_heights)

        if initial_xyzs is None:
            initial_xyzs = np.zeros((num_drones, 3))
            for i in range(num_drones):
                initial_xyzs[i] = [i * 0.3, 0, 0.1]

        # Create underlying aviary
        self._env = BaseAviary(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            initial_xyzs=initial_xyzs,
            neighbourhood_radius=neighbourhood_radius,
            render_mode=render_mode,
        )

        self.render_mode = render_mode
        self._episode_len_sec = 10
        self._step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """13-dim per agent: pos(3) + rpy(3) + vel(3) + angvel(3) + target_h(1)."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """4-dim normalized RPM per agent."""
        return gym.spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self._step_count = 0

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_agent_obs(i)
            infos[agent] = {}

        return observations, infos

    def step(self, actions):
        # Build joint action
        joint_action = np.zeros((self.num_drones, 4))
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                joint_action[i] = np.clip(actions[agent], -1, 1)

        # Convert to RPMs and step
        rpms = np.zeros_like(joint_action)
        for i in range(self.num_drones):
            rpms[i] = self._env._normalizedActionToRPM(joint_action[i])

        self._env._preprocessAction = lambda a: a  # Bypass preprocessing
        self._env.step(rpms.flatten())
        self._step_count += 1

        # Compute per-agent returns
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            observations[agent] = self._get_agent_obs(i)

            # Per-drone reward
            height_err = abs(self._env.pos[i, 2] - self.target_heights[i])
            xy_err = np.linalg.norm(self._env.pos[i, 0:2] - self._env.INIT_XYZS[i, 0:2])
            reward = -height_err - 0.1 * xy_err
            if height_err < 0.05:
                reward += 0.5
            rewards[agent] = float(reward)

            # Per-drone termination
            terminated = (
                self._env.pos[i, 2] < 0.0
                or abs(self._env.rpy[i, 0]) > np.pi / 2
                or abs(self._env.rpy[i, 1]) > np.pi / 2
            )
            terminations[agent] = terminated
            
            if terminated:
                reward -= 100.0
                
            rewards[agent] = float(reward)

            truncated = self._step_count * self._env.CTRL_TIMESTEP >= self._episode_len_sec
            truncations[agent] = truncated

            infos[agent] = {"height_error": float(height_err)}

        # Remove terminated agents
        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]

        return observations, rewards, terminations, truncations, infos

    def _get_agent_obs(self, drone_idx):
        """Get per-agent observation."""
        state = self._env._getDroneStateVector(drone_idx)
        obs = np.hstack([
            state[0:3], state[7:10], state[10:13], state[13:16],
            [self.target_heights[drone_idx]],
        ])
        return obs.astype(np.float32)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

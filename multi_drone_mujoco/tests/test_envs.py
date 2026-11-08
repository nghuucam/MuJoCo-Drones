"""Tests for all environment types."""

import numpy as np
import pytest
import gymnasium as gym

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary
from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
from multi_drone_mujoco.envs.formation_aviary import FormationAviary
from multi_drone_mujoco.envs.race_aviary import RaceAviary
from multi_drone_mujoco.utils.enums import Physics, ActionType


class TestBaseAviary:
    """Tests for BaseAviary."""

    def test_instantiation(self):
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_multi_drone(self):
        env = BaseAviary(num_drones=3, ctrl_freq=240, sim_freq=240)
        obs, _ = env.reset()
        assert env.pos.shape == (3, 3)
        env.close()

    def test_hover_rpm_stability(self):
        """Hover RPM should maintain exact position."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        env.reset()
        z0 = env.pos[0, 2]
        hover_rpm = np.full(4, env.HOVER_RPM)
        for _ in range(240):
            env.step(hover_rpm)
        assert abs(env.pos[0, 2] - z0) < 1e-4
        assert abs(env.vel[0, 2]) < 1e-4
        env.close()

    def test_physics_modes(self):
        """All physics modes should instantiate and step."""
        for phys in [Physics.MJC, Physics.DYN, Physics.MJC_GND, Physics.MJC_DRAG, Physics.MJC_DW, Physics.MJC_GND_DRAG_DW]:
            env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=phys)
            env.reset()
            env.step(np.full(4, env.HOVER_RPM))
            env.close()

    def test_action_space_shape(self):
        env = BaseAviary(num_drones=2, ctrl_freq=240, sim_freq=240)
        env.reset()
        assert env.action_space.shape == (8,)
        env.close()


class TestHoverAviary:
    """Tests for HoverAviary."""

    def test_obs_shape(self):
        env = HoverAviary()
        obs, _ = env.reset()
        assert obs.shape == (12,)
        env.close()

    def test_action_range(self):
        env = HoverAviary()
        env.reset()
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0
        env.close()

    def test_step_returns(self):
        env = HoverAviary()
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_episode_terminates(self):
        """Episode should truncate after EPISODE_LEN_SEC."""
        env = HoverAviary(ctrl_freq=48)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        assert done
        env.close()


class TestVelocityAviary:
    def test_obs_shape(self):
        env = VelocityAviary()
        obs, _ = env.reset()
        assert obs.shape == (16,)
        env.close()


class TestMultiHoverAviary:
    def test_obs_shape(self):
        env = MultiHoverAviary(num_drones=3)
        obs, _ = env.reset()
        assert obs.shape == (39,)
        env.close()


class TestFlyThroughAviary:
    def test_obs_shape(self):
        env = FlyThroughAviary()
        obs, _ = env.reset()
        assert obs.shape == (18,)
        env.close()


class TestFormationAviary:
    def test_obs_shape(self):
        env = FormationAviary(num_drones=3)
        obs, _ = env.reset()
        assert obs.shape == (54,)
        env.close()


class TestRaceAviary:
    def test_obs_shape(self):
        env = RaceAviary()
        obs, _ = env.reset()
        assert obs.shape == (21,)
        env.close()

"""Tests for multi-agent (PettingZoo) wrapper."""

import numpy as np
import pytest

try:
    import pettingzoo
    from multi_drone_mujoco.envs.multi_agent_aviary import MultiAgentAviary
    HAS_PETTINGZOO = True
except (ImportError, ModuleNotFoundError):
    HAS_PETTINGZOO = False


@pytest.mark.skipif(not HAS_PETTINGZOO, reason="pettingzoo not installed")
class TestMultiAgentAviary:
    def test_agents(self):
        env = MultiAgentAviary(num_drones=3)
        env.reset()
        assert len(env.agents) == 3
        env.close()

    def test_step(self):
        env = MultiAgentAviary(num_drones=2)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert len(obs) == 2
        assert len(rewards) == 2
        env.close()

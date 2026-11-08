"""Velocity Aviary: track a desired velocity vector.

Task: follow velocity commands [vx, vy, vz, yaw_rate].
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class VelocityAviary(BaseAviary):
    """Single-drone velocity tracking task."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 10
        self.TARGET_VEL = np.array([0.0, 0.0, 0.0, 0.0])  # Will be randomized

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Randomize target velocity
        if self.np_random is not None:
            self.TARGET_VEL = self.np_random.uniform(-0.5, 0.5, size=4)
            self.TARGET_VEL[3] *= 0.5  # Reduce yaw rate
        return obs, info

    def _actionSpace(self):
        return spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

    def _observationSpace(self):
        # State (12) + target velocity (4) = 16
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def _preprocessAction(self, action):
        action = np.clip(np.array(action).flatten(), -1, 1)
        return self._normalizedActionToRPM(action).reshape(1, 4)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16], self.TARGET_VEL])
        return obs.astype(np.float32)

    def _computeReward(self):
        vel_error = np.linalg.norm(self.vel[0, :3] - self.TARGET_VEL[:3])
        yaw_rate_error = abs(self.ang_v[0, 2] - self.TARGET_VEL[3])
        reward = -vel_error - 0.1 * yaw_rate_error
        # Penalize extreme attitudes
        reward -= 0.1 * (abs(self.rpy[0, 0]) + abs(self.rpy[0, 1]))
        # Bonus for tracking
        if vel_error < 0.05:
            reward += 0.5
            
        if self._computeTerminated():
            reward -= 100.0
            
        return float(reward)

    def _computeTerminated(self):
        if self.pos[0, 2] < 0.0 or self.pos[0, 2] > 5.0:
            return True
        if abs(self.rpy[0, 0]) > np.pi / 2 or abs(self.rpy[0, 1]) > np.pi / 2:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "velocity_error": np.linalg.norm(self.vel[0, :3] - self.TARGET_VEL[:3]),
            "target_vel": self.TARGET_VEL.tolist(),
        }

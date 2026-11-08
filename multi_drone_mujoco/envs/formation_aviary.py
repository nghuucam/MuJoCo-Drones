"""Formation Aviary: multi-drone formation flying.

Task: N drones maintain a desired formation shape while moving.
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class FormationAviary(BaseAviary):
    """Multi-drone formation flying task."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 3,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        formation_offsets=None,
        formation_center_path=None,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 15

        # Formation shape (offsets from center)
        if formation_offsets is None:
            # Equilateral triangle at 0.5m spacing
            angle_step = 2 * np.pi / num_drones
            self.FORMATION_OFFSETS = np.array([
                [0.3 * np.cos(i * angle_step), 0.3 * np.sin(i * angle_step), 0]
                for i in range(num_drones)
            ])
        else:
            self.FORMATION_OFFSETS = np.array(formation_offsets)

        # Path for the formation center (waypoints)
        if formation_center_path is None:
            self.CENTER_PATH = np.array([
                [0, 0, 1.0],
                [1, 0, 1.0],
                [1, 1, 1.2],
                [0, 1, 1.0],
                [0, 0, 1.0],
            ])
        else:
            self.CENTER_PATH = np.array(formation_center_path)

        self.path_idx = 0
        self.path_t = 0.0  # Interpolation parameter

        if initial_xyzs is None:
            center = np.array([0, 0, 0.1])
            initial_xyzs = center + self.FORMATION_OFFSETS
            initial_xyzs[:, 2] = 0.1

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
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
        self.path_idx = 0
        self.path_t = 0.0
        return super().reset(seed=seed, options=options)

    def _get_formation_targets(self):
        """Get current target positions for all drones."""
        # Interpolate center along path
        idx = min(self.path_idx, len(self.CENTER_PATH) - 2)
        t = self.path_t
        center = (1 - t) * self.CENTER_PATH[idx] + t * self.CENTER_PATH[idx + 1]
        return center + self.FORMATION_OFFSETS

    def _advance_path(self):
        """Advance along the path based on formation tracking quality."""
        self.path_t += 0.001  # Slow progression
        if self.path_t >= 1.0:
            self.path_t = 0.0
            self.path_idx = min(self.path_idx + 1, len(self.CENTER_PATH) - 2)

    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(4 * self.NUM_DRONES, dtype=np.float32),
            high=np.ones(4 * self.NUM_DRONES, dtype=np.float32),
        )

    def _observationSpace(self):
        # per drone: pos(3) + rpy(3) + vel(3) + angvel(3) + target(3) + rel_target(3) = 18
        return spaces.Box(low=-np.inf, high=np.inf, shape=(18 * self.NUM_DRONES,), dtype=np.float32)

    def _preprocessAction(self, action):
        action = np.clip(np.array(action).reshape(self.NUM_DRONES, 4), -1, 1)
        rpms = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            rpms[i] = self._normalizedActionToRPM(action[i])
        return rpms

    def _computeObs(self):
        targets = self._get_formation_targets()
        obs_list = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            rel = targets[i] - self.pos[i]
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16], targets[i], rel,
            ]))
        return np.concatenate(obs_list).astype(np.float32)

    def _computeReward(self):
        targets = self._get_formation_targets()
        total = 0.0
        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.pos[i] - targets[i])
            total -= dist
            total -= 0.05 * (abs(self.rpy[i, 0]) + abs(self.rpy[i, 1]))
            if dist < 0.05:
                total += 0.5

        # Bonus for maintaining formation shape (inter-drone distances)
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                desired_dist = np.linalg.norm(self.FORMATION_OFFSETS[i] - self.FORMATION_OFFSETS[j])
                actual_dist = np.linalg.norm(self.pos[i] - self.pos[j])
                total -= 0.5 * abs(actual_dist - desired_dist)

        self._advance_path()
        
        if self._computeTerminated():
            total -= 100.0
            
        return float(total)

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            if self.pos[i, 2] < 0.0:
                return True
            if abs(self.rpy[i, 0]) > np.pi / 2 or abs(self.rpy[i, 1]) > np.pi / 2:
                return True
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        targets = self._get_formation_targets()
        return {
            "formation_errors": [
                float(np.linalg.norm(self.pos[i] - targets[i]))
                for i in range(self.NUM_DRONES)
            ],
            "path_progress": self.path_idx + self.path_t,
        }

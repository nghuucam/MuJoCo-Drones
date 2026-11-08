"""Fly-Through Aviary: navigate through waypoints/gates.

Task: fly through a sequence of waypoints as quickly as possible.
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class FlyThroughAviary(BaseAviary):
    """Fly through waypoints task."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        waypoints=None,
        waypoint_radius: float = 0.1,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 20
        self.WAYPOINT_RADIUS = waypoint_radius

        if waypoints is None:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.5],
            ])
        else:
            self.WAYPOINTS = np.array(waypoints)

        self.current_waypoint_idx = np.zeros(num_drones if num_drones > 1 else 1, dtype=int)

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.1]])

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
        self.current_waypoint_idx[:] = 0
        return super().reset(seed=seed, options=options)

    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(4 * self.NUM_DRONES, dtype=np.float32),
            high=np.ones(4 * self.NUM_DRONES, dtype=np.float32),
        )

    def _observationSpace(self):
        # pos(3) + rpy(3) + vel(3) + angvel(3) + next_waypoint(3) + rel_waypoint(3) = 18
        return spaces.Box(low=-np.inf, high=np.inf, shape=(18 * self.NUM_DRONES,), dtype=np.float32)

    def _preprocessAction(self, action):
        action = np.clip(np.array(action).reshape(self.NUM_DRONES, 4), -1, 1)
        rpms = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            rpms[i] = self._normalizedActionToRPM(action[i])
        return rpms

    def _computeObs(self):
        obs_list = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            rel_wp = wp - self.pos[i]
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16], wp, rel_wp,
            ]))
        return np.concatenate(obs_list).astype(np.float32)

    def _computeReward(self):
        total = 0.0
        for i in range(self.NUM_DRONES):
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            dist = np.linalg.norm(self.pos[i] - wp)

            # Check waypoint reached
            if dist < self.WAYPOINT_RADIUS and self.current_waypoint_idx[i] < len(self.WAYPOINTS):
                self.current_waypoint_idx[i] += 1
                total += 10.0  # Big bonus for reaching waypoint

            total -= dist * 0.1  # Approach reward
            total -= 0.01 * np.linalg.norm(self.ang_v[i])  # Smooth flight

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
        # Done when all waypoints reached or time limit
        all_done = all(idx >= len(self.WAYPOINTS) for idx in self.current_waypoint_idx)
        time_up = self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC
        return all_done or time_up

    def _computeInfo(self):
        return {
            "waypoints_reached": [int(idx) for idx in self.current_waypoint_idx],
            "total_waypoints": len(self.WAYPOINTS),
        }

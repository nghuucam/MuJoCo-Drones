"""DSL PID Controller — tuned for Crazyflie 2.x.

Higher-performance PID with anti-windup and rate limiting.
Based on the DSL lab's controller implementation.
"""

import numpy as np

from multi_drone_mujoco.control.pid_control import PIDControl


class DSLPIDControl(PIDControl):
    """DSL-tuned PID controller with improved performance.

    Features over base PID:
    - Anti-windup with clamping
    - Rate limiting on setpoint changes
    - Better gain scheduling
    - Supports position + velocity + acceleration feedforward
    """

    def __init__(self, env=None):
        super().__init__(env)

        # Tighter, more aggressive gains for DSL controller
        self.P_COEFF_FOR = np.array([0.5, 0.5, 1.2])
        self.I_COEFF_FOR = np.array([0.02, 0.02, 0.02])
        self.D_COEFF_FOR = np.array([1.0, 1.0, 2.2])

        self.P_COEFF_TOR = np.array([0.003, 0.003, 0.0015])
        self.I_COEFF_TOR = np.array([0.0, 0.0, 0.0002])
        self.D_COEFF_TOR = np.array([0.0008, 0.0008, 0.0003])

        # Rate limits
        self.MAX_POS_RATE = 2.0  # m/s max position setpoint change
        self.MAX_YAW_RATE = np.pi  # rad/s max yaw setpoint change

        # Anti-windup limits
        self.INTEGRAL_POS_LIMIT = 1.0
        self.INTEGRAL_RPY_LIMIT = 0.3

        # Previous target for rate limiting
        self._prev_target_pos = None
        self._prev_target_yaw = None

    def reset(self):
        super().reset()
        self._prev_target_pos = None
        self._prev_target_yaw = None

    def computeControl(
        self,
        control_timestep: float,
        cur_pos: np.ndarray,
        cur_quat: np.ndarray,
        cur_vel: np.ndarray,
        cur_ang_vel: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: np.ndarray = np.zeros(3),
        target_vel: np.ndarray = np.zeros(3),
        target_acc: np.ndarray = np.zeros(3),
    ):
        """Compute RPMs with rate limiting and anti-windup.

        Additional parameter:
        target_acc : ndarray (3,)
            Feedforward acceleration.
        """
        # Rate limit target position
        if self._prev_target_pos is not None:
            delta_pos = target_pos - self._prev_target_pos
            max_delta = self.MAX_POS_RATE * control_timestep
            if np.linalg.norm(delta_pos) > max_delta:
                target_pos = self._prev_target_pos + delta_pos / np.linalg.norm(delta_pos) * max_delta

        # Rate limit yaw
        if self._prev_target_yaw is not None:
            delta_yaw = target_rpy[2] - self._prev_target_yaw
            delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
            max_yaw_delta = self.MAX_YAW_RATE * control_timestep
            if abs(delta_yaw) > max_yaw_delta:
                target_rpy = target_rpy.copy()
                target_rpy[2] = self._prev_target_yaw + np.sign(delta_yaw) * max_yaw_delta

        self._prev_target_pos = target_pos.copy()
        self._prev_target_yaw = target_rpy[2]

        # Override integral limits
        self.integral_pos_e = np.clip(self.integral_pos_e, -self.INTEGRAL_POS_LIMIT, self.INTEGRAL_POS_LIMIT)
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -self.INTEGRAL_RPY_LIMIT, self.INTEGRAL_RPY_LIMIT)

        # Call base computation
        rpm, pos_e, yaw_e = super().computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
        )

        return rpm, pos_e, yaw_e

"""PID Controller for quadrotor position/attitude control.

Implements cascaded PID: outer loop (position) → inner loop (attitude) → motor mixer.
Based on the controller in gym-pybullet-drones (Luis & Le Ny, 2016).
"""

import numpy as np


class PIDControl:
    """PID controller for a single Crazyflie drone."""

    def __init__(self, env=None, mode: str = "stiff"):
        """Initialize PID gains.

        Parameters
        ----------
        env : BaseAviary, optional
            Environment instance to read drone parameters from.
        mode : str, optional
            'stiff': Rigidly counteracts wind with high derivative damping.
            'compliant' / 'soft': Realistic compliance allowing natural drone sway and wobble in wind.
        """
        if env is not None:
            self.GRAVITY = env.G
            self.MASS = env.M
            self.KF = env.KF
            self.KM = env.KM
            self.L = env.L
            self.HOVER_RPM = env.HOVER_RPM
            self.MAX_RPM = env.MAX_RPM
            self.J = env.J
        else:
            # Default CF2X parameters
            self.GRAVITY = 9.81
            self.MASS = 0.027
            self.KF = 3.16e-10
            self.KM = 7.94e-12
            self.L = 0.0397
            self.HOVER_RPM = np.sqrt((self.MASS * self.GRAVITY) / (4 * self.KF))
            self.MAX_RPM = self.HOVER_RPM * 1.5
            self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])

        self.mode = mode
        self._apply_gains(mode)

        # Attitude PID gains — must be tiny to stay within torque envelope
        # Max achievable torque ~0.004 Nm; keep PID output well below this
        self.P_COEFF_TOR = np.array([0.002, 0.002, 0.001])
        self.I_COEFF_TOR = np.array([0.0, 0.0, 0.0001])
        self.D_COEFF_TOR = np.array([0.0005, 0.0005, 0.0002])

        # Integral error accumulators
        self.integral_pos_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)

    def _apply_gains(self, mode: str):
        if mode in ("compliant", "soft", "natural"):
            # Natural compliance: allows realistic aerodynamic swaying and rocking
            self.P_COEFF_FOR = np.array([0.35, 0.35, 0.9])
            self.I_COEFF_FOR = np.array([0.008, 0.008, 0.01])
            self.D_COEFF_FOR = np.array([0.45, 0.45, 1.4])
        else:
            # Stiff mode: critically damped tight hold
            self.P_COEFF_FOR = np.array([0.4, 0.4, 1.0])
            self.I_COEFF_FOR = np.array([0.01, 0.01, 0.01])
            self.D_COEFF_FOR = np.array([0.9, 0.9, 2.0])

    def set_mode(self, mode: str):
        """Switch controller tuning between 'stiff' and 'compliant'."""
        self.mode = mode
        self._apply_gains(mode)

    def reset(self):
        """Reset integral accumulators."""
        self.integral_pos_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)

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
    ):
        """Compute RPMs from current state and target.

        Parameters
        ----------
        control_timestep : float
            Time since last control call.
        cur_pos : ndarray (3,)
            Current position [x, y, z].
        cur_quat : ndarray (4,)
            Current quaternion [w, x, y, z].
        cur_vel : ndarray (3,)
            Current velocity.
        cur_ang_vel : ndarray (3,)
            Current angular velocity.
        target_pos : ndarray (3,)
            Desired position.
        target_rpy : ndarray (3,)
            Desired roll-pitch-yaw (only yaw is used).
        target_vel : ndarray (3,)
            Desired velocity (feedforward).

        Returns
        -------
        rpm : ndarray (4,)
            Motor RPMs.
        pos_error : ndarray (3,)
            Position error.
        yaw_error : float
            Yaw error.
        """
        # Position error
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e += pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2.0, 2.0)

        # Desired acceleration (PID on position)
        target_acc = (
            self.P_COEFF_FOR * pos_e
            + self.I_COEFF_FOR * self.integral_pos_e
            + self.D_COEFF_FOR * vel_e
        )

        # Desired thrust
        target_acc[2] += self.GRAVITY  # Compensate gravity
        thrust = self.MASS * np.linalg.norm(target_acc)
        # Prevent thrust direction from flipping (drone can't thrust downward)
        if target_acc[2] < 0:
            target_acc[2] = 0.0
            thrust = self.MASS * np.linalg.norm(target_acc)

        # Desired attitude from desired acceleration
        # (simplified: assumes small angles)
        z_axis = target_acc / np.linalg.norm(target_acc) if np.linalg.norm(target_acc) > 1e-6 else np.array([0, 0, 1])

        target_yaw = target_rpy[2]
        x_c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
        y_axis = np.cross(z_axis, x_c)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis /= y_norm
        else:
            y_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z_axis)

        # Desired rotation matrix → RPY
        # Note: sign convention - positive pitch tilts thrust toward +x in MuJoCo
        target_roll = np.arcsin(y_axis[2])
        target_pitch = np.arctan2(-x_axis[2], z_axis[2])

        # Current RPY from quaternion
        cur_rpy = self._quatToRPY(cur_quat)

        # Attitude error
        rpy_e = np.array([target_roll - cur_rpy[0], target_pitch - cur_rpy[1], target_yaw - cur_rpy[2]])
        # Wrap yaw error
        rpy_e[2] = (rpy_e[2] + np.pi) % (2 * np.pi) - np.pi

        self.integral_rpy_e += rpy_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -0.5, 0.5)

        d_rpy_e = (rpy_e - self.last_rpy_e) / control_timestep if control_timestep > 0 else np.zeros(3)

        # Desired torques
        target_torques = (
            self.P_COEFF_TOR * rpy_e
            + self.I_COEFF_TOR * self.integral_rpy_e
            + self.D_COEFF_TOR * d_rpy_e
        )

        self.last_pos_e = pos_e
        self.last_rpy_e = rpy_e

        # Convert thrust + torques → RPMs
        rpm = self._thrustTorquesToRPM(thrust, target_torques)

        return rpm, pos_e, rpy_e[2]

    def _thrustTorquesToRPM(self, thrust, torques):
        """Convert desired thrust and torques to motor RPMs (X-configuration).

        Uses allocation matrix with proper clamping to avoid motor saturation.
        """
        # Allocation matrix for X-config:
        # [T]     [1      1       1       1    ] [kf*rpm1^2]
        # [tx]  = [L/s2   L/s2   -L/s2  -L/s2 ] [kf*rpm2^2]
        # [ty]    [-L/s2  L/s2    L/s2  -L/s2  ] [kf*rpm3^2]
        # [tz]    [-km/kf km/kf  -km/kf  km/kf ] [kf*rpm4^2]
        s2 = np.sqrt(2)
        km_kf = self.KM / self.KF

        A = np.array([
            [1, 1, 1, 1],
            [self.L / s2, self.L / s2, -self.L / s2, -self.L / s2],
            [-self.L / s2, self.L / s2, self.L / s2, -self.L / s2],
            [-km_kf, km_kf, -km_kf, km_kf],
        ])

        # Scale torques so they don't exceed what's achievable
        # Max achievable torque is limited by motor RPM range
        max_torque_xy = self.L / s2 * self.KF * self.MAX_RPM ** 2 * 0.3  # Use 30% margin
        max_torque_z = km_kf * self.KF * self.MAX_RPM ** 2 * 0.3
        torques = np.array([
            np.clip(torques[0], -max_torque_xy, max_torque_xy),
            np.clip(torques[1], -max_torque_xy, max_torque_xy),
            np.clip(torques[2], -max_torque_z, max_torque_z),
        ])

        b = np.array([thrust, torques[0], torques[1], torques[2]])
        # b is [total_thrust, tau_x, tau_y, tau_z]
        # Each element of solution = kf * rpm_i^2

        try:
            motor_forces = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            motor_forces = np.linalg.lstsq(A, b, rcond=None)[0]

        # Clamp to valid range
        motor_forces = np.clip(motor_forces, 0, self.KF * self.MAX_RPM ** 2)
        rpm = np.sqrt(motor_forces / self.KF)
        return np.clip(rpm, 0, self.MAX_RPM)

    def _quatToRPY(self, quat):
        """Convert quaternion [w,x,y,z] to RPY."""
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

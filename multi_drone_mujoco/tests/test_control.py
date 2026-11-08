"""Tests for PID controllers."""

import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from multi_drone_mujoco.utils.enums import Physics


class TestPIDControl:
    """Tests for PIDControl."""

    def test_hover_convergence(self):
        """PID should converge to target z=1.0 within 10s."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        ctrl = PIDControl(env)
        env.reset()
        target = np.array([0, 0, 1.0])
        for _ in range(2400):
            rpm, _, _ = ctrl.computeControl(
                env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
                env.vel[0], env.ang_v[0], target
            )
            env.step(rpm.flatten())
        error = np.linalg.norm(env.pos[0] - target)
        assert error < 0.1, f"Position error {error:.4f}m too large"
        env.close()

    def test_3d_tracking(self):
        """PID should track a 3D target."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        ctrl = PIDControl(env)
        env.reset()
        target = np.array([0.5, 0.3, 1.0])
        for _ in range(4800):
            rpm, _, _ = ctrl.computeControl(
                env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
                env.vel[0], env.ang_v[0], target
            )
            env.step(rpm.flatten())
        error = np.linalg.norm(env.pos[0] - target)
        assert error < 0.1, f"Position error {error:.4f}m too large"
        env.close()

    def test_no_motor_saturation(self):
        """Motors should not saturate during normal hover."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        ctrl = PIDControl(env)
        env.reset()
        target = np.array([0, 0, 1.0])
        saturated = False
        for _ in range(1200):
            rpm, _, _ = ctrl.computeControl(
                env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
                env.vel[0], env.ang_v[0], target
            )
            if np.any(rpm >= env.MAX_RPM * 0.99) or np.any(rpm <= 1.0):
                saturated = True
                break
            env.step(rpm.flatten())
        assert not saturated, "Motors saturated during hover"
        env.close()

    def test_reset(self):
        ctrl = PIDControl()
        ctrl.integral_pos_e = np.ones(3)
        ctrl.reset()
        assert np.all(ctrl.integral_pos_e == 0)


class TestDSLPIDControl:
    """Tests for DSLPIDControl."""

    def test_convergence(self):
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240, physics=Physics.MJC)
        ctrl = DSLPIDControl(env)
        env.reset()
        target = np.array([0.5, 0.3, 1.0])
        for _ in range(4800):
            rpm, _, _ = ctrl.computeControl(
                env.CTRL_TIMESTEP, env.pos[0], env.quat[0],
                env.vel[0], env.ang_v[0], target
            )
            env.step(rpm.flatten())
        error = np.linalg.norm(env.pos[0] - target)
        assert error < 0.1, f"DSL PID error {error:.4f}m too large"
        env.close()

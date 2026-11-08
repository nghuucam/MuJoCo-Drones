"""Domain Randomization wrapper for sim-to-real transfer.

Randomizes physical parameters, adds sensor noise, and simulates
actuator delays to improve policy robustness for real-world deployment.
"""

import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters.

    Each parameter specifies the range as (min_scale, max_scale) relative
    to the nominal value, or absolute range for additive parameters.
    """
    # Multiplicative randomization (applied as nominal * uniform(lo, hi))
    mass_range: tuple = (0.8, 1.2)           # ±20% mass
    inertia_range: tuple = (0.8, 1.2)        # ±20% inertia
    kf_range: tuple = (0.85, 1.15)           # ±15% thrust coefficient
    km_range: tuple = (0.85, 1.15)           # ±15% torque coefficient
    arm_length_range: tuple = (0.95, 1.05)   # ±5% arm length
    motor_constant_range: tuple = (0.9, 1.1) # ±10% max RPM

    # Sensor noise (additive, per-step)
    pos_noise_std: float = 0.002             # m (GPS-like noise)
    vel_noise_std: float = 0.01              # m/s
    rpy_noise_std: float = 0.005             # rad (IMU gyro bias)
    ang_vel_noise_std: float = 0.02          # rad/s

    # IMU bias (constant per episode, additive)
    gyro_bias_std: float = 0.01              # rad/s constant offset
    acc_bias_std: float = 0.05               # m/s^2 constant offset

    # Actuator dynamics
    action_delay_steps: int = 0              # steps of latency (0-3 typical)
    action_delay_range: tuple = (0, 3)       # randomize delay per episode
    motor_time_constant: float = 0.0        # first-order motor lag (seconds, 0=instant)
    motor_time_constant_range: tuple = (0.0, 0.02)

    # Initial state perturbation
    init_pos_noise_std: float = 0.02         # m
    init_vel_noise_std: float = 0.01         # m/s
    init_rpy_noise_std: float = 0.05         # rad

    # Randomize on each reset
    randomize_every_episode: bool = True


class DomainRandomizationWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies domain randomization.

    Randomizes physics parameters on reset, adds sensor noise per step,
    and optionally simulates actuator delay.

    Example
    -------
    >>> from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    >>> from multi_drone_mujoco.wrappers.domain_randomization import (
    ...     DomainRandomizationWrapper, DomainRandomizationConfig)
    >>> config = DomainRandomizationConfig(mass_range=(0.7, 1.3))
    >>> env = DomainRandomizationWrapper(HoverAviary(), config)
    >>> obs, info = env.reset()
    """

    def __init__(self, env: gym.Env, config: Optional[DomainRandomizationConfig] = None):
        super().__init__(env)
        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng()
        self._action_buffer = []
        self._current_delay = 0
        self._motor_state = None
        self._gyro_bias = np.zeros(3)
        self._acc_bias = np.zeros(3)

        # Store nominal parameters from environment
        self._nominal_mass = env.M
        self._nominal_kf = env.KF
        self._nominal_km = env.KM
        self._nominal_L = env.L
        self._nominal_max_rpm = env.MAX_RPM
        self._nominal_J = env.J.copy()

    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize physics parameters
        if self.config.randomize_every_episode:
            self._randomize_dynamics()

        # Randomize actuator delay
        lo, hi = self.config.action_delay_range
        self._current_delay = self._rng.integers(lo, hi + 1)
        self._action_buffer = []

        # Randomize motor time constant
        lo, hi = self.config.motor_time_constant_range
        self._motor_tau = self._rng.uniform(lo, hi)
        self._motor_state = None

        # Randomize IMU biases (constant for episode)
        self._gyro_bias = self._rng.normal(0, self.config.gyro_bias_std, size=3)
        self._acc_bias = self._rng.normal(0, self.config.acc_bias_std, size=3)

        obs, info = self.env.reset(**kwargs)

        # Perturb initial state
        if self.config.init_pos_noise_std > 0:
            for i in range(self.env.NUM_DRONES):
                self.env.pos[i] += self._rng.normal(0, self.config.init_pos_noise_std, size=3)
                self.env.vel[i] += self._rng.normal(0, self.config.init_vel_noise_std, size=3)

        info["domain_params"] = self._get_domain_params()
        return self._add_obs_noise(obs), info

    def step(self, action):
        # Apply motor lag (first-order filter)
        if self._motor_tau > 0:
            if self._motor_state is None:
                self._motor_state = action.copy()
            alpha = self.env.CTRL_TIMESTEP / (self._motor_tau + self.env.CTRL_TIMESTEP)
            self._motor_state = alpha * action + (1 - alpha) * self._motor_state
            action = self._motor_state.copy()

        # Apply action delay
        if self._current_delay > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self._current_delay:
                action = self._action_buffer.pop(0)
            else:
                # Not enough history yet — use zero/hover action
                action = np.zeros_like(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _randomize_dynamics(self):
        """Randomize physics parameters of the underlying environment."""
        env = self.env

        # Mass
        lo, hi = self.config.mass_range
        mass_scale = self._rng.uniform(lo, hi)
        env.M = self._nominal_mass * mass_scale

        # Inertia
        lo, hi = self.config.inertia_range
        inertia_scale = self._rng.uniform(lo, hi)
        env.J = self._nominal_J * inertia_scale

        # Thrust coefficient
        lo, hi = self.config.kf_range
        env.KF = self._nominal_kf * self._rng.uniform(lo, hi)

        # Torque coefficient
        lo, hi = self.config.km_range
        env.KM = self._nominal_km * self._rng.uniform(lo, hi)

        # Arm length
        lo, hi = self.config.arm_length_range
        env.L = self._nominal_L * self._rng.uniform(lo, hi)

        # Max RPM
        lo, hi = self.config.motor_constant_range
        env.MAX_RPM = self._nominal_max_rpm * self._rng.uniform(lo, hi)

        # Recompute derived quantities
        env.HOVER_RPM = np.sqrt((env.M * env.G) / (4 * env.KF))
        env.MAX_THRUST = 4 * env.KF * env.MAX_RPM ** 2
        env.WEIGHT = env.M * env.G

    def _add_obs_noise(self, obs):
        """Add sensor noise to observations."""
        if obs is None:
            return obs
        noise = self._rng.normal(0, self.config.pos_noise_std, size=obs.shape)
        return (obs + noise).astype(obs.dtype)

    def _get_domain_params(self):
        """Return current randomized parameters for logging."""
        return {
            "mass": self.env.M,
            "kf": self.env.KF,
            "km": self.env.KM,
            "arm_length": self.env.L,
            "max_rpm": self.env.MAX_RPM,
            "action_delay": self._current_delay,
            "motor_tau": self._motor_tau,
            "gyro_bias": self._gyro_bias.tolist(),
        }


# Export wind & obstacle modules for convenient imports
from multi_drone_mujoco.wrappers.wind import WindField, WindConfig, WindModel
from multi_drone_mujoco.wrappers.wind_wrapper import WindWrapper
from multi_drone_mujoco.wrappers.obstacles import (
    ObstacleConfig,
    ObstacleType,
    generate_obstacles,
    obstacles_to_xml,
)

__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizationWrapper",
    "WindField",
    "WindConfig",
    "WindModel",
    "WindWrapper",
    "ObstacleConfig",
    "ObstacleType",
    "generate_obstacles",
    "obstacles_to_xml",
]


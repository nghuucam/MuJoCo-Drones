"""Wind and turbulence models for drone simulation.

Implements multiple wind disturbance models:
- Constant wind field
- Stochastic gusts
- Dryden turbulence model (MIL-F-8785C)
- Sinusoidal (periodic) wind

Wind forces are applied as external disturbances during physics stepping.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class WindModel(Enum):
    """Available wind disturbance models."""
    NONE = "none"
    CONSTANT = "constant"           # Fixed wind vector
    GUST = "gust"                   # Random impulses
    DRYDEN = "dryden"               # Dryden turbulence (continuous)
    SINUSOIDAL = "sinusoidal"       # Periodic gusting
    COMBINED = "combined"           # Constant + Dryden + Gusts


@dataclass
class WindConfig:
    """Configuration for wind disturbance model.

    Parameters
    ----------
    model : WindModel
        Type of wind model to use.
    constant_wind : ndarray (3,)
        Mean wind velocity [wx, wy, wz] in m/s (world frame).
    gust_intensity : float
        Max gust force in N (for micro-drones, typical 0.001-0.01 N).
    gust_probability : float
        Probability of gust occurring each control step.
    gust_duration_steps : int
        How many steps a gust lasts.
    turbulence_intensity : float
        Dryden turbulence intensity (low=0.5, moderate=1.0, severe=2.0).
    altitude : float
        Operating altitude for Dryden model (affects length scale).
    airspeed : float
        Airspeed for Dryden model (m/s, affects filtering).
    sinusoidal_amplitude : float
        Amplitude of sinusoidal wind (N).
    sinusoidal_period : float
        Period of sinusoidal wind (seconds).
    drag_coefficient : float
        Aerodynamic drag factor C_d * A (m^2) for wind force = 0.5*rho*Cd*A*v^2.
    """
    model: WindModel = WindModel.NONE
    constant_wind: Optional[np.ndarray] = None  # m/s in world frame
    gust_intensity: float = 0.040               # N (~15% of Crazyflie weight 0.265 N)
    gust_probability: float = 0.04              # per step
    gust_duration_steps: int = 15
    turbulence_intensity: float = 1.5           # sigma scale
    altitude: float = 1.0                       # m (for Dryden length scale)
    airspeed: float = 1.0                       # m/s
    sinusoidal_amplitude: float = 0.030         # N (~11% of drone weight)
    sinusoidal_period: float = 2.5              # seconds
    drag_coefficient: float = 0.005             # Cd*A including rotor drag & body

    def __post_init__(self):
        if self.constant_wind is None:
            self.constant_wind = np.zeros(3)
        self.constant_wind = np.asarray(self.constant_wind, dtype=np.float64)


class WindField:
    """Computes wind disturbance forces for drone simulation.

    Usage
    -----
    >>> wind = WindField(WindConfig(model=WindModel.DRYDEN, turbulence_intensity=1.5))
    >>> wind.reset(seed=42)
    >>> force = wind.get_force(dt=1/240, position=np.array([0, 0, 1.0]),
    ...                         velocity=np.array([0.1, 0, 0]))
    """

    def __init__(self, config: Optional[WindConfig] = None):
        self.config = config or WindConfig()
        self._rng = np.random.default_rng()
        self._step = 0
        self._dt = 1 / 240

        # Dryden filter states
        self._dryden_state = np.zeros(3)

        # Gust state
        self._gust_active = False
        self._gust_direction = np.zeros(3)
        self._gust_remaining = 0

    def reset(self, seed: Optional[int] = None):
        """Reset wind model state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._dryden_state = np.zeros(3)
        self._gust_active = False
        self._gust_direction = np.zeros(3)
        self._gust_remaining = 0

    def get_force(self, dt: float, position: np.ndarray,
                  velocity: np.ndarray) -> np.ndarray:
        """Compute wind disturbance force vector in world frame.

        Parameters
        ----------
        dt : float
            Timestep (seconds).
        position : ndarray (3,)
            Current drone position (for altitude-dependent effects).
        velocity : ndarray (3,)
            Current drone velocity (for relative wind computation).

        Returns
        -------
        force : ndarray (3,)
            Wind force in Newtons (world frame).
        """
        self._dt = dt
        self._step += 1
        model = self.config.model

        if model == WindModel.NONE:
            return np.zeros(3)

        force = np.zeros(3)

        if model in (WindModel.CONSTANT, WindModel.COMBINED):
            force += self._constant_wind_force(velocity)

        if model in (WindModel.GUST, WindModel.COMBINED):
            force += self._gust_force()

        if model in (WindModel.DRYDEN, WindModel.COMBINED):
            force += self._dryden_force(dt, position)

        if model == WindModel.SINUSOIDAL:
            force += self._sinusoidal_force()

        return force

    def _constant_wind_force(self, velocity: np.ndarray) -> np.ndarray:
        """Force from constant wind using drag model: F = 0.5*rho*Cd*A*v_rel^2."""
        rho = 1.225  # air density kg/m^3
        v_rel = self.config.constant_wind - velocity
        speed = np.linalg.norm(v_rel)
        if speed < 1e-6:
            return np.zeros(3)
        direction = v_rel / speed
        force_mag = 0.5 * rho * self.config.drag_coefficient * speed ** 2
        return force_mag * direction

    def _gust_force(self) -> np.ndarray:
        """Random gust impulses."""
        # Check if new gust starts
        if not self._gust_active:
            if self._rng.random() < self.config.gust_probability:
                self._gust_active = True
                self._gust_remaining = self.config.gust_duration_steps
                # Random direction with bias toward horizontal
                direction = self._rng.normal(size=3)
                direction[2] *= 0.3  # reduce vertical gusts
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                self._gust_direction = direction * self.config.gust_intensity
            else:
                return np.zeros(3)

        if self._gust_active:
            self._gust_remaining -= 1
            if self._gust_remaining <= 0:
                self._gust_active = False
            # Smooth envelope (ramp up/down)
            progress = 1.0 - self._gust_remaining / self.config.gust_duration_steps
            envelope = np.sin(np.pi * progress)  # smooth pulse
            return self._gust_direction * envelope

        return np.zeros(3)

    def _dryden_force(self, dt: float, position: np.ndarray) -> np.ndarray:
        """Dryden continuous turbulence model (simplified).

        Based on MIL-F-8785C. Uses first-order colored noise filters
        with altitude-dependent length scales.
        """
        h = max(position[2], 0.1)  # altitude (clamp to avoid division by zero)
        sigma = self.config.turbulence_intensity

        # Dryden length scales (low altitude approximation)
        L_u = h / (0.177 + 0.000823 * h) ** 1.2
        L_v = L_u
        L_w = h

        # Clamp to reasonable values for micro-drones
        L_u = np.clip(L_u, 0.5, 50.0)
        L_v = np.clip(L_v, 0.5, 50.0)
        L_w = np.clip(L_w, 0.25, 25.0)

        V = max(self.config.airspeed, 0.5)

        # Turbulence intensities (m/s atmospheric turbulence velocity fluctuations)
        sigma_u = sigma * 0.9  # m/s longitudinal turbulence
        sigma_v = sigma * 0.9  # m/s lateral turbulence
        sigma_w = sigma * 0.45 # m/s vertical turbulence

        # First-order filter: dx/dt = -V/L * x + sqrt(2*V/L) * sigma * white_noise
        white = self._rng.normal(size=3)
        tau = np.array([L_u / V, L_v / V, L_w / V])
        sigma_vec = np.array([sigma_u, sigma_v, sigma_w])

        # Discrete-time update
        alpha = np.exp(-dt / tau)
        noise_scale = sigma_vec * np.sqrt(1 - alpha ** 2)
        self._dryden_state = alpha * self._dryden_state + noise_scale * white

        # Convert turbulence velocity to aerodynamic disturbance force:
        # Effective cross-sectional area accounts for body skin drag + spinning rotor drag
        rho = 1.225
        Cd_A = max(self.config.drag_coefficient, 0.006)
        force = 0.5 * rho * Cd_A * self._dryden_state * np.abs(self._dryden_state)
        return force

    def _sinusoidal_force(self) -> np.ndarray:
        """Periodic sinusoidal wind disturbance."""
        t = self._step * self._dt
        amp = self.config.sinusoidal_amplitude
        period = self.config.sinusoidal_period
        omega = 2 * np.pi / period
        # Circular wind pattern in XY plane
        force = np.array([
            amp * np.sin(omega * t),
            amp * np.cos(omega * t),
            amp * 0.3 * np.sin(2 * omega * t),  # smaller vertical
        ])
        return force

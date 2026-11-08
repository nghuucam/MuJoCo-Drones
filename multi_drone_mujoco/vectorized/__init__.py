"""GPU-Vectorized Drone Environment using MuJoCo MJX (JAX backend).

Enables thousands of parallel drone simulations on GPU for massively
accelerated RL training. Compatible with JAX-based RL libraries
(e.g., PureJaxRL, Brax-style training loops).

Requirements: pip install mujoco-mjx jax[cuda12]

Architecture
------------
- Compiles the MuJoCo drone model to XLA via MJX
- Vectorizes across N_envs using jax.vmap
- All physics + reward + reset logic runs on GPU
- Returns batched observations/rewards as JAX arrays
- Optional Gymnasium VectorEnv wrapper for SB3 compatibility

Usage
-----
>>> from multi_drone_mujoco.vectorized.mjx_aviary import MJXVectorAviary
>>> env = MJXVectorAviary(num_envs=4096, num_drones=1, task="hover")
>>> state = env.reset(jax.random.PRNGKey(0))
>>> state, obs, reward, done = env.step(state, action)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

# Lazy imports for JAX/MJX (only required at runtime)
_JAX_AVAILABLE = False
_MJX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit, lax
    _JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import mujoco
    from mujoco import mjx
    _MJX_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        import mujoco
        import mujoco.mjx as mjx
        _MJX_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


def _check_deps():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for GPU-vectorized environments.\n"
            "Install with: pip install 'jax[cuda12]' (GPU) or pip install jax (CPU)"
        )
    if not _MJX_AVAILABLE:
        raise ImportError(
            "MuJoCo MJX is required for GPU-vectorized environments.\n"
            "Install with: pip install mujoco-mjx"
        )


class MJXState(NamedTuple):
    """State container for vectorized simulation."""
    mjx_data: Any          # mjx.Data (batched)
    step_count: Any        # jnp.ndarray (num_envs,)
    rng: Any               # PRNGKey
    done: Any              # jnp.ndarray (num_envs,) bool
    info: Dict[str, Any]   # additional info


class MJXVectorAviary:
    """GPU-vectorized multi-drone aviary using MuJoCo MJX.

    Runs N_envs independent simulations in parallel on GPU using JAX's
    vmap over MJX compiled physics. All compute stays on device.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments (typical: 1024-65536).
    num_drones : int
        Number of drones per environment.
    task : str
        Task type: "hover", "track", "stabilize".
    sim_freq : int
        Physics simulation frequency (Hz).
    ctrl_freq : int
        Control frequency (Hz).
    episode_length : int
        Maximum episode steps.
    target_height : float
        Target hover height for "hover" task.
    backend : str
        JAX backend ("gpu", "cpu"). Auto-detected if None.

    Example
    -------
    >>> env = MJXVectorAviary(num_envs=4096, task="hover")
    >>> rng = jax.random.PRNGKey(0)
    >>> state = env.reset(rng)
    >>> action = jnp.zeros((4096, 4))  # normalized [-1, 1]
    >>> state, obs, reward, done, info = env.step(state, action)
    """

    def __init__(
        self,
        num_envs: int = 4096,
        num_drones: int = 1,
        task: str = "hover",
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        episode_length: int = 500,
        target_height: float = 1.0,
        backend: Optional[str] = None,
    ):
        _check_deps()

        self.num_envs = num_envs
        self.num_drones = num_drones
        self.task = task
        self.sim_freq = sim_freq
        self.ctrl_freq = ctrl_freq
        self.sim_steps_per_ctrl = sim_freq // ctrl_freq
        self.episode_length = episode_length
        self.target_height = target_height

        # Physical constants (Crazyflie 2.x)
        self.mass = 0.027
        self.gravity = 9.81
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.arm_length = 0.0397
        self.max_rpm = 21714.0
        self.hover_rpm = np.sqrt((self.mass * self.gravity) / (4 * self.kf))

        # Build MuJoCo model
        self._xml = self._generate_xml()
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml)

        # Compile to MJX
        self._mjx_model = mjx.put_model(self._mj_model)

        # Shapes
        self.obs_dim = 12 * num_drones  # pos(3) + rpy(3) + vel(3) + angvel(3)
        self.act_dim = 4 * num_drones   # normalized RPM per motor

        # JIT-compile core functions
        self._step_fn = jit(vmap(self._single_step))
        self._reset_fn = jit(vmap(self._single_reset))

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.obs_dim,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.act_dim,)

    def reset(self, rng: Any) -> MJXState:
        """Reset all environments.

        Parameters
        ----------
        rng : PRNGKey
            JAX random key.

        Returns
        -------
        state : MJXState
            Initial state with batched mjx data.
        """
        rngs = random.split(rng, self.num_envs)
        # Initialize mjx data for each env
        mjx_data = mjx.put_data(self._mj_model, mujoco.MjData(self._mj_model))
        batched_data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num_envs, *x.shape)).copy(),
            mjx_data
        )
        state = MJXState(
            mjx_data=batched_data,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            rng=rng,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            info={},
        )
        # Apply per-env randomization via vmap'd reset
        state = self._reset_fn(state, rngs)
        return state

    def step(self, state: MJXState, action: Any) -> Tuple[MJXState, Any, Any, Any, Dict]:
        """Step all environments in parallel.

        Parameters
        ----------
        state : MJXState
            Current batched state.
        action : jnp.ndarray (num_envs, act_dim)
            Normalized actions in [-1, 1].

        Returns
        -------
        state : MJXState
            Next state.
        obs : jnp.ndarray (num_envs, obs_dim)
            Observations.
        reward : jnp.ndarray (num_envs,)
            Rewards.
        done : jnp.ndarray (num_envs,) bool
            Episode termination flags.
        info : dict
            Additional info.
        """
        # Clip actions
        action = jnp.clip(action, -1.0, 1.0)
        state, obs, reward, done = self._step_fn(state, action)
        return state, obs, reward, done, state.info

    def get_obs(self, state: MJXState) -> Any:
        """Extract observations from state."""
        return vmap(self._single_obs)(state.mjx_data)

    def _single_step(self, state: MJXState, action: Any) -> Tuple[MJXState, Any, Any, Any]:
        """Step a single environment (vmapped over batch)."""
        # Convert normalized action to RPM
        rpm = (action + 1) / 2 * self.max_rpm  # [-1,1] → [0, max_rpm]

        # Compute forces from RPM
        forces = self.kf * rpm ** 2
        total_thrust = jnp.sum(forces)

        # Apply thrust along body z-axis via xfrc_applied
        # For single drone, body_id=1 (worldbody=0, drone=1)
        body_id = 1
        data = state.mjx_data

        # Get body rotation matrix
        xmat = data.xmat[body_id].reshape(3, 3)
        thrust_world = xmat @ jnp.array([0.0, 0.0, total_thrust])

        # Compute torques (X-configuration)
        L = self.arm_length
        s2 = jnp.sqrt(2.0)
        tau_x = (forces[0] + forces[1] - forces[2] - forces[3]) * L / s2
        tau_y = (-forces[0] + forces[1] + forces[2] - forces[3]) * L / s2
        tau_z = (-forces[0] + forces[1] - forces[2] + forces[3]) * self.km / self.kf
        torque_body = jnp.array([tau_x, tau_y, tau_z])
        torque_world = xmat @ torque_body

        # Set xfrc_applied
        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[body_id, :3].set(thrust_world)
        xfrc = xfrc.at[body_id, 3:].set(torque_world)
        data = data.replace(xfrc_applied=xfrc)

        # Step physics multiple times per control step
        def physics_step(data, _):
            data = mjx.step(self._mjx_model, data)
            return data, None

        data, _ = lax.scan(physics_step, data, None, length=self.sim_steps_per_ctrl)

        # Extract observation
        obs = self._single_obs(data)

        # Compute reward
        reward = self._compute_reward(data, action)

        # Check termination
        step_count = state.step_count + 1
        done = step_count >= self.episode_length

        new_state = MJXState(
            mjx_data=data,
            step_count=step_count,
            rng=state.rng,
            done=done,
            info=state.info,
        )
        return new_state, obs, reward, done

    def _single_reset(self, state: MJXState, rng: Any) -> MJXState:
        """Reset a single environment (vmapped over batch)."""
        data = mjx.put_data(self._mj_model, mujoco.MjData(self._mj_model))
        # Add small random perturbation to initial position
        pos_noise = random.normal(rng, shape=(3,)) * 0.02
        qpos = data.qpos.at[0:3].add(pos_noise)
        data = data.replace(qpos=qpos)
        return MJXState(
            mjx_data=data,
            step_count=jnp.int32(0),
            rng=rng,
            done=jnp.bool_(False),
            info={},
        )

    def _single_obs(self, data: Any) -> Any:
        """Extract observation from MJX data for single env."""
        body_id = 1
        pos = data.xpos[body_id]
        # Quaternion to RPY (simplified via rotation matrix)
        xmat = data.xmat[body_id].reshape(3, 3)
        roll = jnp.arctan2(xmat[2, 1], xmat[2, 2])
        pitch = jnp.arcsin(-jnp.clip(xmat[2, 0], -1, 1))
        yaw = jnp.arctan2(xmat[1, 0], xmat[0, 0])
        rpy = jnp.array([roll, pitch, yaw])

        vel = data.cvel[body_id, 3:]   # linear velocity
        ang_vel = data.cvel[body_id, :3]  # angular velocity

        return jnp.concatenate([pos, rpy, vel, ang_vel])

    def _compute_reward(self, data: Any, action: Any) -> Any:
        """Compute reward for hover task."""
        body_id = 1
        pos = data.xpos[body_id]

        if self.task == "hover":
            # Distance to target height
            target = jnp.array([0.0, 0.0, self.target_height])
            dist = jnp.linalg.norm(pos - target)
            # Reward: negative distance + action penalty
            reward = -dist - 0.01 * jnp.sum(action ** 2)
        elif self.task == "stabilize":
            # Penalize velocity and angular velocity
            vel = data.cvel[body_id, 3:]
            ang_vel = data.cvel[body_id, :3]
            reward = -jnp.linalg.norm(vel) - 0.5 * jnp.linalg.norm(ang_vel)
        else:
            reward = jnp.float32(0.0)

        return reward

    def _generate_xml(self) -> str:
        """Generate minimal MuJoCo XML for MJX compilation."""
        # MJX needs a simpler model (no meshes, simplified collision)
        mass = self.mass
        ixx, iyy, izz = 1.4e-5, 1.4e-5, 2.17e-5
        L = self.arm_length

        drones = ""
        for d in range(self.num_drones):
            x_offset = d * 0.5
            drones += f"""
    <body name="drone{d}" pos="{x_offset} 0 0.115">
      <freejoint name="drone{d}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="cylinder" size="0.06 0.015" rgba="0.2 0.2 0.8 1" contype="1" conaffinity="1"/>
      <site name="drone{d}_center" pos="0 0 0"/>
    </body>"""

        xml = f"""<mujoco model="mjx_aviary">
  <option integrator="RK4" timestep="{1.0/self.sim_freq}" gravity="0 0 -{self.gravity}"/>
  <compiler autolimits="true"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="10 10 0.05" type="plane" contype="1" conaffinity="1"/>
{drones}
  </worldbody>
</mujoco>"""
        return xml


class MJXVecEnvGymWrapper:
    """Gymnasium VectorEnv-compatible wrapper around MJXVectorAviary.

    Bridges the JAX-native interface to numpy-based Gymnasium API
    for compatibility with SB3 and other standard RL libraries.

    Note: This involves GPU→CPU transfers each step. For maximum
    performance, use the JAX-native interface directly.

    Example
    -------
    >>> env = MJXVecEnvGymWrapper(num_envs=64, task="hover")
    >>> obs = env.reset()
    >>> obs, reward, done, info = env.step(np.zeros((64, 4)))
    """

    def __init__(self, **kwargs):
        _check_deps()
        self._env = MJXVectorAviary(**kwargs)
        self._state = None
        self.num_envs = self._env.num_envs
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self._env.observation_shape, dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=self._env.action_shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.num_envs, *self._env.observation_shape),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0,
            shape=(self.num_envs, *self._env.action_shape),
            dtype=np.float32,
        )

    def reset(self, seed: int = 0):
        rng = random.PRNGKey(seed)
        self._state = self._env.reset(rng)
        obs = self._env.get_obs(self._state)
        return np.asarray(obs)

    def step(self, action: np.ndarray):
        action_jax = jnp.array(action)
        self._state, obs, reward, done, info = self._env.step(self._state, action_jax)
        return (
            np.asarray(obs),
            np.asarray(reward),
            np.asarray(done),
            info,
        )

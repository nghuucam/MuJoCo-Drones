import os
import sys
from typing import NamedTuple, Tuple, Dict, Any
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config_mjx as cfg
from model_xml_builder import generate_static_mjx_xml
from apf_jax import compute_apf_hint_jax

IMPORT_ERROR_MSG = None
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, random
    import mujoco
    try:
        from mujoco import mjx
    except ImportError:
        import mjx
    HAS_JAX = True
except Exception as e:
    HAS_JAX = False
    IMPORT_ERROR_MSG = str(e)


class DroneMJXState(NamedTuple):
    data: Any
    step_count: Any
    goal: Any
    best_dist: Any
    level: Any
    rng: Any


class DroneMJXEnv:
    def __init__(self, num_envs: int = cfg.NUM_ENVS, level: int = 0):
        if not HAS_JAX:
            raise ImportError("JAX hoặc MuJoCo MJX chưa được cài đặt. Hãy cài đặt: pip install jax jaxlib mujoco")

        self.num_envs = num_envs
        self.default_level = level

        xml_str = generate_static_mjx_xml()
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.obs_dim = 18
        self.act_dim = 3

        self.v_step = jit(vmap(self._single_step, in_axes=(0, 0, 0)))
        self.v_reset = jit(vmap(self._single_reset, in_axes=(0, 0)))

    def set_level(self, level: int):
        self.default_level = int(level)

    def _single_reset(self, rng: jnp.ndarray, level: int) -> Tuple[DroneMJXState, jnp.ndarray]:
        rng, r_goal_x, r_goal_y = random.split(rng, 3)

        y_ranges = jnp.array(cfg.GOAL_Y_RANGES)
        y_min = y_ranges[level, 0]
        y_max = y_ranges[level, 1]

        goal_x = random.uniform(r_goal_x, minval=-2.0, maxval=2.0)
        goal_y = random.uniform(r_goal_y, minval=y_min, maxval=y_max)
        goal = jnp.array([goal_x, goal_y, 1.5])

        data = mjx.make_data(self.mjx_model)
        init_qpos = data.qpos.at[0:3].set(jnp.array([0.0, 11.0, 1.0]))
        init_qpos = init_qpos.at[3:7].set(jnp.array([0.7071068, 0.0, 0.0, -0.7071068]))
        data = data.replace(qpos=init_qpos, qvel=jnp.zeros_like(data.qvel))

        init_dist = jnp.linalg.norm(init_qpos[0:3] - goal)

        state = DroneMJXState(
            data=data,
            step_count=jnp.array(0, dtype=jnp.int32),
            goal=goal,
            best_dist=init_dist,
            level=jnp.array(level, dtype=jnp.int32),
            rng=rng
        )

        obs = self._get_obs(state)
        return state, obs

    def _get_obs(self, state: DroneMJXState) -> jnp.ndarray:
        pos = state.data.qpos[0:3]
        vel = state.data.qvel[0:3]
        quat = state.data.qpos[3:7]
        goal = state.goal
        rel_vec = goal - pos
        dist = jnp.linalg.norm(rel_vec)

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.where(jnp.abs(sinp) >= 1.0, jnp.sign(sinp) * (jnp.pi / 2.0), jnp.arcsin(jnp.clip(sinp, -1.0, 1.0)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.atan2(siny_cosp, cosy_cosp)
        rpy = jnp.array([roll, pitch, yaw])

        apf_hint = compute_apf_hint_jax(pos, rpy, goal, state.level)

        obs = jnp.concatenate([
            pos,
            vel,
            rpy,
            goal,
            rel_vec,
            jnp.array([dist]),
            apf_hint
        ])
        return obs

    def _single_step(self, state: DroneMJXState, action: jnp.ndarray, target_level: jnp.ndarray) -> Tuple[DroneMJXState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        action = jnp.clip(action, -1.0, 1.0)
        norm_a, norm_b, norm_d = action[0], action[1], action[2]

        yaw = -1.570796 + norm_a * 0.610865
        pitch = norm_b * 0.523599
        step_dist = 0.2 + (norm_d + 1.0) * 0.65

        target_dx = step_dist * jnp.cos(pitch) * jnp.cos(yaw)
        target_dy = step_dist * jnp.cos(pitch) * jnp.sin(yaw)
        target_dz = step_dist * jnp.sin(pitch)

        cur_pos = state.data.qpos[0:3]
        target_pos = cur_pos + jnp.array([target_dx, target_dy, target_dz])
        target_pos = jnp.clip(
            target_pos,
            jnp.array([-cfg.X_LIMIT, -cfg.Y_LIMIT, cfg.MIN_Z]),
            jnp.array([cfg.X_LIMIT, cfg.Y_LIMIT, cfg.MAX_Z])
        )

        def sub_step(i, d):
            err_pos = target_pos - d.qpos[0:3]
            err_vel = -d.qvel[0:3]
            acc_des = 8.0 * err_pos + 3.0 * err_vel
            force = acc_des * cfg.CF2_MASS
            force = force.at[2].add(cfg.CF2_MASS * cfg.GRAVITY)
            ctrl = jnp.clip(force, -1.5, 1.5)
            d = d.replace(ctrl=ctrl)
            return mjx.step(self.mjx_model, d)

        data = jax.lax.fori_loop(0, cfg.SUBSTEPS, sub_step, state.data)

        new_pos = data.qpos[0:3]
        cur_dist = jnp.linalg.norm(new_pos - state.goal)

        r_time = -0.002 * cfg.SUBSTEPS
        progress = jnp.maximum(0.0, state.best_dist - cur_dist)
        r_prog = 1.0 * progress
        new_best = jnp.minimum(state.best_dist, cur_dist)

        win = cur_dist < cfg.GOAL_THRESHOLD
        over_map = (jnp.abs(new_pos[0]) > cfg.X_LIMIT) | (jnp.abs(new_pos[1]) > cfg.Y_LIMIT) | (new_pos[2] < cfg.MIN_Z) | (new_pos[2] > cfg.MAX_Z)

        ncon = data.ncon
        collision = ncon > 0

        new_step = state.step_count + 1
        timeout = new_step >= cfg.MAX_STEPS

        done = win | over_map | collision | timeout

        remaining = jnp.maximum(0, cfg.MAX_STEPS - new_step)
        r_win = jnp.where(win, 100.0 + 0.5 * remaining, 0.0)
        r_col = jnp.where(collision, -50.0, 0.0)
        r_map = jnp.where(over_map & (~win), -50.0, 0.0)
        r_time_out = jnp.where(timeout & (~win), -50.0, 0.0)

        total_r = r_time + r_prog + r_win + r_col + r_map + r_time_out

        step_state = DroneMJXState(
            data=data,
            step_count=new_step,
            goal=state.goal,
            best_dist=new_best,
            level=state.level,
            rng=state.rng
        )
        step_obs = self._get_obs(step_state)

        rng, r_reset = random.split(state.rng)
        reset_state, reset_obs = self._single_reset(r_reset, target_level)

        next_state = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_state, step_state)
        obs = jnp.where(done, reset_obs, step_obs)

        info = {
            "win": win,
            "collision": collision,
            "over_map": over_map,
            "dist": cur_dist,
            "terminal_observation": step_obs
        }
        return next_state, obs, total_r, done, info

    def reset(self, rng: jnp.ndarray, level: int = None) -> Tuple[DroneMJXState, jnp.ndarray]:
        if level is None:
            level = self.default_level
        levels = jnp.full((self.num_envs,), level, dtype=jnp.int32)
        rngs = random.split(rng, self.num_envs)
        return self.v_reset(rngs, levels)

    def step(self, state: DroneMJXState, actions: jnp.ndarray, level: int = None):
        if level is None:
            level = self.default_level
        levels = jnp.full((self.num_envs,), level, dtype=jnp.int32)
        return self.v_step(state, actions, levels)

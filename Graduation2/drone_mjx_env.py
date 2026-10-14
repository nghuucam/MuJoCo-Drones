import os
import sys
from typing import NamedTuple, Tuple, Dict, Any
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config_mjx as cfg
from model_xml_builder import generate_static_mjx_xml

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, random
    import mujoco
    from mujoco import mjx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class DroneMJXState(NamedTuple):
    data: Any
    step_count: Any
    goal: Any
    best_dist: Any
    level: Any
    rng: Any


class DroneMJXEnv:
    def __init__(self, num_envs: int = cfg.NUM_ENVS, level: int = 6):
        if not HAS_JAX:
            raise ImportError("JAX hoặc MuJoCo MJX chưa được cài đặt. Hãy cài đặt: pip install jax jaxlib mujoco")

        self.num_envs = num_envs
        self.default_level = level

        xml_str = generate_static_mjx_xml()
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.obs_dim = 18
        self.act_dim = 3

        self.v_step = jit(vmap(self._single_step, in_axes=(None, 0, 0)))
        self.v_reset = jit(vmap(self._single_reset, in_axes=(None, 0, 0)))

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
        init_qpos = init_qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
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

        dummy_rpy = jnp.zeros(3)
        dummy_apf = jnp.zeros(2)

        obs = jnp.concatenate([
            pos,
            vel,
            dummy_rpy,
            goal,
            rel_vec,
            jnp.array([dist]),
            dummy_apf
        ])
        return obs

    def _single_step(self, state: DroneMJXState, action: jnp.ndarray) -> Tuple[DroneMJXState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        action = jnp.clip(action, -1.0, 1.0)
        alpha, beta, step_dist = action[0], action[1], (action[2] + 1.0) * 0.5 * 1.3 + 0.2

        yaw = -1.570796 + alpha * 0.610865
        pitch = beta * 0.523599

        target_dx = step_dist * jnp.cos(pitch) * jnp.cos(yaw)
        target_dy = step_dist * jnp.cos(pitch) * jnp.sin(yaw)
        target_dz = step_dist * jnp.sin(pitch)

        cur_pos = state.data.qpos[0:3]
        target_pos = cur_pos + jnp.array([target_dx, target_dy, target_dz])

        def sub_step(i, d):
            kp_pos = 5.0
            kp_vel = 2.0
            err_pos = target_pos - d.qpos[0:3]
            err_vel = -d.qvel[0:3]
            acc_des = kp_pos * err_pos + kp_vel * err_vel

            thrust_z = jnp.clip(acc_des[2] + cfg.GRAVITY, 0.0, 20.0) * (cfg.CF2_MASS / 4.0)
            ctrl_val = (thrust_z / (cfg.CF2_MASS * cfg.GRAVITY)) * 2.0 - 1.0
            ctrl = jnp.array([ctrl_val, acc_des[0]*0.1, acc_des[1]*0.1, 0.0])
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

        next_state = DroneMJXState(
            data=data,
            step_count=new_step,
            goal=state.goal,
            best_dist=new_best,
            level=state.level,
            rng=state.rng
        )

        obs = self._get_obs(next_state)
        info = {
            "win": win,
            "collision": collision,
            "over_map": over_map,
            "dist": cur_dist
        }
        return next_state, obs, total_r, done, info

    def reset(self, rng: jnp.ndarray, level: int = None) -> Tuple[DroneMJXState, jnp.ndarray]:
        if level is None:
            level = self.default_level
        levels = jnp.full((self.num_envs,), level, dtype=jnp.int32)
        rngs = random.split(rng, self.num_envs)
        return self.v_reset(rngs, levels)

    def step(self, state: DroneMJXState, actions: jnp.ndarray):
        return self.v_step(state, actions)

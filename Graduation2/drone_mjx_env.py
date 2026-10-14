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


def yaw_to_quat(yaw):
    return jnp.array([jnp.cos(yaw / 2.0), 0.0, 0.0, jnp.sin(yaw / 2.0)])


def quat_to_yaw(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_contact(data):
    impl = getattr(data, "_impl", None)
    if impl is not None and hasattr(impl, "contact"):
        return impl.contact
    return getattr(data, "contact", None)


def has_penetration(data):
    contact = get_contact(data)
    if contact is None or contact.dist.shape[0] == 0:
        return jnp.array(False)
    return jnp.any(contact.dist < 0.0)


class DroneMJXEnv:
    def __init__(self, num_envs: int = cfg.NUM_ENVS, level: int = 0):
        if not HAS_JAX:
            raise ImportError("JAX hoặc MuJoCo MJX chưa được cài đặt. Hãy cài đặt: pip install jax jaxlib mujoco")

        self.num_envs = num_envs
        self.default_level = level

        xml_str = generate_static_mjx_xml()
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.base_data = mjx.make_data(self.mjx_model)

        self.obs_dim = 18
        self.act_dim = 3

        self.v_step = jit(vmap(self._single_step, in_axes=(0, 0, 0)))
        self.v_reset = jit(vmap(self._single_reset, in_axes=(0, 0)))

    def set_level(self, level: int):
        self.default_level = int(level)

    def _sample_episode(self, rng, level):
        rng, r_goal_x, r_goal_y = random.split(rng, 3)

        y_ranges = jnp.array(cfg.GOAL_Y_RANGES)
        goal_x = random.uniform(r_goal_x, minval=-2.0, maxval=2.0)
        goal_y = random.uniform(r_goal_y, minval=y_ranges[level, 0], maxval=y_ranges[level, 1])
        goal = jnp.array([goal_x, goal_y, cfg.GOAL_Z])

        active_count = jnp.array(cfg.OBSTACLE_COUNTS)[level]
        obs_xy = jnp.array(cfg.FIXED_OBSTACLE_POSITIONS)
        z_pos = jnp.where(jnp.arange(obs_xy.shape[0]) < active_count, 0.0, -100.0)
        obs_mpos = jnp.stack([obs_xy[:, 0], obs_xy[:, 1], z_pos], axis=1)
        mocap_pos = jnp.concatenate([goal[None, :], obs_mpos], axis=0)

        init_qpos = jnp.concatenate([jnp.array(cfg.START_POS), yaw_to_quat(cfg.START_YAW)])
        init_dist = jnp.linalg.norm(init_qpos[0:3] - goal)
        return rng, goal, mocap_pos, init_qpos, init_dist

    def _single_reset(self, rng: jnp.ndarray, level: int) -> Tuple[DroneMJXState, jnp.ndarray]:
        rng, goal, mocap_pos, init_qpos, init_dist = self._sample_episode(rng, level)

        data = self.base_data.replace(
            qpos=init_qpos,
            qvel=jnp.zeros_like(self.base_data.qvel),
            mocap_pos=mocap_pos,
            ctrl=jnp.zeros_like(self.base_data.ctrl)
        )

        state = DroneMJXState(
            data=data,
            step_count=jnp.array(0, dtype=jnp.int32),
            goal=goal,
            best_dist=init_dist,
            level=jnp.array(level, dtype=jnp.int32),
            rng=rng
        )
        return state, self._get_obs(state)

    def _get_obs(self, state: DroneMJXState) -> jnp.ndarray:
        pos = state.data.qpos[0:3]
        vel = state.data.qvel[0:3]
        quat = state.data.qpos[3:7]
        goal = state.goal
        rel_vec = goal - pos
        dist = jnp.linalg.norm(rel_vec)

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        roll = jnp.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))
        yaw = quat_to_yaw(quat)
        rpy = jnp.array([roll, pitch, yaw])

        apf_hint = compute_apf_hint_jax(pos, rpy, goal, state.level)

        return jnp.concatenate([pos, vel, rpy, goal, rel_vec, jnp.array([dist]), apf_hint])

    def _single_step(self, state: DroneMJXState, action: jnp.ndarray, target_level: jnp.ndarray) -> Tuple[DroneMJXState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        action = jnp.clip(action, -1.0, 1.0)
        norm_a, norm_b, norm_d = action[0], action[1], action[2]

        heading = quat_to_yaw(state.data.qpos[3:7])
        move_yaw = heading + norm_a * cfg.MAX_ALPHA_RAD
        pitch = norm_b * cfg.MAX_BETA_RAD
        step_dist = cfg.MIN_STEP_DIST + (norm_d + 1.0) * cfg.STEP_DIST_SCALE

        start_pos = state.data.qpos[0:3]
        delta = step_dist * jnp.array([
            jnp.cos(pitch) * jnp.cos(move_yaw),
            jnp.cos(pitch) * jnp.sin(move_yaw),
            jnp.sin(pitch)
        ])
        target_pos = jnp.clip(
            start_pos + delta,
            jnp.array([-cfg.X_LIMIT, -cfg.Y_LIMIT, cfg.MIN_Z]),
            jnp.array([cfg.X_LIMIT, cfg.Y_LIMIT, cfg.MAX_Z])
        )
        delta = target_pos - start_pos

        new_qpos = state.data.qpos.at[3:7].set(yaw_to_quat(move_yaw))
        new_qvel = state.data.qvel.at[3:6].set(0.0)
        data0 = state.data.replace(qpos=new_qpos, qvel=new_qvel)

        duration = cfg.ACTION_DURATION
        goal = state.goal
        cos_y = jnp.cos(move_yaw)
        sin_y = jnp.sin(move_yaw)

        def sub_step(i, carry):
            d, hit, won = carry
            tau = (i + 1.0) / cfg.SUBSTEPS
            s = 3.0 * tau ** 2 - 2.0 * tau ** 3
            ds = (6.0 * tau - 6.0 * tau ** 2) / duration
            dds = (6.0 - 12.0 * tau) / (duration ** 2)

            ref_pos = start_pos + delta * s
            ref_vel = delta * ds
            ref_acc = delta * dds

            acc_cmd = ref_acc + cfg.TRACK_KP * (ref_pos - d.qpos[0:3]) + cfg.TRACK_KD * (ref_vel - d.qvel[0:3])
            force = acc_cmd * cfg.CF2_MASS
            force = force.at[2].add(cfg.CF2_MASS * cfg.GRAVITY)
            force_body = jnp.array([
                cos_y * force[0] + sin_y * force[1],
                -sin_y * force[0] + cos_y * force[1],
                force[2]
            ])
            ctrl = jnp.clip(force_body, -cfg.MAX_FORCE, cfg.MAX_FORCE)

            d = mjx.step(self.mjx_model, d.replace(ctrl=ctrl))

            active = ~(hit | won)
            now_hit = has_penetration(d)
            now_won = jnp.linalg.norm(d.qpos[0:3] - goal) < cfg.GOAL_THRESHOLD
            hit = hit | (active & now_hit)
            won = won | (active & now_won & ~now_hit)
            return d, hit, won

        data, collision, win = jax.lax.fori_loop(
            0, cfg.SUBSTEPS, sub_step, (data0, jnp.array(False), jnp.array(False))
        )

        new_pos = data.qpos[0:3]
        cur_dist = jnp.linalg.norm(new_pos - goal)
        cur_dist = jnp.where(win, jnp.minimum(cur_dist, cfg.GOAL_THRESHOLD), cur_dist)

        r_time = -0.002 * cfg.SUBSTEPS
        progress = jnp.maximum(0.0, state.best_dist - cur_dist)
        r_prog = 3.0 * progress
        new_best = jnp.minimum(state.best_dist, cur_dist)

        over_map = (
            (jnp.abs(new_pos[0]) > cfg.X_LIMIT) | (jnp.abs(new_pos[1]) > cfg.Y_LIMIT)
            | (new_pos[2] < cfg.MIN_Z) | (new_pos[2] > cfg.MAX_Z)
        ) & ~win & ~collision

        new_step = state.step_count + 1
        timeout = (new_step >= cfg.MAX_STEPS) & ~win & ~collision & ~over_map

        done = win | over_map | collision | timeout

        remaining = jnp.maximum(0, cfg.MAX_STEPS - new_step)
        r_win = jnp.where(win, 100.0 + 0.5 * remaining, 0.0)
        r_col = jnp.where(collision, -50.0, 0.0)
        r_map = jnp.where(over_map, -50.0, 0.0)
        r_time_out = jnp.where(timeout, -20.0, 0.0)

        total_r = r_time + r_prog + r_win + r_col + r_map + r_time_out

        step_state = DroneMJXState(
            data=data,
            step_count=new_step,
            goal=goal,
            best_dist=new_best,
            level=state.level,
            rng=state.rng
        )
        step_obs = self._get_obs(step_state)

        rng, reset_goal, reset_mpos, init_qpos, init_dist = self._sample_episode(state.rng, target_level)

        next_data = data.replace(
            qpos=jnp.where(done, init_qpos, data.qpos),
            qvel=jnp.where(done, jnp.zeros_like(data.qvel), data.qvel),
            mocap_pos=jnp.where(done, reset_mpos, data.mocap_pos),
            ctrl=jnp.where(done, jnp.zeros_like(data.ctrl), data.ctrl)
        )

        next_state = DroneMJXState(
            data=next_data,
            step_count=jnp.where(done, 0, new_step),
            goal=jnp.where(done, reset_goal, goal),
            best_dist=jnp.where(done, init_dist, new_best),
            level=jnp.where(done, target_level, state.level),
            rng=rng
        )
        obs = self._get_obs(next_state)

        info = {
            "win": win,
            "collision": collision,
            "over_map": over_map,
            "timeout": timeout,
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

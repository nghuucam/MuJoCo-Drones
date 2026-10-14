import jax
import jax.numpy as jnp
from jax import jit
import config_mjx as cfg

@jit
def compute_apf_hint_jax(drone_pos: jnp.ndarray, drone_rpy: jnp.ndarray, goal_pos: jnp.ndarray, level: int) -> jnp.ndarray:
    v_goal = (goal_pos - drone_pos)[:2]
    d_goal = jnp.linalg.norm(v_goal)
    u_goal = v_goal / (d_goal + 1e-6)

    obs_positions = jnp.array(cfg.FIXED_OBSTACLE_POSITIONS)
    v_obs = obs_positions - drone_pos[:2]
    dist_obs = jnp.linalg.norm(v_obs, axis=1)

    proj = jnp.dot(v_obs, u_goal)
    in_corridor = (proj > 0.15) & (proj < jnp.minimum(d_goal, 3.5))

    v_perp = v_obs - proj[:, None] * u_goal[None, :]
    dist_perp = jnp.linalg.norm(v_perp, axis=1)

    r_corridor = cfg.CYLINDER_RADIUS + 0.35
    need_deflect = in_corridor & (dist_perp < r_corridor)

    cross_prod = u_goal[0] * v_obs[:, 1] - u_goal[1] * v_obs[:, 0]
    deflect_dir = jnp.where(cross_prod > 0, -1.0, 1.0)
    deflect_angle = jnp.where(need_deflect, deflect_dir * 0.45, 0.0)

    total_deflect = jnp.sum(deflect_angle)
    base_yaw = jnp.atan2(u_goal[1], u_goal[0])
    desired_yaw = base_yaw + total_deflect

    d_surf = jnp.maximum(0.0, dist_obs - cfg.CYLINDER_RADIUS)
    near_mask = (d_surf < 0.6)[:, None]
    rep_mag = jnp.where(d_surf[:, None] < 0.6, (0.6 - d_surf[:, None]) / 0.6, 0.0)
    norm_rep = -v_obs / (dist_obs[:, None] + 1e-6)

    f_rep_xy = jnp.sum(near_mask * rep_mag * norm_rep, axis=0)

    has_rep = jnp.linalg.norm(f_rep_xy) > 1e-3
    v_des = jnp.array([jnp.cos(desired_yaw), jnp.sin(desired_yaw)]) + 1.2 * f_rep_xy
    desired_yaw = jnp.where(has_rep, jnp.atan2(v_des[1], v_des[0]), desired_yaw)

    cur_yaw = drone_rpy[2]
    delta_yaw = (desired_yaw - cur_yaw + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    alpha_hint = jnp.clip(delta_yaw / 0.610865, -1.0, 1.0)

    dz = goal_pos[2] - drone_pos[2]
    desired_pitch = jnp.atan2(dz, d_goal)
    beta_hint = jnp.clip(desired_pitch / 0.523599, -1.0, 1.0)

    return jnp.array([alpha_hint, beta_hint], dtype=jnp.float32)

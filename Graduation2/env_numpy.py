import numpy as np
import config_mjx as cfg


def yaw_to_quat_np(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def quat_to_rpy_np(quat):
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def compute_apf_hint_np(drone_pos, drone_rpy, goal_pos):
    v_goal = (goal_pos - drone_pos)[:2]
    d_goal = np.linalg.norm(v_goal)
    u_goal = v_goal / (d_goal + 1e-6)

    obs_positions = np.array(cfg.FIXED_OBSTACLE_POSITIONS)
    v_obs = obs_positions - drone_pos[:2]
    dist_obs = np.linalg.norm(v_obs, axis=1)

    proj = v_obs @ u_goal
    in_corridor = (proj > 0.15) & (proj < min(d_goal, 3.5))

    v_perp = v_obs - proj[:, None] * u_goal[None, :]
    dist_perp = np.linalg.norm(v_perp, axis=1)

    r_corridor = cfg.CYLINDER_RADIUS + 0.35
    need_deflect = in_corridor & (dist_perp < r_corridor)

    cross_prod = u_goal[0] * v_obs[:, 1] - u_goal[1] * v_obs[:, 0]
    deflect_dir = np.where(cross_prod > 0, -1.0, 1.0)
    deflect_angle = np.where(need_deflect, deflect_dir * 0.45, 0.0)

    desired_yaw = np.arctan2(u_goal[1], u_goal[0]) + np.sum(deflect_angle)

    d_surf = np.maximum(0.0, dist_obs - cfg.CYLINDER_RADIUS)
    near = d_surf < 0.6
    rep_mag = np.where(near, (0.6 - d_surf) / 0.6, 0.0)
    norm_rep = -v_obs / (dist_obs[:, None] + 1e-6)
    f_rep_xy = np.sum(near[:, None] * rep_mag[:, None] * norm_rep, axis=0)

    if np.linalg.norm(f_rep_xy) > 1e-3:
        v_des = np.array([np.cos(desired_yaw), np.sin(desired_yaw)]) + 1.2 * f_rep_xy
        desired_yaw = np.arctan2(v_des[1], v_des[0])

    delta_yaw = (desired_yaw - drone_rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
    alpha_hint = np.clip(delta_yaw / cfg.MAX_ALPHA_RAD, -1.0, 1.0)

    desired_pitch = np.arctan2(goal_pos[2] - drone_pos[2], d_goal)
    beta_hint = np.clip(desired_pitch / cfg.MAX_BETA_RAD, -1.0, 1.0)

    return np.array([alpha_hint, beta_hint], dtype=np.float32)


def build_obs_np(qpos, qvel, goal):
    pos = np.asarray(qpos[0:3], dtype=np.float64)
    vel = np.asarray(qvel[0:3], dtype=np.float64)
    rpy = quat_to_rpy_np(np.asarray(qpos[3:7], dtype=np.float64))
    rel_vec = goal - pos
    dist = np.linalg.norm(rel_vec)
    hint = compute_apf_hint_np(pos, rpy, goal)
    return np.concatenate([pos, vel, rpy, goal, rel_vec, [dist], hint]).astype(np.float32)


def plan_waypoint_np(qpos, action):
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    heading = quat_to_rpy_np(np.asarray(qpos[3:7], dtype=np.float64))[2]
    move_yaw = heading + action[0] * cfg.MAX_ALPHA_RAD
    pitch = action[1] * cfg.MAX_BETA_RAD
    step_dist = cfg.MIN_STEP_DIST + (action[2] + 1.0) * cfg.STEP_DIST_SCALE

    start_pos = np.array(qpos[0:3], dtype=np.float64)
    delta = step_dist * np.array([
        np.cos(pitch) * np.cos(move_yaw),
        np.cos(pitch) * np.sin(move_yaw),
        np.sin(pitch)
    ])
    target = np.clip(
        start_pos + delta,
        [-cfg.X_LIMIT, -cfg.Y_LIMIT, cfg.MIN_Z],
        [cfg.X_LIMIT, cfg.Y_LIMIT, cfg.MAX_Z]
    )
    return start_pos, target - start_pos, move_yaw


def world_to_body_force_np(force, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * force[0] + s * force[1], -s * force[0] + c * force[1], force[2]])


def tracking_force_np(i, start_pos, delta, pos, vel, yaw):
    duration = cfg.ACTION_DURATION
    tau = (i + 1.0) / cfg.SUBSTEPS
    s = 3.0 * tau ** 2 - 2.0 * tau ** 3
    ds = (6.0 * tau - 6.0 * tau ** 2) / duration
    dds = (6.0 - 12.0 * tau) / (duration ** 2)

    acc_cmd = delta * dds + cfg.TRACK_KP * (start_pos + delta * s - pos) + cfg.TRACK_KD * (delta * ds - vel)
    force = acc_cmd * cfg.CF2_MASS
    force[2] += cfg.CF2_MASS * cfg.GRAVITY
    force = np.clip(world_to_body_force_np(force, yaw), -cfg.MAX_FORCE, cfg.MAX_FORCE)
    return force


def apf_pilot_action_np(obs):
    dist = float(obs[15])
    norm_d = np.clip((min(dist, 1.5) - cfg.MIN_STEP_DIST) / cfg.STEP_DIST_SCALE - 1.0, -1.0, 1.0)
    return np.array([obs[16], obs[17], norm_d], dtype=np.float32)

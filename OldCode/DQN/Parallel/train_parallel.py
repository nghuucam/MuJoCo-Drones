import os
import sys
import time
import csv
import traceback
import numpy as np
import torch
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.append(path)

from multi_drone_mujoco.utils.enums import DroneModel
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from mujoco_env import DroneEnv 
from agent import DQN, DroneNet

def get_action_hints(obs, env_goal, direct, sensors=None):
    goal_pos = env_goal[0] if hasattr(env_goal, "__len__") and len(env_goal) > 0 and isinstance(env_goal[0], (list, np.ndarray)) else env_goal
    state = obs[0:3]
    current_dist = np.linalg.norm(state - goal_pos)
    hints = []
    for target in direct:
        future_dist = np.linalg.norm(target - goal_pos)
        progress = current_dist - future_dist
        hints.append(progress)
    hints = np.array(hints, dtype=np.float32)
    max_val = np.max(np.abs(hints)) + 1e-8
    hints = (hints / max_val) * 0.3

    if sensors is not None:
        d = np.array(sensors, dtype=np.float32) * 5.0
        if min(d[3], d[4], d[5]) < 1.8:
            hints[0] = -0.4
        if min(d[7], d[8]) < 1.3:
            hints[1] = -0.4
        if min(d[0], d[1]) < 1.3:
            hints[2] = -0.4
        if min(d[2], d[3]) < 1.5:
            hints[3] = -0.4
        if min(d[5], d[6]) < 1.5:
            hints[4] = -0.4

    return hints

def format_lidar_str(sensor_fracs, max_range=5.0):
    d = np.array(sensor_fracs, dtype=np.float32) * max_range
    def mark(val):
        return f"{val:.1f}⚠️" if val < 1.5 else f"{val:.1f}"

    return (f"  📡 LiDAR (m): [L90: {mark(d[0])} | L67: {mark(d[1])} | L45: {mark(d[2])} | L22: {mark(d[3])} | "
            f"F0: {mark(d[4])} | R22: {mark(d[5])} | R45: {mark(d[6])} | R67: {mark(d[7])} | R90: {mark(d[8])}]")

def get_vec_state(obs, env_goal, action_hints, raycast_sensors):
    pos   = obs[0:3]
    rpy   = obs[7:10]
    vel   = obs[10:13]
    rel_goal = env_goal[0] - pos
    vec = np.concatenate([rpy, vel, rel_goal, action_hints, raycast_sensors]) 
    return np.expand_dims(vec, axis=0)

def get_img_state(rgb_img):
    if isinstance(rgb_img, tuple):
        rgb_img = rgb_img[0]
    img = rgb_img[:, :, :3].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def worker_loop(remote, worker_id):
    """Tiến trình con chạy độc lập 1 môi trường DroneEnv trong MuJoCo cho DQN."""
    try:
        env = DroneEnv(gui=False)
        control = DSLPIDControl(env=env)
    except Exception as e:
        try:
            remote.send(("ERROR", f"Lỗi khởi tạo DroneEnv worker {worker_id}: {traceback.format_exc()}"))
        except Exception:
            pass
        return

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "reset":
                control.reset()
                options = data if data else "reset"
                obs, _ = env.reset(options=options)
                rgb = env._getDroneImages(nth_drone=0)
                sensors = env.get_raycast_sensors()
                remote.send(("OK", (obs, rgb, sensors, env.start, env.goal)))

            elif cmd == "go_up":
                control.reset()
                cur_obs = env._getDroneStateVector(0)
                current_x, current_y = cur_obs[0:2]
                start_time = time.time()
                TIMEOUT = 10.0
                obs_ret = cur_obs

                while True:
                    cur_obs = env._getDroneStateVector(0)
                    cur_pos = cur_obs[0:3]
                    cur_quat = cur_obs[3:7]  
                    cur_vel = cur_obs[10:13]  
                    cur_ang_v = cur_obs[13:16]

                    target_pos = np.array([current_x, current_y, 1.0])
                    action_go_up, _, _ = control.computeControl(
                        control_timestep=env.CTRL_TIMESTEP,
                        cur_pos=cur_pos,
                        cur_quat=cur_quat,
                        cur_vel=cur_vel,
                        cur_ang_vel=cur_ang_v,
                        target_pos=target_pos,
                        target_rpy=np.array([0.0, 0.0, 0.0]),
                    )

                    obs_ret, _, _, _, _, _ = env.step(action_go_up.flatten())
                    if isinstance(obs_ret, tuple):
                        obs_ret = obs_ret[0]

                    if (abs(cur_pos[2] - 1.0) < 0.08 and np.linalg.norm(cur_vel) < 0.20) or (time.time() - start_time > TIMEOUT):
                        break

                env.previous_dist = np.linalg.norm(obs_ret[0:3] - env.goal[0])
                rgb = env._getDroneImages(nth_drone=0)
                sensors = env.get_raycast_sensors()
                remote.send(("OK", (obs_ret, rgb, sensors, env.start, env.goal)))

            elif cmd == "step_action":
                target_pos = data
                accumulated_reward = 0
                terminated = False
                truncated = (False, "NONE", "NONE", "NONE")
                info = ""
                obs = env._getDroneStateVector(0)

                for step in range(200):
                    cur_pos = obs[0:3]
                    cur_quat = obs[3:7]
                    cur_vel = obs[10:13]
                    cur_ang_v = obs[13:16]
                    z_point = cur_pos[2]

                    if z_point - 0.9 < 0:
                        control.reset()
                        cur_obs = env._getDroneStateVector(0)
                        current_x, current_y = cur_obs[0:2]
                        st_time = time.time()
                        while True:
                            cur_obs = env._getDroneStateVector(0)
                            c_pos = cur_obs[0:3]
                            c_quat = cur_obs[3:7]  
                            c_vel = cur_obs[10:13]  
                            c_ang_v = cur_obs[13:16]
                            target_p = np.array([current_x, current_y, 1.0])
                            act_up, _, _ = control.computeControl(
                                control_timestep=env.CTRL_TIMESTEP,
                                cur_pos=c_pos, cur_quat=c_quat,
                                cur_vel=c_vel, cur_ang_vel=c_ang_v,
                                target_pos=target_p, target_rpy=np.array([0.0, 0.0, 0.0])
                            )
                            obs_ret, _, _, _, _, _ = env.step(act_up.flatten())
                            if isinstance(obs_ret, tuple): obs_ret = obs_ret[0]
                            if (abs(c_pos[2] - 1.0) < 0.08 and np.linalg.norm(c_vel) < 0.20) or (time.time() - st_time > 10.0): break
                        
                        cur_pos = obs_ret[0:3]
                        cur_quat = obs_ret[3:7]
                        cur_vel = obs_ret[10:13]
                        cur_ang_v = obs_ret[13:16]
                        env.previous_dist = np.linalg.norm(cur_pos[0:3] - env.goal[0])

                    action_go, _, _ = control.computeControl(
                        control_timestep=env.CTRL_TIMESTEP,
                        cur_pos=cur_pos, cur_quat=cur_quat,
                        cur_vel=cur_vel, cur_ang_vel=cur_ang_v,
                        target_pos=target_pos, target_rpy=np.array([0.0, 0.0, 0.0])
                    )

                    next_obs, img, reward, terminated, truncated, info = env.step(action_go.flatten())
                    if isinstance(next_obs, tuple): next_obs = next_obs[0]
                    accumulated_reward += reward
                    obs = next_obs

                    dist_to_target = np.linalg.norm(cur_pos[0:2] - target_pos[0:2])
                    speed = np.linalg.norm(cur_vel)
                    wobble_speed = np.linalg.norm(cur_ang_v)

                    if step > 20 and dist_to_target < 0.05 and speed < 0.1 and wobble_speed < 0.2:
                        break
                    if terminated or (isinstance(truncated, tuple) and truncated[0]):
                        break

                sensors = env.get_raycast_sensors()
                rgb = env._getDroneImages(nth_drone=0)
                remote.send(("OK", (next_obs, rgb, sensors, accumulated_reward, terminated, truncated, info, env.start, env.goal)))

            elif cmd == "close":
                env.close()
                remote.close()
                break

        except Exception as e:
            try:
                remote.send(("ERROR", f"Lỗi runtime worker {worker_id}: {traceback.format_exc()}"))
            except Exception:
                pass
            break

def main():
    NUM_ENVS = 4
    print(f"🚀 Bắt đầu Huấn luyện Song song DQN với {NUM_ENVS} môi trường MuJoCo...")
    
    os.makedirs(os.path.join(current_dir, "Model"), exist_ok=True)
    
    ctx = mp.get_context("spawn")
    pipes = [ctx.Pipe() for _ in range(NUM_ENVS)]
    workers = []
    
    for i in range(NUM_ENVS):
        parent_conn, child_conn = pipes[i]
        proc = ctx.Process(target=worker_loop, args=(child_conn, i))
        proc.daemon = True
        proc.start()
        workers.append((proc, parent_conn))

    model = DroneNet(n_actions=5, state_vector_dim=23)
    dqn_agent = DQN(model, n_actions=5)

    max_eposide0 = 200
    max_eposide = 400
    max_epsilon = 1.0
    min_epsilon = 0.1
    reduce_epsilon = (max_epsilon - min_epsilon) / max_eposide0
    MAX_ACTIONS = 70

    log_file = os.path.join(current_dir, "drone_flight_log_parallel_DQN.csv")
    log_loss_file = os.path.join(current_dir, "log_loss_parallel_DQN.csv")
    
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Worker_ID', 'Start', 'Goal', 'Collision', 'Win', 'Over_Step', 'Over_Map', 'Epsilon', 'AccumReward'])
    log_f = open(log_file, mode='a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_f)

    if not os.path.exists(log_loss_file):
        with open(log_loss_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Loss'])
    log_loss_f = open(log_loss_file, mode='a', newline='', encoding='utf-8')
    log_loss_writer = csv.writer(log_loss_f)

    for i in range(NUM_ENVS):
        workers[i][1].send(("reset", "begin"))
    
    states_info = []
    for i in range(NUM_ENVS):
        status, payload = workers[i][1].recv()
        if status == "ERROR":
            print(f"❌ Lỗi Worker {i}: {payload}")
            return
        obs, rgb, sensors, start, goal = payload
        workers[i][1].send(("go_up", None))
    
    for i in range(NUM_ENVS):
        status, payload = workers[i][1].recv()
        if status == "ERROR":
            print(f"❌ Lỗi Worker {i}: {payload}")
            return
        obs, rgb, sensors, start, goal = payload
        states_info.append({
            'obs': obs,
            'rgb': rgb,
            'sensors': sensors,
            'start': start,
            'goal': goal,
            'action_count': 0,
            'accum_reward': 0.0
        })

    total_episodes = 0
    total_steps = 0
    episodes_losses = []
    start_time = time.perf_counter()
    RANGE = 0.5

    while total_episodes < max_eposide:
        try:
            actions = []
            s_imgs = []
            s_vecs = []

            for i in range(NUM_ENVS):
                info = states_info[i]
                obs = info['obs']
                rgb = info['rgb']
                sensors = info['sensors']
                env_goal = info['goal']
                x, y, z = obs[0:3]

                direct = [np.array([x, y - RANGE, 1.0]),
                          np.array([x - RANGE, y, 1.0]),
                          np.array([x + RANGE, y, 1.0]),
                          np.array([x + RANGE, y - RANGE, 1.0]),
                          np.array([x - RANGE, y - RANGE, 1.0])]
                
                hints = get_action_hints(obs, env_goal, direct, sensors)
                s_img = get_img_state(rgb)
                s_vec = get_vec_state(obs, env_goal, hints, sensors)
                
                s_imgs.append(s_img)
                s_vecs.append(s_vec)

                _, action_idx = dqn_agent.get_action(s_img, s_vec, max_epsilon)
                actions.append(action_idx)
                
                target_pos = direct[action_idx]
                workers[i][1].send(("step_action", target_pos))

            for i in range(NUM_ENVS):
                status, payload = workers[i][1].recv()
                if status == "ERROR":
                    print(f"❌ Lỗi Worker {i}: {payload}")
                    return
                
                next_obs, rgb_next, sensors_next, reward, terminated, truncated, info_str, start, goal = payload
                
                reward -= 0.10

                last_act = states_info[i].get('last_action', None)
                if last_act is not None:
                    if (last_act == 1 and actions[i] == 2) or \
                       (last_act == 2 and actions[i] == 1) or \
                       (last_act == 3 and actions[i] == 4) or \
                       (last_act == 4 and actions[i] == 3):
                        reward -= 5.0
                states_info[i]['last_action'] = actions[i]

                states_info[i]['action_count'] += 1
                action_count = states_info[i]['action_count']

                if action_count >= MAX_ACTIONS:
                    reward -= 25.0

                states_info[i]['accum_reward'] += reward

                is_truncated_bool = truncated[0] if isinstance(truncated, tuple) else bool(truncated)
                done = is_truncated_bool or terminated or (action_count >= MAX_ACTIONS)

                x1, y1, z1 = next_obs[0:3]
                direct1 = [np.array([x1, y1 - RANGE, 1.0]),
                           np.array([x1 - RANGE, y1, 1.0]),
                           np.array([x1 + RANGE, y1, 1.0]),
                           np.array([x1 + RANGE, y1 - RANGE, 1.0]),
                           np.array([x1 - RANGE, y1 - RANGE, 1.0])]
                
                hints1 = get_action_hints(next_obs, goal, direct1, sensors_next)
                ns_img = get_img_state(rgb_next)
                ns_vec = get_vec_state(next_obs, goal, hints1, sensors_next)

                dqn_agent.store_transition(s_imgs[i], s_vecs[i], actions[i], reward, ns_img, ns_vec, done)
                
                loss = dqn_agent.learn()
                if loss > 0:
                    episodes_losses.append(loss)

                total_steps += 1

                if done:
                    total_episodes += 1
                    WIN = "1" if (isinstance(truncated, tuple) and truncated[1] != "NONE") else "0"
                    OVER_STEP = "1" if action_count >= MAX_ACTIONS else "0"
                    OVER_MAP = "1" if (isinstance(truncated, tuple) and truncated[3] != "NONE") else "0"   
                    COLLISION = "1" if terminated else "0"

                    log_writer.writerow([total_episodes, i, start, goal, COLLISION, WIN, OVER_STEP, OVER_MAP, max_epsilon, states_info[i]['accum_reward']])
                    log_f.flush()

                    avg_loss = np.mean(episodes_losses) if episodes_losses else 0
                    log_loss_writer.writerow([total_episodes, avg_loss])
                    log_loss_f.flush()
                    episodes_losses = []

                    if max_epsilon > min_epsilon:
                        max_epsilon -= reduce_epsilon

                    print(f"🏆 EPISODE {total_episodes}/{max_eposide} [Worker {i}] Hoàn thành (DQN) | Win: {WIN} | Collision: {COLLISION} | Reward: {states_info[i]['accum_reward']:.2f}")

                    workers[i][1].send(("reset", "reset"))
                    status_r, payload_r = workers[i][1].recv()
                    if status_r == "ERROR":
                        print(f"❌ Lỗi Reset Worker {i}: {payload_r}")
                        return
                    obs_r, rgb_r, sensors_r, start_r, goal_r = payload_r

                    workers[i][1].send(("go_up", None))
                    status_up, payload_up = workers[i][1].recv()
                    if status_up == "ERROR":
                        print(f"❌ Lỗi Go_Up Worker {i}: {payload_up}")
                        return
                    obs_up, rgb_up, sensors_up, start_r, goal_r = payload_up
                    
                    states_info[i] = {
                        'obs': obs_up,
                        'rgb': rgb_up,
                        'sensors': sensors_up,
                        'start': start_r,
                        'goal': goal_r,
                        'action_count': 0,
                        'accum_reward': 0.0,
                        'last_action': None
                    }
                else:
                    states_info[i]['obs'] = next_obs
                    states_info[i]['rgb'] = rgb_next
                    states_info[i]['sensors'] = sensors_next

            if total_episodes > 0 and total_episodes % 50 == 0:
                save_path = os.path.join(current_dir, "Model", f"drone_model_parallel_dqn_eposide{total_episodes}.pth")
                dqn_agent.save(save_path)
                buffer_path = os.path.join(current_dir, "Model", f"replay_buffer_parallel_dqn_eposide{total_episodes}.pkl")
                dqn_agent.save_buffer(buffer_path)

        except Exception as e:
            print(f"💥 Lỗi luồng chính: {e}")
            traceback.print_exc()
            break

    end_time = time.perf_counter()
    print(f"\n✅ HOÀN THÀNH HUẤN LUYỆN SONG SONG DQN! TỔNG THỜI GIAN: {end_time - start_time:.2f} GIÂY")

    for i in range(NUM_ENVS):
        try:
            workers[i][1].send(("close", None))
            workers[i][0].join()
        except Exception:
            pass

    log_f.close()
    log_loss_f.close()

if __name__ == "__main__":
    main()

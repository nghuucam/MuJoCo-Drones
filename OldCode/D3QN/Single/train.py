import os
import sys
import time
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.append(path)

from multi_drone_mujoco.utils.enums import DroneModel
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
try:
    from mujoco_env import DroneEnv
    from agent import D3QN, DroneNet
except ImportError:
    from OldCode.D3QN.mujoco_env import DroneEnv 
    from OldCode.D3QN.agent import D3QN, DroneNet

def get_action_hints(obs, env, direct):
    state = obs[0:3]
    current_dist = np.linalg.norm(state - env.goal[0])
    
    hints = []
    for target in direct:
        future_dist = np.linalg.norm(target - env.goal[0])
        progress = current_dist - future_dist
        hints.append(progress)
    
    hints = np.array(hints, dtype=np.float32)
    max_val = np.max(np.abs(hints)) + 1e-8
    hints = (hints / max_val) * 0.3
    
    return hints

def get_vec_state(obs, env, action_hints):
    pos   = obs[0:3]    # 3
    rpy   = obs[7:10]   # 3
    vel   = obs[10:13]  # 3
    ang_v = obs[13:16]  # 3
    hints = action_hints  # 5
    
    four_sensors = env.get_raycast_sensors()

    vec = np.concatenate([pos, rpy, vel, ang_v, hints, four_sensors]) 
    return np.expand_dims(vec, axis=0)

def get_img_state(rgb_img):
    if isinstance(rgb_img, tuple):
        rgb_img = rgb_img[0]
    img = rgb_img[:, :, :3].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def speech_bubble(cur_pos, env):
    pass

def go_up(obs, control, env):
    control.reset()
    cur_obs = env._getDroneStateVector(0)
    current_x, current_y = cur_obs[0:2]

    start_time = time.time()
    TIMEOUT = 10.0

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
            print(f"🚀 Đã bay lên độ cao {cur_pos[2]:.2f}m an toàn (MuJoCo)!")
            break

    return obs_ret

def move(target_pos, obs, control, env):
    accumulated_reward = 0
    
    for step in range(200):
        cur_pos = obs[0:3]
        cur_quat = obs[3:7]
        cur_vel = obs[10:13]
        cur_ang_v = obs[13:16]
        
        z_point = cur_pos[2]
        
        if z_point - 0.9 < 0:
            obs = go_up(obs=obs, control=control, env=env)
            cur_pos = obs[0:3]
            cur_quat = obs[3:7]
            cur_vel = obs[10:13]
            cur_ang_v = obs[13:16]
            state = env._getDroneStateVector(0)
            env.previous_dist = np.linalg.norm(state[0:3] - env.goal[0]) 
            
        action_go, _, _ = control.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=cur_pos, 
            cur_quat=cur_quat,
            cur_vel=cur_vel, 
            cur_ang_vel=cur_ang_v,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, 0.0]) 
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
    
    return obs, img, accumulated_reward, terminated, truncated, info 

def main():
    os.makedirs(os.path.join(current_dir, "Model"), exist_ok=True)

    env = DroneEnv(gui=False)
    model = DroneNet(n_actions=5, state_vector_dim=21)
    d3qn_agent = D3QN(model, n_actions=5)
    control = DSLPIDControl(env=env)
    start_step = 1

    eposide = 1
    max_eposide0 = 200
    max_eposide = 500
    
    max_epsilon =  1.0
    min_epsilon = 0.1
    
    MAX_ACTIONS = 70
    action_count = 0

    accum_reward = 0

    log_file = os.path.join(current_dir, "drone_flight_log_test_D3QN.csv")
    log_entropy_cnn = os.path.join(current_dir, "neutron_array_D3QN.csv")
    log_loss = os.path.join(current_dir, "log_loss_D3QN.csv")
    eposide_losses = []

    reduce_epison = (max_epsilon - min_epsilon) / (max_eposide0 - 0)
    
    print(f"##################      EPISODE {eposide} (MuJoCo)       ######################### ")
    obs, info = env.reset(options="begin")
    obs = go_up(obs=obs, control=control, env=env)
    env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])

    start_time = time.perf_counter()
    rgb = env._getDroneImages(nth_drone=0)

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide', 'Step', 'Start', 'End', 'X', 'Y', 'Z', 
                             'Collision', 'Win', 'Over_Step', 'Over_Map', 'Epsilon', 'Action', 'AccumReward'])
    log_f = open(log_file, mode='a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_f)
    
    if not os.path.exists(log_entropy_cnn):
        with open(log_entropy_cnn, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide', 'Step', 'Start', 'End', 'List_Actions', 'Epsilon', 'Action_Decide'])
    log_f1 = open(log_entropy_cnn, mode='a', newline='', encoding='utf-8')
    log_neutrol = csv.writer(log_f1)
    
    if not os.path.exists(log_loss):
        with open(log_loss, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide', 'Loss'])
    
    log_f2 = open(log_loss, mode='a', newline='', encoding='utf-8')
    log_loss_writer = csv.writer(log_f2)

    while True:
        try:
            print(f"Step: {start_step}")
            x_point, y_point, z_point = obs[0:3]

            if z_point - 0.9 < 0:
                obs = go_up(obs=obs, control=control, env=env)
                x_point, y_point, z_point = obs[0:3]
                env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])
            rgb = env._getDroneImages(nth_drone=0)
            
            RANGE = 0.5
            direct = [np.array([x_point, y_point - RANGE, 1.0]),
                      np.array([x_point - RANGE, y_point, 1.0]),
                      np.array([x_point + RANGE, y_point, 1.0]),
                      np.array([x_point + RANGE, y_point - RANGE, 1.0]),
                      np.array([x_point - RANGE, y_point - RANGE, 1.0])]
            action_hints = get_action_hints(obs, env, direct)
            s_img = get_img_state(rgb)
            s_vec = get_vec_state(obs, env, action_hints)

            list_action, action_idx = d3qn_agent.get_action(s_img, s_vec, max_epsilon)

            action_names = ["Đi thẳng", "Đi phải", "Đi trái", "Đi chéo trái", "Đi chéo phải"]
            print(f"Q-Values: {list_action}  -> Chọn: {action_names[action_idx]}")
            
            target_pos = direct[action_idx]
            next_obs, three_img, reward, terminated, truncated, info = move(target_pos=target_pos, obs=obs, control=control, env=env)
            accum_reward += reward
            action_count += 1
            if isinstance(next_obs, tuple): next_obs = next_obs[0]

            is_truncated_bool = truncated[0] if isinstance(truncated, tuple) else bool(truncated)
            done = is_truncated_bool or terminated or (action_count >= MAX_ACTIONS)
            
            if three_img is not None and not isinstance(three_img, (float, int)):
                rgb = three_img
            else:
                rgb = env._getDroneImages(nth_drone=0)

            x1, y1, z1 = next_obs[0:3]
            direct1 = [np.array([x1, y1 - RANGE, 1.0]),
                       np.array([x1 - RANGE, y1, 1.0]),
                       np.array([x1 + RANGE, y1, 1.0]),
                       np.array([x1 + RANGE, y1 - RANGE, 1.0]),
                       np.array([x1 - RANGE, y1 - RANGE, 1.0])]
            action_hints1 = get_action_hints(next_obs, env, direct1)
            ns_img = get_img_state(rgb)
            ns_vec = get_vec_state(next_obs, env, action_hints1)

            if action_count >= MAX_ACTIONS: 
                reward -= 200
                print("Hết lượt di chuyển!!!\n")
            
            d3qn_agent.store_transition(s_img, s_vec, action_idx, reward, ns_img, ns_vec, done)
            loss = d3qn_agent.learn()
            if loss > 0:
                eposide_losses.append(loss)

            WIN = "1" if (isinstance(truncated, tuple) and truncated[1] != "NONE") else "0"
            OVER_STEP = "1" if action_count >= MAX_ACTIONS else "0"
            OVER_MAP = "1" if (isinstance(truncated, tuple) and truncated[3] != "NONE") else "0"   
            COLLISION = "1" if terminated else "0"

            log_writer.writerow([eposide, start_step, env.start, env.goal, x1, y1, z1, COLLISION, WIN, OVER_STEP, OVER_MAP, max_epsilon, action_names[action_idx], accum_reward])
            log_neutrol.writerow([eposide, start_step, env.start, env.goal, list_action, max_epsilon, action_names[action_idx]])

            start_step += 1

            if done:
                print(f"Màn chơi này thực hiện tổng cộng {action_count} hành động")
                if is_truncated_bool or terminated or (action_count >= MAX_ACTIONS):
                    obs, _ = env.reset(options="reset")
                    start_step = 1
                    eposide += 1
                    
                    avg_loss = np.mean(eposide_losses) if eposide_losses else 0
                    log_loss_writer.writerow([eposide, avg_loss])
                    eposide_losses = []
                    accum_reward = 0

                    if max_epsilon > min_epsilon: max_epsilon -= reduce_epison
                    if WIN == "1": print("Màn chơi thành công tới đích !!!\n")
                    if OVER_STEP == "1": print("Đi quá số bước !!!\n")
                    if OVER_MAP == "1": print("Bay ra ngoài map !!!\n")
                    if COLLISION == "1": print("Bị va chạm với cột !!!\n")
                    print(f"##################      EPISODE {eposide} (MuJoCo)       ######################### ")
                action_count = 0
            else:
                obs = next_obs
                s_img = ns_img
                s_vec = ns_vec

            if eposide > 0 and eposide % 100 == 0:
                save_path = os.path.join(current_dir, "Model", f"drone_model_d3qn_eposide{eposide}.pth")
                d3qn_agent.save(save_path)
            
            if eposide == max_eposide: 
                end_time = time.perf_counter()
                print(f"### THỜI GIAN ĐỂ TRAIN BẰNG THUẬT TOÁN D3QN (MuJoCo) LÀ {end_time-start_time:.2f} GIÂY ###\n")
                break
        
        except Exception as e:
            print(f"\n💥 MÔI TRƯỜNG BỊ KẸT / LỖI: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Khởi động lại episode mới...")
            obs, _ = env.reset(options="reset")
            
            eposide_losses = []
            eposide += 1
            action_count = 1
            accum_reward = 0
            env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])

            print(f"################## BẮT ĐẦU LẠI EPISODE {eposide} #########################\n")
            continue
    
    log_f.close()
    log_f1.close()
    log_f2.close()

if __name__ == "__main__":
    main()

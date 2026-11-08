import os
import sys
import time
import numpy as np
import torch

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

MODEL_PATH = os.path.join(parent_dir, "Single", "Model", "drone_model_dqn_eposide200.pth")

def get_action_hints(obs, env, direct, sensors=None):
    goal_pos = env.goal[0] if hasattr(env, "goal") else (env[0] if isinstance(env, (list, np.ndarray)) else env)
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
    
    if sensors is None and hasattr(env, "get_raycast_sensors"):
        sensors = env.get_raycast_sensors()
        
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

def get_vec_state(obs, env, action_hints):
    pos   = obs[0:3]
    rpy   = obs[7:10]
    vel   = obs[10:13]
    rel_goal = env.goal[0] - pos
    hints = action_hints
    
    raycast_sensors = env.get_raycast_sensors()
    vec = np.concatenate([rpy, vel, rel_goal, hints, raycast_sensors]) 
    return np.expand_dims(vec, axis=0)

def get_img_state(rgb_img):
    if isinstance(rgb_img, tuple):
        rgb_img = rgb_img[0]
    img = rgb_img[:, :, :3].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def go_up(obs, control, env):
    control.reset()
    cur_obs = env._getDroneStateVector(0)
    current_x, current_y = cur_obs[0:2]

    start_time = time.time()
    TIMEOUT = 10.0

    step_count = 0
    while True:
        step_count += 1
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
            print(f"🚀 Đã cất cánh lên độ cao {cur_pos[2]:.2f}m ổn định!")
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
    print("=================================================================")
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH TEST MÔ HÌNH DQN DRONE (MUJOCO GUI)")
    print("=================================================================")
    
    target_model_path = MODEL_PATH
    if not os.path.exists(target_model_path):
        print(f"\n⚠️ KHÔNG TÌM THẤY MODEL TẠI: {target_model_path}")
        print("🔍 Đang tìm kiếm các file model .pth có sẵn...")
        found_models = []
        for search_dir in [os.path.join(parent_dir, "Single", "Model"), os.path.join(parent_dir, "Parallel", "Model")]:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith(".pth"):
                        found_models.append(os.path.join(search_dir, f))
        
        if found_models:
            target_model_path = found_models[0]
            print(f"👉 Tự động sử dụng model tìm thấy: {target_model_path}\n")
        else:
            print("❌ Không tìm thấy bất kỳ file model .pth nào! Vui lòng huấn luyện hoặc cung cấp đường dẫn model đúng.")
            return

    env = DroneEnv(gui=True)
    model = DroneNet(n_actions=5, state_vector_dim=23)
    dqn_agent = DQN(model, n_actions=5)
    
    dqn_agent.load(target_model_path)
    control = DSLPIDControl(env=env)

    MAX_TEST_EPISODES = 20
    MAX_ACTIONS_PER_EPISODE = 70
    action_names = ["Đi thẳng", "Đi phải", "Đi trái", "Đi chéo trái", "Đi chéo phải"]

    test_stats = {
        'win': 0,
        'collision': 0,
        'over_map': 0,
        'over_step': 0,
        'rewards': [],
        'steps': []
    }

    for ep in range(1, MAX_TEST_EPISODES + 1):
        print(f"\n🎮 [TEST EPISODE {ep}/{MAX_TEST_EPISODES}] (Vật cản ngẫu nhiên)")
        
        obs, _ = env.reset(options="reset")
        control.reset()
        obs = go_up(obs=obs, control=control, env=env)
        env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])
        
        accum_reward = 0
        action_count = 0
        
        print(f"📍 Điểm xuất phát: {env.start[0][:2]} | Điểm đích: {env.goal[0][:2]}")
        print(f"📏 Khoảng cách ban đầu tới đích: {env.previous_dist:.2f}m")

        while action_count < MAX_ACTIONS_PER_EPISODE:
            action_count += 1
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

            q_values, action_idx = dqn_agent.get_action(s_img, s_vec, eps=0.0)

            print(f"  👉 Bước {action_count}: Q-Values={q_values} -> Chọn: {action_names[action_idx]}")

            target_pos = direct[action_idx]
            next_obs, next_img, reward, terminated, truncated, info = move(target_pos, obs, control, env)

            reward -= 0.10

            accum_reward += reward
            obs = next_obs

            is_truncated_bool = truncated[0] if isinstance(truncated, tuple) else truncated
            
            if terminated or is_truncated_bool:
                break

        win = env.win
        collision = env.collision
        over_map = env.over_map
        over_step = (action_count >= MAX_ACTIONS_PER_EPISODE) and not win and not collision

        if win:
            test_stats['win'] += 1
            print(f"  🏆 THÀNH CÔNG: Drone đã tới đích an toàn trong {action_count} bước!")
        elif collision:
            test_stats['collision'] += 1
            print(f"  💥 THẤT BẠI: Drone đã va chạm vật cản!")
        elif over_map:
            test_stats['over_map'] += 1
            print(f"  🌐 THẤT BẠI: Drone bay ra ngoài bản đồ!")
        else:
            test_stats['over_step'] += 1
            print(f"  ⏱️ THẤT BẠI: Vượt quá số bước tối đa ({MAX_ACTIONS_PER_EPISODE})!")

        test_stats['rewards'].append(accum_reward)
        test_stats['steps'].append(action_count)

        print(f"📊 Kết thúc Episode {ep} | Tổng bước: {action_count} | Tổng Reward: {accum_reward:.2f}")

    total = MAX_TEST_EPISODES
    w_cnt = test_stats['win']
    c_cnt = test_stats['collision']
    om_cnt = test_stats['over_map']
    os_cnt = test_stats['over_step']
    avg_rew = float(np.mean(test_stats['rewards'])) if test_stats['rewards'] else 0.0
    avg_steps = float(np.mean(test_stats['steps'])) if test_stats['steps'] else 0.0

    print("\n" + "=" * 65)
    print(f"📋 BÁO CÁO TỔNG KẾT KIỂM THỬ DQN ({total} EPISODES)")
    print("=" * 65)
    print(f"🏆 Tỷ lệ Chiến thắng (Win Rate)  : {w_cnt}/{total} ({w_cnt/total*100:.2f}%)")
    print(f"💥 Tỷ lệ Va chạm (Collision Rate): {c_cnt}/{total} ({c_cnt/total*100:.2f}%)")
    print(f"🌐 Tỷ lệ Ra ngoài map (Over_Map) : {om_cnt}/{total} ({om_cnt/total*100:.2f}%)")
    print(f"⏱️ Tỷ lệ Hết bước (Over_Step)   : {os_cnt}/{total} ({os_cnt/total*100:.2f}%)")
    print("-" * 65)
    print(f"⭐ Điểm thưởng trung bình        : {avg_rew:.2f}")
    print(f"👣 Số bước bay trung bình        : {avg_steps:.1f} bước/episode")
    print("=" * 65)
    print("✅ HOÀN THÀNH CHƯƠNG TRÌNH TEST.\n")

if __name__ == "__main__":
    main()

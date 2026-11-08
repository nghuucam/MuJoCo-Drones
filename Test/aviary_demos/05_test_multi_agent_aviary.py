"""Kiểm tra môi trường: MultiAgentAviary (Chuẩn PettingZoo ParallelEnv Đa tác tử).

Mục tiêu nhiệm vụ:
- Cung cấp giao diện chuẩn PettingZoo ParallelEnv phục vụ huấn luyện MARL (Multi-Agent RL như MAPPO, QMIX, MADDPG).
- Mỗi drone là một tác tử độc lập (drone0, drone1, drone2) với không gian quan sát (13 chiều)
  và không gian hành động (4 chiều) riêng biệt.
- Nhiệm vụ: Từng drone tự bay lên và duy trì trạng thái lơ lửng (hover) tại các cao độ mục tiêu khác nhau
  được chỉ định riêng cho từng tác tử (ví dụ: drone0 lên 0.7m, drone1 lên 0.95m, drone2 lên 1.2m).

Cách chạy:
    python 05_test_multi_agent_aviary.py              # Chạy mô phỏng PID đa tác tử
    python 05_test_multi_agent_aviary.py --gui        # Mở cửa sổ 3D MuJoCo trực quan
    python 05_test_multi_agent_aviary.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 05_test_multi_agent_aviary.py --random     # Thử nghiệm với hành động ngẫu nhiên
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.multi_agent_aviary import MultiAgentAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils import rpm_to_normalized_action, save_gif, parse_demo_args


def run_multi_agent_demo(gui: bool = False, record: bool = False, steps: int = 350, use_random: bool = False, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    NUM_AGENTS = 3
    
    print("=" * 85)
    print(f" KHỞI TẠO MÔI TRƯỜNG: MultiAgentAviary (PettingZoo ParallelEnv - {NUM_AGENTS} Tác tử độc lập)")
    print("=" * 85)
    
    env = MultiAgentAviary(
        num_drones=NUM_AGENTS,
        render_mode=render_mode,
        ctrl_freq=48
    )
    
    observations, infos = env.reset()
    ctrls = [PIDControl(env._env) for _ in range(NUM_AGENTS)]
    
    print(f"[*] Danh sách tác tử (Agents): {env.possible_agents}")
    print(f"[*] Không gian quan sát tác tử mẫu: {env.observation_space(env.possible_agents[0])}")
    print(f"[*] Không gian hành động tác tử mẫu: {env.action_space(env.possible_agents[0])}")
    print("[*] Phân công cao độ mục tiêu cho từng tác tử (Target Altitude):")
    for i, agent in enumerate(env.possible_agents):
        print(f"    - {agent}: Z_target = {env.target_heights[i]:.2f} m (Vị trí ban đầu: {env._env.INIT_XYZS[i].tolist()})")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Độ cao drone0 (m)':<18} | {'Độ cao drone1 (m)':<18} | {'Độ cao drone2 (m)':<18} | {'Tổng thưởng':<12}")
    print("-" * 85)
    
    frames = []
    start_time = time.time()
    
    for step in range(steps):
        actions = {}
        
        for i, agent in enumerate(env.possible_agents):
            if agent not in env.agents:
                continue
                
            if use_random:
                actions[agent] = env.action_space(agent).sample()
            else:
                target = np.array([
                    env._env.INIT_XYZS[i, 0],
                    env._env.INIT_XYZS[i, 1],
                    env.target_heights[i]
                ])
                rpm, _, _ = ctrls[i].computeControl(
                    control_timestep=env._env.CTRL_TIMESTEP,
                    cur_pos=env._env.pos[i],
                    cur_quat=env._env.quat[i],
                    cur_vel=env._env.vel[i],
                    cur_ang_vel=env._env.ang_v[i],
                    target_pos=target
                )
                actions[agent] = rpm_to_normalized_action(rpm, env._env)
                
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render
        if gui:
            env.render()
            time.sleep(max(0.0, env._env.CTRL_TIMESTEP - (time.time() - start_time) % env._env.CTRL_TIMESTEP))
        elif record and (step % 2 == 0):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
        if step % 30 == 0 or step == steps - 1:
            z0 = f"{env._env.pos[0, 2]:.2f} (đích: {env.target_heights[0]:.2f})"
            z1 = f"{env._env.pos[1, 2]:.2f} (đích: {env.target_heights[1]:.2f})"
            z2 = f"{env._env.pos[2, 2]:.2f} (đích: {env.target_heights[2]:.2f})"
            total_r = sum(rewards.values()) if rewards else 0.0
            print(f"{step:<6} | {z0:<18} | {z1:<18} | {z2:<18} | {total_r:<+12.2f}")
            
        # Kiểm tra nếu tất cả tác tử đã dừng
        if len(env.agents) == 0:
            print(f"[!] Tất cả tác tử đã kết thúc tại bước {step}!")
            break
            
    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "multi_agent_aviary.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: MultiAgentAviary")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_demo_args("Kiểm tra môi trường MultiAgentAviary (PettingZoo Đa tác tử)")
    run_multi_agent_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        use_random=args.random,
        camera=args.camera
    )

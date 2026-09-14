"""Kiểm tra môi trường: FormationAviary (Bay đội hình).

Mục tiêu nhiệm vụ:
- Nhiều drone (mặc định 3 drone) duy trì một hình dạng đội hình hình học (ví dụ: tam giác đều)
  trong khi tâm của toàn bộ đội hình di chuyển dọc theo quỹ đạo không gian.
- Hàm phần thưởng (Reward) vừa đánh giá độ bám mục tiêu của từng drone, vừa phạt nếu khoảng cách
  giữa các drone trong đội hình bị sai lệch (mất liên kết đội hình).

Cách chạy:
    python 03_test_formation_aviary.py              # Chạy mô phỏng PID 3 drone bay đội hình
    python 03_test_formation_aviary.py --gui        # Mở cửa sổ 3D MuJoCo trực quan
    python 03_test_formation_aviary.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 03_test_formation_aviary.py --random     # Thử nghiệm với hành động ngẫu nhiên
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.formation_aviary import FormationAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils import rpm_to_normalized_action, save_gif, parse_demo_args


def run_formation_demo(gui: bool = False, record: bool = False, steps: int = 400, use_random: bool = False, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    NUM_DRONES = 3
    
    print("=" * 80)
    print(f" KHỞI TẠO MÔI TRƯỜNG: FormationAviary (Bay đội hình {NUM_DRONES} Drones)")
    print("=" * 80)
    
    env = FormationAviary(
        num_drones=NUM_DRONES,
        render_mode=render_mode,
        ctrl_freq=48
    )
    
    obs, info = env.reset()
    ctrls = [PIDControl(env) for _ in range(NUM_DRONES)]
    
    print(f"[*] Số lượng drone: {NUM_DRONES}")
    print(f"[*] Không gian quan sát tổng hợp: {env.observation_space}")
    print(f"[*] Không gian hành động tổng hợp: {env.action_space}")
    print(f"[*] Vector bù đắp hình học (Formation Offsets relative to center):")
    for d in range(NUM_DRONES):
        print(f"    - Drone {d}: {env.FORMATION_OFFSETS[d].round(3).tolist()}")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone 0':<20} | {'Tọa độ Drone 1':<20} | {'Tọa độ Drone 2':<20} | {'Thưởng':<8}")
    print("-" * 80)
    
    frames = []
    start_time = time.time()
    
    for step in range(steps):
        targets = env._get_formation_targets()
        
        if use_random:
            action = env.action_space.sample()
        else:
            all_act = []
            for d in range(NUM_DRONES):
                rpm, _, _ = ctrls[d].computeControl(
                    control_timestep=env.CTRL_TIMESTEP,
                    cur_pos=env.pos[d],
                    cur_quat=env.quat[d],
                    cur_vel=env.vel[d],
                    cur_ang_vel=env.ang_v[d],
                    target_pos=targets[d]
                )
                all_act.append(rpm_to_normalized_action(rpm, env))
            action = np.concatenate(all_act)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        if gui:
            env.render()
            time.sleep(max(0.0, env.CTRL_TIMESTEP - (time.time() - start_time) % env.CTRL_TIMESTEP))
        elif record and (step % 2 == 0):
            frame = env.render(camera_mode=camera)
            if frame is not None:
                frames.append(frame)
                
        if step % 30 == 0 or step == steps - 1:
            p0 = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            p1 = f"[{env.pos[1, 0]:.2f}, {env.pos[1, 1]:.2f}, {env.pos[1, 2]:.2f}]"
            p2 = f"[{env.pos[2, 0]:.2f}, {env.pos[2, 1]:.2f}, {env.pos[2, 2]:.2f}]"
            print(f"{step:<6} | {p0:<20} | {p1:<20} | {p2:<20} | {reward:<+8.2f}")
            
        if terminated or truncated:
            print(f"[!] Kết thúc tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break
            
    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "formation_aviary.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: FormationAviary")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_demo_args("Kiểm tra môi trường FormationAviary (Bay đội hình)")
    run_formation_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        use_random=args.random,
        camera=args.camera
    )

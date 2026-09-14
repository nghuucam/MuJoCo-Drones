"""Kiểm tra môi trường: VelocityAviary (Bám đuổi vận tốc).

Mục tiêu nhiệm vụ:
- Điều khiển drone bám theo vector vận tốc mục tiêu [vx, vy, vz, yaw_rate].
- Thích hợp cho các bài toán lập kế hoạch cấp cao (high-level trajectory/velocity tracking).

Cách chạy:
    python 01_test_velocity_aviary.py              # Chạy mô phỏng PID và in thông số
    python 01_test_velocity_aviary.py --gui        # Mở cửa sổ 3D MuJoCo trực quan
    python 01_test_velocity_aviary.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 01_test_velocity_aviary.py --random     # Thử nghiệm với hành động ngẫu nhiên (RL baseline)
"""

import os
import sys
import time
import numpy as np

# Thêm đường dẫn project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils import rpm_to_normalized_action, save_gif, parse_demo_args


def run_velocity_demo(gui: bool = False, record: bool = False, steps: int = 300, use_random: bool = False, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 70)
    print(" KHỞI TẠO MÔI TRƯỜNG: VelocityAviary (Bám đuổi vận tốc)")
    print("=" * 70)
    
    env = VelocityAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]])
    )
    
    obs, info = env.reset()
    
    # Thiết lập vận tốc mục tiêu mong muốn: tiến tới trước 0.4m/s, bay ngang 0.2m/s, bay lên 0.1m/s
    TARGET_COMMAND = np.array([0.4, 0.2, 0.1, 0.0])
    env.TARGET_VEL = TARGET_COMMAND.copy()
    
    print(f"[*] Không gian quan sát (Observation Space): {env.observation_space}")
    print(f"[*] Không gian hành động (Action Space):      {env.action_space}")
    print(f"[*] Vận tốc mục tiêu (Target Velocity):     vx={TARGET_COMMAND[0]:.2f} m/s, vy={TARGET_COMMAND[1]:.2f} m/s, vz={TARGET_COMMAND[2]:.2f} m/s")
    print(f"[*] Chế độ điều khiển:                       {'Ngẫu nhiên (Random Sampling)' if use_random else 'Bộ điều khiển PID (Autonomous PID Tracking)'}")
    print("-" * 70)
    print(f"{'Bước (Step)':<12} | {'Vận tốc thực tế (vx, vy, vz)':<30} | {'Sai số (m/s)':<14} | {'Phần thưởng':<12}")
    print("-" * 70)
    
    ctrl = PIDControl(env)
    target_pos = env.pos[0].copy()
    frames = []
    
    start_time = time.time()
    
    for step in range(steps):
        if use_random:
            action = env.action_space.sample()
        else:
            # Tích phân vị trí mục tiêu theo vận tốc mong muốn
            target_pos += env.TARGET_VEL[:3] * env.CTRL_TIMESTEP
            rpm, _, _ = ctrl.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=env.pos[0],
                cur_quat=env.quat[0],
                cur_vel=env.vel[0],
                cur_ang_vel=env.ang_v[0],
                target_pos=target_pos,
                target_vel=env.TARGET_VEL[:3]
            )
            action = rpm_to_normalized_action(rpm, env)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        if gui:
            env.render()
            time.sleep(max(0.0, env.CTRL_TIMESTEP - (time.time() - start_time) % env.CTRL_TIMESTEP))
        elif record and (step % 2 == 0):
            frame = env.render(camera_mode=camera)
            if frame is not None:
                frames.append(frame)
                
        # In thông số mỗi 30 bước (khoảng 0.6 giây)
        if step % 30 == 0 or step == steps - 1:
            cur_vel = env.vel[0]
            vel_str = f"[{cur_vel[0]:+5.2f}, {cur_vel[1]:+5.2f}, {cur_vel[2]:+5.2f}]"
            vel_err = info.get("velocity_error", np.linalg.norm(cur_vel - env.TARGET_VEL[:3]))
            print(f"{step:<12} | {vel_str:<30} | {vel_err:<14.3f} | {reward:<+12.2f}")
            
        if terminated or truncated:
            print(f"[!] Tập mô phỏng kết thúc tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "velocity_aviary.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 70)
    print(" HOÀN THÀNH TEST: VelocityAviary")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_demo_args("Kiểm tra môi trường VelocityAviary (Bám đuổi vận tốc)")
    run_velocity_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        use_random=args.random,
        camera=args.camera
    )

"""Kiểm tra môi trường: RaceAviary (Đua qua cổng).

Mục tiêu nhiệm vụ:
- Điều khiển drone bay qua chuỗi các cổng đua (race gates) trên đường đua vòng tròn (circuit)
  càng nhanh càng tốt.
- Phần thưởng lớn (+20 điểm) khi vượt qua mỗi cổng, kèm thưởng tỷ lệ thuận với tốc độ bay (speed bonus).
- Theo dõi số cổng đã vượt qua (gates_passed) và số vòng đua hoàn thành (laps_completed).

Cách chạy:
    python 04_test_race_aviary.py              # Chạy mô phỏng PID đua qua các cổng
    python 04_test_race_aviary.py --gui        # Mở cửa sổ 3D MuJoCo trực quan
    python 04_test_race_aviary.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 04_test_race_aviary.py --random     # Thử nghiệm với hành động ngẫu nhiên
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.race_aviary import RaceAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils import rpm_to_normalized_action, save_gif, parse_demo_args


def run_race_demo(gui: bool = False, record: bool = False, steps: int = 500, use_random: bool = False, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG: RaceAviary (Đua qua cổng)")
    print("=" * 80)
    
    env = RaceAviary(
        num_drones=1,
        render_mode=render_mode,
        ctrl_freq=48,
        gate_radius=0.25,
        initial_xyzs=np.array([[0.3, 0.0, 0.5]])
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Không gian quan sát: {env.observation_space}")
    print(f"[*] Không gian hành động: {env.action_space}")
    print(f"[*] Danh sách {len(env.GATES)} Cổng đua (Racing Gates):")
    for idx, g in enumerate(env.GATES):
        print(f"    - Cổng {idx}: [X={g[0]:.2f}, Y={g[1]:.2f}, Z={g[2]:.2f}] m")
    print(f"[*] Bán kính cổng: {env.GATE_RADIUS} m")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Vị trí hiện tại':<22} | {'Cổng đích':<16} | {'Tốc độ':<10} | {'Đã qua':<8} | {'Thưởng':<8}")
    print("-" * 80)
    
    frames = []
    last_gate_count = 0
    start_time = time.time()
    
    for step in range(steps):
        gate_idx = env.gates_passed[0] % len(env.GATES)
        target_gate = env.GATES[gate_idx]
        
        if use_random:
            action = env.action_space.sample()
        else:
            rpm, _, _ = ctrl.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=env.pos[0],
                cur_quat=env.quat[0],
                cur_vel=env.vel[0],
                cur_ang_vel=env.ang_v[0],
                target_pos=target_gate
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
                
        # Thông báo khi qua được cổng mới
        current_gate_count = env.gates_passed[0]
        if current_gate_count > last_gate_count:
            lap = current_gate_count // len(env.GATES)
            print(f"  >>> [GATE CLEARED!] Drone vượt qua cổng {last_gate_count % len(env.GATES)}! Tổng cổng: {current_gate_count} (Vòng {lap}) <<<")
        last_gate_count = current_gate_count
        
        speed = np.linalg.norm(env.vel[0])
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            g_str = f"Gate {gate_idx}"
            print(f"{step:<6} | {p_str:<22} | {g_str:<16} | {speed:<8.2f} m/s | {current_gate_count:<8} | {reward:<+8.2f}")
            
        if terminated or truncated:
            print(f"[!] Kết thúc cuộc đua: Tổng số cổng đã qua: {env.gates_passed[0]}, Vòng hoàn thành: {env.gates_passed[0] // len(env.GATES)}")
            break
            
    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "race_aviary.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: RaceAviary")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_demo_args("Kiểm tra môi trường RaceAviary (Đua qua cổng)")
    run_race_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        use_random=args.random,
        camera=args.camera
    )

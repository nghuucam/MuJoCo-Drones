"""Kiểm tra môi trường: FlyThroughAviary (Bay qua các điểm waypoint).

Mục tiêu nhiệm vụ:
- Điều khiển drone bay tuần tự qua một chuỗi các điểm mốc (waypoints) trong không gian 3D.
- Khi drone tiếp cận trong bán kính WAYPOINT_RADIUS (mặc định 0.1m), môi trường chuyển sang waypoint tiếp theo và thưởng điểm lớn (+10 điểm).

Cách chạy:
    python 02_test_fly_through_aviary.py              # Chạy mô phỏng PID bay qua waypoint
    python 02_test_fly_through_aviary.py --gui        # Mở cửa sổ 3D MuJoCo trực quan
    python 02_test_fly_through_aviary.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 02_test_fly_through_aviary.py --random     # Thử nghiệm với hành động ngẫu nhiên
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils import rpm_to_normalized_action, save_gif, parse_demo_args


def run_fly_through_demo(gui: bool = False, record: bool = False, steps: int = 400, use_random: bool = False, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 75)
    print(" KHỞI TẠO MÔI TRƯỜNG: FlyThroughAviary (Bay qua các điểm Waypoint)")
    print("=" * 75)
    
    # Định nghĩa lộ trình waypoint mẫu rõ ràng trong không gian
    custom_waypoints = np.array([
        [0.0, 0.0, 0.8],   # Cất cánh lên 0.8m
        [0.8, 0.0, 1.0],   # Tiến về phía trước
        [0.8, 0.8, 1.2],   # Rẽ phải và bay lên 1.2m
        [0.0, 0.8, 1.0],   # Lượn sang trái
        [0.0, 0.0, 0.6],   # Quay về gốc và hạ thấp dần
    ])
    
    env = FlyThroughAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        waypoints=custom_waypoints,
        waypoint_radius=0.15,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]])
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Không gian quan sát: {env.observation_space}")
    print(f"[*] Không gian hành động: {env.action_space}")
    print(f"[*] Danh sách {len(env.WAYPOINTS)} Waypoint hành trình:")
    for idx, wp in enumerate(env.WAYPOINTS):
        print(f"    - WP {idx}: [X={wp[0]:.2f}, Y={wp[1]:.2f}, Z={wp[2]:.2f}] m")
    print(f"[*] Bán kính nhận diện mốc (Waypoint Radius): {env.WAYPOINT_RADIUS} m")
    print("-" * 75)
    print(f"{'Bước':<8} | {'Vị trí hiện tại (X, Y, Z)':<25} | {'Mục tiêu WP':<18} | {'Khoảng cách':<12} | {'Thưởng':<8}")
    print("-" * 75)
    
    frames = []
    last_wp_idx = -1
    start_time = time.time()
    
    for step in range(steps):
        wp_idx = min(env.current_waypoint_idx[0], len(env.WAYPOINTS) - 1)
        target_wp = env.WAYPOINTS[wp_idx]
        
        if use_random:
            action = env.action_space.sample()
        else:
            rpm, _, _ = ctrl.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=env.pos[0],
                cur_quat=env.quat[0],
                cur_vel=env.vel[0],
                cur_ang_vel=env.ang_v[0],
                target_pos=target_wp
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
                
        # Thông báo khi vừa chạm được một waypoint mới
        new_wp_idx = env.current_waypoint_idx[0]
        if new_wp_idx > last_wp_idx and last_wp_idx != -1:
            print(f"  >>> [CHÚC MỪNG] Drone đã chạm mốc Waypoint {last_wp_idx}! Chuyển sang Waypoint {new_wp_idx} <<<")
        last_wp_idx = new_wp_idx
        
        dist = np.linalg.norm(env.pos[0] - target_wp)
        
        if step % 25 == 0 or step == steps - 1:
            pos_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            wp_str = f"WP{wp_idx}: {target_wp.tolist()}"
            print(f"{step:<8} | {pos_str:<25} | {wp_str:<18} | {dist:<12.3f} m | {reward:<+8.2f}")
            
        if terminated or truncated:
            print(f"[!] Kết thúc hành trình: Đã hoàn thành {env.current_waypoint_idx[0]}/{len(env.WAYPOINTS)} Waypoints!")
            break
            
    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "fly_through_aviary.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 75)
    print(" HOÀN THÀNH TEST: FlyThroughAviary")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_demo_args("Kiểm tra môi trường FlyThroughAviary (Bay qua các điểm Waypoint)")
    run_fly_through_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        use_random=args.random,
        camera=args.camera
    )

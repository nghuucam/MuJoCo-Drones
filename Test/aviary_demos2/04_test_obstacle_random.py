"""Kiểm tra môi trường Chướng ngại vật: RANDOM (Ngẫu nhiên hỗn hợp đa hình khối).

Đặc điểm môi trường:
- Tự động sinh hỗn hợp ngẫu nhiên các loại hình học cơ bản (Spheres, Cylinders, Boxes).
- Mỗi vật cản có kích thước, màu sắc ngẫu nhiên rực rỡ và góc nghiêng 3 chiều (Euler rotation).
- Mô phỏng không gian bừa bộn (Random Clutter) thử thách khả năng tránh chướng ngại vật phức tạp của thuật toán DRL.

Cách chạy:
    python 04_test_obstacle_random.py --gui        # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 04_test_obstacle_random.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 04_test_obstacle_random.py --num-obstacles 30  # Tăng mật độ vật cản
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.wrappers.obstacles import ObstacleConfig, ObstacleType, generate_obstacles, obstacles_to_xml
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils2 import rpm_to_normalized_action, save_gif, parse_obstacle_args


def run_random_demo(gui: bool = False, record: bool = False, steps: int = 300, num_obstacles: int = 25, seed: int = 42, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG CHƯỚNG NGẠI VẬT: CHỦ ĐỀ NGẪU NHIÊN (RANDOM CLUTTER)")
    print("=" * 80)
    
    config = ObstacleConfig(
        obstacle_type=ObstacleType.RANDOM,
        num_obstacles=num_obstacles,
        arena_size=(2.5, 2.5, 2.0),
        min_spacing=0.35,
        safe_zone_radius=0.6,
        safe_zone_centers=np.array([[0.0, 0.0, 0.5]]),
        seed=seed,
    )
    clutter_items = generate_obstacles(config)
    obs_xml = obstacles_to_xml(clutter_items)
    
    initial_pos = np.array([[0.0, 0.0, 0.5]])
    env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=initial_pos,
        custom_xml=obs_xml
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    # Phân loại số lượng hình khối
    box_cnt = sum(1 for o in clutter_items if o.geom_type == "box")
    cyl_cnt = sum(1 for o in clutter_items if o.geom_type == "cylinder")
    sph_cnt = sum(1 for o in clutter_items if o.geom_type == "sphere")
    
    print(f"[*] Tổng số vật cản ngẫu nhiên: {len(clutter_items)}")
    print(f"    - Khối hộp (Boxes):    {box_cnt}")
    print(f"    - Khối trụ (Cylinders): {cyl_cnt}")
    print(f"    - Khối cầu (Spheres):   {sph_cnt}")
    print(f"[*] Kích thước đấu trường: ±{config.arena_size[0]}m x ±{config.arena_size[1]}m, Cao: {config.arena_size[2]}m")
    print(f"[*] Bán kính vùng an toàn cất cánh: {config.safe_zone_radius}m (Seed: {seed})")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone (X, Y, Z)':<24} | {'Mục tiêu bay':<20} | {'Khoảng cách tới vật gần nhất':<30}")
    print("-" * 80)
    
    # Lộ trình bay lượn
    waypoints = [
        np.array([0.0, 0.0, 0.8]),
        np.array([0.8, -0.7, 0.9]),
        np.array([0.9, 0.8, 1.1]),
        np.array([-0.7, 0.8, 0.9]),
        np.array([-0.8, -0.7, 1.0]),
        np.array([0.0, 0.0, 0.8]),
    ]
    wp_idx = 0
    
    frames = []
    start_time = time.time()
    
    item_centers = np.array([o.position for o in clutter_items])
    
    for step in range(steps):
        target = waypoints[wp_idx]
        if np.linalg.norm(env.pos[0] - target) < 0.2 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            
        rpm, _, _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=env.pos[0],
            cur_quat=env.quat[0],
            cur_vel=env.vel[0],
            cur_ang_vel=env.ang_v[0],
            target_pos=target
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
                
        # Khoảng cách tới vật cản gần nhất
        dists = np.linalg.norm(item_centers - env.pos[0], axis=1)
        min_dist = np.min(dists) if len(dists) > 0 else 999.0
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            t_str = f"WP{wp_idx}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            print(f"{step:<6} | {p_str:<24} | {t_str:<20} | {min_dist:<30.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "obstacle_random.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: Chủ đề RANDOM (Ngẫu nhiên)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_obstacle_args("RANDOM (Ngẫu nhiên hỗn hợp)")
    run_random_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        camera=args.camera
    )

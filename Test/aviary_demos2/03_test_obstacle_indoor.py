"""Kiểm tra môi trường Chướng ngại vật: INDOOR (Trong nhà / Phòng kín).

Đặc điểm môi trường:
- Tự động dựng một căn phòng kín với 4 bức tường xung quanh và trần nhà bao bọc.
- Bên trong phòng bố trí các nội thất: Bàn gỗ (tables), cột trụ nhà (columns), kệ tủ (shelves).
- Thử thách điều khiển drone trong không gian bị chặn trần và tường (hiệu ứng dội khí downwash và không gian chật hẹp).

Cách chạy:
    python 03_test_obstacle_indoor.py --gui        # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 03_test_obstacle_indoor.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 03_test_obstacle_indoor.py --num-obstacles 15  # Tăng số đồ đạc trong phòng
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


def run_indoor_demo(gui: bool = False, record: bool = False, steps: int = 300, num_obstacles: int = 15, seed: int = 42, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG CHƯỚNG NGẠI VẬT: CHỦ ĐỀ TRONG NHÀ (INDOOR)")
    print("=" * 80)
    
    config = ObstacleConfig(
        obstacle_type=ObstacleType.INDOOR,
        num_obstacles=num_obstacles,
        arena_size=(2.0, 2.0, 1.8),  # Phòng 4m x 4m, trần cao 1.8m
        min_spacing=0.4,
        safe_zone_radius=0.5,
        safe_zone_centers=np.array([[0.0, 0.0, 0.4]]),
        seed=seed,
    )
    indoor_items = generate_obstacles(config)
    obs_xml = obstacles_to_xml(indoor_items)
    
    initial_pos = np.array([[0.0, 0.0, 0.3]])
    env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=initial_pos,
        custom_xml=obs_xml
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Cấu trúc phòng kín: 4 Vách tường + 1 Trần nhà + {len(indoor_items) - 5} Vật dụng nội thất")
    print(f"[*] Kích thước phòng: Chiều dài={config.arena_size[0]*2}m, Chiều rộng={config.arena_size[1]*2}m, Độ cao trần={config.arena_size[2]}m")
    print(f"[*] Vùng cất cánh an toàn tâm phòng: {config.safe_zone_radius}m (Seed: {seed})")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone (X, Y, Z)':<24} | {'Mục tiêu phòng':<20} | {'Khoảng cách tới trần':<24}")
    print("-" * 80)
    
    # Lộ trình bay lượn tránh cột và bàn trong phòng
    ceiling_z = config.arena_size[2]
    waypoints = [
        np.array([0.0, 0.0, 0.6]),
        np.array([0.7, 0.5, 0.8]),
        np.array([0.6, -0.6, 0.7]),
        np.array([-0.6, -0.6, 0.9]),
        np.array([-0.6, 0.6, 0.75]),
        np.array([0.0, 0.0, 0.6]),
    ]
    wp_idx = 0
    
    frames = []
    start_time = time.time()
    
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
                
        dist_to_ceiling = ceiling_z - env.pos[0, 2]
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            t_str = f"WP{wp_idx}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            print(f"{step:<6} | {p_str:<24} | {t_str:<20} | {dist_to_ceiling:<24.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "obstacle_indoor.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: Chủ đề INDOOR (Trong nhà)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_obstacle_args("INDOOR (Trong nhà)")
    run_indoor_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        camera=args.camera
    )

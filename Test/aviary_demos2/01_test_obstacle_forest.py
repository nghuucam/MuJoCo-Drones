"""Kiểm tra môi trường Chướng ngại vật: FOREST (Rừng cây).

Đặc điểm môi trường:
- Tự động sinh ngẫu nhiên rừng cây gồm các thân cây hình trụ (Cylinder Trunks) màu gỗ
  và tán lá xanh (Foliage Crowns) che phủ bên trên.
- Thiết lập vùng an toàn (Safe Zone) tại vị trí xuất phát của drone để không bị kẹt.
- Drone bay lượn lách qua các hàng cây theo quỹ đạo thử nghiệm.

Cách chạy:
    python 01_test_obstacle_forest.py --gui        # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 01_test_obstacle_forest.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 01_test_obstacle_forest.py --num-obstacles 35  # Tăng mật độ cây rừng
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.wrappers.obstacles import ObstacleConfig, ObstacleType, generate_obstacles, obstacles_to_xml, Obstacle
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils2 import rpm_to_normalized_action, save_gif, parse_obstacle_args


def build_forest_obstacles(num_obstacles=25, seed=42, arena_size=(2.5, 2.5, 2.0)):
    """Tạo chướng ngại vật rừng cây với thân gỗ và tán lá xanh."""
    rng = np.random.default_rng(seed)
    config = ObstacleConfig(
        obstacle_type=ObstacleType.FOREST,
        num_obstacles=num_obstacles,
        arena_size=arena_size,
        min_spacing=0.35,
        safe_zone_radius=0.6,
        safe_zone_centers=np.array([[0.0, 0.0, 0.5]]),
        seed=seed,
    )
    base_trees = generate_obstacles(config)
    
    # Bổ sung tán cây (Green Foliage Spheres) trên đỉnh từng thân cây
    enhanced_trees = []
    for tree in base_trees:
        enhanced_trees.append(tree)
        # Thêm tán lá
        r_trunk, h_half = tree.size
        crown_r = r_trunk * rng.uniform(2.2, 3.2)
        crown_pos = np.array([tree.position[0], tree.position[1], h_half * 2 + crown_r * 0.7])
        crown_rgba = np.array([
            rng.uniform(0.1, 0.25),
            rng.uniform(0.55, 0.85),
            rng.uniform(0.15, 0.35),
            0.95
        ])
        enhanced_trees.append(Obstacle(
            geom_type="sphere",
            position=crown_pos,
            size=np.array([crown_r]),
            rgba=crown_rgba,
        ))
    return enhanced_trees, config


def run_forest_demo(gui: bool = False, record: bool = False, steps: int = 300, num_obstacles: int = 25, seed: int = 42, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG CHƯỚNG NGẠI VẬT: CHỦ ĐỀ RỪNG CÂY (FOREST)")
    print("=" * 80)
    
    trees, config = build_forest_obstacles(num_obstacles=num_obstacles, seed=seed)
    obs_xml = obstacles_to_xml(trees)
    
    initial_pos = np.array([[0.0, 0.0, 0.5]])
    env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=initial_pos,
        custom_xml=obs_xml
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Tổng số đối tượng cây rừng được tạo: {len(trees)} (thân cây + tán lá)")
    print(f"[*] Kích thước rừng cây (Arena): ±{config.arena_size[0]}m x ±{config.arena_size[1]}m, Cao: {config.arena_size[2]}m")
    print(f"[*] Bán kính vùng an toàn xuất phát: {config.safe_zone_radius}m (Seed: {seed})")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone (X, Y, Z)':<24} | {'Mục tiêu bay':<20} | {'Khoảng cách tới cây gần nhất':<28}")
    print("-" * 80)
    
    # Lộ trình bay lượn số 8 giữa các hàng cây
    waypoints = [
        np.array([0.0, 0.0, 0.8]),
        np.array([0.8, 0.6, 0.9]),
        np.array([0.0, 1.2, 0.8]),
        np.array([-0.8, 0.6, 0.9]),
        np.array([0.0, 0.0, 0.8]),
        np.array([0.8, -0.6, 0.9]),
        np.array([0.0, -1.2, 0.8]),
        np.array([-0.8, -0.6, 0.9]),
    ]
    wp_idx = 0
    
    frames = []
    start_time = time.time()
    
    # Lấy danh sách tọa độ XY của thân cây để tính khoảng cách
    tree_positions = np.array([t.position for t in trees if t.geom_type == "cylinder"])
    
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
                
        # Tính khoảng cách tới thân cây gần nhất
        dists = np.linalg.norm(tree_positions[:, :2] - env.pos[0, :2], axis=1)
        min_tree_dist = np.min(dists) if len(dists) > 0 else 999.0
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            t_str = f"WP{wp_idx}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            print(f"{step:<6} | {p_str:<24} | {t_str:<20} | {min_tree_dist:<28.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "obstacle_forest.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: Chủ đề FOREST (Rừng cây)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_obstacle_args("FOREST (Rừng cây)")
    run_forest_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        camera=args.camera
    )

"""Kiểm tra môi trường Chướng ngại vật: URBAN (Đô thị / Hẻm nhà cao tầng).

Đặc điểm môi trường:
- Tự động sinh ngẫu nhiên các tòa nhà hình khối hộp (Box Buildings) với các kích thước
  chiều dài, chiều rộng và chiều cao khác nhau (Urban Canyon).
- Các tòa nhà mang tông màu xám bê tông, kính xanh, đá xây dựng thực tế.
- Drone bay luồn lách dọc theo các khe hẻm đô thị và hẻm phố giữa các khối nhà cao tầng.

Cách chạy:
    python 02_test_obstacle_urban.py --gui        # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 02_test_obstacle_urban.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 02_test_obstacle_urban.py --num-obstacles 20  # Thay đổi số lượng tòa nhà
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


def build_urban_obstacles(num_obstacles=18, seed=42, arena_size=(2.5, 2.5, 2.0)):
    """Tạo các tòa nhà đô thị với màu sắc hiện đại và đèn tín hiệu trên nóc."""
    rng = np.random.default_rng(seed)
    config = ObstacleConfig(
        obstacle_type=ObstacleType.URBAN,
        num_obstacles=num_obstacles,
        arena_size=arena_size,
        min_spacing=0.45,
        safe_zone_radius=0.65,
        safe_zone_centers=np.array([[0.0, 0.0, 0.5]]),
        seed=seed,
    )
    base_buildings = generate_obstacles(config)
    
    # Bổ sung đèn tín hiệu nóc tòa nhà (Rooftop Aviation Warning Lights)
    enhanced_buildings = []
    for b in base_buildings:
        enhanced_buildings.append(b)
        sx, sy, sz_half = b.size
        # Đèn chớp đỏ trên đỉnh nóc tòa nhà cao
        if sz_half * 2 > 0.8:
            light_pos = np.array([b.position[0], b.position[1], sz_half * 2 + 0.02])
            enhanced_buildings.append(Obstacle(
                geom_type="sphere",
                position=light_pos,
                size=np.array([0.025]),
                rgba=np.array([1.0, 0.1, 0.1, 0.9]),
            ))
    return enhanced_buildings, config


def run_urban_demo(gui: bool = False, record: bool = False, steps: int = 300, num_obstacles: int = 18, seed: int = 42, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG CHƯỚNG NGẠI VẬT: CHỦ ĐỀ ĐÔ THỊ (URBAN)")
    print("=" * 80)
    
    buildings, config = build_urban_obstacles(num_obstacles=num_obstacles, seed=seed)
    obs_xml = obstacles_to_xml(buildings)
    
    initial_pos = np.array([[0.0, 0.0, 0.5]])
    env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=initial_pos,
        custom_xml=obs_xml
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Số lượng tòa nhà và đối tượng đô thị: {len(buildings)}")
    print(f"[*] Kích thước khu đô thị: ±{config.arena_size[0]}m x ±{config.arena_size[1]}m, Chiều cao tối đa: {config.arena_size[2]}m")
    print(f"[*] Vùng an toàn xuất phát ngã tư: {config.safe_zone_radius}m (Seed: {seed})")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone (X, Y, Z)':<24} | {'Mục tiêu bay':<20} | {'Khoảng cách tới tòa nhà gần nhất':<32}")
    print("-" * 80)
    
    # Lộ trình bay tuần tra qua các con phố / hẻm hẹp
    waypoints = [
        np.array([0.0, 0.0, 0.7]),
        np.array([1.0, 0.0, 0.9]),
        np.array([1.0, 1.0, 1.1]),
        np.array([0.0, 1.0, 0.8]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, -0.5, 0.8]),
        np.array([0.0, -0.8, 0.7]),
        np.array([0.0, 0.0, 0.7]),
    ]
    wp_idx = 0
    
    frames = []
    start_time = time.time()
    
    building_centers = np.array([b.position for b in buildings if b.geom_type == "box"])
    
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
                
        # Khoảng cách tới tòa nhà gần nhất
        dists = np.linalg.norm(building_centers[:, :2] - env.pos[0, :2], axis=1)
        min_bld_dist = np.min(dists) if len(dists) > 0 else 999.0
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            t_str = f"WP{wp_idx}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            print(f"{step:<6} | {p_str:<24} | {t_str:<20} | {min_bld_dist:<32.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "obstacle_urban.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: Chủ đề URBAN (Đô thị)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_obstacle_args("URBAN (Đô thị)")
    run_urban_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        camera=args.camera
    )

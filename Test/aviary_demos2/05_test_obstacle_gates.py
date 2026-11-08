"""Kiểm tra môi trường Chướng ngại vật: GATES (Hệ thống cổng đua liên hoàn).

Đặc điểm môi trường:
- Tự động sinh ngẫu nhiên chuỗi các khung cổng (Gates) dọc theo quỹ đạo bay vòng tròn/ellipse.
- Mỗi cổng được tạo từ các thanh trụ đứng (Pillars) và xà ngang (Crossbar) màu đỏ/cam nổi bật,
  ở giữa có vùng tâm cổng (Portal Window) phát sáng.
- Drone bay lượn lần lượt xuyên qua các cổng, kiểm tra kỹ năng bay luồn lách qua các khe hở hẹp.

Cách chạy:
    python 05_test_obstacle_gates.py --gui        # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 05_test_obstacle_gates.py --record     # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 05_test_obstacle_gates.py --num-obstacles 6   # Tạo 6 cổng đua
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


def build_enhanced_gates(num_gates=6, seed=42, arena_size=(2.5, 2.5, 2.0)):
    """Tạo chuỗi cổng đua với khung cổng và vùng tâm cổng phát sáng."""
    rng = np.random.default_rng(seed)
    config = ObstacleConfig(
        obstacle_type=ObstacleType.GATES,
        num_obstacles=num_gates,
        arena_size=arena_size,
        seed=seed,
    )
    base_gate_parts = generate_obstacles(config)
    
    # Tính tọa độ tâm của các cổng để drone bay qua và tạo portal phát sáng
    n_gates = min(num_gates, 10)
    xh, yh, zh = arena_size
    gate_centers = []
    
    enhanced_parts = list(base_gate_parts)
    for i in range(n_gates):
        angle = 2 * np.pi * i / n_gates
        radius = min(xh, yh) * 0.6
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)
        gate_h = 1.0 + 0.2 * np.sin(angle) # Cao độ cổng nhấp nhô lượn sóng
        gate_w = 0.5
        yaw = angle + np.pi / 2
        
        gate_center = np.array([cx, cy, gate_h * 0.65])
        gate_centers.append(gate_center)
        
        # Thêm tâm cổng phát sáng xanh ngọc (Portal Zone)
        enhanced_parts.append(Obstacle(
            geom_type="box",
            position=gate_center,
            size=np.array([0.005, gate_w / 2 * 0.85, gate_h * 0.3]),
            rgba=np.array([0.1, 1.0, 0.4, 0.25]),
            euler=np.array([0, 0, yaw]),
        ))
        
    return enhanced_parts, gate_centers, config


def run_gates_demo(gui: bool = False, record: bool = False, steps: int = 350, num_obstacles: int = 6, seed: int = 42, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 80)
    print(" KHỞI TẠO MÔI TRƯỜNG CHƯỚNG NGẠI VẬT: CHỦ ĐỀ CỔNG ĐUA (GATES)")
    print("=" * 80)
    
    gate_parts, gate_centers, config = build_enhanced_gates(num_gates=num_obstacles, seed=seed)
    obs_xml = obstacles_to_xml(gate_parts)
    
    initial_pos = np.array([[0.0, 0.0, 0.5]])
    env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        initial_xyzs=initial_pos,
        custom_xml=obs_xml
    )
    
    obs, info = env.reset()
    ctrl = PIDControl(env)
    
    print(f"[*] Tổng số cổng đua tạo ngẫu nhiên: {len(gate_centers)} cổng")
    for gi, gc in enumerate(gate_centers):
        print(f"    - Cổng {gi}: [X={gc[0]:.2f}, Y={gc[1]:.2f}, Z={gc[2]:.2f}] m")
    print(f"[*] Tổng số chi tiết khung cổng 3D: {len(gate_parts)}")
    print("-" * 80)
    print(f"{'Bước':<6} | {'Tọa độ Drone (X, Y, Z)':<24} | {'Cổng đích':<16} | {'Khoảng cách tới tâm cổng':<28}")
    print("-" * 80)
    
    # Thứ tự bay qua các cổng
    waypoints = [np.array([0.0, 0.0, 0.6])] + gate_centers + [gate_centers[0]]
    wp_idx = 0
    
    frames = []
    start_time = time.time()
    
    for step in range(steps):
        target = waypoints[wp_idx]
        dist_to_gate = np.linalg.norm(env.pos[0] - target)
        
        if dist_to_gate < 0.25 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            if wp_idx <= len(gate_centers):
                print(f"  >>> [GATE PASSED!] Drone đã bay xuyên qua cổng {wp_idx - 1}! Chuyển sang cổng {wp_idx % len(gate_centers)} <<<")
            
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
                
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{env.pos[0, 0]:.2f}, {env.pos[0, 1]:.2f}, {env.pos[0, 2]:.2f}]"
            g_str = f"Cổng {wp_idx % len(gate_centers)}"
            print(f"{step:<6} | {p_str:<24} | {g_str:<16} | {dist_to_gate:<28.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "obstacle_gates.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 80)
    print(" HOÀN THÀNH TEST: Chủ đề GATES (Cổng đua)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_obstacle_args("GATES (Cổng đua liên hoàn)")
    run_gates_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        camera=args.camera
    )

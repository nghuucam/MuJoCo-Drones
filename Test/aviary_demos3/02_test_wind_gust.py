"""Kiểm tra WindWrapper: Mô hình gió giật ngẫu nhiên (STOCHASTIC GUSTS).

Đặc điểm kiểm tra:
- Các cơn gió giật đột ngột với hướng ngẫu nhiên và cường độ xung lực tức thời (Gust Force).
- Mỗi cơn gió giật kéo dài trong một số bước ngắn (gust_duration_steps = 15 bước ~ 0.3 giây).
- Kiểm tra độ bền vững (Robustness) và khả năng hồi phục cân bằng của hệ thống điều khiển sau khi bị gió giật.

Cách chạy:
    python 02_test_wind_gust.py --gui                   # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 02_test_wind_gust.py --record                # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 02_test_wind_gust.py --gust-intensity 0.015  # Tăng cường độ lực gió giật
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.wrappers.wind_wrapper import WindWrapper
from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils3 import rpm_to_normalized_action, save_gif, get_windsock_xml, parse_wind_args


def run_gust_wind_demo(gui: bool = False, record: bool = False, steps: int = 350, gust_intensity: float = 0.010, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 85)
    print(f" KIỂM TRA WINDWRAPPER: MÔ HÌNH GIÓ GIẬT BẤT NGỜ (GUST FORCE = {gust_intensity:.4f} N)")
    print("=" * 85)
    
    wind_cfg = WindConfig(
        model=WindModel.GUST,
        gust_intensity=gust_intensity,
        gust_probability=0.04,         # 4% xác suất mỗi bước xảy ra cơn gió giật
        gust_duration_steps=15,        # Kéo dài khoảng 0.3 giây
    )
    
    custom_xml = get_windsock_xml()
    base_env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        target_height=1.0,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        custom_xml=custom_xml
    )
    
    env = WindWrapper(base_env, wind_config=wind_cfg)
    obs, info = env.reset()
    ctrl = PIDControl(base_env)
    target_pos = np.array([0.0, 0.0, 1.0])
    
    print(f"[*] Cường độ gió giật cực đại: {gust_intensity:.4f} N")
    print(f"[*] Xác suất xuất hiện gió giật: {wind_cfg.gust_probability * 100:.1f}% / bước (Thời lượng: {wind_cfg.gust_duration_steps} steps)")
    print(f"[*] Tọa độ đích Hover: {target_pos.tolist()} m")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Vị trí Drone (X, Y, Z)':<24} | {'Vận tốc tức thời (m/s)':<24} | {'Trạng thái gió'}")
    print("-" * 85)
    
    frames = []
    start_time = time.time()
    last_gust_status = False
    
    for step in range(steps):
        rpm, _, _ = ctrl.computeControl(
            control_timestep=base_env.CTRL_TIMESTEP,
            cur_pos=base_env.pos[0],
            cur_quat=base_env.quat[0],
            cur_vel=base_env.vel[0],
            cur_ang_vel=base_env.ang_v[0],
            target_pos=target_pos
        )
        action = rpm_to_normalized_action(rpm, base_env)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Kiểm tra trạng thái gió giật từ wind_field
        is_gusting = env.wind_field._gust_active
        if is_gusting and not last_gust_status:
            g_dir = env.wind_field._gust_direction
            print(f"  >>> [GUST ALERT!] Cơn gió giật xuất hiện! Hướng: [{g_dir[0]:.2f}, {g_dir[1]:.2f}, {g_dir[2]:.2f}] <<<")
        last_gust_status = is_gusting
        
        # Render
        if gui:
            base_env.render()
            time.sleep(max(0.0, base_env.CTRL_TIMESTEP - (time.time() - start_time) % base_env.CTRL_TIMESTEP))
        elif record and (step % 2 == 0):
            frame = base_env.render(camera_mode=camera)
            if frame is not None:
                frames.append(frame)
                
        vel_mag = np.linalg.norm(base_env.vel[0])
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            v_str = f"{vel_mag:.3f} m/s"
            status_str = "GIÓ GIẬT MẠNH (GUST)" if is_gusting else "Bình thường (Lặng gió)"
            print(f"{step:<6} | {p_str:<24} | {v_str:<24} | {status_str}")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "wind_gust.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: Mô hình gió GUST")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_wind_args("STOCHASTIC GUSTS (Gió giật ngẫu nhiên)")
    run_gust_wind_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        gust_intensity=args.gust_intensity,
        camera=args.camera
    )

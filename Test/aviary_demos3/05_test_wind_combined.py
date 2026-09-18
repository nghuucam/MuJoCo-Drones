"""Kiểm tra WindWrapper: Mô hình gió bão kết hợp (COMBINED EXTREME WIND & WEATHER).

Đặc điểm kiểm tra:
- Thử thách điều kiện thời tiết khắc nghiệt nhất:
  + Gió nền thổi liên tục (Constant Wind: 1.5 m/s)
  + Nhiễu loạn khí quyển ngẫu nhiên liên tục (Dryden Turbulence: 1.5)
  + Gió giật bất thình lình (Sudden Gusts: 0.012 N)
- Kiểm tra giới hạn bám giữ vị trí (Position Holding Limit) và năng lượng phản ứng của 4 động cơ.

Cách chạy:
    python 05_test_wind_combined.py --gui               # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 05_test_wind_combined.py --record            # Lưu hoạt ảnh GIF vào thư mục gifs/
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


def run_combined_wind_demo(gui: bool = False, record: bool = False, steps: int = 350, camera: str = "track", soft_pid: bool = False):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 85)
    print(" KIỂM TRA WINDWRAPPER: MÔ HÌNH GIÓ BÃO KẾT HỢP (COMBINED EXTREME WEATHER)")
    print(f" Chế độ điều khiển PID: {'MỀM / DỄ LUNG LAY (Compliant)' if soft_pid else 'CỨNG / BÁM CHẶT (Stiff)'}")
    print("=" * 85)
    
    wind_cfg = WindConfig(
        model=WindModel.COMBINED,
        constant_wind=np.array([2.8, 1.2, 0.0]),  # Gió thổi chéo X-Y ~3.0 m/s
        turbulence_intensity=1.8,                 # Nhiễu loạn Dryden chân thực
        gust_intensity=0.040,                     # Gió giật mạnh 0.040 N (~15% trọng lượng drone)
        gust_probability=0.04,
        gust_duration_steps=15,
        drag_coefficient=0.006
    )
    
    custom_xml = get_windsock_xml(wind_dir=wind_cfg.constant_wind)
    base_env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        target_height=1.0,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        custom_xml=custom_xml
    )
    
    env = WindWrapper(base_env, wind_config=wind_cfg)
    obs, info = env.reset()
    ctrl = PIDControl(base_env, mode="compliant" if soft_pid else "stiff")
    target_pos = np.array([0.0, 0.0, 1.0])
    
    print(f"[*] Thành phần gió kết hợp:")
    print(f"    - Gió nền (Constant):   {wind_cfg.constant_wind.tolist()} m/s")
    print(f"    - Nhiễu loạn (Dryden):  {wind_cfg.turbulence_intensity} (Cường độ cao)")
    print(f"    - Gió giật (Gust):      {wind_cfg.gust_intensity} N (Xác suất 4%)")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Vị trí Drone (X, Y, Z)':<24} | {'Góc nghiêng Roll/Pitch (°)':<28} | {'Sai số đích (m)':<16}")
    print("-" * 85)
    
    frames = []
    start_time = time.time()
    
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
        
        # Render
        if gui:
            base_env.render()
            time.sleep(max(0.0, base_env.CTRL_TIMESTEP - (time.time() - start_time) % base_env.CTRL_TIMESTEP))
        elif record and (step % 2 == 0):
            frame = base_env.render(camera_mode=camera)
            if frame is not None:
                frames.append(frame)
                
        roll_deg = np.degrees(base_env.rpy[0, 0])
        pitch_deg = np.degrees(base_env.rpy[0, 1])
        pos_err = np.linalg.norm(base_env.pos[0] - target_pos)
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            att_str = f"R={roll_deg:+5.1f}°, P={pitch_deg:+5.1f}°"
            print(f"{step:<6} | {p_str:<24} | {att_str:<28} | {pos_err:<16.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "wind_combined.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: Mô hình gió COMBINED")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_wind_args("COMBINED EXTREME WIND (Gió kết hợp cực hạn)")
    run_combined_wind_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        camera=args.camera,
        soft_pid=args.soft_pid
    )

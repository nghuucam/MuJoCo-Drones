"""Kiểm tra WindWrapper: Mô hình nhiễu loạn khí quyển Dryden (MIL-F-8785C DRYDEN TURBULENCE).

Đặc điểm kiểm tra:
- Mô phỏng phổ nhiễu loạn khí động học tiêu chuẩn hàng không quân sự (Dryden Gust/Turbulence Model).
- Tạo ra các dao động liên tục, ngẫu nhiên có tương quan thời gian theo 3 trục không gian (u, v, w).
- Tái hiện chân thực cảm giác drone bay ngoài trời có gió xoáy và luồng khí nhiễu động.

Cách chạy:
    python 03_test_wind_dryden.py --gui               # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 03_test_wind_dryden.py --record            # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 03_test_wind_dryden.py --turbulence 2.0    # Tăng cường độ nhiễu loạn gió bão
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


def run_dryden_wind_demo(gui: bool = False, record: bool = False, steps: int = 350, turbulence: float = 1.2, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 85)
    print(f" KIỂM TRA WINDWRAPPER: MÔ HÌNH NHIỄU LOẠN KHÍ QUYỂN DRYDEN (TURBULENCE = {turbulence:.2f})")
    print("=" * 85)
    
    wind_cfg = WindConfig(
        model=WindModel.DRYDEN,
        turbulence_intensity=turbulence,
        altitude=1.0,
        airspeed=0.8,
        drag_coefficient=0.001
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
    
    print(f"[*] Cường độ nhiễu loạn khí quyển: {turbulence:.2f} (Thang đo Dryden MIL-F-8785C)")
    print(f"[*] Độ cao hoạt động: {wind_cfg.altitude}m, Vận tốc dòng khí: {wind_cfg.airspeed} m/s")
    print(f"[*] Tọa độ đích Hover: {target_pos.tolist()} m")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Vị trí Drone (X, Y, Z)':<24} | {'Vận tốc góc Yaw/Pitch/Roll':<28} | {'Sai số (m)':<12}")
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
                
        pos_err = np.linalg.norm(base_env.pos[0] - target_pos)
        ang_v = base_env.ang_v[0]
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            ang_str = f"[{ang_v[0]:+5.2f}, {ang_v[1]:+5.2f}, {ang_v[2]:+5.2f}] rad/s"
            print(f"{step:<6} | {p_str:<24} | {ang_str:<28} | {pos_err:<12.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "wind_dryden.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: Mô hình gió DRYDEN")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_wind_args("DRYDEN TURBULENCE (Nhiễu loạn khí quyển Dryden)")
    run_dryden_wind_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        turbulence=args.turbulence,
        camera=args.camera
    )

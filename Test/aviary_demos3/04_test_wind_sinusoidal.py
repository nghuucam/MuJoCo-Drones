"""Kiểm tra WindWrapper: Mô hình gió dao động điều hòa hình sin (SINUSOIDAL OSCILLATING WIND).

Đặc điểm kiểm tra:
- Lực gió biến thiên tuần hoàn theo hàm sin: F_wind(t) = A * sin(2 * pi * t / T).
- Chu kỳ dao động T (sinusoidal_period) và biên độ lực A (sinusoidal_amplitude).
- Quan sát chuyển động lắc lư tuần hoàn của drone khi chịu luồng gió đảo chiều đều đặn (như quạt gió quay đảo chiều).

Cách chạy:
    python 04_test_wind_sinusoidal.py --gui               # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 04_test_wind_sinusoidal.py --record            # Lưu hoạt ảnh GIF vào thư mục gifs/
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


def run_sinusoidal_wind_demo(gui: bool = False, record: bool = False, steps: int = 350, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    PERIOD_SEC = 2.5
    AMPLITUDE_N = 0.006
    
    print("=" * 85)
    print(f" KIỂM TRA WINDWRAPPER: MÔ HÌNH GIÓ ĐIỀU HÒA HÌNH SIN (PERIOD = {PERIOD_SEC}s, AMP = {AMPLITUDE_N}N)")
    print("=" * 85)
    
    wind_cfg = WindConfig(
        model=WindModel.SINUSOIDAL,
        sinusoidal_amplitude=AMPLITUDE_N,
        sinusoidal_period=PERIOD_SEC,
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
    
    print(f"[*] Biên độ lực gió (Amplitude): {AMPLITUDE_N} N")
    print(f"[*] Chu kỳ dao động (Period): {PERIOD_SEC} giây")
    print(f"[*] Tọa độ đích Hover: {target_pos.tolist()} m")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Thời gian (s)':<14} | {'Vị trí Drone (X, Y, Z)':<24} | {'Góc nghiêng Pitch':<20} | {'Pha gió'}")
    print("-" * 85)
    
    frames = []
    start_time = time.time()
    
    for step in range(steps):
        t = step * base_env.CTRL_TIMESTEP
        phase = np.sin(2 * np.pi * t / PERIOD_SEC)
        
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
                
        pitch_deg = np.degrees(base_env.rpy[0, 1])
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            pitch_str = f"{pitch_deg:+6.2f}°"
            phase_str = f"sin(ωt) = {phase:+5.2f}"
            print(f"{step:<6} | {t:<14.2f} | {p_str:<24} | {pitch_str:<20} | {phase_str}")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "wind_sinusoidal.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: Mô hình gió SINUSOIDAL")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_wind_args("SINUSOIDAL WIND (Gió điều hòa hình sin)")
    run_sinusoidal_wind_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        camera=args.camera
    )

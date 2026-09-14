"""Kiểm tra WindWrapper: Mô hình gió thổi liên tục (CONSTANT WIND FIELD).

Đặc điểm kiểm tra:
- Luồng gió cố định thổi ngang theo trục X với vận tốc xác định (ví dụ: 1.5 m/s hoặc 2.5 m/s).
- Drone phải bay lơ lửng (hover) tại điểm đích [0.0, 0.0, 1.0].
- Quan sát bộ điều khiển PID bù gió: Drone phải chủ động nghiêng cánh góc pitch (Pitch Tilt Angle)
  về phía trước để tạo lực đẩy chống lại sức gió thổi lùi.

Cách chạy:
    python 01_test_wind_constant.py --gui               # Xem trực tiếp trong cửa sổ 3D MuJoCo
    python 01_test_wind_constant.py --record            # Lưu hoạt ảnh GIF vào thư mục gifs/
    python 01_test_wind_constant.py --wind-speed 2.5    # Tăng tốc độ gió lên 2.5 m/s
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


def run_constant_wind_demo(gui: bool = False, record: bool = False, steps: int = 300, wind_speed: float = 1.5, camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 85)
    print(f" KIỂM TRA WINDWRAPPER: MÔ HÌNH GIÓ THỔI LIÊN TỤC (CONSTANT WIND = {wind_speed:.2f} m/s)")
    print("=" * 85)
    
    wind_vector = np.array([wind_speed, 0.0, 0.0])
    wind_cfg = WindConfig(
        model=WindModel.CONSTANT,
        constant_wind=wind_vector,
        drag_coefficient=0.001
    )
    
    # Tạo môi trường Hover kèm cột cờ gió
    custom_xml = get_windsock_xml(wind_dir=wind_vector)
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
    
    print(f"[*] Vector gió cài đặt: wx={wind_vector[0]:.2f} m/s, wy={wind_vector[1]:.2f} m/s, wz={wind_vector[2]:.2f} m/s")
    print(f"[*] Tọa độ đích Hover: {target_pos.tolist()} m")
    print(f"[*] Cơ chế kiểm tra: Quan sát góc nghiêng bù gió (Pitch/Roll Tilt) và độ trôi vị trí của Drone")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Vị trí Drone (X, Y, Z)':<24} | {'Góc nghiêng Pitch (độ)':<24} | {'Độ lệch vị trí':<18}")
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
                
        pitch_deg = np.degrees(base_env.rpy[0, 1])
        pos_err = np.linalg.norm(base_env.pos[0] - target_pos)
        
        if step % 30 == 0 or step == steps - 1:
            p_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            pitch_str = f"{pitch_deg:+6.2f}° (nghiêng bù gió)"
            print(f"{step:<6} | {p_str:<24} | {pitch_str:<24} | {pos_err:<18.3f} m")
            
        if terminated or truncated:
            print(f"[!] Dừng tại bước {step} (Terminated={terminated}, Truncated={truncated})")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "wind_constant.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(" HOÀN THÀNH TEST: Mô hình gió CONSTANT")
    print("=" * 85)


if __name__ == "__main__":
    args = parse_wind_args("CONSTANT WIND (Gió thổi liên tục)")
    run_constant_wind_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        wind_speed=args.wind_speed,
        camera=args.camera
    )

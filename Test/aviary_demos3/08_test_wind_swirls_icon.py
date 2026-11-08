"""Trực quan hóa luồng gió 3D (Wind Gust Swirls) theo biểu tượng khí động học.

Mô phỏng drone bay lơ lửng (hover) trong luồng gió với các cụm biểu tượng xoáy gió 3D
(3 dòng khí uốn lượn có vòng cuộn xoắn ốc ở đầu mút) bay quanh drone và đổi hướng theo vector gió.

Cách chạy:
    python 08_test_wind_swirls_icon.py --gui            # Mở cửa sổ 3D tương tác xem trực tiếp
    python 08_test_wind_swirls_icon.py --record         # Xuất ảnh động GIF vào gifs/wind_swirls_icon.gif
    python 08_test_wind_swirls_icon.py --wind-speed 3.0 # Đổi tốc độ gió sang 3.0 m/s
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
from controller_utils3 import (
    rpm_to_normalized_action,
    save_gif,
    get_windsock_xml,
    parse_wind_args,
)


def run_swirls_demo(gui: bool = False, record: bool = False, steps: int = 300, wind_speed: float = 3.5, soft_pid: bool = False):
    render_mode = "human" if gui else ("rgb_array" if record else None)

    print("=" * 85)
    print(" DEMO TRỰC QUAN HÓA CUỘN XOÁY GIÓ 3D (WIND GUST SWIRLS - USER ICON)")
    print(f" Chế độ điều khiển PID: {'MỀM / DỄ LUNG LAY (Compliant)' if soft_pid else 'CỨNG / BÁM CHẶT (Stiff)'}")
    print("=" * 85)
    print(f"[*] Tốc độ gió: {wind_speed:.2f} m/s")
    print("[*] Biểu tượng: 3 dòng khí uốn cong với các vòng xoáy tròn 3D ở đầu mút bao bọc quanh Drone")
    print("-" * 85)

    wind_vector = np.array([wind_speed, 0.0, 0.0])
    wind_cfg = WindConfig(
        model=WindModel.CONSTANT,
        constant_wind=wind_vector,
        drag_coefficient=0.005,
    )

    custom_xml = get_windsock_xml(wind_dir=wind_vector, wind_speed=wind_speed)
    base_env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        target_height=1.0,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        custom_xml=custom_xml,
    )

    env = WindWrapper(base_env, wind_config=wind_cfg)
    obs, info = env.reset()
    ctrl = PIDControl(base_env, mode="compliant" if soft_pid else "stiff")
    target_pos = np.array([0.0, 0.0, 1.0])

    frames = []
    print(f"{'Bước':<6} | {'Vị trí Drone (X, Y, Z)':<24} | {'Góc nghiêng Pitch (°)':<22} | {'Trạng thái'}")
    print("-" * 85)

    for step in range(steps):
        rpm, _, _ = ctrl.computeControl(
            control_timestep=base_env.CTRL_TIMESTEP,
            cur_pos=base_env.pos[0],
            cur_quat=base_env.quat[0],
            cur_vel=base_env.vel[0],
            cur_ang_vel=base_env.ang_v[0],
            target_pos=target_pos,
        )
        action = rpm_to_normalized_action(rpm, base_env)
        obs, reward, terminated, truncated, info = env.step(action)

        if gui:
            base_env.render()
            time.sleep(max(0.0, base_env.CTRL_TIMESTEP))
            if hasattr(base_env, "_viewer") and base_env._viewer is not None and not base_env._viewer.is_running():
                print("\n[!] Cửa sổ xem 3D MuJoCo đã được người dùng đóng.")
                break

        if record and step % 2 == 0:
            frame = base_env.render(camera_mode="fixed")
            frames.append(frame)

        if step % 50 == 0 or step == steps - 1:
            pos_str = f"[{base_env.pos[0, 0]:.2f}, {base_env.pos[0, 1]:.2f}, {base_env.pos[0, 2]:.2f}]"
            pitch_deg = np.degrees(base_env.rpy[0, 1])
            print(f"{step:<6} | {pos_str:<24} | {pitch_deg:>+8.2f}°              | Đang bay trong luồng xoáy gió 3D")

    if record:
        gif_path = os.path.join(os.path.dirname(__file__), "gifs", "wind_swirls_icon.gif")
        save_gif(frames, gif_path, fps=24)

    env.close()
    print("=" * 85)
    print(" HOÀN THÀNH DEMO TRỰC QUAN HÓA XOÁY GIÓ 3D")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    args = parse_wind_args("Demo Trực quan hóa Biểu tượng Cuộn Xoáy Gió 3D")
    run_swirls_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        wind_speed=args.wind_speed,
        soft_pid=args.soft_pid,
    )

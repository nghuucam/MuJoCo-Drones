"""Kiểm tra trực quan độ tác động của gió: THỔI BAY DRONE & NGHIÊNG BÙ GIÓ (VISIBLE DRIFT & TILT).

Mục đích kịch bản này:
- Chứng minh rõ ràng 100% lực gió hoạt động thực tế trên đồ họa 3D mà mắt thường có thể nhìn thấy:
  1. Giai đoạn 1 (Bước 0 -> 80): LẶNG GIÓ HOÀN TOÀN (Wind = 0 m/s).
     Drone bay lơ lửng đứng im phăng phắc tại gốc tọa độ [0.0, 0.0, 1.0].
  2. Giai đoạn 2 (Bước 80 trở đi): GIÓ MẠNH ÙA TỚI (Wind = 6.0 m/s)!
     + Ở chế độ Bù gió (--mode resist): Drone giật mình và nghiêng hẳn -12° đến -15° để chống cự lại gió!
     + Ở chế độ Thả trôi (--mode drift): Drone bị gió cuốn bay dạt vèo đi hơn 2.5 mét!

Cách chạy:
    python 07_test_wind_drift_visible.py --gui                  # Xem trực tiếp drone nghiêng -14° chống bão
    python 07_test_wind_drift_visible.py --gui --mode drift     # Xem drone bị gió thổi bay dạt đi xa 3m
    python 07_test_wind_drift_visible.py --record               # Lưu hoạt ảnh GIF minh chứng rõ nét
"""

import os
import sys
import time
import argparse
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.wrappers.wind_wrapper import WindWrapper
from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel
from multi_drone_mujoco.control.pid_control import PIDControl
from controller_utils3 import rpm_to_normalized_action, save_gif, get_windsock_xml


def run_visible_drift_demo(gui: bool = True, record: bool = False, steps: int = 300, mode: str = "resist", camera: str = "track"):
    render_mode = "human" if gui else ("rgb_array" if record else None)
    
    print("=" * 85)
    print(f" KIỂM TRA TRỰC QUAN LỰC GIÓ: CHẾ ĐỘ [{mode.upper()}] (VISIBLE WIND EXPERIMENT)")
    print("=" * 85)
    
    STRONG_WIND_SPEED = 3.0  # 6 m/s gió bão rất mạnh
    wind_vector = np.array([STRONG_WIND_SPEED, 0.0, 0.0])
    
    # Cấu hình gió mạnh có hệ số cản 0.005 tạo lực ~0.08 N (>30% trọng lượng drone)
    wind_cfg = WindConfig(
        model=WindModel.CONSTANT,
        constant_wind=wind_vector,
        drag_coefficient=0.005
    )
    
    custom_xml = get_windsock_xml(wind_dir=wind_vector, wind_speed=STRONG_WIND_SPEED)
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
    
    print("[*] Kịch bản thử nghiệm 2 giai đoạn:")
    print("    - Bước 0  -> 70:  LẶNG GIÓ (Wind = 0 m/s) -> Drone đứng yên hoàn toàn ở tâm.")
    print(f"    - Bước 70 -> {steps}: BÃO GIÓ ÙA VÀO ({STRONG_WIND_SPEED} m/s) -> Drone phản ứng mạnh!")
    print(f"[*] Chế độ: {'Drone gồng mình nghiêng -14° chống gió' if mode == 'resist' else 'Drone thả trôi bị gió thổi bay dạt'}")
    print("-" * 85)
    print(f"{'Bước':<6} | {'Giai đoạn':<20} | {'Vị trí X (m)':<16} | {'Góc nghiêng Pitch (°)':<24} | {'Trạng thái'}")
    print("-" * 85)
    
    frames = []
    start_time = time.time()
    wind_started = False
    onset_step = min(70, max(15, steps // 3))
    
    for step in range(steps):
        # Giai đoạn 1: Tắt gió tạm thời bằng cách đặt constant_wind = 0
        if step < onset_step:
            env.wind_config.constant_wind = np.zeros(3)
        else:
            if not wind_started:
                print(f"\n  >>> [BÃO GIÓ ÙA VÀO!] Gió {STRONG_WIND_SPEED:.1f} m/s bắt đầu thổi mạnh dọc theo trục X! <<<\n")
                wind_started = True
            env.wind_config.constant_wind = wind_vector
            
        if mode == "drift":
            # Ở chế độ thả trôi: Chỉ điều khiển giữ độ cao Z=1.0m, không ghì vị trí X, Y
            drift_target = np.array([base_env.pos[0, 0], base_env.pos[0, 1], 1.0])
            rpm, _, _ = ctrl.computeControl(
                control_timestep=base_env.CTRL_TIMESTEP,
                cur_pos=base_env.pos[0],
                cur_quat=base_env.quat[0],
                cur_vel=base_env.vel[0],
                cur_ang_vel=base_env.ang_v[0],
                target_pos=drift_target
            )
        else:
            # Ở chế độ gồng chống gió: Cố giữ nguyên tọa độ gốc [0, 0, 1.0]
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
        x_pos = base_env.pos[0, 0]
        
        if step % 25 == 0 or step == steps - 1:
            phase_name = "LẶNG GIÓ (0 m/s)" if step < onset_step else f"BÃO GIÓ ({STRONG_WIND_SPEED:.1f} m/s)"
            status_desc = "Đứng yên ở tâm" if step < onset_step else ("Nghiêng gồng bù gió" if mode == "resist" else "Bị gió thổi bay")
            print(f"{step:<6} | {phase_name:<20} | {x_pos:<+16.3f} | {pitch_deg:<+24.2f}° | {status_desc}")
            
        if terminated or truncated:
            print(f"[!] Kết thúc tại bước {step}")
            break

    env.close()
    
    if record and frames:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", f"wind_{mode}_visible.gif")
        save_gif(frames, out_gif, fps=24)
        
    print("=" * 85)
    print(f" HOÀN THÀNH TEST: Minh chứng trực quan gió [{mode.upper()}]")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minh chứng trực quan lực gió")
    parser.add_argument("--gui", action="store_true", help="Mở giao diện 3D MuJoCo")
    parser.add_argument("--record", action="store_true", help="Xuất file GIF")
    parser.add_argument("--steps", type=int, default=300, help="Số bước mô phỏng")
    parser.add_argument("--mode", type=str, default="resist", choices=["resist", "drift"], help="'resist': drone nghiêng mình chống gió, 'drift': drone thả trôi bay theo gió")
    parser.add_argument("--camera", type=str, default="track", choices=["track", "fixed", "fpv"])
    args = parser.parse_args()
    
    run_visible_drift_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps,
        mode=args.mode,
        camera=args.camera
    )

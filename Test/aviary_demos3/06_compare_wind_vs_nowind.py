"""Kịch bản đánh giá & So sánh đối chuẩn: KHÔNG GIÓ (Baseline) vs CÓ GIÓ (Wind Active).

Mục đích:
- Chạy 2 lần thử nghiệm liên tiếp với cùng một bộ điều khiển PID và mục tiêu bay lơ lửng:
  1. Thử nghiệm 1: Môi trường lý tưởng không có gió (No Wind).
  2. Thử nghiệm 2: Môi trường có gió bão nhiễu loạn mạnh (Wind Active).
- Thu thập và so sánh trực tiếp các chỉ số định lượng:
  + Sai số bám vị trí trung bình (Mean Position Error)
  + Sai số cực đại (Max Drift Error)
  + Góc nghiêng thân drone trung bình (Mean Tilt Angle)
  + Tốc độ quay động cơ trung bình (Mean Motor RPM Effort)
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


def run_benchmark_trial(with_wind=False, steps=250, record=False, gui=False):
    target_pos = np.array([0.0, 0.0, 1.0])
    
    if with_wind:
        wind_cfg = WindConfig(
            model=WindModel.COMBINED,
            constant_wind=np.array([1.5, 0.0, 0.0]),
            turbulence_intensity=1.2,
            gust_intensity=0.008,
            gust_probability=0.03
        )
        custom_xml = get_windsock_xml(wind_dir=wind_cfg.constant_wind)
    else:
        wind_cfg = WindConfig(model=WindModel.NONE)
        custom_xml = get_windsock_xml()
        
    render_mode = "human" if gui else ("rgb_array" if record else None)
    base_env = HoverAviary(
        render_mode=render_mode,
        ctrl_freq=48,
        target_height=1.0,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        custom_xml=custom_xml
    )
    
    env = WindWrapper(base_env, wind_config=wind_cfg)
    obs, info = env.reset(seed=42)
    ctrl = PIDControl(base_env)
    
    frames = []
    pos_errors = []
    rpms_list = []
    tilts_list = []
    
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
        
        if gui:
            base_env.render()
            time.sleep(base_env.CTRL_TIMESTEP)
        elif record and (step % 2 == 0):
            f = base_env.render(camera_mode="track")
            if f is not None:
                frames.append(f)
                
        # Ghi nhận chỉ số sau bước ổn định ban đầu (sau step 50)
        if step >= 50:
            pos_err = np.linalg.norm(base_env.pos[0] - target_pos)
            tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (base_env.quat[0, 1]**2 + base_env.quat[0, 2]**2), -1, 1)))
            pos_errors.append(pos_err)
            rpms_list.append(np.mean(base_env.data.actuator_force))
            tilts_list.append(tilt_deg)
            
    env.close()
    
    metrics = {
        "mean_pos_err": float(np.mean(pos_errors)),
        "max_pos_err": float(np.max(pos_errors)),
        "mean_tilt": float(np.mean(tilts_list)),
        "frames": frames,
    }
    return metrics


def run_comparison_demo(gui: bool = False, record: bool = False, steps: int = 250):
    print("=" * 80)
    print(" BẮT ĐẦU ĐÁNH GIÁ ĐỐI CHUẨN: KHÔNG GIÓ (BASELINE) vs CÓ GIÓ (WINDWRAPPER)")
    print("=" * 80)
    
    print("\n>>> [1/2] Đang chạy mô phỏng: KHÔNG CÓ GIÓ (No Wind)...")
    res_no_wind = run_benchmark_trial(with_wind=False, steps=steps, record=record, gui=gui)
    
    print(">>> [2/2] Đang chạy mô phỏng: CÓ GIÓ MẠNH (Wind Active)...")
    res_wind = run_benchmark_trial(with_wind=True, steps=steps, record=record, gui=gui)
    
    print("\n" + "=" * 80)
    print(f"{'Chỉ số đo lường (Metrics)':<35} | {'Không có gió (No Wind)':<20} | {'Có gió (Wind Active)':<20}")
    print("=" * 80)
    print(f"{'Sai số vị trí trung bình (Mean Error)':<35} | {res_no_wind['mean_pos_err']:<18.4f} m | {res_wind['mean_pos_err']:<18.4f} m")
    print(f"{'Độ dạt vị trí cực đại (Max Drift)':<35} | {res_no_wind['max_pos_err']:<18.4f} m | {res_wind['max_pos_err']:<18.4f} m")
    print(f"{'Góc nghiêng thân trung bình (Tilt)':<35} | {res_no_wind['mean_tilt']:<18.2f}° | {res_wind['mean_tilt']:<18.2f}°")
    
    drift_increase = ((res_wind['mean_pos_err'] - res_no_wind['mean_pos_err']) / max(1e-5, res_no_wind['mean_pos_err'])) * 100
    tilt_increase = res_wind['mean_tilt'] - res_no_wind['mean_tilt']
    
    print("-" * 80)
    print(f"[ĐÁNH GIÁ]: Dưới tác động của gió, độ trôi vị trí tăng thêm: +{drift_increase:.1f}%, drone nghiêng thêm: +{tilt_increase:.2f}° để bù lực gió.")
    print("=" * 80)
    
    if record and res_wind["frames"]:
        out_gif = os.path.join(os.path.dirname(__file__), "gifs", "compare_benchmark.gif")
        save_gif(res_wind["frames"], out_gif, fps=24)


if __name__ == "__main__":
    args = parse_wind_args("SO SÁNH ĐỐI CHUẨN: KHÔNG GIÓ vs CÓ GIÓ")
    run_comparison_demo(
        gui=args.gui,
        record=args.record,
        steps=args.steps
    )

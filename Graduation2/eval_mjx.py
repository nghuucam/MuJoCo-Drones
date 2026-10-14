import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config_mjx as cfg
from model_xml_builder import generate_static_mjx_xml
from env_numpy import (
    yaw_to_quat_np,
    build_obs_np,
    plan_waypoint_np,
    tracking_force_np,
    apf_pilot_action_np,
)


def load_ppo(model_path):
    try:
        from sbx import PPO
        return PPO.load(model_path)
    except Exception as e_sbx:
        try:
            from stable_baselines3 import PPO
            return PPO.load(model_path)
        except Exception as e_sb3:
            print(f"⚠️ Không nạp được model (SBX: {e_sbx} | SB3: {e_sb3})")
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy xem mô phỏng 3D Drone bay tránh cọc gai trên máy tính (Graduation2)")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file model .zip đã huấn luyện")
    parser.add_argument("--level", type=int, default=0, help="Cấp độ Curriculum thử nghiệm (0 đến 6, Mặc định: 0)")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episode chạy xem (Mặc định: 5)")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ hiển thị (1.0 = thời gian thực, 0.5 = tua chậm)")
    return parser.parse_args()


def has_penetration_cpu(d):
    return any(d.contact[k].dist < 0.0 for k in range(d.ncon))


def reset_episode(mj_model, mj_data, level):
    y_min, y_max = cfg.GOAL_Y_RANGES[level]
    goal = np.array([np.random.uniform(-2.0, 2.0), np.random.uniform(y_min, y_max), cfg.GOAL_Z])

    obs_xy = np.array(cfg.FIXED_OBSTACLE_POSITIONS)
    z_pos = np.where(np.arange(len(obs_xy)) < cfg.OBSTACLE_COUNTS[level], 0.0, -100.0)
    obs_mpos = np.stack([obs_xy[:, 0], obs_xy[:, 1], z_pos], axis=1)

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.mocap_pos[:] = np.concatenate([goal[None, :], obs_mpos], axis=0)
    mj_data.qpos[0:3] = cfg.START_POS
    mj_data.qpos[3:7] = yaw_to_quat_np(cfg.START_YAW)
    mujoco.mj_forward(mj_model, mj_data)
    return goal


def main():
    args = parse_args()

    print("=" * 80)
    print("🛸 GIAO DIỆN 3D MUJOCO QUAN SÁT DRONE BAY TỰ HÀNH (GRADUATION2)")
    print("=" * 80)
    print(f"🎯 Cấp độ Curriculum: Level {args.level}")
    print(f"📊 Số cọc gai xuất hiện: {cfg.OBSTACLE_COUNTS[args.level]}")

    model = None
    if args.model_path:
        if os.path.exists(args.model_path):
            print(f"📦 Đang nạp Model từ: {args.model_path}")
            model = load_ppo(args.model_path)
            if model is not None:
                print("✅ Đã nạp thành công mạng nơ-ron PPO!")
        else:
            print(f"⚠️ Không tìm thấy file model: {args.model_path}")
    if model is None:
        print("💡 Chạy ở chế độ [PHI CÔNG DẪN ĐƯỜNG APF] (dùng đúng gợi ý APF trong observation lúc train)")

    print("-" * 80)
    print("🚀 Đang khởi động cửa sổ đồ họa 3D MuJoCo Viewer...")

    mj_model = mujoco.MjModel.from_xml_string(generate_static_mjx_xml())
    mj_data = mujoco.MjData(mj_model)
    drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    frame_dt = 1.0 / cfg.SIM_FREQ / max(args.speed, 1e-3)

    stats = {"win": 0, "collision": 0, "over_map": 0, "timeout": 0}

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = drone_body_id
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = -90.0

        for ep in range(args.episodes):
            if not viewer.is_running():
                break

            goal = reset_episode(mj_model, mj_data, args.level)
            print(f"\n▶️ --- EPISODE {ep + 1}/{args.episodes} (Level {args.level}) ---")
            print(f"   🎯 Đích: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}] | Cự ly ban đầu: {np.linalg.norm(mj_data.qpos[0:3] - goal):.2f}m")

            outcome = "timeout"
            for step in range(cfg.MAX_STEPS):
                if not viewer.is_running():
                    outcome = None
                    break

                obs = build_obs_np(mj_data.qpos, mj_data.qvel, goal)
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = apf_pilot_action_np(obs)

                start_pos, delta, move_yaw = plan_waypoint_np(mj_data.qpos, action)
                mj_data.qpos[3:7] = yaw_to_quat_np(move_yaw)
                mj_data.qvel[3:6] = 0.0

                hit = False
                won = False
                for sub in range(cfg.SUBSTEPS):
                    mj_data.ctrl[:] = tracking_force_np(sub, start_pos, delta, mj_data.qpos[0:3].copy(), mj_data.qvel[0:3].copy(), move_yaw)
                    mujoco.mj_step(mj_model, mj_data)
                    viewer.sync()
                    time.sleep(frame_dt)

                    if has_penetration_cpu(mj_data):
                        hit = True
                        break
                    if np.linalg.norm(mj_data.qpos[0:3] - goal) < cfg.GOAL_THRESHOLD:
                        won = True
                        break

                new_pos = mj_data.qpos[0:3]
                cur_dist = np.linalg.norm(new_pos - goal)
                over_map = (abs(new_pos[0]) > cfg.X_LIMIT) or (abs(new_pos[1]) > cfg.Y_LIMIT) or (new_pos[2] < cfg.MIN_Z) or (new_pos[2] > cfg.MAX_Z)

                if step % 5 == 0 or won or hit or over_map:
                    print(f"   Bước {step+1:>3}: Pos=[{new_pos[0]:+5.2f}, {new_pos[1]:+5.2f}, {new_pos[2]:4.2f}] | Cách đích: {cur_dist:.2f}m")

                if won:
                    outcome = "win"
                    print(f"   🎉 CHIẾN THẮNG! Drone đã chạm đích thành công ở bước {step+1}!")
                    break
                if hit:
                    outcome = "collision"
                    print(f"   💥 VA CHẠM CỌC GAI ở bước {step+1}!")
                    break
                if over_map:
                    outcome = "over_map"
                    print(f"   ⚠️ VĂNG KHỎI BẢN ĐỒ ở bước {step+1}!")
                    break

            if outcome is None:
                break
            if outcome == "timeout":
                print(f"   ⌛ HẾT GIỜ sau {cfg.MAX_STEPS} bước!")
            stats[outcome] += 1
            time.sleep(1.0)

    total = sum(stats.values())
    print("\n" + "=" * 80)
    if total > 0:
        print(f"📈 Kết quả {total} episode: Thắng {stats['win']} ({100.0 * stats['win'] / total:.0f}%) | Đâm {stats['collision']} | Văng {stats['over_map']} | Hết giờ {stats['timeout']}")
    print("🏁 ĐÃ HOÀN TẤT ĐÁNH GIÁ MÔ PHỎNG 3D!")


if __name__ == "__main__":
    main()

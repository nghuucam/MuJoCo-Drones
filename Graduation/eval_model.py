import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO

MODELS_DIR = os.path.join(current_dir, "models")


def get_default_model_path():
    final_path = os.path.join(MODELS_DIR, "drone_ppo_curriculum_ver4_final.zip")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(MODELS_DIR):
        zips = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
        if zips:
            zips.sort(key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True)
            return os.path.join(MODELS_DIR, zips[0])
    return final_path


def evaluate():
    parser = argparse.ArgumentParser(description="Đánh giá / Xem thử Drone hoạt động với Model đã huấn luyện (Ver4)")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file model (.zip). Mặc định: tự tìm model mới nhất")
    parser.add_argument("--level", type=int, default=6, help="Cấp độ Curriculum cần test (0 đến 6, Mặc định: 6)")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episode test (Mặc định: 5)")
    parser.add_argument("--no-gui", action="store_true", help="Tắt giao diện 3D (Chỉ in kết quả)")
    parser.add_argument("--show-marker", action="store_true", help="Bật lại con trỏ chấm trắng khổng lồ trên đầu Drone (Mặc định: TẮT)")
    parser.add_argument("--quiet", action="store_true", help="Ẩn bảng in tọa độ từng bước (Mặc định: HIỆN bảng tọa độ chi tiết)")
    args = parser.parse_args()

    model_path = args.model_path if (args.model_path and os.path.exists(args.model_path)) else get_default_model_path()

    print("=" * 80)
    print("🛸 ĐÁNH GIÁ MÔ HÌNH DRONE HUẤN LUYỆN VER4 (EVALUATION)")
    print("=" * 80)
    print(f"📦 Đang nạp Model từ: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY FILE MODEL TẠI: {model_path}")
        print("💡 Vui lòng kiểm tra lại đường dẫn file .zip trong biến MODEL_PATH ở đầu file.")
        return

    gui = not args.no_gui
    print(f"🎮 Giao diện 3D GUI: {'BẬT' if gui else 'TẮT'}")
    print(f"🎯 Cấp độ thử nghiệm: Level {args.level}")
    print(f"🔵 Thân Drone thật: {'ẨN (Chấm trắng bật)' if args.show_marker else 'HIỆN (Vỏ màu xanh thực tế 6cm, Chấm trắng đã TẮT)'}")
    print("-" * 80)

    env = DronePPOCurriculumEnv(gui=gui, show_marker=args.show_marker)
    env.set_level(args.level)

    model = PPO.load(model_path, env=env)
    print("✅ Đã nạp thành công mô hình PPO!")

    success_count = 0
    total_rewards = []

    for ep in range(args.episodes):
        print(f"\n▶️ --- EPISODE TEST {ep + 1}/{args.episodes} (Level {args.level}) ---")
        obs, info = env.reset()

        goal_p = env.goal[0]
        cur_p = env.pos[0]
        init_dist = float(np.linalg.norm(cur_p - goal_p))
        print(f"   🎯 Tọa độ Đích      : X={goal_p[0]:+6.2f}, Y={goal_p[1]:+6.2f}, Z={goal_p[2]:5.2f}m")
        print(f"   🛫 Vị trí Xuất phát : X={cur_p[0]:+6.2f}, Y={cur_p[1]:+6.2f}, Z={cur_p[2]:5.2f}m (Cách đích: {init_dist:.2f}m)")

        if not args.quiet:
            print(f"   {'Bước':<6} | {'Tọa độ Drone [X, Y, Z]':<26} | {'Cách đích':<10} | {'Độ cao ΔZ':<10} | {'Reward':<8}")
            print(f"   {'-'*6} + {'-'*26} + {'-'*10} + {'-'*10} + {'-'*8}")

        ep_reward = 0.0
        step_count = 0

        for step in range(config.MAX_STEPS):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            cur_p = env.pos[0]
            dist_goal = float(np.linalg.norm(cur_p - goal_p))
            delta_z = float(abs(cur_p[2] - goal_p[2]))

            ep_reward += reward
            step_count += 1

            if not args.quiet:
                coord_str = f"[{cur_p[0]:+6.2f}, {cur_p[1]:+6.2f}, {cur_p[2]:5.2f}m]"
                print(f"   #{step_count:03d}   | {coord_str:<26} | {dist_goal:7.2f}m   | {delta_z:7.2f}m   | {reward:+7.2f}")

            if hasattr(env, "_fpv_window") and env._fpv_window is not None:
                try:
                    env._fpv_window.title(f"FPV | Drone: [{cur_p[0]:+.2f}, {cur_p[1]:+.2f}, {cur_p[2]:.2f}] | Đích: {dist_goal:.2f}m")
                except Exception:
                    pass

            if gui:
                time.sleep(0.08)

            if terminated or truncated:
                break

        is_win = bool(info.get("win", False))
        is_col = bool(info.get("collision", False))
        is_overmap = bool(info.get("over_map", False))

        cur_p = env.pos[0]
        goal_p = env.goal[0]
        final_dist = float(np.linalg.norm(cur_p - goal_p))

        if is_win:
            success_count += 1
            status_str = "🎉 THÀNH CÔNG (Tới Đích!)"
        elif is_col:
            status_str = "💥 VA CHẠM (Đâm Cột/Gai)"
        elif is_overmap:
            status_str = "🚩 VĂNG MAP / VƯỢT QUÁ VỊ TRÍ ĐÍCH (Over Map)"
        elif truncated:
            status_str = f"⏱️ HẾT GIỜ (Đạt tối đa {config.MAX_STEPS} bước - Truncated)"
        else:
            status_str = "🛑 KẾT THÚC (Khác)"

        total_rewards.append(ep_reward)
        print(f"   - Kết quả: {status_str}")
        print(f"   - Vị trí Drone cuối : X={cur_p[0]:.2f}, Y={cur_p[1]:.2f}, Z={cur_p[2]:.2f}")
        print(f"   - Vị trí Đích       : X={goal_p[0]:.2f}, Y={goal_p[1]:.2f}, Z={goal_p[2]:.2f}")
        print(f"   - Khoảng cách đích  : {final_dist:.2f}m (Yêu cầu thắng: < {config.GOAL_THRESHOLD}m)")
        print(f"   - Tổng số bước: {step_count} steps | Phần thưởng: {ep_reward:.2f}")

    print("\n" + "=" * 80)
    print("📊 TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ:")
    print(f"   • Tổng số Episode test : {args.episodes}")
    print(f"   • Số lần tới đích thành công: {success_count}/{args.episodes} ({success_count / args.episodes * 100:.1f}%)")
    print(f"   • Phần thưởng trung bình  : {np.mean(total_rewards):.2f}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    evaluate()

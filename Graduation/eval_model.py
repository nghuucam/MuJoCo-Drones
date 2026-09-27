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

# ==============================================================================
# 🔑 TỰ ĐỘNG TÌM FILE MODEL MỚI NHẤT TRONG THƯ MỤC MODELS
# ==============================================================================
MODELS_DIR = os.path.join(current_dir, "models")


def get_default_model_path():
    final_path = os.path.join(MODELS_DIR, "drone_ppo_curriculum_ver4_final.zip")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(MODELS_DIR):
        zips = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
        if zips:
            # Lấy file zip mới nhất theo thời gian tạo
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

    # 1. Khởi tạo môi trường (mặc định tắt chấm trắng để quan sát chính xác thân Drone thật)
    env = DronePPOCurriculumEnv(gui=gui, show_marker=args.show_marker)
    env.set_level(args.level)

    # 2. Nạp mô hình PPO đã huấn luyện
    model = PPO.load(model_path, env=env)
    print("✅ Đã nạp thành công mô hình PPO!")

    success_count = 0
    total_rewards = []

    for ep in range(args.episodes):
        print(f"\n▶️ --- EPISODE TEST {ep + 1}/{args.episodes} (Level {args.level}) ---")
        obs, info = env.reset()

        ep_reward = 0.0
        step_count = 0

        for step in range(config.MAX_STEPS):
            # Mạng nơ-ron dự đoán hành động (deterministic=True cho kết quả tối ưu nhất)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            step_count += 1

            if gui:
                time.sleep(0.08)  # Trễ nhẹ để quan sát chuyển động mượt mà trên 3D GUI

            if terminated or truncated:
                break

        is_win = info.get("win", False)
        is_col = info.get("collision", False)

        if is_win:
            success_count += 1
            status_str = "🎉 THÀNH CÔNG (Tới Đích!)"
        elif is_col:
            status_str = "💥 VA CHẠM (Đâm Cột/Gai)"
        else:
            status_str = "⏱️ HẾT GIỜ (Truncated)"

        total_rewards.append(ep_reward)
        print(f"   - Kết quả: {status_str}")
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

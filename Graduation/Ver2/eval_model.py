import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO

# ==============================================================================
# 🔑 ĐƯỜNG DẪN TỚI FILE MODEL CẦN TEST (BẠN CHỈ CẦN SỬA 1 DÒNG NÀY):
# ==============================================================================
MODEL_PATH = r"c:\KLTN\MuJoCo-Drones\Graduation\Ver2\models\drone_ppo_curriculum_final.zip"
# ==============================================================================


def evaluate():
    parser = argparse.ArgumentParser(description="Đánh giá / Xem thử Drone hoạt động với Model đã huấn luyện")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Đường dẫn file model (.zip)")
    parser.add_argument("--level", type=int, default=0, help="Cấp độ Curriculum cần test (0, 1, 2, 3)")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episode test (Mặc định: 5)")
    parser.add_argument("--no-gui", action="store_true", help="Tắt giao diện 3D (Chỉ in kết quả)")
    args = parser.parse_args()

    model_path = args.model_path if os.path.exists(args.model_path) else MODEL_PATH

    print("=" * 80)
    print("🛸 ĐÁNH GIÁ MÔ HÌNH DRONE HUẤN LUYỆN (EVALUATION)")
    print("=" * 80)
    print(f"📦 Đang nạp Model từ: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY FILE MODEL TẠI: {model_path}")
        print("💡 Vui lòng kiểm tra lại đường dẫn file .zip trong biến MODEL_PATH ở đầu file.")
        return

    gui = not args.no-gui
    print(f"🎮 Giao diện 3D GUI: {'BẬT' if gui else 'TẮT'}")
    print(f"🎯 Cấp độ thử nghiệm: Level {args.level}")
    print("-" * 80)

    # 1. Khởi tạo môi trường
    env = DronePPOCurriculumEnv(gui=gui)
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

        for step in range(80):
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

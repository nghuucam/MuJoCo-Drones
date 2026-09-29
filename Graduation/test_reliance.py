import os
import sys
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO
from eval_model import get_default_model_path


def run_mode(env, model, mode_name, num_episodes, base_seed=42):
    """Chạy đánh giá cho một chế độ thử nghiệm can thiệp tín hiệu (Ablation Mode)."""
    wins = 0
    collisions = 0
    overmaps = 0
    timeouts = 0
    rewards = []
    steps = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep * 100
        obs, info = env.reset(seed=ep_seed)

        ep_reward = 0.0
        step_count = 0

        for step in range(config.MAX_STEPS):
            obs_eval = {
                "rgb": obs["rgb"].copy(),
                "state": obs["state"].copy()
            }

            if mode_name == "ZERO_HINT":
                obs_eval["state"][16] = 0.0
                obs_eval["state"][17] = 0.0

            elif mode_name == "BLINDFOLD":
                obs_eval["rgb"] = np.zeros_like(obs["rgb"])

            elif mode_name == "ADVERSARIAL":
                obs_eval["state"][16] = -float(obs["state"][16])

            action, _ = model.predict(obs_eval, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        is_win = bool(info.get("win", False))
        is_col = bool(info.get("collision", False))
        is_overmap = bool(info.get("over_map", False))

        if is_win:
            wins += 1
        elif is_col:
            collisions += 1
        elif is_overmap:
            overmaps += 1
        else:
            timeouts += 1

        rewards.append(ep_reward)
        steps.append(step_count)

    return {
        "win_rate": (wins / num_episodes) * 100.0,
        "collision_rate": (collisions / num_episodes) * 100.0,
        "overmap_rate": (overmaps / num_episodes) * 100.0,
        "timeout_rate": (timeouts / num_episodes) * 100.0,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
    }


def main():
    parser = argparse.ArgumentParser(description="Chẩn đoán mức độ ỷ lại vào Hint của Drone (Ablation Diagnostic)")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file model .zip (Mặc định: model mới nhất)")
    parser.add_argument("--level", type=int, default=2, help="Level kiểm thử (Mặc định: Level 2 - có 6 cột gai)")
    parser.add_argument("--episodes", type=int, default=5, help="Số episode mỗi bài test (Mặc định: 5)")
    parser.add_argument("--seed", type=int, default=100, help="Random seed gốc (Giúp cố định bản đồ để so sánh công bằng)")
    args = parser.parse_args()

    model_path = args.model_path if (args.model_path and os.path.exists(args.model_path)) else get_default_model_path()

    print("=" * 85)
    print("🔬 BÀI THỬ NGHIỆM CHẨN ĐOÁN MỨC ĐỘ Ỷ LẠI VÀO HINT & THỊ GIÁC FPV (ABLATION STUDY)")
    print("=" * 85)
    print(f"📦 Mô hình đánh giá : {os.path.basename(model_path)}")
    print(f"🎯 Cấp độ thử nghiệm: Level {args.level} ({config.OBSTACLE_COUNTS.get(args.level, 0)} cột gai)")
    print(f"🔁 Số lượt mỗi mode : {args.episodes} episodes (Cùng hạt giống seed={args.seed} để đối chứng)")
    print("-" * 85)

    env = DronePPOCurriculumEnv(gui=False, show_marker=False)
    env.set_level(args.level)
    model = PPO.load(model_path, env=env)

    modes = [
        ("BASELINE", "1. Chuẩn mực (Full Camera + Full Hint)"),
        ("ZERO_HINT", "2. Cắt Hint (Chỉ dùng Camera FPV, Hint = 0)"),
        ("BLINDFOLD", "3. Bịt mắt (Bịt đen Camera, Chỉ dùng Hint)"),
        ("ADVERSARIAL", "4. Lừa Hint (Đảo ngược Hint, Ép đâm gai)")
    ]

    results = {}
    for mode_key, mode_desc in modes:
        print(f"⏳ Đang chạy: {mode_desc}...")
        results[mode_key] = run_mode(env, model, mode_key, args.episodes, base_seed=args.seed)

    env.close()

    print("\n" + "=" * 85)
    print(f"{'Chế độ thử nghiệm':<35} | {'Tỷ lệ Thắng':<12} | {'Va chạm':<10} | {'Văng map':<10} | {'Thưởng TB':<10}")
    print("-" * 85)
    labels = {
        "BASELINE": "1. Baseline (Chuẩn: Camera + Hint)",
        "ZERO_HINT": "2. Cắt Hint (Thực lực Camera)",
        "BLINDFOLD": "3. Bịt mắt Camera (Chỉ Hint)",
        "ADVERSARIAL": "4. Đánh lừa Hint (Đối kháng)"
    }
    for k in ["BASELINE", "ZERO_HINT", "BLINDFOLD", "ADVERSARIAL"]:
        res = results[k]
        print(f"{labels[k]:<35} | {res['win_rate']:>10.1f}% | {res['collision_rate']:>8.1f}% | {res['overmap_rate']:>8.1f}% | {res['mean_reward']:>10.1f}")
    print("=" * 85)

    base_win = results["BASELINE"]["win_rate"]
    no_hint_win = results["ZERO_HINT"]["win_rate"]
    no_hint_col = results["ZERO_HINT"]["collision_rate"]
    blind_win = results["BLINDFOLD"]["win_rate"]
    adv_col = results["ADVERSARIAL"]["collision_rate"]

    print("\n🧠 KẾT QUẢ PHÂN TÍCH CHẨN ĐOÁN:")
    
    if no_hint_win < 0.3 * base_win or no_hint_col >= 60.0:
        reliance_status = "🔴 RẤT CAO (BỊ Ỷ LẠI NẶNG NỀ)"
        reliance_detail = "Khi bị cắt Hint, tỷ lệ va chạm tăng vọt hoặc tỷ lệ thắng sụt giảm nghiêm trọng. Drone chưa thể tự bay an toàn nếu thiếu 'hoa tiêu' APF."
    elif no_hint_win >= 0.7 * base_win:
        reliance_status = "🟢 RẤT THẤP (ĐỘC LẬP TỰ CHỦ CAO)"
        reliance_detail = "Khi cắt Hint, Drone vẫn duy trì tỷ lệ thắng tốt và tự luồn lách né gai thành công bằng Camera FPV."
    else:
        reliance_status = "🟡 TRUNG BÌNH (CÓ PHỤ THUỘC MỘT PHẦN)"
        reliance_detail = "Drone có khả năng nhận biết vật cản qua Camera FPV nhưng vẫn dựa vào Hint để duy trì quỹ đạo tối ưu."

    print(f"   • Mức độ ỷ lại vào Hint : {reliance_status}")
    print(f"     -> {reliance_detail}")

    if blind_win >= 0.8 * base_win:
        vision_status = "🔴 MÙ THỊ GIÁC (CAMERA BỊ BỎ RƠI)"
        vision_detail = "Dù bịt đen hoàn toàn Camera, Drone vẫn bay về đích tốt như bình thường. Mạng nơ-ron hầu như KHÔNG trích xuất thông tin từ ảnh FPV."
    elif blind_win <= 0.3 * base_win:
        vision_status = "🟢 THỊ GIÁC HOẠT ĐỘNG TỐT (CAMERA ĐÓNG VAI TRÒ CHỦ CHỐT)"
        vision_detail = "Khi bị bịt mắt, Drone bị mất phương hướng hoặc va chạm ngay, chứng tỏ ảnh Camera FPV là đầu vào sống còn."
    else:
        vision_status = "🟡 THỊ GIÁC HOẠT ĐỘNG MỘT PHẦN"
        vision_detail = "Drone có sử dụng tín hiệu từ ảnh FPV nhưng chưa khai thác hết tiềm năng."

    print(f"   • Mức độ dùng Camera FPV: {vision_status}")
    print(f"     -> {vision_detail}")

    print(f"   • Tỷ lệ bị lừa đâm gai  : {adv_col:.1f}%")
    if adv_col >= 60.0:
        print("     -> Drone mù quáng tin theo Hint lừa đảo dù mắt camera nhìn thấy gai trước mặt.")
    else:
        print("     -> Drone có khả năng phản kháng lại Hint sai lệch nhờ quan sát thị giác.")
    print("=" * 85)


if __name__ == "__main__":
    main()

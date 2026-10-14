import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config_mjx as cfg
from mjx_gym_wrapper import MJXGymEnv
from curriculum_callback import MJXCurriculumCallback

try:
    from sbx import PPO
    HAS_SBX = True
except ImportError:
    HAS_SBX = False


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện PPO JAX (SBX) cho Drone trên GPU bằng MuJoCo MJX (Graduation2)")
    parser.add_argument("--exp-name", type=str, default="sbx_ppo_run", help="Tên thí nghiệm riêng biệt")
    parser.add_argument("--num-envs", type=int, default=128, help="Số lượng drone song song trên GPU (Mặc định: 128)")
    parser.add_argument("--start-level", type=int, default=0, help="Cấp độ Curriculum khởi đầu (0 đến 6, Mặc định: 0)")
    parser.add_argument("--no-curriculum", action="store_true", help="Vô hiệu hóa Curriculum, cố định tại start-level")
    parser.add_argument("--window-size", type=int, default=200, help="Số episode đánh giá win rate Curriculum (Mặc định: 200)")
    parser.add_argument("--threshold-advance", type=float, default=0.8, help="Tỉ lệ thắng tối thiểu để thăng cấp (Mặc định: 0.8)")
    parser.add_argument("--threshold-retreat", type=float, default=0.2, help="Tỉ lệ thắng tối thiểu để không bị hạ cấp (Mặc định: 0.2)")
    parser.add_argument("--timesteps", type=int, default=3000000, help="Tổng số bước huấn luyện PPO (Mặc định: 3,000,000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (Mặc định: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=128, help="Số bước lấy mẫu mỗi worker trước khi update (Mặc định: 128)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size PPO (Mặc định: 256)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Số epochs cập nhật mạng (Mặc định: 10)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (Mặc định: 0.99)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range (Mặc định: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient (Mặc định: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🚀 HUẤN LUYỆN SBX (STABLE-BASELINES-JAX) PPO TRÊN GPU VỚI MUJOCO MJX (GRADUATION2)")
    print("=" * 80)
    print(f"🏷️ Thí nghiệm : {args.exp_name}")
    print(f"🎮 Số drone song song (Num Envs): {args.num_envs:,}")
    print(f"🎯 Cấp độ khởi đầu (Start Level): {args.start_level}")
    print(f"📚 Chế độ Curriculum            : {'TẮT (Cố định)' if args.no_curriculum else 'BẬT (Tự động thăng/hạ cấp)'}")
    if not args.no_curriculum:
        print(f"   - Cửa sổ đánh giá             : {args.window_size} episodes")
        print(f"   - Ngưỡng thăng cấp (Win Rate) : >= {args.threshold_advance*100:.0f}%")
        print(f"   - Ngưỡng hạ cấp (Win Rate)   : <= {args.threshold_retreat*100:.0f}%")
    print(f"📊 Tổng số bước (Timesteps)     : {args.timesteps:,}")

    if not HAS_SBX:
        print("\n❌ LỖI: Thư viện SBX chưa được cài đặt!")
        print("💡 Hãy chạy trên Colab:")
        print("   !pip install sbx-rl flax \"jax[cuda12]\"")
        return

    models_dir = os.path.join(current_dir, "models", args.exp_name)
    logs_dir = os.path.join(current_dir, "logs", args.exp_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"📁 Thư mục Models : {models_dir}")
    print(f"📁 Thư mục Logs   : {logs_dir}")
    print("-" * 80)

    print(f"⏳ Đang khởi tạo môi trường MJXGymEnv với {args.num_envs} drone...")
    env = MJXGymEnv(num_envs=args.num_envs, level=args.start_level)
    print("✅ Đã khởi tạo môi trường GPU thành công!")

    curriculum_cb = None
    if not args.no_curriculum:
        curriculum_cb = MJXCurriculumCallback(
            window_size=args.window_size,
            threshold_advance=args.threshold_advance,
            threshold_retreat=args.threshold_retreat,
            num_levels=7,
            start_level=args.start_level,
            verbose=1
        )

    print(f"📦 Đang biên dịch mạng PPO JAX / Flax...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        tensorboard_log=logs_dir
    )

    print("\n🏁 BẮT ĐẦU HUẤN LUYỆN PPO TRÊN GPU...")
    start_t = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=curriculum_cb)
        final_path = os.path.join(models_dir, f"drone_sbx_ppo_{args.exp_name}_final")
        model.save(final_path)
        elapsed = time.time() - start_t
        print("\n" + "=" * 80)
        print(f"🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG sau {elapsed / 60:.2f} phút!")
        print(f"💾 Model đã được lưu tại: {final_path}.zip")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n🛑 Phát hiện Ctrl+C! Đang lưu model...")
        interrupted_path = os.path.join(models_dir, f"drone_sbx_ppo_{args.exp_name}_interrupted")
        model.save(interrupted_path)
        print(f"💾 Đã lưu an toàn tại: {interrupted_path}.zip")


if __name__ == "__main__":
    main()

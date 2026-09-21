import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_env import DronePPOEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank: int, seed: int = 0):
    """Hàm tạo môi trường con cho mỗi tiến trình worker độc lập."""
    def _init():
        env = DronePPOEnv(gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện song song PPO cho Drone với SubprocVecEnv (SB3)")
    parser.add_argument("--num-cpu", type=int, default=4,
                        help="Số lượng tiến trình / môi trường chạy song song (Mặc định: 4, tối đa số core CPU)")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Tổng số bước huấn luyện PPO (Mặc định: 200,000)")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Chu kỳ lưu checkpoint (Mặc định: mỗi 10,000 bước)")
    parser.add_argument("--n-steps", type=int, default=128,
                        help="Số bước rollout trên mỗi môi trường trước mỗi lần cập nhật PPO (Mặc định: 128)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size khi huấn luyện mạng nơ-ron (Mặc định: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Tốc độ học Learning Rate (Mặc định: 3e-4)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Thư mục lưu checkpoint và nhật ký TensorBoard
    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 85)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SONG SONG: DRONE PPO (STABLE-BASELINES3 + SUBPROCVECENV)")
    print("=" * 85)
    print(f"💻 Số lượng tiến trình chạy song song (Workers): {args.num_cpu}")
    print(f"🎯 Tổng số bước huấn luyện (Total Timesteps): {args.timesteps:,}")
    print(f"📦 Rollout Steps / Env: {args.n_steps} (Tổng steps mỗi lần update = {args.n_steps * args.num_cpu})")
    print(f"📊 Thư mục Logs TensorBoard: {logs_dir}")
    print(f"💾 Thư mục lưu Models: {models_dir}")
    print("-" * 85)

    # 1. Khởi tạo môi trường song song đa tiến trình (SubprocVecEnv)
    print(f"⏳ Đang khởi tạo {args.num_cpu} tiến trình môi trường MuJoCo độc lập...")
    env_fns = [make_env(rank=i, seed=42) for i in range(args.num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    # Bọc VecMonitor để theo dõi Episode Return và Success Rate
    vec_env = VecMonitor(vec_env, filename=os.path.join(logs_dir, "monitor.csv"))
    print("✅ Đã khởi tạo thành công tất cả các tiến trình!")

    # 2. Kiểm tra TensorBoard
    try:
        import tensorboard
        tb_log = logs_dir
    except ImportError:
        tb_log = None
        print("⚠️ Chưa cài tensorboard. Chạy không ghi log TensorBoard.")

    # 3. Kiểm tra thanh tiến trình
    try:
        import tqdm
        import rich
        has_progress_bar = True
    except ImportError:
        has_progress_bar = False

    # 4. Cấu hình Callback tự động lưu checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.num_cpu),
        save_path=models_dir,
        name_prefix=f"drone_ppo_parallel_{args.num_cpu}cpu"
    )

    # 5. Khởi tạo mô hình PPO với CnnPolicy
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log
    )

    print("\n🧠 Cấu trúc mạng PPO:")
    print(model.policy)
    print("\n" + "=" * 85)
    print("🏁 BẮT ĐẦU HUẤN LUYỆN (Nhấn Ctrl+C để dừng an toàn bất kỳ lúc nào)...")
    print("=" * 85)

    start_time = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=has_progress_bar
        )
        final_save_path = os.path.join(models_dir, "drone_ppo_parallel_final.zip")
        model.save(final_save_path)
        elapsed = time.time() - start_time
        print("\n" + "=" * 85)
        print(f"🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG sau {elapsed/60:.2f} phút!")
        print(f"💾 Model đã được lưu tại: {final_save_path}")
        print("=" * 85)
    except KeyboardInterrupt:
        print("\n🛑 Phát hiện thao tác dừng (Ctrl+C)! Đang lưu checkpoint khẩn cấp...")
        interrupted_path = os.path.join(models_dir, "drone_ppo_parallel_interrupted.zip")
        model.save(interrupted_path)
        print(f"💾 Đã lưu model an toàn tại: {interrupted_path}")
    finally:
        print("🧹 Đang đóng các tiến trình con...")
        vec_env.close()
        print("✅ Đã giải phóng toàn bộ tài nguyên!")


if __name__ == "__main__":
    mp.freeze_support()
    main()

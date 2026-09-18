import os
import sys
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_env import DronePPOEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def train_drone_ppo():
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN DRONE VỚI THUẬT TOÁN PPO (STABLE-BASELINES3)")
    print("=" * 80)

    # Thư mục lưu model và tensorboard logs
    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Khởi tạo môi trường
    env = DronePPOEnv(gui=True)

    # Callback tự động lưu checkpoint sau mỗi 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=models_dir,
        name_prefix="drone_ppo_model"
    )

    # Kiểm tra TensorBoard
    try:
        import tensorboard
        tb_log_path = logs_dir
        print(f"📊 TensorBoard đã sẵn sàng! Ghi log tại: {tb_log_path}")
    except ImportError:
        tb_log_path = None
        print("⚠️ Chưa cài tensorboard. Chạy không ghi log TensorBoard (hoặc chạy 'pip install tensorboard').")

    # Cấu hình mạng PPO với CnnPolicy cho ảnh RGB FPV
    # Lưu ý: Mỗi bước PPO tương ứng với 1 bước di chuyển cấp cao của Drone
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log_path
    )

    print("\n🧠 Kiến trúc Policy:")
    print(model.policy)
    print("\n" + "-" * 80)
    print("⏳ Bắt đầu vòng lặp huấn luyện...")

    # Kiểm tra thanh tiến trình
    try:
        import tqdm
        import rich
        has_progress_bar = True
    except ImportError:
        has_progress_bar = False

    TOTAL_TIMESTEPS = 100000
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=has_progress_bar
        )
        final_model_path = os.path.join(models_dir, "drone_ppo_final.zip")
        model.save(final_model_path)
        print(f"\n🎉 Huấn luyện hoàn tất! Model cuối cùng đã lưu tại: {final_model_path}")
    except KeyboardInterrupt:
        print("\n🛑 Người dùng dừng huấn luyện! Đang lưu checkpoint hiện tại...")
        interrupt_path = os.path.join(models_dir, "drone_ppo_interrupted.zip")
        model.save(interrupt_path)
        print(f"💾 Đã lưu model tại: {interrupt_path}")
    finally:
        env.close()

if __name__ == "__main__":
    train_drone_ppo()

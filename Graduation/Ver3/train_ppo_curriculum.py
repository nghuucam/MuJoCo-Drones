import os
import sys
import time
import argparse
import numpy as np
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import config
from flight_logger_callback import FlightLoggerCallback


class SingleEnvCurriculumCallback(BaseCallback):
    """Callback theo dõi Success Rate và chuyển cấp Curriculum cho 1 môi trường đơn."""

    def __init__(
        self,
        window_size: int = 20,
        threshold_advance: float = 0.8,
        threshold_retreat: float = 0.2,
        num_levels: int = 4,
        start_level: int = 0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.threshold_advance = threshold_advance
        self.threshold_retreat = threshold_retreat
        self.num_levels = num_levels
        self.current_level = start_level

        self.episode_outcomes = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # Ghi chỉ số Level hiện tại vào TensorBoard
        self.logger.record("curriculum/level", float(self.current_level))

        for info in self.locals.get("infos", []):
            if "is_success" in info and ("terminal_observation" in info or info.get("collision") or info.get("win") or info.get("step_count", 0) >= config.MAX_STEPS):
                is_succ = float(info.get("is_success", False))
                self.episode_outcomes.append(is_succ)
                self._maybe_adjust_level()

        return True

    def _maybe_adjust_level(self):
        if len(self.episode_outcomes) < self.window_size:
            return

        success_rate = np.mean(list(self.episode_outcomes))

        if success_rate >= self.threshold_advance:
            if self.current_level < self.num_levels - 1:
                self.current_level += 1
                self.episode_outcomes.clear()
                self.training_env.envs[0].env.set_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n🎉 [CURRICULUM UP] Success Rate đạt {success_rate * 100:.1f}% >= 80%!")
                    print(f"🚀 TỰ ĐỘNG CHUYỂN SANG CURRICULUM LEVEL {self.current_level}!\n")

        elif success_rate <= self.threshold_retreat:
            if self.current_level > 0:
                self.current_level -= 1
                self.episode_outcomes.clear()
                self.training_env.envs[0].env.set_level(self.current_level)
                if self.verbose > 0:
                    print(f"\n⚠️ [CURRICULUM DOWN] Success Rate tụt xuống {success_rate * 100:.1f}% <= 20%!")
                    print(f"📉 HẠ BỚT ĐỘ KHÓ VỀ LEVEL {self.current_level} để drone học lại nền tảng!\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện PPO đơn tiến trình cho Drone với Curriculum Learning & Cột gai (Ver2)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Tổng số bước huấn luyện PPO (Mặc định: 500,000)")
    parser.add_argument("--save-freq", type=int, default=20000, help="Chu kỳ lưu checkpoint (Mặc định: 20,000 bước)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size PPO (Mặc định: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (Mặc định: 3e-4)")
    parser.add_argument("--gui", action="store_true", help="Bật cửa sổ đồ họa 3D MuJoCo GUI trực tiếp khi train")
    return parser.parse_args()


def main():
    args = parse_args()

    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 85)
    print("🌵 HUẤN LUYỆN ĐƠN TIẾN TRÌNH PPO CURRICULUM + ẢNH FPV & TỌA ĐỘ ĐÍCH (GRADUATION / VER3)")
    print("=" * 85)
    print(f"🎯 Tổng số bước huấn luyện: {args.timesteps:,}")
    print(f"🖥️ Chế độ GUI 3D: {'BẬT (Hiển thị)' if args.gui else 'TẮT (Chạy ngầm headless)'}")
    print(f"📈 Tham số Curriculum: Window=20 | Advance=0.8 (80%) | Retreat=0.2 (20%) | Levels=0..3")
    print(f"🛡️ Margin va chạm gai (Collision Margin): {config.COLLISION_MARGIN}m")
    print("-" * 85)

    print("⏳ Đang khởi tạo môi trường MuJoCo Ver3...")
    raw_env = DronePPOCurriculumEnv(gui=args.gui)
    env = Monitor(raw_env, filename=os.path.join(logs_dir, "single_monitor_ver3.csv"))

    curriculum_cb = SingleEnvCurriculumCallback(
        window_size=20,
        threshold_advance=0.8,
        threshold_retreat=0.2,
        num_levels=4,
        start_level=0,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_dir,
        name_prefix="drone_ppo_ver3_multiinput"
    )

    flight_log_path = os.path.join(logs_dir, "drone_flight_log_ver3.csv")
    flight_logger_cb = FlightLoggerCallback(
        log_path=flight_log_path,
        verbose=1
    )
    print(f"📊 Nhật ký bay chi tiết (Excel CSV): {flight_log_path}")

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=128,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.03,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=logs_dir
    )

    print("\n🏁 BẮT ĐẦU HUẤN LUYỆN ĐƠN TIẾN TRÌNH VER3 (Nhấn Ctrl+C để dừng an toàn bất kỳ lúc nào)...")
    print("=" * 85)

    start_t = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[curriculum_cb, checkpoint_cb, flight_logger_cb]
        )
        final_path = os.path.join(models_dir, "drone_ppo_ver3_multiinput_final.zip")
        model.save(final_path)
        elapsed = time.time() - start_t
        print("\n" + "=" * 85)
        print(f"🎉 HUẤN LUYỆN VER3 HOÀN TẤT THÀNH CÔNG sau {elapsed / 60:.2f} phút!")
        print(f"💾 Model đã được lưu tại: {final_path}")
        print("=" * 85)
    except KeyboardInterrupt:
        print("\n🛑 Phát hiện Ctrl+C! Đang lưu checkpoint khẩn cấp...")
        interrupted_path = os.path.join(models_dir, "drone_ppo_ver3_multiinput_interrupted.zip")
        model.save(interrupted_path)
        print(f"💾 Đã lưu model an toàn tại: {interrupted_path}")
    finally:
        env.close()
        print("🧹 Đã đóng môi trường an toàn!")


if __name__ == "__main__":
    main()

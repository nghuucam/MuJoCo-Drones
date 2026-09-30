import os
import sys
import time
import argparse
import numpy as np
from collections import deque
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

for path in [current_dir, parent_dir, workspace_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import config
from flight_logger_callback import FlightLoggerCallback


class GlobalCurriculumCallback(BaseCallback):
    """Callback theo dõi Success Rate toàn cục để chuyển cấp Curriculum cho tất cả các worker CPU."""

    def __init__(
        self,
        window_size: int = 100,
        threshold_advance: float = 0.8,
        threshold_retreat: float = 0.2,
        num_levels: int = len(config.GOAL_Y_RANGES),
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
        self.advance_streak = 0

    def _on_step(self) -> bool:
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
                self._update_env_levels()
                if self.verbose > 0:
                    print(f"\n🎉 [CURRICULUM UP] Success Rate đạt {success_rate * 100:.1f}% >= 80%!")
                    print(f"🚀 TỰ ĐỘNG CHUYỂN SANG CURRICULUM LEVEL {self.current_level} cho tất cả các CPU!\n")

        elif success_rate <= self.threshold_retreat:
            if self.current_level > 0:
                self.current_level -= 1
                self.episode_outcomes.clear()
                self._update_env_levels()
                if self.verbose > 0:
                    print(f"\n⚠️ [CURRICULUM DOWN] Success Rate tụt xuống {success_rate * 100:.1f}% <= 20%!")
                    print(f"📉 HẠ BỚT ĐỘ KHÓ VỀ LEVEL {self.current_level} để drone học lại nền tảng!\n")

    def _update_env_levels(self):
        self.training_env.env_method("set_level", self.current_level)


def make_env(rank: int, seed: int = 0):
    def _init():
        env = DronePPOCurriculumEnv(gui=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện song song PPO cho Drone với Curriculum Learning & Cột trụ Gai (Ver4)")
    # 1. Quản lý tài nguyên & Lưu trữ
    parser.add_argument("--num-cpu", type=int, default=4, help="Số lượng CPU workers chạy song song (Mặc định: 4)")
    parser.add_argument("--timesteps", type=int, default=2000000, help="Tổng số bước huấn luyện PPO (Mặc định: 2,000,000)")
    parser.add_argument("--save-freq", type=int, default=10000, help="Chu kỳ lưu checkpoint tổng thể (Mặc định: 10,000 bước)")
    parser.add_argument("--resume-model", type=str, default=None, help="Đường dẫn file model .zip để tiếp tục huấn luyện (Resume)")

    # 2. Siêu tham số Mạng nơ-ron PPO
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học Learning rate (Mặc định: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=512, help="Số bước lấy mẫu mỗi worker trước khi update (Mặc định: 512)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size PPO (Mặc định: 256)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Số epochs cập nhật mạng mỗi chu kỳ (Mặc định: 10)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Hệ số chiết khấu phần thưởng tương lai Discount Factor (Mặc định: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Hệ số GAE Lambda (Mặc định: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Biên cắt tỷ lệ chính sách PPO Clipping (Mặc định: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Hệ số Entropy khuyến khích khám phá (Mặc định: 0.001)")
    parser.add_argument("--target-kl", type=float, default=0.05, help="Ngưỡng Target KL divergence ngắt sớm (Mặc định: 0.05)")

    # 3. Tham số Học chương trình Curriculum Learning
    parser.add_argument("--start-level", type=int, default=0, help="Level khởi đầu từ 0 đến 6 (Mặc định: 0)")
    parser.add_argument("--window-size", type=int, default=100, help="Cửa sổ đánh giá số episode gần nhất (Mặc định: 100)")
    parser.add_argument("--threshold-advance", type=float, default=0.8, help="Tỷ lệ thắng tối thiểu để thăng cấp Level (Mặc định: 0.8)")
    parser.add_argument("--threshold-retreat", type=float, default=0.2, help="Tỷ lệ thắng tối thiểu để không bị hạ Level (Mặc định: 0.2)")

    return parser.parse_args()


def main():
    args = parse_args()

    buffer_size = args.n_steps * args.num_cpu
    assert buffer_size % args.batch_size == 0, (
        f"Lỗi cấu hình: Buffer size ({buffer_size} = {args.n_steps} n_steps * {args.num_cpu} CPU) "
        f"phải chia hết cho batch-size ({args.batch_size})!"
    )

    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 85)
    print("🌵 HUẤN LUYỆN SONG SONG PPO CURRICULUM + ẢNH FPV & TỌA ĐỘ ĐÍCH (GRADUATION / VER4)")
    print("=" * 85)
    print(f"💻 Số lượng CPU Workers: {args.num_cpu} | Buffer size: {buffer_size} mẫu")
    print(f"📦 Batch Size: {args.batch_size} | N_Steps: {args.n_steps} | N_Epochs: {args.n_epochs}")
    print(f"⚙️ LR: {args.lr} | Gamma: {args.gamma} | GAE: {args.gae_lambda} | Clip: {args.clip_range} | Ent: {args.ent_coef}")
    print(f"🎯 Tổng số bước huấn luyện: {args.timesteps:,}")
    print(f"📈 Curriculum: Level khởi đầu={args.start_level} | Window={args.window_size} | Advance={args.threshold_advance} | Retreat={args.threshold_retreat}")
    print(f"🛡️ Margin va chạm gai (Collision Margin): {config.COLLISION_MARGIN}m")
    print(f"🎯 Bán kính đích (Goal Threshold): {config.GOAL_THRESHOLD}m")
    if args.resume_model:
        print(f"🔄 Tiếp tục huấn luyện từ (Resume): {args.resume_model}")
    print("-" * 85)

    print(f"⏳ Đang khởi tạo {args.num_cpu} tiến trình môi trường MuJoCo Ver4...")
    env_fns = [make_env(rank=i, seed=42) for i in range(args.num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(logs_dir, "monitor_ver4.csv"))
    print("✅ Đã khởi tạo thành công tất cả các CPU workers!")

    curriculum_cb = GlobalCurriculumCallback(
        window_size=args.window_size,
        threshold_advance=args.threshold_advance,
        threshold_retreat=args.threshold_retreat,
        num_levels=len(config.GOAL_Y_RANGES),
        start_level=args.start_level,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.num_cpu),
        save_path=models_dir,
        name_prefix=f"drone_ppo_curriculum_{args.num_cpu}cpu_ver4"
    )

    flight_log_path = os.path.join(logs_dir, "drone_flight_log_parallel_ver4.csv")
    flight_logger_cb = FlightLoggerCallback(
        log_path=flight_log_path,
        verbose=1
    )
    print(f"📊 Nhật ký bay chi tiết (Excel CSV): {flight_log_path}")

    if args.resume_model and os.path.exists(args.resume_model):
        print(f"📦 Đang nạp trọng số mô hình cũ từ: {args.resume_model}")
        model = PPO.load(
            args.resume_model,
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            tensorboard_log=logs_dir
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=args.lr,         
            n_steps=args.n_steps,                
            batch_size=args.batch_size,             
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            target_kl=args.target_kl,             
            ent_coef=args.ent_coef,             
            verbose=1,
            tensorboard_log=logs_dir
        )

    print("\n🏁 BẮT ĐẦU HUẤN LUYỆN CURRICULUM VER4 (Nhấn Ctrl+C để dừng an toàn bất kỳ lúc nào)...")
    print("=" * 85)

    start_t = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[curriculum_cb, checkpoint_cb, flight_logger_cb]
        )
        final_path = os.path.join(models_dir, "drone_ppo_curriculum_ver4_final.zip")
        model.save(final_path)
        elapsed = time.time() - start_t
        print("\n" + "=" * 85)
        print(f"🎉 HUẤN LUYỆN VER4 HOÀN TẤT THÀNH CÔNG sau {elapsed / 60:.2f} phút!")
        print(f"💾 Model đã được lưu tại: {final_path}")
        print("=" * 85)
    except KeyboardInterrupt:
        print("\n🛑 Phát hiện Ctrl+C! Đang lưu checkpoint khẩn cấp...")
        interrupted_path = os.path.join(models_dir, "drone_ppo_curriculum_ver4_interrupted.zip")
        model.save(interrupted_path)
        print(f"💾 Đã lưu model an toàn tại: {interrupted_path}")
    finally:
        vec_env.close()
        print("🧹 Đã đóng tất cả các tiến trình con!")


if __name__ == "__main__":
    mp.freeze_support()
    main()

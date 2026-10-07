import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

for path in [current_dir, parent_dir, workspace_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config
from drone_ppo_curriculum_env import DronePPOCurriculumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def make_env(rank: int, seed: int = 0, initial_level: int = 0):
    def _init():
        env = DronePPOCurriculumEnv(gui=False)
        env.set_level(initial_level)
        env.reset(seed=seed + rank)
        return env
    return _init


class OptunaTrialCallback(BaseCallback):
    """Callback theo dõi tiến trình và báo điểm thưởng định kỳ về cho Optuna để hỗ trợ Pruning."""
    def __init__(self, trial: optuna.Trial, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        self.recent_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total_reward" in info:
                self.recent_rewards.append(info["total_reward"])

        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            mean_r = np.mean(self.recent_rewards[-50:]) if self.recent_rewards else -50.0

            # Báo cáo điểm về cho Optuna tại bước này
            self.trial.report(mean_r, self.num_timesteps)

            # Kiểm tra xem Optuna có quyết định cắt tỉa trial này sớm không
            if self.trial.should_prune():
                return False  # Dừng trial sớm

        return True


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Định nghĩa dải giá trị tìm kiếm cho các siêu tham số PPO."""
    lr = trial.suggest_float("learning_rate", 3e-5, 2e-4, log=True)
    n_epochs = trial.suggest_int("n_epochs", 3, 8)
    clip_range = trial.suggest_categorical("clip_range", [0.15, 0.2, 0.25])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.92, 0.95, 0.98])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99])

    return {
        "learning_rate": lr,
        "n_epochs": n_epochs,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
    }


def objective(trial: optuna.Trial, args) -> float:
    params = sample_ppo_params(trial)

    buffer_size = args.n_steps * args.num_cpu
    assert buffer_size % args.batch_size == 0

    env_fns = [make_env(rank=i, seed=args.seed + trial.number * 10, initial_level=args.start_level) for i in range(args.num_cpu)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env.env_method("set_level", args.start_level)

    if args.resume_model and os.path.exists(args.resume_model):
        model = PPO.load(
            args.resume_model,
            env=vec_env,
            learning_rate=params["learning_rate"],
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            target_kl=args.target_kl,
            device=args.device
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=params["learning_rate"],
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            target_kl=args.target_kl,
            device=args.device,
            verbose=0
        )

    trial_cb = OptunaTrialCallback(trial, eval_freq=10000)

    is_pruned = False
    try:
        model.learn(total_timesteps=args.trial_timesteps, callback=trial_cb)
    except Exception as e:
        vec_env.close()
        raise e
    finally:
        vec_env.close()

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    final_reward = np.mean(trial_cb.recent_rewards[-50:]) if trial_cb.recent_rewards else -100.0
    return float(final_reward)


def main():
    parser = argparse.ArgumentParser(description="Tối ưu hóa siêu tham số PPO bằng Optuna (Graduation / Ver4)")
    parser.add_argument("--resume-model", type=str, default=None, help="Đường dẫn file model .zip để tinh chỉnh tiếp (Resume Tuning)")
    parser.add_argument("--n-trials", type=int, default=15, help="Số lượng thử nghiệm (Mặc định: 15)")
    parser.add_argument("--trial-timesteps", type=int, default=100000, help="Số bước cho mỗi trial (Mặc định: 100,000)")
    parser.add_argument("--num-cpu", type=int, default=4, help="Số lượng worker CPU song song mỗi trial (Mặc định: 4)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size cố định (Mặc định: 256)")
    parser.add_argument("--n-steps", type=int, default=512, help="N_steps cố định (Mặc định: 512)")
    parser.add_argument("--start-level", type=int, default=0, help="Level khởi đầu thử nghiệm (Mặc định: 0)")
    parser.add_argument("--target-kl", type=float, default=0.04, help="Target KL ngắt sớm (Mặc định: 0.04)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị tính toán")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--study-name", type=str, default="drone_ppo_optuna_study", help="Tên phiên nghiên cứu Optuna")
    args = parser.parse_args()

    optuna_dir = os.path.join(current_dir, "optuna_results")
    os.makedirs(optuna_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(optuna_dir, 'optuna_study.db')}"

    print("=" * 85)
    print("🔬 BẮT ĐẦU TỐI ƯU HÓA SIÊU THAM SỐ PPO BẰNG OPTUNA")
    print("=" * 85)
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Dùng CPU"
    print(f"🚀 Thiết bị: {args.device} [{gpu_info}] | Số Trials: {args.n_trials} | Steps/Trial: {args.trial_timesteps:,}")
    print(f"💻 Workers CPU: {args.num_cpu} | Buffer/Batch: {args.n_steps * args.num_cpu} / {args.batch_size}")
    print(f"📁 Lưu trữ Database SQLite: {storage_path}")
    print("-" * 85)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=30000, interval_steps=10000)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n🛑 Nhận tín hiệu dừng! Đang lưu kết quả hiện tại...")

    print("\n" + "=" * 85)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA HOÀN TẤT")
    print("=" * 85)
    print(f"Điểm số tốt nhất (Best Mean Reward): {study.best_value:.2f}")
    print("Bộ siêu tham số tối ưu nhất (Best Hyperparameters):")
    for key, value in study.best_params.items():
        print(f"   ⭐ {key:<16}: {value}")

    # Xuất ra file CSV tổng hợp
    df_results = study.trials_dataframe()
    csv_file = os.path.join(optuna_dir, "optuna_trials_summary.csv")
    df_results.to_csv(csv_file, index=False)
    print(f"\n📊 Đã lưu bảng tổng hợp tất cả các trial tại: {csv_file}")

    # Xuất biểu đồ HTML tương tác (Plotly)
    try:
        import plotly
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )
        fig_hist = plot_optimization_history(study)
        fig_hist.write_html(os.path.join(optuna_dir, "optimization_history.html"))

        fig_imp = plot_param_importances(study)
        fig_imp.write_html(os.path.join(optuna_dir, "param_importances.html"))

        fig_par = plot_parallel_coordinate(study)
        fig_par.write_html(os.path.join(optuna_dir, "parallel_coordinate.html"))

        print(f"📈 Đã xuất 3 biểu đồ HTML tương tác vào thư mục: {optuna_dir}")
    except Exception as e:
        print(f"⚠️ Không thể xuất biểu đồ HTML: {e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config_mjx as cfg
from drone_mjx_env import DroneMJXEnv, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import random


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện song song PPO cho Drone trên 100% GPU bằng MuJoCo MJX (Graduation2)")
    parser.add_argument("--exp-name", type=str, default="mjx_run", help="Tên thí nghiệm")
    parser.add_argument("--num-envs", type=int, default=1024, help="Số lượng môi trường song song trên GPU (Mặc định: 1024)")
    parser.add_argument("--level", type=int, default=6, help="Cấp độ cọc gai (0 đến 6, Mặc định: 6)")
    parser.add_argument("--timesteps", type=int, default=5000000, help="Tổng số bước huấn luyện (Mặc định: 5,000,000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tốc độ học (Mặc định: 3e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🚀 HUẤN LUYỆN DRONE SONG SONG HÀNG NGHÌN CON TRÊN GPU BẰNG MUJOCO MJX (GRADUATION2)")
    print("=" * 80)
    print(f"🏷️ Thí nghiệm : {args.exp_name}")
    print(f"🎮 Số drone song song (Num Envs): {args.num_envs:,}")
    print(f"🎯 Cấp độ bản đồ (Level)        : {args.level}")
    print(f"📊 Tổng số bước (Timesteps)     : {args.timesteps:,}")

    if not HAS_JAX:
        print("\n❌ LỖI: JAX hoặc MuJoCo MJX chưa được cài đặt trong môi trường này!")
        print("💡 Hãy chạy: pip install jax jaxlib mujoco")
        return

    devices = jax.devices()
    print(f"💻 Thiết bị JAX phát hiện: {devices}")
    print("-" * 80)

    print(f"⏳ Đang JIT-compile mô hình MuJoCo MJX cho {args.num_envs} drone...")
    start_compile = time.time()
    env = DroneMJXEnv(num_envs=args.num_envs, level=args.level)
    rng = random.PRNGKey(args.seed)
    rng, r_init = random.split(rng)
    state, obs = env.reset(r_init, level=args.level)
    print(f"✅ Biên dịch XLA hoàn tất sau {time.time() - start_compile:.2f}s!")

    print("\n🏁 BẮT ĐẦU CHẠY THỬ NGHIỆM SIMULATION TỐC ĐỘ CAO TRÊN GPU...")
    steps_done = 0
    start_train = time.time()

    while steps_done < args.timesteps:
        rng, r_act = random.split(rng)
        actions = random.uniform(r_act, shape=(args.num_envs, env.act_dim), minval=-1.0, maxval=1.0)
        state, obs, rewards, dones, info = env.step(state, actions)

        steps_done += args.num_envs
        if steps_done % (args.num_envs * 10) == 0:
            elapsed = time.time() - start_train
            fps = steps_done / max(1e-6, elapsed)
            mean_rew = float(jnp.mean(rewards))
            wins = int(jnp.sum(info["win"]))
            cols = int(jnp.sum(info["collision"]))
            print(f"   Step: {steps_done:>8,} | FPS: {fps:>8.0f} | TB Reward: {mean_rew:>+6.2f} | Thắng: {wins:>4} | Đâm: {cols:>4}")

    print("=" * 80)
    print(f"🎉 HOÀN THÀNH {args.timesteps:,} STEPS sau {time.time() - start_train:.2f}s!")
    print(f"🚀 Tốc độ trung bình đạt: {args.timesteps / (time.time() - start_train):.0f} FPS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
